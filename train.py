import datetime
import os
import time
import warnings
import random
import math
import copy

import presets
import torch
import torchvision
import transforms
import utils
import numpy as np
from torch import nn
from torch.utils.data.dataloader import default_collate

from ptflops import get_model_complexity_info

from torchvision.transforms.functional import InterpolationMode


from tensorboardX import SummaryWriter
from lr_scheduler import LinearCosineLR

curdir = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def get_grad_norm(model:nn.Module):
    maxg,ming,avg = -10000,10000,0
    numel = 0
    for na,p in model.named_parameters():
        if p.grad is not None:
            maxg = max(maxg, torch.max(p.grad))
            ming = min(ming, torch.min(p.grad))
            numel += p.grad.numel()
            avg += torch.mean(p.grad.abs().sum())
    avg = avg/numel
    
    # global gradwriter,ssidx
    # gradwriter.add_scalar('grad maxg', maxg, ssidx)
    # gradwriter.add_scalar('grad ming', ming, ssidx)
    # gradwriter.add_scalar('grad avg', avg, ssidx)
    # ssidx += 1


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == math.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm




def train_one_epoch(model:nn.Module, criterion, optimizer, data_loader, device, epoch, args, lr_scheduler=None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.7f}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value:.1f}"))
    metric_logger.add_meter("acc1", utils.SmoothedValue(window_size=100, fmt="{median:.2f} ({global_avg:.2f})"))
    metric_logger.add_meter("acc5", utils.SmoothedValue(window_size=100, fmt="{median:.2f} ({global_avg:.2f})"))

    header = f"Epoch-{epoch}:"
    freq = max(len(data_loader)//args.print_freq, 1)
    for idx, (image, target) in enumerate(metric_logger.log_every(data_loader, freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(image)
            loss = criterion(output, target)
        skip = False
        if torch.isnan(output).any(): 
            print(f'epoch-{epoch}-{idx}: output value is nan') 
            skip=True
        if torch.isnan(loss): 
            print(f'epoch-{epoch}-{idx}: loss value is nan')
            skip=True
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # get_grad_norm(model)
        if skip:
            # exit()
            continue
        if args.clip_grad_norm is not None:
            # we should unscale the gradients of optimizer's assigned params if do gradient clipping
            scaler.unscale_(optimizer)
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            metric_logger.otstr = f'  gradnorm: {total_norm.item():.4f}'
        elif idx%freq==0:
            total_norm = ampscaler_get_grad_norm(model.parameters())
            metric_logger.otstr = f'  gradnorm: {total_norm.item():.4f}'
        scaler.step(optimizer)
        scaler.update()
        if args.lr_scheduler == "linearcosine": lr_scheduler.step()
        
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        metric_logger.meters["loss"].update(loss.item(), n=batch_size)
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
    torch.cuda.empty_cache()
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg

@torch.no_grad()
def evaluate(model, criterion, data_loader, device, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("acc1", utils.SmoothedValue(window_size=100, fmt="{median:.2f} ({global_avg:.2f})"))
    metric_logger.add_meter("acc5", utils.SmoothedValue(window_size=100, fmt="{median:.2f} ({global_avg:.2f})"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=100, fmt="{value:.1f}"))
    header = f"Test: {log_suffix}"
    freq = max(len(data_loader)//4, 1)
    num_processed_samples = 0
    for image, target in metric_logger.log_every(data_loader, freq, header):
        start_time = time.time()
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(image)
        loss = criterion(output, target)
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        # FIXME need to take into account that the datasets
        # could have been padded in distributed setup
        batch_size = image.shape[0]
        metric_logger.meters["loss"].update(loss.item(), n=batch_size)
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        num_processed_samples += batch_size
        # break
    # gather the stats from all processes
    torch.cuda.empty_cache()
    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    metric_logger.synchronize_between_processes()
    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg


def load_data(args):
    # Data loading code
    train_dir,val_dir = os.path.join(args.data_path, "train"), os.path.join(args.data_path, "val")
    # print(train_dir, val_dir)
    print("Loading data")
    val_resize_size, crop_size = args.val_resize_size, args.crop_size
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    ra_magnitude = args.ra_magnitude
    augmix_severity = args.augmix_severity
    dataset = torchvision.datasets.ImageFolder(
        train_dir,
        presets.ClassificationPresetTrain(
            crop_size=crop_size,
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
            ra_magnitude=ra_magnitude,
            augmix_severity=augmix_severity,
        ),
    )
    print("Took", time.time() - st)

    print("Loading validation data")
    preprocessing = presets.ClassificationPresetEval(
        crop_size=crop_size, resize_size=val_resize_size, interpolation=interpolation
    )
    dataset_test = torchvision.datasets.ImageFolder(val_dir, preprocessing)

    return dataset, dataset_test



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True # very slow
    


def main(args):
    # setup_seed(42)
    torch.backends.cudnn.benchmark = True
    args.dstype = os.path.basename(args.data_path)
    args.output_dir = os.path.join(args.output_dir, args.dstype)
    os.makedirs(args.output_dir, exist_ok=True)
    utils.init_distributed_mode(args)
    if args.distributed:
        args.device = 'cuda'
    device = torch.device(args.device)
    dataset,dataset_test = load_data(args)
    num_classes = len(dataset.classes)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    collate_fn = None
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))
    
    tr_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, sampler=train_sampler, collate_fn=collate_fn, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.workers, sampler=test_sampler, pin_memory=True)

    print("Creating model")
    from models import sdgformer,metaformer,cmt
    if args.model in sdgformer.__dict__:
        model = sdgformer.__dict__[args.model](num_classes=num_classes)
    elif args.model in metaformer.__dict__:
        model = metaformer.__dict__[args.model](num_classes=num_classes)
    elif args.model in cmt.__dict__:
        model = cmt.__dict__[args.model](num_classes=num_classes, img_size=args.crop_size)
    elif 'swinv3' in args.model:
        from models import swinv3
        model = swinv3.__dict__[args.model](num_classes=num_classes)
    else:
        # , weights='pretrained' if args.stage=='test' else None
        model = torchvision.models.__dict__[args.model](num_classes=num_classes)
    
    
    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")
    
    
    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min)
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    elif args.lr_scheduler == "linearcosine":
        main_lr_scheduler = LinearCosineLR(optimizer, args.lr, args.epochs, len(tr_loader), min_lr=args.lr_min, warmup_epoch=args.lr_warmup_epochs, warmup_decay=args.lr_warmup_decay)
    elif args.lr_scheduler == "multistep":
        main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    else:
        raise RuntimeError(f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR are supported.")
    
    if args.lr_warmup_epochs > 0 and args.lr_scheduler not in ["linearcosine", "multistep"]:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs)
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs)
        else:
            raise RuntimeError(f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported.")
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs])
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    best_loss,best_acc = 10000,0
    if args.stage in ['resume', 'test']:
        checkpoint = torch.load(args.weight_path, map_location=device)
        if 'model' in checkpoint: model_without_ddp.load_state_dict(checkpoint["model"])
        else: model_without_ddp.load_state_dict(checkpoint)
        if  args.stage=='resume' and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
        best_loss,best_acc = evaluate(model, criterion, val_loader, device=device)
        if args.stage=='test': return

    prefix = f'{args.model}'
    if utils.is_main_process():
        tsuffix = prefix+'_'+time.strftime("%m%d%H%M", time.localtime())
        global writer,widx
        widx = 0
        writer = SummaryWriter(os.path.join(args.output_dir, 'log/logtb', tsuffix))
        bkpath = os.path.join(args.output_dir, 'backup', tsuffix)
        os.makedirs(bkpath, exist_ok=True)
        os.system(f'cp {curdir}/models/*.py {bkpath}/')
        logfl = os.path.join(args.output_dir, 'log/logtext', tsuffix+f'{args.log_postfix}.log')
        logger = utils.get_logger('base', logfl, stdout=True)
        logger.info(f"Creating model: {args.model}")
        message = '\n'.join([f'{k:<20}: {v}' for k, v in vars(args).items()])
        logger.info(message)
        with torch.no_grad():
            macs, params = get_model_complexity_info(model, (3, args.crop_size, args.crop_size), as_strings=True, print_per_layer_stat=False, verbose=False)
        logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))
        logger.info(f"Creating optimizer: {optimizer}")
        
    print("Start training")
    patience_time = 0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        st = time.time()
        nowlr = optimizer.param_groups[0]['lr']
        tr_loss,tr_acc = train_one_epoch(model, criterion, optimizer, tr_loader, device, epoch, args, lr_scheduler, scaler)
        if args.lr_scheduler != "linearcosine": lr_scheduler.step()
        val_loss,val_acc = evaluate(model, criterion, val_loader, device=device)
        
        if best_acc<val_acc: 
            patience_time = 0
            best_acc = val_acc
        if best_loss>val_loss: 
            patience_time = 0
            best_loss = val_loss
        else: patience_time += 1
        if not utils.is_main_process(): continue
        writer.add_scalars('loss', {'train':tr_loss, 'val':val_loss}, global_step=epoch)
        writer.add_scalars('acc', {'train':tr_acc, 'val':val_acc}, global_step=epoch)
        
        logger.info(f'epoch--{epoch} finish: time--{time.time()-st:.2f}, lr--{nowlr:.7f}, loss-{tr_loss:.4f} {val_loss:.4f}, acc-{tr_acc:.2f} {val_acc:.2f} best-{best_loss:.4f} {best_acc:.2f}')
        
        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
            'meter':[val_loss, val_acc],
        }
        if best_acc==val_acc:
            utils.save_checkpoint(checkpoint, filename=os.path.join(args.output_dir, 'checkpoints', prefix, f'bestacc_model.pth'))
        if best_acc==val_acc:
            utils.save_checkpoint(checkpoint, filename=os.path.join(args.output_dir, 'checkpoints', prefix, f'best_model.pth'))
        utils.save_checkpoint(checkpoint, filename=os.path.join(args.output_dir, 'checkpoints', prefix, f'model.pth'))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    parser.add_argument("--data_path", default="/temp/dataset/imagenet", type=str, help="dataset path")
    parser.add_argument("--output_dir", default="output", type=str, help="path to save outputs")
    parser.add_argument("--log_postfix", default="", type=str, help="path to save outputs")
    parser.add_argument("--model", default="sdgformer_tiny", type=str, help="model name")
    parser.add_argument("--device", default="cuda:0", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-bs", "--batch_size", default=128, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--epochs", default=120, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=8, type=int, metavar="N", help="number of data loading workers (default: 16)")
    parser.add_argument("--print_freq", default=10, type=int, help="print frequency")

    parser.add_argument("--label_smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing")
    parser.add_argument("--val_resize_size", default=256, type=int, help="the resize size used for validation (default: 256)")
    parser.add_argument("--crop_size", default=224, type=int, help="the random crop size used for training (default: 224)")
    parser.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)")
    parser.add_argument("--ra_magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix_severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--auto_augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--random_erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")
    parser.add_argument("--mixup_alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix_alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")

    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight_decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay",)
    parser.add_argument("--norm_weight_decay", default=None, type=float, help="weight decay for Normalization layers (default: None, same value as --wd)",)
    parser.add_argument("--bias_weight_decay", default=None, type=float, help="weight decay for bias parameters of all layers (default: None, same value as --wd)",)
    parser.add_argument("--clip_grad_norm", default=10.0, type=float, help="the maximum gradient norm (default None)")

    parser.add_argument("--lr_scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr_warmup_epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr_warmup_method", default="constant", type=str, help="the warmup method (default: constant)")
    parser.add_argument("--lr_warmup_decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr_step_size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr_gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr_min", default=1e-6, type=float, help="minimum lr of lr schedule (default: 0.0)")
    
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--weight_path", '-wp', default="", type=str, help="path of checkpoint")
    # train scheme parameters
    parser.add_argument('--stage', '-st', type=str, default='train', help='(train, test, resume)')
  

    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", type=str, help="url used to set up distributed training")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
