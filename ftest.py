import os,sys,time,random,math,shutil
import torch,torchvision
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from torch import nn,optim
from PIL import Image
from tqdm import tqdm
from torch.autograd.function import once_differentiable
import utils
from train import setup_seed
from ptflops import get_model_complexity_info

torch.set_printoptions(precision=2, linewidth=256)

def cleanbk():
    def subfun(dir):
        flst = os.listdir(f'{dir}/log/logtext')
        validlst = []
        for fl in flst:
            validlst.append(fl.split('.')[0])
        
        flst = os.listdir(f'{dir}/log/logtb')
        for fl in flst:
            if fl not in validlst: shutil.rmtree(f'{dir}/log/logtb/{fl}')
        flst = os.listdir(f'{dir}/backup')
        for fl in flst:
            if fl not in validlst: shutil.rmtree(f'{dir}/backup/{fl}')
    subfun('output/imagenet')
    subfun('output/mini-imagenet')

    
def main():
    ln = nn.LayerNorm(10)
    print(ln.state_dict())


  
@torch.no_grad()
def ModelSpeedTest(model, bs=128, duration=30, res=(224, 224)):
    device = 'cuda:0' if next(model.parameters()).is_cuda else 'cpu'
    total = 0
    st = time.time()
    results = []
    while True:
        x = torch.rand((bs, 3, res[0], res[1]), device=device)
        out = model(x)
        results.append(out)
        total += bs
        et = time.time()
        if et-st>duration:
            break
    torch.cuda.synchronize()
    return results, total/(et-st)


def ModelEval(model: nn.Module, bs=64, model_name='resnet50', res=(3, 224, 224)):
    macs, params = get_model_complexity_info(model, input_res=res, as_strings=True, print_per_layer_stat=False, verbose=False)
    cfps,gfps = 0, 0
    # _,cfps = ModelSpeedTest(model, bs=2, duration=10, res=res[1:])
    # _,gfps = ModelSpeedTest(model.cuda(), bs, res=res[1:])
    print(model_name, macs, params, f'cpu: {cfps:.2f}, gpu: {gfps:.2f}')


def test1():
    from models import sdgformer,cmt,metaformer
    with torch.no_grad():
        # # model = torchvision.models.maxvit_t()
        # # ModelEval(model, 16, 'maxvit_t')
        # # model = torchvision.models.swin_t()
        # # ModelEval(model, 16, 'swin_t')
        # model = sdgformer.sdgformer_tiny()
        # ModelEval(model, 64, 'sdgformer_tiny')
        # model = sdgformer.sdgformer_small()
        # ModelEval(model, 32, 'sdgformer_small')
        # model = sdgformer.sdgformer_base()
        # ModelEval(model, 32, 'sdgformer_base')
        # # from models import sdgformericme
        # # model = sdgformericme.sdgformer_tiny()
        # # ModelEval(model, 64, 'sdgformer_tiny')
        # # model = sdgformericme.sdgformer_small()
        # # ModelEval(model, 32, 'sdgformer_small')
        # # from models import sdgformerbk
        # # model = sdgformerbk.sdgformer_tiny()
        # # ModelEval(model, 64, 'sdgformer_tiny')
        # # model = sdgformerbk.sdgformer_small()
        # # ModelEval(model, 32, 'sdgformer_small')
        # # model = sdgformerbk.sdgformer_base()
        # # ModelEval(model, 16, 'sdgformer_base')
        # exit()

        model = sdgformer.sdgformer_small(num_classes=100)
        ModelEval(model, 32, 'sdgformer_small')
        model = torchvision.models.regnet_y_1_6gf(num_classes=100)
        ModelEval(model, 32, 'regnet_y_1_6gf')
        model = torchvision.models.efficientnet_b3(num_classes=100)
        ModelEval(model, 32, 'efficientnet_b3')
        model = torchvision.models.resnet50(num_classes=100)
        ModelEval(model, 16, 'resnet50')
        model = torchvision.models.convnext_tiny(num_classes=100)
        ModelEval(model, 16, 'convnext_tiny')
        model = torchvision.models.swin_t(num_classes=100)
        ModelEval(model, 16, 'swin_t')
        exit()

        model = sdgformer.sdgformer_tiny()
        ModelEval(model, 64, 'sdgformer_tiny')
        model = torchvision.models.regnet_x_800mf()
        ModelEval(model, 64, 'regnet_x_800mf')
        model = torchvision.models.regnet_y_800mf()
        ModelEval(model, 64, 'regnet_y_800mf')
        model = torchvision.models.efficientnet_b2()
        ModelEval(model, 64, 'efficientnet_b2')
        model = cmt.cmt_ti()
        ModelEval(model, 64, 'cmt_ti', res=(3, 160, 160))

        model = sdgformer.sdgformer_small()
        ModelEval(model, 32, 'sdgformer_small')
        model = torchvision.models.regnet_x_1_6gf()
        ModelEval(model, 32, 'regnet_x_1_6gf')
        model = torchvision.models.regnet_y_1_6gf()
        ModelEval(model, 32, 'regnet_y_1_6gf')
        model = torchvision.models.efficientnet_b3()
        ModelEval(model, 32, 'efficientnet_b3')
        model = cmt.cmt_xs()
        ModelEval(model, 32, 'cmt_xs', res=(3, 192, 192))
        model = metaformer.poolformerv2_s12()
        ModelEval(model, 32, 'poolformerv2_s12')

        model = sdgformer.sdgformer_base()
        ModelEval(model, 16, 'sdgformer_base')
        model = torchvision.models.efficientnet_b4()
        ModelEval(model, 16, 'efficientnet_b4')
        model = torchvision.models.resnet50()
        ModelEval(model, 16, 'resnet50')
        model = torchvision.models.convnext_tiny()
        ModelEval(model, 16, 'convnext_tiny')
        model = torchvision.models.swin_t()
        ModelEval(model, 16, 'swin_t')
        model = torchvision.models.maxvit_t()
        ModelEval(model, 16, 'maxvit_t')
    # x = torch.randn((10,))*5
    # print(x)
    # x = torch.clamp(x, -3, 3)/6
    # print(x)


def test2():
    lines = open('data/imagenet_val.txt').readlines()
    for ln in tqdm(lines):
        ln = ln.strip().split(' ')[0].split('/')
        os.makedirs(f'/temp/dataset/imagenet/val/{ln[0]}', exist_ok=True)
        shutil.move(f'/temp/dataset/{ln[1]}', f'/temp/dataset/imagenet/val/{ln[0]}/{ln[1]}')


def test3():
    from einops import rearrange
    bksz = (2,2)
    x2 = torch.linspace(1, 16, 16).view(1, 1, 4, 4)
    print(x2)
    gx2 = rearrange(x2, 'b c (h p1) (w p2) -> (b c) (h w) p1 p2', h=bksz[0], w=bksz[1])
    print(gx2)


def test4():
    model = torchvision.models.resnet18()
    print(list(model.modules()))
    exit()
    from PIL import Image
    from torchvision.transforms.functional import InterpolationMode
    import presets
    from models import sdgformer
    val_resize_size,crop_size = 224,224
    interpolation = InterpolationMode('bicubic')
    preprocessing = presets.ClassificationPresetEval(
        crop_size=crop_size, resize_size=val_resize_size, interpolation=interpolation
    )
    def readimg(fp):
        img = Image.open(fp).convert("RGB")
        img = preprocessing(img)
        return img
    imgs = torch.stack([readimg('data/n01532829/n0153282900000317.jpg'), readimg('data/n02108089/n0210808900001095.jpg')])
    print(imgs.shape)
    model = sdgformer.sdgformer_small(num_classes=100)
    chechpoints = torch.load('output/mini-imagenet/checkpoints/sdgformer_small-85.23/bestacc_model.pth')
    model.load_state_dict(chechpoints['model'])
    out = model(imgs)
    print(torch.argmax(out, dim=-1))
    



if __name__=='__main__':
    funname = 'main'
    if len(sys.argv)>1:
        funname = sys.argv[1]
        sys.argv.remove(funname)
    globals()[funname]()