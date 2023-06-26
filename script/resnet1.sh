set -x

torchrun --nproc_per_node 8 train.py --auto_augment eig --batch_size 128\
 --model resnet50  --epochs 120 --amp --lr 0.4 --lr_min 1e-4 --wd 0.00005\
 --lr_scheduler cosineannealinglr --lr_warmup_method linear --lr_warmup_epochs 5 --lr_warmup_decay 1e-2 $*

