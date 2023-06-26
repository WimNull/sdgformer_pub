set -x

torchrun --nproc_per_node 8 train.py --batch_size 128\
 --model regnet_x_800mf --epochs 100 --amp --lr 0.8 --lr_min 1e-5 --wd 0.00005\
 --lr_scheduler cosineannealinglr --lr_warmup_method linear --lr_warmup_epochs 5 --lr_warmup_decay 0.1 $*

#  torchrun --nproc_per_node 8 train.py --batch_size 128\
#  --model regnet_x_800mf --epochs 100 --amp --lr 0.8 --lr_min 1e-4 --wd 0.00005\
#  --lr_scheduler linearcosine --lr_warmup_method linear --lr_warmup_epochs 5 --lr_warmup_decay 1e-2 $*