set -x
torchrun --nproc_per_node 8 train.py --model swin_t --epochs 120 --batch_size 128 --opt adamw --lr 0.001 --weight_decay 0.05\
 --lr_scheduler cosineannealinglr --lr_min 0.00001 --lr_warmup_method linear --lr_warmup_epochs 5 --lr_warmup_decay 0.01 --amp\
 --label_smoothing 0.1 --clip_grad_norm 5.0 --interpolation bicubic --auto_augment ta_wide $*