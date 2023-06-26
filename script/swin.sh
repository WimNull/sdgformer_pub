torchrun --nproc_per_node=8 train.py --model swin_t --epochs 300 --batch_size 128 --opt adamw --lr 0.001 --weight_decay 0.05\
 --norm_weight_decay 0.0  --bias_weight_decay 0.0 --lr_scheduler cosineannealinglr\
 --lr_min 0.00001 --lr_warmup_method linear  --lr_warmup_epochs 20 --lr_warmup_decay 0.01 --amp --label_smoothing 0.1\
 --mixup_alpha 0.8 --clip_grad_norm 5.0 --cutmix_alpha 1.0 --random_erase 0.25 --interpolation bicubic --auto_augment ta_wide --val_resize_size 224