CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  --gt_dir ./nyu_output/depths --rgb_dir ./nyu_output/images \
  --depth_scale 1000 --no-depth_is_z --dmax 10 \
  --epochs 2000 --early_stop --es_metric val_mae \
  --bs 1 --accum 8 \
  --lr 1e-4 --lr_wave 1e-5 \
  --amp --amp_dtype bf16 \
  --use_mamba --no-mamba_force_fp32 \
  --train_size 320x240 \
  --save_dir runs/nyu_ddp_mamba_bs1a8 --clean_save
