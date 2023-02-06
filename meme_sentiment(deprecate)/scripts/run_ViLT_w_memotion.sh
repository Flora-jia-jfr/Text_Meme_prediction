python3 run_ViLT.py \
  --batch_size=256 \
  --n_epoch=50 \
  --n_sentiment=5 \
  --save_every=2 \
  --output_dir=output/ViLT/ \
  --val_every=5 \
  --dataset_filepath=data/memotion_dataset_7k/memotion_dataset_7k.pkl \
#  --checkpoint=ViLT_model_w_memotion.ckpt
