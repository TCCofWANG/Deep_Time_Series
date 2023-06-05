export CUDA_VISIBLE_DEVICES=0

#cd ..


# exchange
python -u main.py \
  --train True \
  --resume False \
  --loss quantile \
  --seed 1 \
  --data_name 'exchange' \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --d_mark 4 \
  --d_feature 8 \
  --c_out 8 \
  --features 'M' \
  --d_model 512 \
  --d_ff 1024 \
  --lr 0.001 \
  --batch_size 8 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 2 \

