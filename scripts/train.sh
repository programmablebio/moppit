export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

torchrun --nproc_per_node=6 train_bindevaluator.py -o train_bindevaluator_0 \
-lr 1e-3 \
-n_layers 8 \
-d_model 128 \
-d_hidden 128 \
-n_head 8 \
-d_inner 64 \
-batch_size 32 \
--max_epochs 30 \
--dropout 0.3 \
--grad_clip 0.5 \
--kl_weight 0.1
