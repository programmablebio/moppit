export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

torchrun --nproc_per_node=6 --master_port=25142 finetune_bindevaluator.py -o finetune_bindevaluator_0 \
-sm './model_path/pretrained_BindEvaluator.ckpt' \
-lr 1e-4 \
-n_layers 8 \
-d_model 128 \
-d_hidden 128 \
-n_head 8 \
-d_inner 64 \
-batch_size 32 \
--max_epochs 50 \
-dropout 0.5 \
--grad_clip 1 \
--kl_weight 1
