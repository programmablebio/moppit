export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

torchrun --nproc_per_node=6 test_bindevaluator.py \
-sm './model_path/finetuned_BindEvaluator.ckpt' \ # Checkpoints are in the huggingface repo
-n_layers 8 \
-d_model 128 \
-d_hidden 128 \
-n_head 8 \
-d_inner 64 \
-batch_size 32 \
--kl_weight 1
