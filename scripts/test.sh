export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

torchrun --nproc_per_node=6 test_bindevaluator.py \
-sm '/home/tc415/moPPIt/moppit/model_path/finetune_bindevaluator_0/model-epoch=30-val_mcc=0.60-val_loss=0.51.ckpt' \
-n_layers 8 \
-d_model 128 \
-d_hidden 128 \
-n_head 8 \
-d_inner 64 \
-batch_size 32 \
--kl_weight 1
