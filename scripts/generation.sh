export CUDA_VISBILE_DEVICES=7

python moppit.py \
-sm '/home/tc415/moPPIt/moppit/model_path/finetune_bindevaluator_0/model-epoch=29-val_mcc=0.60-val_loss=0.51.ckpt' \
-n_layers 8 \
-d_model 128 \
-d_hidden 128 \
-n_head 8 \
-d_inner 64 \
--protein_seq 'IVEGSDAEIGMSPWQVMLFRKSPQELLCGASLISDRWVLTAAHCLLYPPWDKNFTENDLLVRIGKHSRTRYERNIEKISMLEKIYIHPRYNWRENLDRDIALMKLKKPVAFSDYIHPVCLPDRETAASLLQAGYKGRVTGWGNLKETGQPSVLQVVNLPIVERPVCKDSTRIRITDNMFCAGYKPDEGKRGDACEGDSGGPFVMKSPFNNRWYQMGIVSWGEGCDRDGKYGFYTHVFRLKKWIQKVIDQFGE' \
--peptide_length 11 \
--motif '[18,23,59,67,68,69,70,76,77]' \ # amino acid indices start from 0 on a protein sequence
--top_k 3 \
--num_binders 50 \
--num_display 10
