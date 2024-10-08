# moPPIt: De novo generation of motif-specific binders with protein language models

![image/png](https://cdn-uploads.huggingface.co/production/uploads/649ef40be56dc456b7a36649/QuO8YvTMdCJtKgg5KIEUt.png)


![image/png](https://cdn-uploads.huggingface.co/production/uploads/649ef40be56dc456b7a36649/JWMZZy9VG2ldAPONQz5Z_.png)

Motif-specific targeting of protein-protein interactions (PPIs) is crucial for developing highly selective therapeutics, yet remains a significant challenge in drug discovery. The ability to precisely target specific motifs or epitopes within these proteins is essential for modulating their function while minimizing off-target effects, but current methods struggle to achieve this specificity without structural information. In this work, we introduce a motif-specific PPI targeting algorithm, moPPIt, for de novo generation of motif-specific peptide binders using only protein sequence information. At the core of moPPIt is BindEvaluator, a transformer-based model that interpolates protein language model embeddings via a series of multi-headed self-attention blocks, with a key focus on local interaction changes. Trained on over 510,000 PPI-hotspot triplets from the PPIRef dataset, BindEvaluator accurately predicts binding hotspots between two proteins with a test AUC > 0.94, improving to AUC > 0.96 when fine-tuned on peptide-protein pairs. By combining BindEvaluator with our PepMLM peptide generator and genetic algorithm-based optimization, moPPIt generates peptides that bind specifically to user-defined motifs on target proteins.

---
**Colab Notebook for Binding Site Prediction and Motif-Specific Binder Generation**: https://colab.research.google.com/drive/1SL3H_vI1y6qccce3vLOo0W2EpxIF4Xik?usp=sharing

**Colab Notebook for PeptiDerive**: https://colab.research.google.com/drive/1aCODZ-WRwhxr-u8nEB6ZrdrhIOTz7-UF?usp=sharing

---

# 0. Conda Environment Preparation

```
conda env create -f environment.yml

conda activate moppit
```

# 1. Dataset Preparation

Pre-training dataset: `dataset/pretrain_dataset.csv`

Fine-tuning dataset: `dataset/finetune_dataset.csv`

To accelerate training and fine-tuning, datasets need to be processed  into HUggingFace Dataset in advance.

Before pre-training, run:
```
python dataset/pretrain_prebatching.py -dataset_pth dataset/pretrain_dataset.csv -output_dir dataset
```

Before fine-tuning, run:
```
python dataset/pretrain_prebatching.py -dataset_pth dataset/finetune_dataset.csv -output_dir dataset
```

The processed datasets will be saved in `output_dir` 

# 2. Model Training and Fine-tuning

To train BindEvaluator with dilated CNN modules, run `scripts/train.sh`

To fine-tune the pre-trained BindEvaluator, run `scripts/finetune.sh`

To test the performance of BindEvaluator, run `scripts/test.sh`

Ensure you adjust the hyper-parameters according to your specific requirements.

# 3. Binding site prediction

Protein-protein interaction binding sites can be predicted using the pre-trained BindEvaluator (`model_path/pretrained_BindEvaluator.ckpt`)

Peptide-protein interaction binding sites can be predicted using the fine-tuned BindEvaluator (`model_path/finetuned_BindEvaluator.ckpt`)

We provide an example script to use BindEvaluator to predict binding sites (`scripts/predict.sh`)
``` txt
usage: python predict_motifs.py -sm MODEL_PATH -target Target -binder Binder
                        [-gt] [-n_layers] [-d_model] [-d_hidden] [-n_head] [-d_inner]

arguments:
  -sm         The path to the BindEvaluator model weights
  -target     Target protein sequence
  -binder     Binder sequence
  -gt         Ground Truth binding motifs if known. If specified, the prediction accuracy, F1 score, and MCC score will be calculated.
  -n_layers, -d_model, -d_hidden, -n_head, -d_inner   Model parameters for BindEvaluator, which should be the same as the model specified in -sm used
```

# 4. Motif-Specific Binder Generation

We provide an example script to use moPPIt for generating motif-specific binders based on a target sequence (`scripts/generation.sh`)
``` txt
usage: python moppit.py -sm MODEL_PATH --protein_seq PROTEIN --peptide_length LENGTH --motif MOTIF
                        [--top_k] [--num_binders] [--num_display] [-max_iterations] [-n_layers] [-d_model] [-d_hidden] [-n_head] [-d_inner]

arguments:
  -sm               The path to the BindEvaluator model weights
  --protein_seq     Target protein sequence
  --peptide_length  The length for the generated binders
  --motif           The binding motifs
  --top_k           Sampling argument for each position used in PepMLM
  --num_binders     The size of the pool of candidates in the genetic algorithm
  --num_display     The number of top binders to display after each generation
  -max_iterations   Maximum no improvement iterations
  -n_layers, -d_model, -d_hidden, -n_head, -d_inner   Model parameters for BindEvaluator, which should be the same as the model specified in -sm used
```

Please sign the academic-only, non-commercial license to access moPPIt. 

## Repository Authors

[Tong Chen](mailto:tong.chen2@duke.edu), Visiting Student at Duke University <br>
[Pranam Chatterjee](mailto:pranam.chatterjee@duke.edu), Assistant Professor at Duke University 

Reach out to us with any questions!
