import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, accuracy_score
from argparse import ArgumentParser
import os
import torch.distributed as dist
import pandas as pd
from models import * 


def parse_motif(motif: str) -> list:
    parts = motif.split(',')
    result = []

    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))

    return result


class PeptideModel(pl.LightningModule):
    def __init__(self, n_layers, d_model, d_hidden, n_head,
                 d_k, d_v, d_inner, dropout=0.2,
                 learning_rate=0.00001, max_epochs=15, kl_weight=1):
        super(PeptideModel, self).__init__()

        self.esm_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        # freeze all the esm_model parameters
        for param in self.esm_model.parameters():
            param.requires_grad = False

        self.repeated_module = RepeatedModule3(n_layers, d_model, d_hidden,
                                               n_head, d_k, d_v, d_inner, dropout=dropout)

        self.final_attention_layer = MultiHeadAttentionSequence(n_head, d_model,
                                                                d_k, d_v, dropout=dropout)

        self.final_ffn = FFN(d_model, d_inner, dropout=dropout)

        self.output_projection_prot = nn.Linear(d_model, 1)

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.kl_weight = kl_weight

        self.classification_threshold = nn.Parameter(torch.tensor(0.5))  # Initial threshold
        self.historical_memory = 0.9
        self.class_weights = torch.tensor([3.000471363174231, 0.5999811490272925])  # binding_site weights, non-bidning site weights

    def forward(self, binder_tokens, target_tokens):
        peptide_sequence = self.esm_model(**binder_tokens).last_hidden_state
        protein_sequence = self.esm_model(**target_tokens).last_hidden_state

        prot_enc, sequence_enc, sequence_attention_list, prot_attention_list, \
            seq_prot_attention_list, seq_prot_attention_list = self.repeated_module(peptide_sequence,
                                                                                    protein_sequence)

        prot_enc, final_prot_seq_attention = self.final_attention_layer(prot_enc, sequence_enc, sequence_enc)

        prot_enc = self.final_ffn(prot_enc)

        prot_enc = self.output_projection_prot(prot_enc)

        return prot_enc


def calculate_score(target_sequence, binder_sequence, model, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    anchor_tokens = tokenizer(target_sequence, return_tensors='pt', padding=True, truncation=True, max_length=40000)
    positive_tokens = tokenizer(binder_sequence, return_tensors='pt', padding=True, truncation=True, max_length=40000)

    anchor_tokens['attention_mask'][0][0] = 0
    anchor_tokens['attention_mask'][0][-1] = 0
    positive_tokens['attention_mask'][0][0] = 0
    positive_tokens['attention_mask'][0][-1] = 0

    target_tokens = {'input_ids': anchor_tokens["input_ids"].to(device),
                     'attention_mask': anchor_tokens["attention_mask"].to(device)}
    binder_tokens = {'input_ids': positive_tokens['input_ids'].to(device),
                     'attention_mask': positive_tokens['attention_mask'].to(device)}

    model.eval()

    prediction = model(binder_tokens, target_tokens).squeeze(-1)[0][1:-1]
    prediction = torch.sigmoid(prediction)

    return prediction, model.classification_threshold


def compute_metrics(true_residues, predicted_residues, length):
    # Initialize the true and predicted lists with 0
    true_list = [0] * length
    predicted_list = [0] * length

    # Set the values to 1 based on the provided lists
    for index in true_residues:
        true_list[index] = 1
    for index in predicted_residues:
        predicted_list[index] = 1

    # Compute the metrics
    accuracy = accuracy_score(true_list, predicted_list)
    f1 = f1_score(true_list, predicted_list)
    mcc = matthews_corrcoef(true_list, predicted_list)

    return accuracy, f1, mcc


def main():
    parser = ArgumentParser()
    parser.add_argument("-sm", required=True, help="File containing initial params", type=str)
    parser.add_argument("-batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-n_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("-d_model", type=int, default=64, help="Dimension of model")
    parser.add_argument("-d_hidden", type=int, default=128, help="Dimension of CNN block")
    parser.add_argument("-n_head", type=int, default=6, help="Number of heads")
    parser.add_argument("-d_inner", type=int, default=64)
    parser.add_argument("-target", type=str)
    parser.add_argument("-binder", type=str)
    parser.add_argument("-gt", type=str, default=None, help="Ground Truth Binding Motifs")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = PeptideModel.load_from_checkpoint(args.sm,
                                            n_layers=args.n_layers,
                                            d_model=args.d_model,
                                            d_hidden=args.d_hidden,
                                            n_head=args.n_head,
                                            d_k=64,
                                            d_v=128,
                                            d_inner=64).to(device)

    prediction, _ = calculate_score(args.target, args.binder, model, args)
    print(prediction)

    binding_site = []
    for i in range(len(prediction)):
        if prediction[i] >= 0.5:
            binding_site.append(i)

    print("Prediction: ", binding_site)

    if args.gt is not None:
        L = len(args.target)
        # print(L)
        gt = parse_motif(args.gt)
        print("Ground Truth: ", gt)

        acc, f1, mcc = compute_metrics(gt, binding_site, L)
        print(f"Accuracy={acc}\tF1={f1}\tMCC={mcc}")

    print("Prediction Logits: ", prediction[binding_site])


if __name__ == "__main__":
    main()
