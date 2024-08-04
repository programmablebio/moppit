import pandas as pd
from Bio import SeqIO
import io
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from torch.distributions.categorical import Categorical
import numpy as np
import os
from argparse import ArgumentParser

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Load the model and tokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("ChatterjeeLab/PepMLM-650M")
model = AutoModelForMaskedLM.from_pretrained("ChatterjeeLab/PepMLM-650M").to(device)


def compute_pseudo_perplexity(model, tokenizer, protein_seq, binder_seq):
    """
    For alternative computation of PPL (in batch/matrix format), please check our GitHub repo:
    https://github.com/programmablebio/pepmlm/blob/main/scripts/generation.py
    """
    sequence = protein_seq + binder_seq
    tensor_input = tokenizer.encode(sequence, return_tensors='pt').to(model.device)
    total_loss = 0

    # Loop through each token in the binder sequence
    for i in range(-len(binder_seq)-1, -1):
        # Create a copy of the original tensor
        masked_input = tensor_input.clone()

        # Mask one token at a time
        masked_input[0, i] = tokenizer.mask_token_id
        # Create labels
        labels = torch.full(tensor_input.shape, -100).to(model.device)
        labels[0, i] = tensor_input[0, i]

        # Get model prediction and loss
        with torch.no_grad():
            outputs = model(masked_input, labels=labels)
            total_loss += outputs.loss.item()

    # Calculate the average loss
    avg_loss = total_loss / len(binder_seq)

    # Calculate pseudo perplexity
    pseudo_perplexity = np.exp(avg_loss)
    return pseudo_perplexity


def generate_peptide_for_single_sequence(protein_seq, peptide_length = 15, top_k = 3, num_binders = 4):

    peptide_length = int(peptide_length)
    top_k = int(top_k)
    num_binders = int(num_binders)

    binders_with_ppl = []

    for _ in range(num_binders):
        # Generate binder
        masked_peptide = '<mask>' * peptide_length
        input_sequence = protein_seq + masked_peptide
        inputs = tokenizer(input_sequence, return_tensors="pt").to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits
        mask_token_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        logits_at_masks = logits[0, mask_token_indices]

        # Apply top-k sampling
        top_k_logits, top_k_indices = logits_at_masks.topk(top_k, dim=-1)
        probabilities = torch.nn.functional.softmax(top_k_logits, dim=-1)
        predicted_indices = Categorical(probabilities).sample()
        predicted_token_ids = top_k_indices.gather(-1, predicted_indices.unsqueeze(-1)).squeeze(-1)

        generated_binder = tokenizer.decode(predicted_token_ids, skip_special_tokens=True).replace(' ', '')

        # Compute PPL for the generated binder
        ppl_value = compute_pseudo_perplexity(model, tokenizer, protein_seq, generated_binder)

        # Add the generated binder and its PPL to the results list
        binders_with_ppl.append([generated_binder, ppl_value])

    return binders_with_ppl


def generate_peptide(input_seqs, peptide_length=15, top_k=3, num_binders=4):
    if isinstance(input_seqs, str):  # Single sequence
        binders = generate_peptide_for_single_sequence(input_seqs, peptide_length, top_k, num_binders)
        return pd.DataFrame(binders, columns=['Binder', 'Pseudo Perplexity'])

    elif isinstance(input_seqs, list):  # List of sequences
        results = []
        for seq in input_seqs:
            binders = generate_peptide_for_single_sequence(seq, peptide_length, top_k, num_binders)
            for binder, ppl in binders:
                results.append([seq, binder, ppl])
        return pd.DataFrame(results, columns=['Input Sequence', 'Binder', 'Pseudo Perplexity'])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-s', type=str, required=True)
    args = parser.parse_args()

    peptide_length = 13
    top_k = 2
    num_binders = 50
    peptide_df = generate_peptide(args.s, peptide_length, top_k, num_binders)
    peptide_df = peptide_df.drop_duplicates(subset='Binder')
    peptide_df = peptide_df.sort_values(by='Pseudo Perplexity')
    print(peptide_df)
