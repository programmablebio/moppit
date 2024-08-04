import random
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pepmlm import generate_peptide, compute_pseudo_perplexity
from predict_motifs import calculate_score, PeptideModel
from argparse import ArgumentParser
import numpy as np
import torch
import os

AA = 'ARNDCEQGHILKMFPSTWYV'


def generate_random_seq(length):
    global AA
    s = ''
    for i in range(length):
        s += random.choice(AA)
    return s

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


def cal_score(binder_seq, protein_seq, motif, model, args):
    prediction, threshold = calculate_score(protein_seq, binder_seq, model, args)
    binding_score = 0
    score = 0
    for pos in motif:
        if prediction[pos] < 0.5:
            score += 1

    for i in range(len(prediction)):
        if i not in motif and prediction[i] >= 0.5:
            score += 0.5

    return score


class Binder(object):
    def __init__(self, binder_seq, model, pepmlm, tokenizer, args):
        self.binder_seq = binder_seq
        self.protein_seq = args.protein_seq
        self.motif = parse_motif(args.motif.strip('[]'))
        self.model = model
        self.args = args
        self.pepmlm = pepmlm
        self.tokenizer = tokenizer
        self.score = cal_score(binder_seq, self.protein_seq, self.motif, self.model, self.args)
        self.ppl = compute_pseudo_perplexity(self.pepmlm, self.tokenizer, self.protein_seq, self.binder_seq)

    def mutated_aa(self):
        global AA
        mutated_aa = random.choice(AA)
        return mutated_aa

    def mutate_seq(self, binder_seq):
        position = random.randint(0, len(binder_seq) - 1)
        mutated_seq = binder_seq[:position] + self.mutated_aa() + binder_seq[position + 1:]
        return mutated_seq

    def mate(self, par2):
        assert len(par2.binder_seq) == len(self.binder_seq)

        child = ''
        for i in range(len(par2.binder_seq)):
            a1 = self.binder_seq[i]
            a2 = par2.binder_seq[i]

            prob = random.random()
            if prob < 0.45:
                child += a1
            elif prob < 0.90:
                child += a2
            else:
                child += self.mutated_aa()

        return Binder(child, self.model, self.pepmlm, self.tokenizer, args)


def main(args):
    print(parse_motif(args.motif.strip('[]')))
    random.seed(args.seed)
    binders = []
    generation = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = PeptideModel.load_from_checkpoint(args.sm,
                                              n_layers=args.n_layers,
                                              d_model=args.d_model,
                                              d_hidden=args.d_hidden,
                                              n_head=args.n_head,
                                              d_k=64,
                                              d_v=128,
                                              d_inner=64).to(device)

    tokenizer = AutoTokenizer.from_pretrained("ChatterjeeLab/PepMLM-650M")
    pepmlm = AutoModelForMaskedLM.from_pretrained("ChatterjeeLab/PepMLM-650M").to(device)

    temp_binders = []
    count = 2
    while len(temp_binders) < args.num_binders and count >= 0:
        temp_binders += generate_peptide(args.protein_seq, args.peptide_length, args.top_k, args.num_binders-len(temp_binders))['Binder'].tolist()
        temp_binders = [seq for seq in temp_binders if 'X' not in seq]
        temp_binders = list(set(temp_binders))
        count -= 1

    for n in range(0, args.num_binders-len(temp_binders)):
        temp_binders.append(generate_random_seq(args.peptide_length))
    temp_binders = list(set(temp_binders))

    print(f"Pool Size = {len(temp_binders)}")

    for binder_seq in temp_binders:
        binders.append(Binder(binder_seq, model, pepmlm, tokenizer, args))

    binders = sorted(binders, key=lambda binder: (binder.score, binder.ppl))
    for m in range(min(args.num_display, len(binders))):
        print(f"Generation: -1\tBinder: {binders[m].binder_seq}\tScore: {binders[m].score}\tPPL: {binders[m].ppl}")

    for m in range(min(args.num_display, len(binders))):
            print(f"{binders[m].binder_seq}")

    no_improvement_generations = 0
    max_tolerance = 20
    previous_score = 10000
    previous_ppl = 10000
    threshold = int(0.1*len(parse_motif(args.motif.strip('[]'))))

    print(f"Threshold: {threshold}")

    while no_improvement_generations < max_tolerance:
        # binders = sorted(binders, key=lambda binder: (binder.score, binder.ppl))
        previous_score = binders[0].score
        previous_ppl = binders[0].ppl

        # if binders[0].score <= threshold:
        #     break

        new_binders = []

        s = int((10*len(binders))/100)
        new_binders.extend(binders[:s])

        s = int((90*len(binders))/100)
        half = int(len(binders) / 2)
        for _ in range(s):
            par1 = random.choice(binders[:half])
            par2 = random.choice(binders[:half])
            new_binders.append(par1.mate(par2))

        for k in range(1, len(new_binders)):
            if new_binders[k].binder_seq == new_binders[k-1].binder_seq:
                new_seq = new_binders[k].mutate_seq(new_binders[k].binder_seq)
                new_binders[k] = Binder(new_seq, model, pepmlm, tokenizer, args)

        new_binders = sorted(new_binders, key=lambda binder: (binder.score, binder.ppl))

        for m in range(min(args.num_display, len(new_binders))):
            print(f"Generation: {generation}\tBinder: {new_binders[m].binder_seq}\tScore: {new_binders[m].score}\tPPL: {new_binders[m].ppl}")

        if new_binders[0].score < previous_score or new_binders[0].ppl < previous_ppl:
            no_improvement_generations = 0
            print(f"Generation: {generation}\tImproved!")
            # print(f"Generation: {generation}\tBinder: {new_binders[0].binder_seq}\tScore: {new_binders[0].score}\tPPL: {new_binders[0].ppl}")
        else:
            no_improvement_generations += 1
            print(f"Generation: {generation}\tNo improvement {no_improvement_generations} generations")

        for m in range(min(args.num_display, len(new_binders))):
            print(f"{new_binders[m].binder_seq}")

        binders = new_binders
        generation += 1

    print(f"Generation: {generation}\tBinder: {binders[0].binder_seq}\tScore: {binders[0].score}\tPPL: {binders[0].ppl}")



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--protein_seq', type=str, required=True)
    parser.add_argument('--peptide_length', type=int, required=True)
    parser.add_argument('--motif', type=str, required=True)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--num_binders', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("-sm", required=True, help="File containing initial params", type=str)
    parser.add_argument("-batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-n_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("-d_model", type=int, default=64, help="Dimension of model")
    parser.add_argument("-d_hidden", type=int, default=128, help="Dimension of CNN block")
    parser.add_argument("-n_head", type=int, default=6, help="Number of heads")
    parser.add_argument("-d_inner", type=int, default=64)
    parser.add_argument("--num_display", type=int, default=1)
    args = parser.parse_args()
    main(args)
