import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from argparse import ArgumentParser
import os
import torch.distributed as dist

from models import *  # Import your model and other necessary classes/functions here


def collate_fn(batch):
    # Unpack the batch
    anchors = []
    positives = []
    # negatives = []
    binding_sites = []

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    for b in batch:
        anchors.append(b['anchors'])
        positives.append(b['positives'])
        # negatives.append(b['negatives'])
        binding_sites.append(b['binding_site'])

    # Collate the tensors using torch's pad_sequence
    anchor_input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.Tensor(item['input_ids']).squeeze(0) for item in anchors], batch_first=True, padding_value=tokenizer.pad_token_id)
    anchor_attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.Tensor(item['attention_mask']).squeeze(0) for item in anchors], batch_first=True, padding_value=0)

    positive_input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.Tensor(item['input_ids']).squeeze(0) for item in positives], batch_first=True, padding_value=tokenizer.pad_token_id)
    positive_attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.Tensor(item['attention_mask']).squeeze(0) for item in positives], batch_first=True, padding_value=0)

    n, max_length = anchor_input_ids.shape[0], anchor_input_ids.shape[1]
    site = torch.zeros(n, max_length)
    for i in range(len(binding_sites)):
        binding_site = binding_sites[i]
        site[i, binding_site] = 1

    # Return the collated batch
    return {
        'anchor_input_ids': anchor_input_ids.int(),
        'anchor_attention_mask': anchor_attention_mask.int(),
        'positive_input_ids': positive_input_ids.int(),
        'positive_attention_mask': positive_attention_mask.int(),
        # 'negative_input_ids': negative_input_ids.int(),
        # 'negative_attention_mask': negative_attention_mask.int(),
        'binding_site': site
    }


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size: int = 128):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def test_dataloader(self):
        # test_dataset = load_from_disk('/home/tc415/muPPIt/dataset/test/correct_test_dataset_drop_500')
        test_dataset = load_from_disk('/home/tc415/muPPIt/dataset/test/correct_pepnn_biolip_test')
        return DataLoader(test_dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=8, pin_memory=True)


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
        # self.class_weights = torch.tensor([3.000471363174231, 0.5999811490272925])  # binding_site weights, non-bidning site weights
        self.class_weights = torch.tensor([7.478236497659688, 0.5358256702941844])
        self.kl_weight = kl_weight

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

    def test_step(self, batch, batch_idx):
        target_tokens = {'input_ids': batch['anchor_input_ids'].to(self.device),
                         'attention_mask': batch['anchor_attention_mask'].to(self.device)}
        binder_tokens = {'input_ids': batch['positive_input_ids'].to(self.device),
                         'attention_mask': batch['positive_attention_mask'].to(self.device)}
        binding_site = batch['binding_site'].to(self.device)
        mask = target_tokens['attention_mask']

        outputs_nodes = self.forward(binder_tokens, target_tokens).squeeze(-1)

        weight = self.class_weights[0] * binding_site + self.class_weights[1] * (1 - binding_site)
        bce_loss = F.binary_cross_entropy_with_logits(outputs_nodes, binding_site, weight=weight, reduction='none')
        masked_bce_loss = bce_loss * mask
        mean_bce_loss = masked_bce_loss.sum() / mask.sum()

        kl_loss = self.compute_kl_loss(outputs_nodes, binding_site, mask)

        mean_loss = mean_bce_loss + self.kl_weight * kl_loss

        sigmoid_outputs = torch.sigmoid(outputs_nodes)
        total = mask.sum()

        predict = (sigmoid_outputs >= 0.5).float()
        correct = ((predict == binding_site) * mask).sum()
        accuracy = correct / total

        outputs_nodes_flat = sigmoid_outputs[mask.bool()].float().cpu().detach().numpy().flatten()
        binding_site_flat = binding_site[mask.bool()].float().cpu().detach().numpy().flatten()
        predictions_flat = predict[mask.bool()].float().cpu().detach().numpy().flatten()

        auc = roc_auc_score(binding_site_flat, outputs_nodes_flat)
        f1 = f1_score(binding_site_flat, predictions_flat)
        mcc = matthews_corrcoef(binding_site_flat, predictions_flat)

        self.log('test_loss', mean_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_bce_loss', mean_bce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_auc', auc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_f1', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_mcc', mcc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def compute_kl_loss(self, outputs, targets, mask):
        log_probs = F.log_softmax(outputs, dim=-1)
        target_probs = targets.float()
        kl_loss = F.kl_div(log_probs, target_probs, reduction='none')
        masked_kl_loss = kl_loss * mask
        mean_kl_loss = masked_kl_loss.sum() / mask.sum()
        return mean_kl_loss

def main():
    parser = ArgumentParser()
    parser.add_argument("-sm", required=True, help="File containing initial params", type=str)
    parser.add_argument("-batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-n_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("-d_model", type=int, default=64, help="Dimension of model")
    parser.add_argument("-n_head", type=int, default=6, help="Number of heads")
    parser.add_argument("-d_inner", type=int, default=64)
    parser.add_argument("-d_hidden", type=int, default=128, help="Dimension of CNN block")
    parser.add_argument("--kl_weight", type=float, default=1)
    parser.add_argument("-dropout", type=float, default=0.2)
    parser.add_argument("--max_epochs", type=int, default=15, help="Max number of epochs to train")


    args = parser.parse_args()
    print(args.sm)

    # Initialize the process group for distributed training
    dist.init_process_group(backend='nccl')

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    data_module = CustomDataModule(tokenizer, args.batch_size)

    model = PeptideModel.load_from_checkpoint(args.sm,
                                              n_layers=args.n_layers,
                                              d_model=args.d_model,
                                              d_hidden=args.d_hidden,
                                              n_head=args.n_head,
                                              d_k=64,
                                              d_v=128,
                                              d_inner=args.d_inner,
                                              dropout=args.dropout,
                                              learning_rate=args.lr,
                                              max_epochs=args.max_epochs,
                                              kl_weight=args.kl_weight)

    trainer = pl.Trainer(accelerator='gpu',
                         devices=[0,1,2,3,4,5],
                         strategy='ddp',
                         precision='bf16')

    results = trainer.test(model, datamodule=data_module)

    print(results)


if __name__ == "__main__":
    main()
