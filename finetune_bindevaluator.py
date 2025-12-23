from pytorch_lightning.strategies import DDPStrategy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_from_disk
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, \
    Timer, TQDMProgressBar, LearningRateMonitor, StochasticWeightAveraging, GradientAccumulationScheduler
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import _LRScheduler
from transformers.optimization import get_cosine_schedule_with_warmup
from argparse import ArgumentParser
import os
import uuid
import numpy as np
import torch.distributed as dist
from models import *
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import Adam, AdamW
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
import gc


os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def collate_fn(batch):
    # Unpack the batch
    anchors = []
    positives = []
    binding_sites = []

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    for b in batch:
        anchors.append(b['anchors'])
        positives.append(b['positives'])
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
        'binding_site': site
    }


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, tokenizer, batch_size: int = 128):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn,
                          num_workers=8, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=8,
                          pin_memory=True)

    def setup(self, stage=None):
        if stage == 'test' or stage is None:
            test_dataset = load_from_disk('/home/tc415/moPPIt/dataset/pep_prot_test')
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=collate_fn,
                                              num_workers=8, pin_memory=True)


class CosineAnnealingWithWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr, max_lr, min_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        super(CosineAnnealingWithWarmup, self).__init__(optimizer, last_epoch)
        print(f"SELF BASE LRS = {self.base_lrs}")

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup phase from base_lr to max_lr
            return [self.base_lr + (self.max_lr - self.base_lr) * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]

        # Cosine annealing phase from max_lr to min_lr
        progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        decayed_lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay

        return [decayed_lr for base_lr in self.base_lrs]

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
        self.class_weights = torch.tensor([7.478236497659688, 0.5358256702941844])  # binding_site weights, non-bidning site weights

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

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        lr = opt.param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

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

        # print('logging')
        self.log('bce_loss', mean_bce_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('kl_loss', kl_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_loss', mean_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return mean_loss

    def validation_step(self, batch, batch_idx):
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

        # Calculate predictions and apply mask
        sigmoid_outputs = torch.sigmoid(outputs_nodes)
        total = mask.sum()

        # self.update_class_thresholds(sigmoid_outputs, binding_site, mask)
        # self.log('threshold', self.classification_threshold, on_epoch=True)

        predict = (sigmoid_outputs >= 0.5).float()
        correct = ((predict == binding_site) * mask).sum()
        accuracy = correct / total

        # Compute AUC
        outputs_nodes_flat = sigmoid_outputs[mask.bool()].float().cpu().detach().numpy().flatten()
        binding_site_flat = binding_site[mask.bool()].float().cpu().detach().numpy().flatten()
        predictions_flat = predict[mask.bool()].float().cpu().detach().numpy().flatten()

        auc = roc_auc_score(binding_site_flat, outputs_nodes_flat)
        f1 = f1_score(binding_site_flat, predictions_flat)
        mcc = matthews_corrcoef(binding_site_flat, predictions_flat)

        self.log('val_loss', mean_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_bce_loss', mean_bce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_auc', auc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_f1', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_mcc', mcc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def compute_kl_loss(self, outputs, targets, mask):
        log_probs = F.log_softmax(outputs, dim=-1)
        target_probs = targets.float()
        kl_loss = F.kl_div(log_probs, target_probs, reduction='none')
        masked_kl_loss = kl_loss * mask
        mean_kl_loss = masked_kl_loss.sum() / mask.sum()
        return mean_kl_loss

    def configure_optimizers(self):
        print(f"MAX STEPS = {self.max_epochs}")
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.95))
        
        base_lr = 0
        max_lr = self.learning_rate
        min_lr = 0.1 * self.learning_rate

        schedulers = CosineAnnealingWithWarmup(optimizer, warmup_steps=80, total_steps=2160,
                                              base_lr=base_lr, max_lr=max_lr, min_lr=min_lr)

        lr_schedulers = {
            "scheduler": schedulers,
            "name": 'learning_rate_logs',
            "interval": 'step',  # The scheduler updates the learning rate at every step (not epoch)
            'frequency': 1  # The scheduler updates the learning rate after every batch
        }
        return [optimizer], [lr_schedulers]

    def on_training_epoch_end(self, outputs):
        gc.collect()
        torch.cuda.empty_cache()
        super().training_epoch_end(outputs)

    # def on_validation_epoch_end(self, outputs):
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     super().validation_epoch_end(outputs)




def main():
    parser = ArgumentParser()

    parser.add_argument("-o", dest="output_file", help="File for output of model parameters", required=True, type=str)
    parser.add_argument("-d", dest="dataset", required=False, type=str, default="pepnn",
                        help="Which dataset to train on, pepnn, pepbind, or interpep")
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("-n_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("-d_model", type=int, default=64, help="Dimension of model")
    parser.add_argument("-d_hidden", type=int, default=128, help="Dimension of CNN block")
    parser.add_argument("-n_head", type=int, default=6, help="Number of heads")
    parser.add_argument("-d_inner", type=int, default=64)
    parser.add_argument("-dropout", type=float, default=0.2)
    # parser.add_argument("-sm", dest="saved_model", help="File containing initial params", required=False, type=str,
    #                     default=None)
    parser.add_argument("-sm", default=None, help="File containing initial params", type=str)
    parser.add_argument("--max_epochs", type=int, default=15, help="Max number of epochs to train")
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--kl_weight", type=float, default=1)
    args = parser.parse_args()

    print(args.max_epochs)

    # Initialize the process group for distributed training
    dist.init_process_group(backend="nccl")

    train_dataset = load_from_disk('/home/tc415/moPPIt/dataset/train/correct_pepnn_biolip_train')
    val_dataset = load_from_disk('/home/tc415/moPPIt/dataset/val/correct_pepnn_biolip_val')
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    data_module = CustomDataModule(train_dataset, val_dataset, tokenizer=tokenizer, batch_size=args.batch_size)

    model = PeptideModel(args.n_layers, args.d_model, args.d_hidden, args.n_head, 64, 128, args.d_inner, dropout=args.dropout,
                         learning_rate=args.lr, max_epochs=args.max_epochs, kl_weight=args.kl_weight)
    if args.sm:
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

    run_id = str(uuid.uuid4())

    logger = WandbLogger(project=f"bind_evaluator",
                         name=f"finetune_bindevaluator_lr={args.lr}_dropout={args.dropout}_nlayers={args.n_layers}_dmodel={args.d_model}_nhead={args.n_head}_dinner={args.d_inner}",
                         job_type='model-training',
                         id=run_id)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_mcc',
        dirpath=args.output_file,
        filename='model-{epoch:02d}-{val_mcc:.2f}-{val_loss:.2f}',
        save_top_k=-1,
        mode='max',
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=True,
        mode='min'
    )

    accumulator = GradientAccumulationScheduler(scheduling={0:4, 5:2, 40:1})

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        strategy='ddp_find_unused_parameters_true',
        precision='bf16',
        logger=logger,
        devices=[0,1,2,3,4,5],
        callbacks=[checkpoint_callback, accumulator, early_stopping_callback],
        gradient_clip_val=args.grad_clip
    )

    trainer.fit(model, datamodule=data_module)

    best_model_path = checkpoint_callback.best_model_path
    print(best_model_path)


if __name__ == "__main__":
    main()
