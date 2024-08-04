from torch import nn
from .modules import *
import pdb

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out


class DilatedCNN(nn.Module):
    def __init__(self, d_model, d_hidden):
        super(DilatedCNN, self).__init__()
        self.first_ = nn.ModuleList()
        self.second_ = nn.ModuleList()
        self.third_ = nn.ModuleList()

        dilation_tuple = (1, 2, 3)
        dim_in_tuple = (d_model, d_hidden, d_hidden)
        dim_out_tuple = (d_hidden, d_hidden, d_hidden)

        for i, dilation_rate in enumerate(dilation_tuple):
            self.first_.append(ConvLayer(dim_in_tuple[i], dim_out_tuple[i], kernel_size=3, padding=dilation_rate,
                                         dilation=dilation_rate))

        for i, dilation_rate in enumerate(dilation_tuple):
            self.second_.append(ConvLayer(dim_in_tuple[i], dim_out_tuple[i], kernel_size=5, padding=2*dilation_rate,
                                          dilation=dilation_rate))

        for i, dilation_rate in enumerate(dilation_tuple):
            self.third_.append(ConvLayer(dim_in_tuple[i], dim_out_tuple[i], kernel_size=7, padding=3*dilation_rate,
                                         dilation=dilation_rate))

    def forward(self, protein_seq_enc):
        # pdb.set_trace()
        protein_seq_enc = protein_seq_enc.transpose(1, 2)    # protein_seq_enc's shape: B*L*d_model -> B*d_model*L

        first_embedding = protein_seq_enc
        second_embedding = protein_seq_enc
        third_embedding = protein_seq_enc

        for i in range(len(self.first_)):
            first_embedding = self.first_[i](first_embedding)

        for i in range(len(self.second_)):
            second_embedding = self.second_[i](second_embedding)

        for i in range(len(self.third_)):
            third_embedding = self.third_[i](third_embedding)

        # pdb.set_trace()

        protein_seq_enc = first_embedding + second_embedding + third_embedding

        return protein_seq_enc.transpose(1, 2)


class ReciprocalLayerwithCNN(nn.Module):

    def __init__(self, d_model, d_inner, d_hidden, n_head, d_k, d_v):
        super().__init__()

        self.cnn = DilatedCNN(d_model, d_hidden)

        self.sequence_attention_layer = MultiHeadAttentionSequence(n_head, d_hidden,
                                                                   d_k, d_v)

        self.protein_attention_layer = MultiHeadAttentionSequence(n_head, d_hidden,
                                                                  d_k, d_v)

        self.reciprocal_attention_layer = MultiHeadAttentionReciprocal(n_head, d_hidden,
                                                                       d_k, d_v)

        self.ffn_seq = FFN(d_hidden, d_inner)

        self.ffn_protein = FFN(d_hidden, d_inner)

    def forward(self, sequence_enc, protein_seq_enc):
        # pdb.set_trace()  # protein_seq_enc.shape = B * L * d_model
        protein_seq_enc = self.cnn(protein_seq_enc)
        prot_enc, prot_attention = self.protein_attention_layer(protein_seq_enc, protein_seq_enc, protein_seq_enc)

        seq_enc, sequence_attention = self.sequence_attention_layer(sequence_enc, sequence_enc, sequence_enc)

        prot_enc, seq_enc, prot_seq_attention, seq_prot_attention = self.reciprocal_attention_layer(prot_enc,
                                                                                                    seq_enc,
                                                                                                    seq_enc,
                                                                                                    prot_enc)
        prot_enc = self.ffn_protein(prot_enc)

        seq_enc = self.ffn_seq(seq_enc)

        return prot_enc, seq_enc, prot_attention, sequence_attention, prot_seq_attention, seq_prot_attention


class ReciprocalLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v):
        
        super().__init__()
        
        self.sequence_attention_layer = MultiHeadAttentionSequence(n_head, d_model,
                                                                d_k, d_v)
        
        self.protein_attention_layer = MultiHeadAttentionSequence(n_head, d_model,
                                                               d_k, d_v)
        
        self.reciprocal_attention_layer = MultiHeadAttentionReciprocal(n_head, d_model,
                                                                           d_k, d_v)
        
        
        
        self.ffn_seq = FFN(d_model, d_inner)
        
        self.ffn_protein = FFN(d_model, d_inner)

    def forward(self, sequence_enc, protein_seq_enc):
        prot_enc, prot_attention = self.protein_attention_layer(protein_seq_enc, protein_seq_enc, protein_seq_enc)
        
        seq_enc, sequence_attention = self.sequence_attention_layer(sequence_enc, sequence_enc, sequence_enc)
        
        
        prot_enc, seq_enc, prot_seq_attention, seq_prot_attention = self.reciprocal_attention_layer(prot_enc,
                                                                                   seq_enc,
                                                                                   seq_enc,
                                                                                   prot_enc)
        prot_enc = self.ffn_protein(prot_enc)
        
        seq_enc = self.ffn_seq(seq_enc)
        
        
        
        return prot_enc, seq_enc, prot_attention, sequence_attention, prot_seq_attention, seq_prot_attention



