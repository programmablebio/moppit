from torch import nn
import numpy as np
import torch
import torch.nn.functional as F


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x
   

    

    
class MultiHeadAttentionSequence(nn.Module):
    
    
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        
        super().__init__()
        
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, n_head*d_k)
        self.W_K = nn.Linear(d_model, n_head*d_k)
        self.W_V = nn.Linear(d_model, n_head*d_v)
        self.W_O = nn.Linear(n_head*d_v, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        
        batch, len_q, _ = q.size()
        batch, len_k, _ = k.size()
        batch, len_v, _ = v.size()

        Q = self.W_Q(q).view([batch, len_q, self.n_head, self.d_k])
        K = self.W_K(k).view([batch, len_k, self.n_head, self.d_k])
        V = self.W_V(v).view([batch, len_v, self.n_head, self.d_v])

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2).transpose(2, 3)
        V = V.transpose(1, 2)

        attention = torch.matmul(Q, K)

        attention = attention / np.sqrt(self.d_k)

        attention = F.softmax(attention, dim=-1)
        
        output = torch.matmul(attention, V)

        output = output.transpose(1, 2).reshape([batch, len_q, self.d_v*self.n_head])
            
        output = self.W_O(output)

        output = self.dropout(output)
        
        output = self.layer_norm(output + q)
        
        return output, attention
        
class MultiHeadAttentionReciprocal(nn.Module):
    
    
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        
        super().__init__()
        
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        
        
        self.W_Q = nn.Linear(d_model, n_head*d_k)
        self.W_K = nn.Linear(d_model, n_head*d_k)
        self.W_V = nn.Linear(d_model, n_head*d_v)
        self.W_O = nn.Linear(n_head*d_v, d_model)
        self.W_V_2 = nn.Linear(d_model, n_head*d_v)
        self.W_O_2 = nn.Linear(n_head*d_v, d_model)
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.layer_norm_2 = nn.LayerNorm(d_model)
        
        self.dropout_2 = nn.Dropout(dropout)
    
    
 

    def forward(self, q, k, v, v_2):
        
        batch, len_q, _ = q.size()
        batch, len_k, _ = k.size()
        batch, len_v, _ = v.size()
        batch, len_v_2, _ = v_2.size()        
        
            
        Q = self.W_Q(q).view([batch, len_q, self.n_head, self.d_k])
        K = self.W_K(k).view([batch, len_k, self.n_head, self.d_k])
        V = self.W_V(v).view([batch, len_v, self.n_head, self.d_v])
        V_2 = self.W_V_2(v_2).view([batch, len_v_2, self.n_head, self.d_v])
        
        
           
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2).transpose(2, 3)
        V = V.transpose(1, 2)
        V_2 = V_2.transpose(1,2) 
        
        attention = torch.matmul(Q, K)
       
            
        attention = attention /np.sqrt(self.d_k)
        
        attention_2 = attention.transpose(-2, -1)
        
        
       
        attention = F.softmax(attention, dim=-1)
        
        attention_2 = F.softmax(attention_2, dim=-1)
    
        
        output = torch.matmul(attention, V)
        
        output_2 = torch.matmul(attention_2, V_2)
            
        output = output.transpose(1, 2).reshape([batch, len_q, self.d_v*self.n_head])
       
        output_2 = output_2.transpose(1, 2).reshape([batch, len_k, self.d_v*self.n_head])
            
        output = self.W_O(output)
        
        output_2 = self.W_O_2(output_2)
        
        output = self.dropout(output)
        
        output = self.layer_norm(output + q)
        
        output_2 = self.dropout(output_2)
        
        output_2 = self.layer_norm(output_2 + k)
        
        
        
       
        
        return output, output_2, attention, attention_2
        
    
class FFN(nn.Module):
    
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        
        self.layer_1 = nn.Conv1d(d_in, d_hid,1)
        self.layer_2 = nn.Conv1d(d_hid, d_in,1)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_in)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        residual = x 
        output = self.layer_1(x.transpose(1, 2))
        
        output = self.relu(output)
        
        output = self.layer_2(output)
        
        output = self.dropout(output)
        
        output = self.layer_norm(output.transpose(1, 2)+residual)
        
        return output

