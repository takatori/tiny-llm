import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # 出力次元数をヘッド数で分割

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out) # Linear層を追加して、ヘッドの出力を組み合わせる
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # num_heads次元を追加して行列を暗黙的に分割
        # 続いて、最後の次元を展開し、形状を(b, num_tokens, d_out)から(b, num_tokens, num_heads, head_dim)に変形
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # 形状を(b, num_tokens, num_heads, head_dim)から(b, num_heads, num_tokens, head_dim)に変形
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3) # 各ヘッドのドット積を計算

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] # マスクをトークン数で切り捨て

        attn_scores.masked_fill_(mask_bool, -torch.inf) # Attentionスコアを埋めるためにマスクを使う

        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1) 
        attn_weights = self.dropout(attn_weights)

        # テンソルの形状は(b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out) # ヘッドを結合

        context_vec = self.out_proj(context_vec) # 線形射影を追加

        return context_vec
