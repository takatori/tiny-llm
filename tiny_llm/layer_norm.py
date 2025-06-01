import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 # 正規化時に0除算を防ぐために分散に加算される小さな定数
        self.scale = nn.Parameter(torch.ones(emb_dim)) # 訓練可能なパラメータ
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # 訓練可能なパラメータ

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # 埋め込み次元nが非常に大きいため、分散の計算で不偏推定を使用しない
        normalized_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * normalized_x + self.shift
