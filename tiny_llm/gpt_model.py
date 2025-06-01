import torch
import torch.nn as nn

from tiny_llm.transormer_block import TransformerBlock
from tiny_llm.layer_norm import LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate_emb"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear( # バイアスのない線形出力ヘッドを定義: Transfomerブロックの出力をトークナイザの語彙空間に射影    
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx) # 入力トークンインデックスを密ベクトルに変換

        pos_embeds = self.pos_emb( # 位置埋め込み適用
            torch.arange(seq_len, device=in_idx.device) # デバイス設定: 入力データがCPUとGPUのどちらにあるかに応じて、モデルをどちらかのデバイスで訓練できる
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x) # このロジットは、次に来るトークンの正規化されていない確率を表す
        return logits
