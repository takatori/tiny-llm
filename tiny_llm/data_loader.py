import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt) # テキスト全体をトークン化

        for i in range(0, len(token_ids) - max_length, stride): # スライディングウィンドウを使ってmax_lengthの長さのシーケンスに分割
            input_chunk = token_ids[i : i+max_length]
            target_chunk = token_ids[i+1 : i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self): # データセットに含まれている行の総数を返す
        return len(self.input_ids)
    
    def __getitem__(self, idx): # データセットから一行返す
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dateloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2") # トークナイザを初期化
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) # データセットを作成
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last, # 指定されたbatch_sizeよりも最後のバッチが短い場合に、訓練中の損失値のスパイクを防ぐためにそのバッチを除外する
        num_workers=num_workers, # 前処理に使うCPUプロセスの数
    )
    return dataloader    