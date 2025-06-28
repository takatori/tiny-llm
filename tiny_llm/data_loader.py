import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
import pandas as pd

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


class SpamDataset(Dataset):
    """
    訓練データセット内で、最も長いシーケンスを特定し、テキストメッセージをエンコードし、パディングトークンをつかって
    すべてのシーケンスを最も長いシーケンスと同じ長さにパディングする    
    """

    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50216):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data['Text']
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[:self.max_length] # シーケンスがmax_lengthを超える場合は切り捨てる
                for encoded_text in self.encoded_texts
            ]
        self.encoded_texts = [ # シーケンスを最も長いシーケンスと同じ長さにパディング
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]['Label']
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
