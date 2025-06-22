import tiktoken
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def generate_text_simple(model, idx, max_new_tokens, context_size): # idxの形状はコンテキストに対応するインデックスの(batch, n_token)配列    

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] # サポートされているコンテキストサイズを超える場合は現在のコンテキストを切り詰める。例えば、LLMがトークンを5つだけサポートしていて、コンテキストのサイズが10の場合は、最後の5つのトークンのみを使用する
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]  # 最後のタイムステップのみに着目し、(batch, n_token, vocab_size)が(batch, vocab_size)になるようにする

        probas = torch.softmax(logits, dim=-1)  # 確率分布に変換 probasの形状は(batch, vocab_size)

        idx_next = torch.argmax(probas, dim=-1, keepdim=True) # idx_nextの形状は(batch, 1)

        idx = torch.cat((idx, idx_next), dim=1)  # サンプリングしたインデックスを実行中のシーケンスに追加。idxの形状は(batch, n_token+1)になる

    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # バッチ次元を追加
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # バッチ次元を削除
    return tokenizer.decode(flat.tolist())

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device) # deviceを指定すると、データをモデルと同じデバイスに転送できる
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0,1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader) # num_batchesが指定されていない場合は、すべてのバッチを反復処理
    else:
        num_batches = min(num_batches, len(data_loader)) # num_batchesがデータローダーのバッチ数を超えている場合は、データローダーのバッチ数と一致するように調整
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches # 全バッチの損失の平均を計算



def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    "LLMの事前学習を行うためのメイン関数"

    train_losses, val_losses, track_tokens_seen = [], [], [] # 損失と監視のトークンを追跡するためにリストを初期化
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 前回のバッチの反復処理で計算された損失の勾配をゼロにリセット
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # 損失の勾配を計算
            optimizer.step() # 損失の勾配を使ってモデルの重みを更新
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0: # オプションの評価ステップ
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)
    
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    モデルを更新するたびに訓練データセットと検証データセッでの損失を出力することで、訓練によってモデルが改善されたかどうかを評価できるようにする
    モデルを評価モードに切り替えて勾配の追跡とドロップアウトを無効にした上で、訓練データセットと検証データセットでの損失を計算する
    """
    model.eval() # 安定した再現性のある結果を得るために、評価中はドロップアウトを無効化
    with torch.no_grad(): # 計算量を減らすために、評価には必要のない勾配の追跡を無効にする
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    訓練中のモデルの性能を判断するために、モデルが生成した具体的なテキストサンプルを出力する
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", ""))
    model.train()



def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5,3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, label="Validation loss", linestyle="-.")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) 

    ax2 = ax1.twiny() # 同じy軸を共有する新しいx軸を作成
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

def softmax_with_temperature(logits, temperature):
    """
    ソフトマックス関数を温度パラメータで調整して、確率分布を生成する
    温度が高いほど、分布は平坦になり、低いほど鋭くなる
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0")
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)  # 確率分布に変換

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None: # top_kサンプリングでロジットをフィルタリング
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf"), device=logits.device),
                logits            
            )

        if temperature > 0.0: # 温度スケーリングを適用
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id: # シーケンス終了トークンが検出された場合は、生成を早期終了
            break
        
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def assign(left, right):
    """
    2つのテンソルの形状が同じかどうかをチェックし、同じである場合は、訓練可能なPyTorchパラメータとしてrightテンソルを返す
    """
    if left.shape != right.shape:
        raise ValueError(f"Shapes do not match: {left.shape} != {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))



def load_weights_into_gpt(gpt, params):
    """
    GPTモデルのコードにOpenAIの重みを読み込む
    """
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"]) # OpenAIのオリジナルのGPT-2モデルは、トークン埋め込み層のパラメータを出力層で再利用するbことでパラメータの総数を減らしている(重み共有)