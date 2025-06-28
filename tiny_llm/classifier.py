import torch

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    outputs = model(input_batch)
    logits = outputs[:, -1, :]  # 最後のトークンの出力を取得
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss =0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

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


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  
                

            predicted_labels = torch.argmax(logits, dim=-1)
            
            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()  # 正しい予測の数をカウント
            )
        else:
            break

    return correct_predictions / num_examples


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    """
    スパムを分類するためのモデルのファインチューニング
    """
    train_losses, val_losses, train_accs, val_accs = [], [], [], [] # 損失と既に見たサンプルを追跡するためにリストを初期化    
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # 前回のバッチの反復処理で計算された損失の勾配をリセット
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # 損失の勾配を計算
            optimizer.step() # 損失の勾配を使ってモデルの重みを更新
            examples_seen += input_batch.shape[0] # バッチのサイズを加算して、見たサンプルの数を更新
            global_step += 1

            if global_step % eval_freq == 0: # オプションの評価ステップ
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")

        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)   
    
    return train_losses, val_losses, train_accs, val_accs, examples_seen



def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):

    model.eval()

    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]
    
    input_ids = input_ids[:min(max_length, supported_context_length)] # シーケンスが長すぎる場合は切り詰める

    input_ids += [pad_token_id] * (max_length - len(input_ids)) # 最も長いシーケンスと同じ長さにパディング

    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0) # バッチ次元を追加

    with torch.no_grad(): # モデルを評価モードにし、勾配の追跡を無効にして計算量を減らす
        logits = model(input_tensor.to(device))[:, -1, :]  # 最後のトークンの出力を取得

    predicted_label = torch.argmax(logits, dim=-1).item()  # 最も可能性の高いラベルを取得

    return "spam" if predicted_label == 1 else "not spam"