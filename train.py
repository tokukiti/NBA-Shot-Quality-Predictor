import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import sys
import time
import os
# --- 冒頭の import 群にこれを追加 ---
import matplotlib.pyplot as plt

from model import STGAT      # ← 正しくはこれ（同じ場所の model.py を使う）s

# --- 設定 ---
DATA_PATH = 'dataset_50games.pt'        # さっき決めた名前に合わせる
SAVE_MODEL_PATH = 'stgat_model_50.pth'  # モデル名変更
SAVE_CSV_PATH = 'evaluation_results_50.csv'

HIDDEN_DIM = 64
LR = 0.001
EPOCHS = 50  # エポック数は50のままでOK（データが減っても学習回数は確保したい）
ACCUMULATION_STEPS = 16 

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed()
    
    if not torch.cuda.is_available():
        print("エラー: GPUが検出されません。")
        sys.exit(1)
    
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    print("Loading dataset...")
    if not os.path.exists(DATA_PATH):
        print(f"エラー: データファイル {DATA_PATH} が見つかりません。")
        sys.exit(1)

    try:
        full_dataset = torch.load(DATA_PATH, weights_only=False)
    except Exception as e:
        print(f"警告: weights_only=False でのロードに失敗しました。通常ロードを試みます。Error: {e}")
        full_dataset = torch.load(DATA_PATH)

    # ラベル分布の確認と重み計算
    # play_data['label'] が tensor か int かを確認しつつ取得
    labels = []
    for d in full_dataset:
        l = d['label']
        if isinstance(l, torch.Tensor):
            labels.append(l.item())
        else:
            labels.append(l)

    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    # ゼロ除算防止
    pos_weight_val = neg_count / (pos_count + 1e-5)
    pos_weight = torch.tensor([pos_weight_val]).to(device)
    print(f"Stats: Make={pos_count}, Miss={neg_count}, Weight={pos_weight_val:.2f}")

    train_data, test_data = train_test_split(full_dataset, test_size=0.2, stratify=labels, random_state=42)

    # データセット内の特徴量次元を取得 (PyGのDataオブジェクト構造に依存)
    # dataset[0]['graphs'] がリスト(時系列)か単一グラフかによってアクセスが変わる可能性があります
    # ここでは dataset[0]['graphs'] が PyGのDataオブジェクトのリストまたはBatchであると仮定
    sample_graph = full_dataset[0]['graphs']
    if isinstance(sample_graph, list):
        num_node_features = sample_graph[0].num_node_features
        num_edge_features = sample_graph[0].num_edge_features
    else:
        num_node_features = sample_graph.num_node_features
        num_edge_features = sample_graph.num_edge_features
    
    print(f"Input Features: Node={num_node_features}, Edge={num_edge_features}")

    model = STGAT(node_features=num_node_features, 
                    edge_features=num_edge_features, 
                    hidden_channels=HIDDEN_DIM, 
                    out_channels=1).to(device)
                    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    print(f"\nStarting training for {EPOCHS} epochs...")
    start_time = time.time()

    # ★★★ 追加 1: Loss履歴保存用リスト ★★★
    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        random.shuffle(train_data)
        
        loop = tqdm(train_data, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for i, play_data in enumerate(loop):
            graphs = play_data['graphs']
            
            # ラベルの処理
            label = play_data['label']
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label, dtype=torch.float)
            label = label.float().to(device)

            # modelの入力形式に合わせて graphs を処理
            # STGATがリスト(時系列グラフ)を受け取る仕様と仮定
            # ※ GPU転送が必要な場合はここで行う (graphs内の各Dataを .to(device))
            # model内部で .to(device) している場合は不要ですが、念のため
            # (PyGのDataオブジェクトは .to() が再帰的に効く)
            
            # forward
            out = model(graphs) 
            
            loss = criterion(out.view(-1), label.view(-1))
            loss = loss / ACCUMULATION_STEPS
            loss.backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # 損失の記録 (累積ステップで割ったものを戻して加算)
            total_loss += loss.item() * ACCUMULATION_STEPS

        avg_loss = total_loss / len(train_data)
        
        # ★★★ 追加 2: リストに記録 ★★★
        loss_history.append(avg_loss)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f} (Total Time: {elapsed:.0f}s)")

    # --- ループを抜けた直後 (Evaluateの前) にグラフ保存 ---
    
    # ★★★ 追加 3: Lossグラフの描画と保存 ★★★
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), loss_history, marker='o', label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('training_loss_curve.png') # 画像として保存
    plt.close()
    print("学習曲線を 'training_loss_curve.png' に保存しました")

    # --- 保存と評価 ---
    print("\nEvaluating & Saving...")
    model.eval()
    
    # 修正: 空リストで初期化
    results_list = []
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for play_data in tqdm(test_data, desc="Evaluating"):
            graphs = play_data['graphs']
            
            # ラベル取得
            l_val = play_data['label']
            if isinstance(l_val, torch.Tensor):
                l_val = l_val.item()
            
            game_id = play_data.get('game_id', 'unknown')
            event_id = play_data.get('event_id', -1)
            
            logits = model(graphs)
            prob = torch.sigmoid(logits).item()
            pred = 1 if prob >= 0.5 else 0
            
            results_list.append({
                'game_id': game_id,
                'event_id': event_id,
                'actual': int(l_val),
                'predicted': pred,
                'prob_make': round(prob, 4),
                'correct': (int(l_val) == pred)
            })
            
            all_labels.append(l_val)
            all_probs.append(prob)
            all_preds.append(pred)

    # CSV出力
    df_res = pd.DataFrame(results_list)
    df_res.to_csv(SAVE_CSV_PATH, index=False)
    print(f"評価結果を保存しました: {SAVE_CSV_PATH}")

    # モデル保存
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"モデルを保存しました: {SAVE_MODEL_PATH}")

    # --- 評価指標の計算 (F1, AUC追加) ---
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # ROC-AUC はクラスが1種類しかないとエラーになるため例外処理を追加
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        print("警告: テストデータにクラスが1種類しかないため、ROC-AUCを計算できませんでした (0.0とします)")
        auc = 0.0

    print("\n=== Final Result ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}  <-- 重要 (不均衡データ)")
    print(f"ROC-AUC  : {auc:.4f}  <-- 重要 (確率の信頼度)")

if __name__ == "__main__":
    main()