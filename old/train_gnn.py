import torch
import torch.nn.functional as F
from torch.nn import LSTM, Linear, Dropout
from torch_geometric.nn import GATv2Conv
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd # ★ 分析のために pandas をインポート
import numpy as np  # ★ 確率計算のために numpy をインポート

# ==============================================================================
# モデル定義 (変更なし)
# ==============================================================================
class SpatioTemporalGAT(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_classes, hidden_dim=32, heads=2):
        super(SpatioTemporalGAT, self).__init__()
        self.gat_conv1 = GATv2Conv(num_node_features, hidden_dim, heads=heads, edge_dim=num_edge_features)
        self.gat_conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=num_edge_features)
        self.lstm = LSTM(input_size=hidden_dim * heads, hidden_size=64, batch_first=True)
        self.dropout = Dropout(p=0.5)
        self.linear = Linear(64, num_classes)

    def forward(self, play_data_graphs):
        moment_embeddings = []
        for moment_graph in play_data_graphs:
            x, edge_index, edge_attr = moment_graph.x, moment_graph.edge_index, moment_graph.edge_attr
            x = self.gat_conv1(x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)
            x = self.gat_conv2(x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)
            moment_embedding = x.mean(dim=0)
            moment_embeddings.append(moment_embedding)
        
        play_sequence = torch.stack(moment_embeddings).unsqueeze(0)
        lstm_out, _ = self.lstm(play_sequence)
        final_representation = lstm_out[0, -1, :]
        final_representation = self.dropout(final_representation)
        out = self.linear(final_representation)
        # ★ 分析のために、log_softmax の前の「生の」スコアも返すように変更
        #   F.log_softmax(out, dim=0)
        return out # 生のスコア（ロジット）を返す

# ==============================================================================
# データ準備 (v5.1)
# ==============================================================================
print("--- データセット(v5.1)を準備しています ---")
# ★ v5.1 の入力ファイル
GRAPH_DATA_FILE = 'final_play_sequence_dataset_v5.1_with_ID.pt' 
MODEL_SAVE_PATH = 'gat_lstm_model_v5.pth' # ★ 保存するモデルのファイル名
EVALUATION_CSV_PATH = 'evaluation_results_v5.csv' # ★ 出力するCSVのファイル名

try:
    all_plays = torch.load(GRAPH_DATA_FILE, weights_only=False)
except FileNotFoundError:
    print(f"エラー: {GRAPH_DATA_FILE}が見つかりません。build_graph_dataset.py (v5.1) を先に実行してください。")
    exit()
except ImportError:
    print("\n--- !!! エラー !!! ---")
    print("torch.loadでエラーが発生しました。pandasが環境にない可能性があります。")
    print("pip install pandas を実行してから、もう一度お試しください。")
    exit()

if not all_plays:
    print("エラー: データセットが空です。")
    exit()

dataset = all_plays
labels = [d['label'].item() for d in dataset]
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42, stratify=labels)
print(f"データセットの準備完了。訓練データ: {len(train_data)}プレー, テストデータ: {len(test_data)}プレー")

# ==============================================================================
# 訓練 (v5.1)
# ==============================================================================
print("\n--- モデルの訓練を開始します (v5.1: 加速度対応) ---")

NUM_NODE_FEATURES = all_plays[0]['graphs'][0].num_node_features
NUM_EDGE_FEATURES = all_plays[0]['graphs'][0].num_edge_features
print(f"ノード特徴量: {NUM_NODE_FEATURES}次元, エッジ特徴量: {NUM_EDGE_FEATURES}次元")

model = SpatioTemporalGAT(
    num_node_features=NUM_NODE_FEATURES, 
    num_edge_features=NUM_EDGE_FEATURES, 
    num_classes=2
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
num_epochs = 50 
for epoch in range(num_epochs):
    total_loss = 0
    pbar = tqdm(train_data, desc=f"エポック {epoch+1}/{num_epochs}")
    valid_plays_count = 0 
    
    for play in pbar:
        optimizer.zero_grad()
        raw_scores = model(play['graphs']) # ★ 生のスコアを受け取る
        label = play['label'] 
        
        # ★ 損失計算のために、ここで log_softmax と nll_loss を適用
        #   (F.cross_entropy を使うとワンステップでできます)
        loss = F.cross_entropy(raw_scores.unsqueeze(0), label)
        
        if torch.isnan(loss) or torch.isinf(loss): continue
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        valid_plays_count += 1
        if valid_plays_count > 0:
            pbar.set_postfix({'avg_loss': total_loss / valid_plays_count})
            
    avg_epoch_loss = total_loss / valid_plays_count if valid_plays_count > 0 else 0
    print(f"エポック {epoch+1}/{num_epochs} 完了, 平均損失: {avg_epoch_loss:.4f} ({valid_plays_count}プレーで訓練)")

# ★★★ v5.1 変更点：モデルの保存 ★★★
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\n--- 訓練完了。モデルを '{MODEL_SAVE_PATH}' に保存しました ---")


# ==============================================================================
# 評価と分析 (v5.1)
# ==============================================================================
print(f"\n--- モデルの性能を評価し、'{EVALUATION_CSV_PATH}' に出力します ---")
model.eval()
correct = 0
results_list = [] # ★ 分析結果をためるリスト

with torch.no_grad():
    for play in tqdm(test_data, desc="テストデータを評価中"):
        
        raw_scores = model(play['graphs']) # (1, 2) のテンソル [失敗スコア, 成功スコア]
        
        # ★ v5.1 変更点：確率と予測の計算
        # F.softmaxで生のスコアを確率（0%〜100%）に変換
        probabilities = F.softmax(raw_scores, dim=0) 
        
        # 予測ラベル (0か1)
        pred_label = probabilities.argmax(dim=0).item()
        # 正解ラベル (0か1)
        actual_label = play['label'].item()
        # プレーID
        game_id = play['game_id']
        event_id = play['event_id']
        
        # AIが「成功(1)」と予測した確率
        prob_success = probabilities[1].item() 
        
        is_correct = (pred_label == actual_label)
        if is_correct:
            correct += 1
            
        # ★ 分析結果をリストに追加
        results_list.append({
            'game_id': game_id,
            'event_id': event_id,
            'predicted_label': pred_label,
            'actual_label': actual_label,
            'is_correct': is_correct,
            'probability_success': prob_success # 「成功」である確率
        })

accuracy = (correct / len(test_data) * 100) if len(test_data) > 0 else 0
print(f"\n✅ 評価完了！")
print(f"テストデータ {len(test_data)}プレーに対する正解率 (Accuracy): {accuracy:.2f}%")

# ★★★ v5.1 変更点：CSVファイルへの保存 ★★★
try:
    results_df = pd.DataFrame(results_list)
    # 確率が高い順、または間違えた順などでソートすると分析しやすい
    results_df = results_df.sort_values(by='probability_success', ascending=False)
    
    results_df.to_csv(EVALUATION_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"✅ 分析結果を '{EVALUATION_CSV_PATH}' に保存しました。")
    print("\n--- 分析のヒント ---")
    print("・'is_correct' が 'False' の行をフィルタリングして、AIが間違えたプレーを特定できます。")
    print("・'probability_success' が 0.4 〜 0.6 のプレーは、AIが「迷った」プレーです。")
    print("・'probability_success' が 0.9 以上なのに 'actual_label' が 0 のプレーは、AIが「強く間違えた」プレーです。")
    
except Exception as e:
    print(f"\n❌ CSVファイルの保存に失敗しました: {e}")
    print("分析結果（一部）:")
    print(results_list[:5]) # エラーの場合は、結果の先頭5件だけ表示