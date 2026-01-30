import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# model.py から読み込む設定に変更
from model import STGAT
from torch_geometric.data import Batch

# --- 設定 ---
MODEL_PATH = 'stgat_model_v11.pth'
DATA_PATH = 'dataset_v11_pipeline.pt'
OUTPUT_IMAGE = 'attention_gravity.png'
TARGET_INDEX = 10  # 可視化したいプレーのインデックス（適当な数字に変えて試してください）

def load_model_and_data():
    print("Loading data and model...")
    if not os.path.exists(DATA_PATH):
        print(f"エラー: {DATA_PATH} が見つかりません。")
        sys.exit(1)
        
    # weights_only=False は古いPyTorchの警告回避のため
    try:
        dataset = torch.load(DATA_PATH, weights_only=False)
    except:
        dataset = torch.load(DATA_PATH)
    
    # グラフデータの形式確認 (List か PyG Object か)
    sample_graph = dataset[0]['graphs']
    if isinstance(sample_graph, list):
        num_node_features = sample_graph[0].num_node_features
        num_edge_features = sample_graph[0].num_edge_features
    else:
        num_node_features = sample_graph.num_node_features
        num_edge_features = sample_graph.num_edge_features
    
    # モデルの初期化 (train2.py, model2.py とパラメータを合わせる)
    model = STGAT(node_features=num_node_features, 
                  edge_features=num_edge_features, 
                  hidden_channels=64, 
                  out_channels=1,
                  heads=3) # model2.pyのデフォルトheadsに合わせる
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"✅ Loaded model from {MODEL_PATH}")
    else:
        print(f"エラー: モデルファイル {MODEL_PATH} がありません。先に train2.py を実行してください。")
        sys.exit(1)
        
    model.eval()
    return model, dataset

def visualize_gravity(model, play_data):
    graphs = play_data['graphs']
    # シュート直前（最後のフレーム）のグラフを使用
    target_graph = graphs[-1] 
    
    # モデルに入力してAttentionを取得
    # model2.py の forward は data_list を受け取る仕様
    # 1フレームだけをリストに入れて渡す
    input_list = [target_graph]
    
    with torch.no_grad():
        # model2.py の仕様に合わせて return_attention=True
        # 戻り値: out, att_edge_index, att_weights
        _, edge_index, att_weights = model(input_list, return_attention=True)

    # --- Attention Weightsの処理 ---
    # att_weights shape: [num_edges, heads] -> ヘッド平均をとる
    if att_weights is not None:
        att_weights_mean = att_weights.mean(dim=1).cpu().numpy()
        edge_index_np = edge_index.cpu().numpy()
    else:
        print("Error: Attention weights not returned.")
        return

    # ノード情報の取得
    x = target_graph.x.cpu().numpy()
    
    # --- 座標と属性の特定 (make_dataset.py の仕様に依存) ---
    # 想定: x = [x, y, vx, vy, ax, ay, dist_basket, dist_ball, is_offense, ...]
    pos = x[:, 0:2] # 0,1番目が座標 (x, y)
    
    # is_offense (8番目と仮定)
    is_offense = x[:, 8] if x.shape[1] > 8 else np.zeros(len(x))

    # ボール保持者（シューター）の特定
    # dist_ball (7番目) が 0 に近いノードを探す
    dist_ball = x[:, 7] if x.shape[1] > 7 else np.zeros(len(x))
    ball_holder_idx = np.argmin(dist_ball)
    
    print(f"Visualizing Play: Label={play_data['label']}")
    print(f"Ball Holder Node Index: {ball_holder_idx}")

    # --- お絵描き (Matplotlib) ---
    plt.figure(figsize=(10, 8))
    
    # コートの背景画像があれば読み込む（なければ白紙）
    if os.path.exists("court.png"):
        img = plt.imread("court.png")
        plt.imshow(img, extent=[0, 94, 0, 50], zorder=0, alpha=0.5)

    # ノードの描画
    for i in range(len(x)):
        # 色分け: オフェンス=赤, ディフェンス=青, ボールマン=オレンジ
        if i == ball_holder_idx:
            c = 'orange'
            lbl = 'Ball'
            s = 300
        elif is_offense[i] == 1:
            c = 'red'
            lbl = 'Off'
            s = 150
        else:
            c = 'blue'
            lbl = 'Def'
            s = 150
            
        plt.scatter(pos[i, 0], pos[i, 1], c=c, s=s, edgecolors='black', zorder=10)
        # 番号を表示
        plt.text(pos[i, 0], pos[i, 1]+1.5, str(i), fontsize=9, ha='center', color='black', fontweight='bold')

    # エッジ（Attention線）の描画
    # 「誰が(Source) → 誰に(Target) 注目したか」
    # GATの仕様上、矢印の向きは Target(更新されるノード) <- Source(情報源)
    # ここでは、「ボールマン(Target)に対して、周囲(Source)がどれだけ重要だったか」を見ます。
    
    target_node = ball_holder_idx # ボールマンへの注目を見たい
    max_alpha = att_weights_mean.max()
    
    for k in range(edge_index_np.shape[1]):
        src = edge_index_np[0, k] # 情報源（見ている人）
        dst = edge_index_np[1, k] # 更新対象（見られている人＝ボールマン）
        weight = att_weights_mean[k]
        
        # ボールマン(dst) に向かうエッジだけを描画
        if dst == target_node and src != dst:
            # 線の太さと透明度をAttentionに比例させる
            alpha_val = (weight / max_alpha) if max_alpha > 0 else 0
            
            # 閾値（あまりに薄い線は描かない）
            if alpha_val > 0.2: 
                plt.plot([pos[src, 0], pos[dst, 0]], [pos[src, 1], pos[dst, 1]], 
                         color='green', linewidth=weight*20, alpha=alpha_val, zorder=5)
                
                # 重みの数値を線の中央に表示
                mid_x = (pos[src, 0] + pos[dst, 0]) / 2
                mid_y = (pos[src, 1] + pos[dst, 1]) / 2
                plt.text(mid_x, mid_y, f"{weight:.2f}", color='green', fontsize=8, fontweight='bold', zorder=15)

    plt.title(f"Attention Weights: Who is looking at Ball Holder (Node {ball_holder_idx})?")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    # コートの範囲（NBA標準）
    plt.xlim(-5, 100)
    plt.ylim(-5, 55)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(OUTPUT_IMAGE)
    plt.show()
    print(f"✅ Attention map saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    model, dataset = load_model_and_data()
    
    if len(dataset) > TARGET_INDEX:
        play_data = dataset[TARGET_INDEX]
        visualize_gravity(model, play_data)
    else:
        print("エラー: TARGET_INDEX がデータサイズを超えています。")