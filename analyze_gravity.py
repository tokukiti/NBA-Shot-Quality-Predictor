import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import json
# model.py を読み込むように変更
from model import STGAT
from torch_geometric.data import Data, Batch

# --- 設定 ---
MODEL_PATH = 'stgat_model_v11.pth'
DATA_CSV = 'cleaned_shots_data_v2.csv'
TRACKING_DIR = './data/2016.NBA.Raw.SportVU.Game.Logs'
OUTPUT_DIR = './gravity_analysis'

# ★重要: 学習時と同じパラメータ (64)
HIDDEN_DIM = 64  
NUM_NODE_FEATURES = 9
NUM_EDGE_FEATURES = 1

# コート設定 (正規化用)
HALF_COURT_X = 47.0
COURT_LENGTH = 94.0
COURT_WIDTH = 50.0
BASKET_COORDS = np.array([88.75, 25.0])
FPS = 25
PRE_EVENT_SEC = 4.0 
POST_EVENT_SEC = 1.0
MIN_FRAMES = 25

# チームID辞書
TEAM_ABBR_TO_ID = {
    'ATL': 1610612737, 'BOS': 1610612738, 'CLE': 1610612739, 'NOP': 1610612740,
    'CHI': 1610612741, 'DAL': 1610612742, 'DEN': 1610612743, 'GSW': 1610612744,
    'HOU': 1610612745, 'LAC': 1610612746, 'LAL': 1610612747, 'MIA': 1610612748,
    'MIL': 1610612749, 'MIN': 1610612750, 'BKN': 1610612751, 'NYK': 1610612752,
    'ORL': 1610612753, 'IND': 1610612754, 'PHI': 1610612755, 'PHX': 1610612756,
    'POR': 1610612757, 'SAC': 1610612758, 'SAS': 1610612759, 'OKC': 1610612760,
    'TOR': 1610612761, 'UTA': 1610612762, 'MEM': 1610612763, 'WAS': 1610612764,
    'DET': 1610612765, 'CHA': 1610612766
}

# ==========================================
# データ処理関数 (make_dataset_v11-2.py のロジックを移植・統合)
# ==========================================

def get_game_moments(json_path):
    if not os.path.exists(json_path): return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        all_moments = []
        for event in data['events']:
            all_moments.extend(event.get('moments', []))
        return all_moments
    except:
        return None

def extract_time_window(moments, target_q, target_sec):
    indices_in_q = [i for i, m in enumerate(moments) if m[0] == target_q]
    if not indices_in_q: return None
    best_idx = -1
    min_diff = 9999.0
    for i in indices_in_q:
        time_remaining = moments[i][2]
        diff = abs(time_remaining - target_sec)
        if diff < min_diff:
            min_diff = diff
            best_idx = i
        if diff < 0.1: break
    if best_idx == -1 or min_diff > 2.0: return None
    start_idx = max(0, best_idx - int(PRE_EVENT_SEC * FPS))
    end_idx = min(len(moments), best_idx + int(POST_EVENT_SEC * FPS))
    if end_idx - start_idx < MIN_FRAMES: return None
    return moments[start_idx : end_idx]

def calculate_features_normalized(play_df, offense_team_id):
    """学習時(v11-2)と同じ正規化処理を行ってグラフデータを作成する"""
    # 確実にソート
    play_df = play_df.sort_values(by=['player_id', 'moment_index'])
    play_df['dt'] = 0.04 
    
    # 速度・加速度計算
    play_df['vx'] = play_df['x'].diff() / play_df['dt']
    play_df['vy'] = play_df['y'].diff() / play_df['dt']
    play_df['ax'] = play_df['vx'].diff() / play_df['dt']
    play_df['ay'] = play_df['vy'].diff() / play_df['dt']
    play_df.fillna(0, inplace=True)

    # コート反転処理 (左→右攻めに統一)
    ball_rows = play_df[play_df['player_id'] == -1]
    if ball_rows.empty: return None, None
    ball_start_x = ball_rows['x'].iloc[0]
    
    if ball_start_x < HALF_COURT_X:
        play_df['x'] = 94.0 - play_df['x']
        play_df['y'] = 50.0 - play_df['y']
        play_df['vx'] = -play_df['vx']
        play_df['vy'] = -play_df['vy']
        play_df['ax'] = -play_df['ax']
        play_df['ay'] = -play_df['ay']

    # ★ 正規化 (学習データ v11-2 と合わせる)
    play_df['x_norm'] = play_df['x'] / COURT_LENGTH
    play_df['y_norm'] = play_df['y'] / COURT_WIDTH
    play_df['vx_norm'] = play_df['vx'] / 10.0 
    play_df['vy_norm'] = play_df['vy'] / 10.0
    play_df['ax_norm'] = play_df['ax'] / 10.0
    play_df['ay_norm'] = play_df['ay'] / 10.0

    play_df['is_offense'] = play_df.apply(
        lambda row: 1.0 if (row['team_id'] == offense_team_id or row['player_id'] == -1) else 0.0, 
        axis=1
    )

    moment_graphs = []
    # moment_index ごとにグラフ作成
    for _, m_df in play_df.groupby('moment_index'):
        ball_df = m_df[m_df['player_id'] == -1]
        players_df = m_df[m_df['player_id'] != -1]
        
        if ball_df.empty or len(players_df) < 5: continue

        # ボール特徴量
        b_row = ball_df.iloc[0]
        b_coords = np.array([b_row['x'], b_row['y']])
        b_coords_norm = np.array([b_row['x_norm'], b_row['y_norm']])
        b_dist_basket = np.linalg.norm(b_coords - BASKET_COORDS) / COURT_LENGTH

        ball_feat = np.array([[
            b_coords_norm[0], b_coords_norm[1], 
            b_row['vx_norm'], b_row['vy_norm'], 
            b_row['ax_norm'], b_row['ay_norm'], 
            b_dist_basket, 
            0.0, 
            1.0 
        ]])

        # 選手特徴量
        p_coords = players_df[['x', 'y']].values
        p_coords_norm = players_df[['x_norm', 'y_norm']].values
        
        dist_basket = np.linalg.norm(p_coords - BASKET_COORDS, axis=1) / COURT_LENGTH
        dist_ball = np.linalg.norm(p_coords - b_coords, axis=1) / COURT_LENGTH
        
        player_feats = np.column_stack([
            p_coords_norm,
            players_df[['vx_norm', 'vy_norm', 'ax_norm', 'ay_norm']].values,
            dist_basket, dist_ball, players_df[['is_offense']].values
        ])
        
        # 結合 (Ball + Players)
        all_feats = np.vstack([ball_feat, player_feats])
        x = torch.tensor(all_feats, dtype=torch.float)
        
        # 全結合エッジ
        num_nodes = len(all_feats)
        edge_index, edge_attr = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
                    dist = np.linalg.norm(all_feats[i, :2] - all_feats[j, :2])
                    edge_attr.append([dist])
        
        data = Data(
            x=x, 
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(), 
            edge_attr=torch.tensor(edge_attr, dtype=torch.float)
        )
        moment_graphs.append(data)
    
    if not moment_graphs: return None, None
    return moment_graphs, play_df

# ==========================================
# メイン処理
# ==========================================

def main(game_id, event_id):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. モデルロード
    print("Loading model...")
    model = STGAT(node_features=NUM_NODE_FEATURES, 
                  edge_features=NUM_EDGE_FEATURES, 
                  hidden_channels=HIDDEN_DIM, 
                  out_channels=1).to(device)
    
    # weights_only=True 対応 (警告回避)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    except:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
    model.eval()

    # 2. データ準備
    print(f"Preparing data for Game {game_id}, Event {event_id}...")
    
    # CSV情報
    df = pd.read_csv(DATA_CSV, dtype={'GAME_ID': str})
    if 'EVENTNUM' not in df.columns:
        df['EVENTNUM'] = df.index 

    row = df[(df['GAME_ID'] == game_id) & (df['EVENTNUM'] == event_id)]
    if row.empty:
        print("Error: Event not found in CSV.")
        return
    row = row.iloc[0]

    # トラッキングデータ
    json_path = os.path.join(TRACKING_DIR, f"{game_id}.json")
    moments = get_game_moments(json_path)
    if not moments:
        print("Error: Tracking JSON not found.")
        return

    # 時間切り出し
    shot_moments = extract_time_window(moments, row['Quarter'], row['SecLeft'])
    if not shot_moments:
        print("Error: Time window extraction failed.")
        return

    # チームID特定
    is_away_play = pd.notna(row['AwayPlay']) and str(row['AwayPlay']).strip() != ""
    team_abbr = row['AwayTeam'] if is_away_play else row['HomeTeam']
    offense_team_id = TEAM_ABBR_TO_ID.get(team_abbr)

    # DataFrame作成 (ここで生データを作る)
    flat_rows = []
    for i, m in enumerate(shot_moments):
        for entity in m[5]:
            # entity: [teamid, playerid, x, y, z]
            flat_rows.append([m[0], m[2], m[3], entity[0], entity[1], entity[2], entity[3], entity[4], i])
    
    play_df_raw = pd.DataFrame(flat_rows, columns=['quarter', 'game_clock', 'shot_clock', 'team_id', 'player_id', 'x', 'y', 'z', 'moment_index'])
    
    # ★ ここで正規化済みのグラフデータを作成
    # (calculate_kinematics_and_graph はもう使わない)
    graphs, play_df_processed = calculate_features_normalized(play_df_raw, offense_team_id)
    
    if not graphs:
        print("Error: Graph creation failed.")
        return

    # 3. 推論 & Attention取得
    print("Running inference with Attention...")
    with torch.no_grad():
        logits, (edge_index, att_weights) = model(graphs, return_attention=True)
        prob = torch.sigmoid(logits).item()

    print(f"Predicted Success Probability: {prob:.4f}")

    # 4. Attention (Gravity) 可視化
    # 最初のフレームの平均Attentionを使用
    att_mean = att_weights.mean(dim=1).cpu().numpy()
    edge_index = edge_index.cpu().numpy()
    
    # ノードマッピング (play_df_processed の並び順に基づく)
    first_frame_df = play_df_processed[play_df_processed['moment_index'] == 0]
    
    node_map = {}
    
    # ハンドラー特定 (ボールに一番近いオフェンス)
    ball_row = first_frame_df[first_frame_df['player_id'] == -1].iloc[0]
    min_dist = 100
    handler_node_id = -1
    
    # 行番号(i) をノードIDとする
    for i in range(len(first_frame_df)):
        row_data = first_frame_df.iloc[i]
        pid = int(row_data['player_id'])
        tid = int(row_data['team_id'])
        
        is_offense = (tid == offense_team_id) or (pid == -1)
        role = "Ball" if pid == -1 else ("Off" if is_offense else "Def")
        node_map[i] = {'pid': pid, 'role': role}
        
        if role == 'Off':
            dist = np.sqrt((row_data['x'] - ball_row['x'])**2 + (row_data['y'] - ball_row['y'])**2)
            if dist < min_dist:
                min_dist = dist
                handler_node_id = i

    print(f"Handler Node ID: {handler_node_id}")

    # 行列作成
    num_nodes = len(first_frame_df)
    attn_matrix = np.zeros((num_nodes, num_nodes))
    num_edges_per_frame = num_nodes * (num_nodes - 1)
    
    for k in range(num_edges_per_frame):
        src = edge_index[0, k] # 見られる側 (Handler)
        tgt = edge_index[1, k] # 見る側 (Defender)
        
        if k < len(att_mean):
            attn_matrix[tgt, src] = att_mean[k]

    # ヒートマップ
    labels = [f"{i}:{node_map[i]['role']}" for i in range(num_nodes)]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_matrix, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, cmap="Reds")
    plt.title(f"Gravity Analysis (GAT Attention)\nGame {game_id} Event {event_id}\nProb: {prob:.2%}")
    plt.xlabel("Source Node (Attention Target)")
    plt.ylabel("Target Node (Observer)")
    
    if handler_node_id != -1:
        # ハンドラーの列を強調
        plt.axvline(x=handler_node_id + 0.5, color='blue', linewidth=3)
        plt.text(handler_node_id + 0.5, num_nodes + 0.5, "Handler", color='blue', ha='center', fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"Gravity_{game_id}_{event_id}.png")
    plt.savefig(save_path)
    print(f"✅ Gravity Heatmap saved: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_id', type=str, required=True)
    parser.add_argument('--event_id', type=int, required=True)
    args = parser.parse_args()
    
    gid = args.game_id.zfill(10)
    main(gid, args.event_id)