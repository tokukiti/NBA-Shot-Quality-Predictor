import pandas as pd
import numpy as np
import json
import os
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# --- 設定 ---
KAGGLE_PBP_CSV = 'NBA_PBP_2015-16.csv'
TRACKING_DIR = './data/2016.NBA.Raw.SportVU.Game.Logs'
OUTPUT_PT = 'dataset_v10_strict_filter.pt' # ★ v10: 厳密フィルタ版

# コート設定
BASKET_COORDS = np.array([88.75, 25.0])
HALF_COURT_X = 47.0
FPS = 25

# 時間切り出し設定 (CSV時刻基準)
PRE_EVENT_SEC = 5.0 
POST_EVENT_SEC = 1.0
MIN_FRAMES = 25

# チームID変換辞書
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

def load_and_prep_csv():
    print(f"Loading CSV: {KAGGLE_PBP_CSV} ...")
    try:
        df = pd.read_csv(KAGGLE_PBP_CSV, encoding='ISO-8859-1')
    except FileNotFoundError:
        print(f"エラー: {KAGGLE_PBP_CSV} が見つかりません。")
        exit()

    # URLからGAME_ID生成
    unique_urls = df['URL'].unique()
    url_to_id = {url: f"002150{str(i + 1).zfill(4)}" for i, url in enumerate(unique_urls)}
    df['GAME_ID'] = df['URL'].map(url_to_id)
    
    # データ欠損試合の除外
    missing_games = ['0021500006', '0021500008', '0021500014']
    df = df[~df['GAME_ID'].isin(missing_games)]

    # 対象試合の絞り込み (最初の20試合)
    unique_game_ids = df['GAME_ID'].unique()
    target_game_ids = unique_game_ids[:20] if len(unique_game_ids) >= 20 else unique_game_ids
    print(f"Target Games ({len(target_game_ids)} matches):")
    print(target_game_ids)

    df = df[df['GAME_ID'].isin(target_game_ids)].copy()

    # ★★★ 厳密なフィルタリング ★★★
    print(f"Original CSV rows: {len(df)}")
    
    # 1. ShotOutcome が 'make' か 'miss' のものだけ残す
    if 'ShotOutcome' in df.columns:
        df['ShotOutcome'] = df['ShotOutcome'].astype(str).str.lower().str.strip()
        df = df[df['ShotOutcome'].isin(['make', 'miss'])]
    
    # 2. Shooter名が入っているものだけ残す (リバウンド等を除外)
    if 'Shooter' in df.columns:
        df = df[df['Shooter'].notna()]
        
    print(f"Filtered Shot rows: {len(df)}")

    # 数値変換
    df['SecLeft'] = pd.to_numeric(df['SecLeft'], errors='coerce').fillna(0).astype(int)
    df['Quarter'] = pd.to_numeric(df['Quarter'], errors='coerce').fillna(1).astype(int)
    
    # EVENTNUM生成 (フィルタリング後に行うと番号が飛びますが、IDとしてのユニーク性は保たれます)
    # 元のCSVの行番号をEVENTNUMとして保持したい場合は、フィルタ前に振るべきですが、
    # ここでは単純に「このデータセット内でのID」として扱います。
    df['EVENTNUM'] = range(len(df))
    
    return df, target_game_ids

def get_game_moments(json_path):
    if not os.path.exists(json_path): return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        all_moments = []
        for event in data['events']:
            all_moments.extend(event.get('moments', []))
        return all_moments
    except Exception as e:
        print(f"JSON Error: {e}")
        return None

def extract_time_window(moments, target_q, target_sec):
    """CSVの時間(target_sec)と一致するフレームを探し、前後を切り出す"""
    # 該当クォーターのみ抽出
    indices_in_q = [i for i, m in enumerate(moments) if m[0] == target_q]
    if not indices_in_q: return None

    # 最も時間が近いフレームを探す
    best_idx = -1
    min_diff = 9999.0
    
    for i in indices_in_q:
        time_remaining = moments[i][2]
        diff = abs(time_remaining - target_sec)
        if diff < min_diff:
            min_diff = diff
            best_idx = i
        if diff < 0.1: break # 0.1秒以内の誤差なら即決
            
    # 2秒以上ズレてたら対象外とする (全く別の時間帯を拾うのを防ぐ)
    if best_idx == -1 or min_diff > 2.0: 
        return None

    # 切り出し
    start_idx = max(0, best_idx - int(PRE_EVENT_SEC * FPS))
    end_idx = min(len(moments), best_idx + int(POST_EVENT_SEC * FPS))
    
    if end_idx - start_idx < MIN_FRAMES:
        return None
        
    return moments[start_idx : end_idx]

def calculate_kinematics_and_graph(play_df, label, game_id, event_id, offense_team_id):
    play_df = play_df.sort_values(by=['player_id', 'moment_index'])
    play_df['dt'] = 0.04 
    
    play_df['vx'] = play_df['x'].diff() / play_df['dt']
    play_df['vy'] = play_df['y'].diff() / play_df['dt']
    play_df['ax'] = play_df['vx'].diff() / play_df['dt']
    play_df['ay'] = play_df['vy'].diff() / play_df['dt']
    play_df.fillna(0, inplace=True)

    ball_start_x = play_df[play_df['player_id'] == -1]['x'].iloc[0]
    if ball_start_x < HALF_COURT_X:
        play_df['x'] = 94.0 - play_df['x']
        play_df['y'] = 50.0 - play_df['y']
        play_df['vx'] = -play_df['vx']
        play_df['vy'] = -play_df['vy']
        play_df['ax'] = -play_df['ax']
        play_df['ay'] = -play_df['ay']

    play_df['is_offense'] = play_df.apply(
        lambda row: 1.0 if (row['team_id'] == offense_team_id or row['player_id'] == -1) else 0.0, 
        axis=1
    )

    moment_graphs = []
    for _, m_df in play_df.groupby('moment_index'):
        ball_df = m_df[m_df['player_id'] == -1]
        players_df = m_df[m_df['player_id'] != -1]
        
        if ball_df.empty or len(players_df) < 10: continue

        p_coords = players_df[['x', 'y']].values
        b_coords = ball_df[['x', 'y']].iloc[0].values

        dist_basket = np.linalg.norm(p_coords - BASKET_COORDS, axis=1)
        dist_ball = np.linalg.norm(p_coords - b_coords, axis=1)
        
        node_feats = np.column_stack([
            players_df[['x', 'y', 'vx', 'vy', 'ax', 'ay']].values,
            dist_basket, dist_ball, players_df[['is_offense']].values
        ])
        
        x = torch.tensor(node_feats, dtype=torch.float)
        
        # エッジ作成 (全結合)
        num_nodes = len(players_df)
        edge_index, edge_attr = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
                    dist = np.linalg.norm(node_feats[i, :2] - node_feats[j, :2])
                    edge_attr.append([dist])
        
        data = Data(
            x=x, 
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(), 
            edge_attr=torch.tensor(edge_attr, dtype=torch.float)
        )
        moment_graphs.append(data)
    
    if not moment_graphs: return None

    return {
        'game_id': game_id,
        'event_id': event_id,
        'label': torch.tensor([label], dtype=torch.long), 
        'graphs': moment_graphs
    }

def main():
    pbp_df, target_game_ids = load_and_prep_csv()
    final_dataset = []

    print("\n--- Processing Games (Strict Shot Filtering) ---")
    
    for game_id in target_game_ids:
        game_shots = pbp_df[pbp_df['GAME_ID'] == game_id]
        if game_shots.empty: continue

        json_path = os.path.join(TRACKING_DIR, f"{game_id}.json")
        if not os.path.exists(json_path):
            print(f"Skipping {game_id}: JSON not found.")
            continue
            
        print(f"Processing {game_id}: {len(game_shots)} shots...")
        
        all_moments = get_game_moments(json_path)
        if all_moments is None: continue
            
        valid_count = 0
        
        for _, row in tqdm(game_shots.iterrows(), total=len(game_shots), leave=False):
            target_q = row['Quarter']
            target_sec = row['SecLeft']
            event_id = row['EVENTNUM']
            
            is_away_play = pd.notna(row['AwayPlay']) and str(row['AwayPlay']).strip() != ""
            team_abbr = row['AwayTeam'] if is_away_play else row['HomeTeam']
            offense_team_id = TEAM_ABBR_TO_ID.get(team_abbr)
            
            if offense_team_id is None: continue

            label = 1 if row['ShotOutcome'] == 'make' else 0
            
            shot_moments = extract_time_window(all_moments, target_q, target_sec)
            
            if shot_moments:
                flat_rows = []
                for i, m in enumerate(shot_moments):
                    for entity in m[5]:
                        flat_rows.append([
                            m[0], m[2], m[3], 
                            entity[0], entity[1], entity[2], entity[3], entity[4], 
                            i 
                        ])
                
                play_df = pd.DataFrame(flat_rows, columns=[
                    'quarter', 'game_clock', 'shot_clock', 
                    'team_id', 'player_id', 'x', 'y', 'z', 'moment_index'
                ])
                
                graph_data = calculate_kinematics_and_graph(play_df, label, game_id, event_id, offense_team_id)
                if graph_data:
                    final_dataset.append(graph_data)
                    valid_count += 1
        
        print(f" -> {game_id}: Created {valid_count} valid shot graphs.")

    if final_dataset:
        torch.save(final_dataset, OUTPUT_PT)
        print(f"\n✅ 完了！ {len(final_dataset)} プレーを保存しました: {OUTPUT_PT}")
    else:
        print("\n❌ データが生成されませんでした。")

if __name__ == "__main__":
    main()