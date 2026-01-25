import pandas as pd
import numpy as np
import json
import os
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import math

# --- 設定 ---
INPUT_CSV = 'cleaned_shots_data_v2.csv' 
if not os.path.exists(INPUT_CSV): INPUT_CSV = 'cleaned_shots_data.csv'

TRACKING_DIR = './data/2016.NBA.Raw.SportVU.Game.Logs'
OUTPUT_PT = 'dataset_v17_classification_v6.pt' # ★ v17: FT静止判定 & 距離フィルタ

FPS = 25
HALF_COURT_X = 47.0
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

def get_dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# ==========================================
# ★ v6 フィルタリングロジック ★
# ==========================================

def is_static_play(moments, threshold=2.0):
    """
    フリースロー除去用:
    プレー開始直後の全選手の平均移動距離が極端に少ない場合、静止状態(FT)とみなす。
    threshold: 平均移動距離(ft)の許容値
    """
    if len(moments) < 10: return True
    
    # 最初の10フレーム(約0.4秒)の動きを見る
    start_frame = moments[0]
    check_frame = moments[9] # 10フレーム後
    
    total_move_dist = 0
    player_count = 0
    
    # 選手IDごとに位置を取得して比較
    start_players = {p[1]: [p[2], p[3]] for p in start_frame[5][1:]}
    
    for p in check_frame[5][1:]:
        pid = p[1]
        if pid in start_players:
            pos_start = start_players[pid]
            pos_curr = [p[2], p[3]]
            total_move_dist += get_dist(pos_start, pos_curr)
            player_count += 1
            
    if player_count == 0: return True
    
    avg_move = total_move_dist / player_count
    
    # 全員合わせて平均 0.5フィート(15cm)も動いていないなら静止とみなす
    # フリースロー中はほぼ不動、通常のプレーなら誰かが走り出しているはず
    return avg_move < 0.5 

def is_garbage_range(moments, basket_coords):
    """
    バックコート除去用:
    シュート時のボール位置がリングから遠すぎる場合(35ft以上)は除外
    """
    # 最後のフレーム（シュート時付近）
    last_frame = moments[-1]
    ball = last_frame[5][0]
    ball_pos = [ball[2], ball[3]]
    
    dist = get_dist(ball_pos, basket_coords)
    
    # 3ポイントライン(約24ft)よりはるか遠く、35ft以上ならロゴショットかバックコート
    if dist > 35.0:
        return True
    return False

# ==========================================
# ★ 戦術判定 (3クラス版) ★
# ==========================================
def identify_play_type_v6(moments, offense_team_id):
    """
    Return: 0:Other, 1:PnR, 2:PnP
    """
    first_ball = moments[0][5][0]
    ball_x = first_ball[2]
    basket_coords = [5.35, 25.0] if ball_x < HALF_COURT_X else [88.65, 25.0]

    # ★ フィルタリング実行
    # 1. 静止チェック (FT除去)
    if is_static_play(moments):
        return -1 # 除外フラグ
        
    # 2. 距離チェック (バックコート除去)
    if is_garbage_range(moments, basket_coords):
        return -1 # 除外フラグ

    max_screen_frames = 0
    screener_start_pos = None
    screener_end_pos = None
    
    for i, m in enumerate(moments):
        ball = m[5][0]
        ball_pos = [ball[2], ball[3]]
        
        offense_players = []
        defense_players = []
        for entity in m[5][1:]:
            team_id, pid, x, y, z = entity
            pos = [x, y]
            if team_id == offense_team_id:
                offense_players.append({'id': pid, 'pos': pos})
            else:
                defense_players.append({'id': pid, 'pos': pos})
        
        # ハンドラー特定
        current_handler = None
        min_dist = 100
        for p in offense_players:
            d = get_dist(ball_pos, p['pos'])
            if d < min_dist:
                min_dist = d
                current_handler = p
        
        if min_dist > 4.0: continue
        handler_pos = current_handler['pos']
        
        # スクリーン判定
        is_screen_now = False
        for p in offense_players:
            if p['id'] == current_handler['id']: continue
            
            dist_mate = get_dist(handler_pos, p['pos'])
            # 距離条件
            if dist_mate < 7.0:
                nearest_def = 100
                for d in defense_players:
                    dd = get_dist(handler_pos, d['pos'])
                    if dd < nearest_def: nearest_def = dd
                
                if nearest_def < 7.0:
                    is_screen_now = True
                    if max_screen_frames == 0:
                        screener_start_pos = p['pos']
                    screener_end_pos = p['pos']
                    break
        
        if is_screen_now:
            max_screen_frames += 1
    
    # 分類
    if max_screen_frames >= 10:
        if screener_start_pos and screener_end_pos:
            start_dist = get_dist(screener_start_pos, basket_coords)
            end_dist = get_dist(screener_end_pos, basket_coords)
            delta = start_dist - end_dist
            
            if delta > 2.0:
                return 1 # PnR
            else:
                return 2 # PnP
        return 1
    
    return 0 # Other

# ==========================================
# メイン処理 (変更点なし)
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

def calculate_kinematics_and_graph(play_df, label, game_id, event_id, offense_team_id, play_type_label):
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
    play_df['is_offense'] = play_df.apply(lambda row: 1.0 if (row['team_id'] == offense_team_id or row['player_id'] == -1) else 0.0, axis=1)
    moment_graphs = []
    for _, m_df in play_df.groupby('moment_index'):
        ball_df = m_df[m_df['player_id'] == -1]
        players_df = m_df[m_df['player_id'] != -1]
        if ball_df.empty or len(players_df) < 10: continue
        p_coords = players_df[['x', 'y']].values
        b_coords = ball_df[['x', 'y']].iloc[0].values
        basket_normalized = np.array([88.75, 25.0])
        dist_basket = np.linalg.norm(p_coords - basket_normalized, axis=1)
        dist_ball = np.linalg.norm(p_coords - b_coords, axis=1)
        node_feats = np.column_stack([players_df[['x', 'y', 'vx', 'vy', 'ax', 'ay']].values, dist_basket, dist_ball, players_df[['is_offense']].values])
        x = torch.tensor(node_feats, dtype=torch.float)
        num_nodes = len(players_df)
        edge_index, edge_attr = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
                    dist = np.linalg.norm(node_feats[i, :2] - node_feats[j, :2])
                    edge_attr.append([dist])
        data = Data(x=x, edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(), edge_attr=torch.tensor(edge_attr, dtype=torch.float))
        moment_graphs.append(data)
    if not moment_graphs: return None
    return {'game_id': game_id, 'event_id': event_id, 'label': torch.tensor([label], dtype=torch.long), 'play_type_label': torch.tensor([play_type_label], dtype=torch.long), 'graphs': moment_graphs}

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"エラー: {INPUT_CSV} がありません。")
        return
    df = pd.read_csv(INPUT_CSV, dtype={'GAME_ID': str})
    unique_games = df['GAME_ID'].unique()
    target_games = unique_games[:20] 
    df = df[df['GAME_ID'].isin(target_games)]
    
    print(f"--- Generating Classification Dataset (v17: Strict Filtering) ---")
    
    final_dataset = []
    skipped_ft = 0
    
    for game_id in tqdm(target_games, desc="Games"):
        game_shots = df[df['GAME_ID'] == game_id]
        json_filename = f"{game_id}.json"
        json_path = os.path.join(TRACKING_DIR, json_filename)
        all_moments = get_game_moments(json_path)
        if all_moments is None: continue
        
        for _, row in game_shots.iterrows():
            target_q = row['Quarter']
            target_sec = row['SecLeft']
            is_away_play = pd.notna(row['AwayPlay']) and str(row['AwayPlay']).strip() != ""
            team_abbr = row['AwayTeam'] if is_away_play else row['HomeTeam']
            offense_team_id = TEAM_ABBR_TO_ID.get(team_abbr)
            if offense_team_id is None: continue
            
            shot_moments = extract_time_window(all_moments, target_q, target_sec)
            
            if shot_moments:
                # ★ v6ロジック (除外判定含む)
                play_type = identify_play_type_v6(shot_moments, offense_team_id)
                
                if play_type == -1: # 除外フラグ
                    skipped_ft += 1
                    continue
                
                flat_rows = []
                for i, m in enumerate(shot_moments):
                    for entity in m[5]:
                        flat_rows.append([m[0], m[2], m[3], entity[0], entity[1], entity[2], entity[3], entity[4], i])
                play_df = pd.DataFrame(flat_rows, columns=['quarter', 'game_clock', 'shot_clock', 'team_id', 'player_id', 'x', 'y', 'z', 'moment_index'])
                label = 1 if row['ShotOutcome'] == 'make' else 0
                graph_data = calculate_kinematics_and_graph(play_df, label, game_id, row['EVENTNUM'], offense_team_id, play_type)
                if graph_data: final_dataset.append(graph_data)

    if final_dataset:
        torch.save(final_dataset, OUTPUT_PT)
        print(f"\n✅ 完了！ 保存先: {OUTPUT_PT}")
        print(f"🚫 削除されたFT/バックコート: {skipped_ft}件")
        
        labels = [d['play_type_label'].item() for d in final_dataset]
        from collections import Counter
        counts = Counter(labels)
        print("【戦術ラベルの内訳 (v6)】")
        print(f"  0: Other (No Screen) = {counts[0]}")
        print(f"  1: Pick & Roll       = {counts[1]}")
        print(f"  2: Pick & Pop        = {counts[2]}")
    else:
        print("\n❌ データが生成されませんでした。")

if __name__ == "__main__":
    main()