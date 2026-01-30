import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.lines import Line2D
import os
import argparse
import json
from model import STGAT
from torch_geometric.data import Data

# --- 設定 ---
MODEL_PATH = 'stgat_model_v11.pth'
DATA_CSV = 'cleaned_shots_data_v2.csv'
TRACKING_DIR = './data/2016.NBA.Raw.SportVU.Game.Logs'
OUTPUT_DIR = './final_graph_analysis_v4' # フォルダ名更新

HIDDEN_DIM = 64
NUM_NODE_FEATURES = 9
NUM_EDGE_FEATURES = 1
COURT_LENGTH = 94.0
COURT_WIDTH = 50.0
HALF_COURT_X = 47.0
BASKET_COORDS = np.array([88.75, 25.0])
FPS = 25
PRE_EVENT_SEC = 4.0 
POST_EVENT_SEC = 1.0
MIN_FRAMES = 25

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

# --- 関数群 ---
def parse_time_string(time_str):
    try:
        parts = time_str.split(':')
        if len(parts) == 2: return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 1: return float(parts[0])
    except: pass
    return None

def draw_half_court(ax=None, color='black', lw=2):
    if ax is None: ax = plt.gca()
    hoop = Circle((-41.75, 0), radius=0.75, linewidth=lw, color=color, fill=False)
    backboard = Rectangle((-41.75 - 0.75, -3), 0, 6, linewidth=lw, color=color)
    outer_box = Rectangle((-47, -8), 19, 16, linewidth=lw, color=color, fill=False)
    inner_box = Rectangle((-47, -6), 19, 12, linewidth=lw, color=color, fill=False)
    top_free_throw = Arc((-47 + 19, 0), 12, 12, theta1=270, theta2=90, linewidth=lw, color=color, fill=False)
    bottom_free_throw = Arc((-47 + 19, 0), 12, 12, theta1=90, theta2=270, linewidth=lw, color=color, linestyle='dashed')
    restricted = Arc((-41.75, 0), 8, 8, theta1=270, theta2=90, linewidth=lw, color=color)
    corner_three_a = Rectangle((-47, -22), 14, 0, linewidth=lw, color=color)
    corner_three_b = Rectangle((-47, 22), 14, 0, linewidth=lw, color=color)
    three_arc = Arc((-41.75, 0), 47.5, 47.5, theta1=292, theta2=68, linewidth=lw, color=color)
    center_outer_arc = Arc((0, 0), 12, 12, theta1=90, theta2=270, linewidth=lw, color=color)
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw, bottom_free_throw, restricted, corner_three_a, corner_three_b, three_arc, center_outer_arc]
    outer_lines = Rectangle((-47, -25), 47, 50, linewidth=lw, color=color, fill=False)
    court_elements.append(outer_lines)
    for element in court_elements: ax.add_patch(element)
    return ax

def get_game_moments(json_path):
    if not os.path.exists(json_path): return None
    try:
        with open(json_path, 'r') as f: data = json.load(f)
        all_moments = []
        for event in data['events']: all_moments.extend(event.get('moments', []))
        return all_moments
    except: return None

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
    play_df = play_df.sort_values(by=['player_id', 'moment_index'])
    play_df['dt'] = 0.04 
    play_df['vx'] = play_df['x'].diff() / play_df['dt']
    play_df['vy'] = play_df['y'].diff() / play_df['dt']
    play_df.fillna(0, inplace=True)
    ball_rows = play_df[play_df['player_id'] == -1]
    if ball_rows.empty: return None, None
    ball_start_x = ball_rows['x'].iloc[0]
    if ball_start_x > HALF_COURT_X:
        play_df['x'] = 94.0 - play_df['x']
        play_df['y'] = 50.0 - play_df['y']
        play_df['vx'] = -play_df['vx']
        play_df['vy'] = -play_df['vy']
    play_df['x_norm'] = play_df['x'] / COURT_LENGTH
    play_df['y_norm'] = play_df['y'] / COURT_WIDTH
    play_df['vx_norm'] = play_df['vx'] / 10.0 
    play_df['vy_norm'] = play_df['vy'] / 10.0
    play_df['ax_norm'] = 0; play_df['ay_norm'] = 0 
    play_df['is_offense'] = play_df.apply(lambda row: 1.0 if (row['team_id'] == offense_team_id or row['player_id'] == -1) else 0.0, axis=1)
    
    moment_graphs = []
    for _, m_df in play_df.groupby('moment_index'):
        ball_df = m_df[m_df['player_id'] == -1]
        players_df = m_df[m_df['player_id'] != -1]
        if ball_df.empty or len(players_df) < 5: continue
        
        b_row = ball_df.iloc[0]
        b_coords_norm = np.array([b_row['x_norm'], b_row['y_norm']])
        b_dist_basket = np.linalg.norm(np.array([b_row['x'], b_row['y']]) - BASKET_COORDS) / COURT_LENGTH
        ball_feat = np.array([[b_coords_norm[0], b_coords_norm[1], b_row['vx_norm'], b_row['vy_norm'], 0, 0, b_dist_basket, 0.0, 1.0]])
        
        p_coords = players_df[['x', 'y']].values
        p_coords_norm = players_df[['x_norm', 'y_norm']].values
        b_coords = np.array([b_row['x'], b_row['y']])
        dist_basket = np.linalg.norm(p_coords - BASKET_COORDS, axis=1) / COURT_LENGTH
        dist_ball = np.linalg.norm(p_coords - b_coords, axis=1) / COURT_LENGTH
        player_feats = np.column_stack([p_coords_norm, players_df[['vx_norm', 'vy_norm']].values, np.zeros((len(players_df), 2)), dist_basket, dist_ball, players_df[['is_offense']].values])
        
        all_feats = np.vstack([ball_feat, player_feats])
        x = torch.tensor(all_feats, dtype=torch.float)
        num_nodes = len(all_feats)
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j: edge_index.append([i, j])
        edge_attr = torch.zeros((len(edge_index), 1), dtype=torch.float)
        data = Data(x=x, edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(), edge_attr=edge_attr)
        moment_graphs.append(data)
    return moment_graphs, play_df

# ==========================================
# メイン処理
# ==========================================
def main(game_id, event_id, target_time_str, target_prob):
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    target_sec = parse_time_string(target_time_str)
    if target_sec is None:
        print(f"Error: Invalid time format")
        return

    print("Loading model...")
    model = STGAT(node_features=NUM_NODE_FEATURES, edge_features=NUM_EDGE_FEATURES, hidden_channels=HIDDEN_DIM, out_channels=1).to(device)
    try: model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    except: model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print(f"Processing Game {game_id}, Event {event_id} at {target_time_str}...")
    
    df = pd.read_csv(DATA_CSV, dtype={'GAME_ID': str})
    if 'EVENTNUM' not in df.columns: df['EVENTNUM'] = df.index 
    row = df[(df['GAME_ID'] == game_id) & (df['EVENTNUM'] == event_id)]
    if row.empty: return
    row = row.iloc[0]

    json_path = os.path.join(TRACKING_DIR, f"{game_id}.json")
    moments = get_game_moments(json_path)
    shot_moments = extract_time_window(moments, row['Quarter'], row['SecLeft'])
    if not shot_moments: return

    is_away_play = pd.notna(row['AwayPlay']) and str(row['AwayPlay']).strip() != ""
    team_abbr = row['AwayTeam'] if is_away_play else row['HomeTeam']
    offense_team_id = TEAM_ABBR_TO_ID.get(team_abbr)

    flat_rows = []
    for i, m in enumerate(shot_moments):
        for entity in m[5]:
            flat_rows.append([m[0], m[2], m[3], entity[0], entity[1], entity[2], entity[3], entity[4], i])
    play_df_raw = pd.DataFrame(flat_rows, columns=['quarter', 'game_clock', 'shot_clock', 'team_id', 'player_id', 'x', 'y', 'z', 'moment_index'])
    
    graphs, play_df_processed = calculate_features_normalized(play_df_raw, offense_team_id)

    play_df_processed['time_diff'] = (play_df_processed['game_clock'] - target_sec).abs()
    best_match_df = play_df_processed.sort_values('time_diff').head(1)
    if best_match_df.empty: return
    target_moment_index = best_match_df['moment_index'].values[0]
    actual_time = best_match_df['game_clock'].values[0]
    
    target_graph = graphs[target_moment_index]
    
    with torch.no_grad():
        _, (edge_index, att_weights) = model([target_graph], return_attention=True)
    
    target_frame_df = play_df_processed[play_df_processed['moment_index'] == target_moment_index].reset_index(drop=True)
    ball_row = target_frame_df[target_frame_df['player_id'] == -1].iloc[0]
    handler_node_idx = -1
    min_dist = 100
    node_positions = []
    node_roles = []
    
    for i, row_data in target_frame_df.iterrows():
        plot_x = row_data['x'] - 47.0 
        plot_y = row_data['y'] - 25.0
        node_positions.append((plot_x, plot_y))
        pid = row_data['player_id']
        tid = row_data['team_id']
        is_offense = (tid == offense_team_id) or (pid == -1)
        role = 'Ball' if pid == -1 else ('Off' if is_offense else 'Def')
        node_roles.append(role)
        if role == 'Off':
            d = np.sqrt((row_data['x'] - ball_row['x'])**2 + (row_data['y'] - ball_row['y'])**2)
            if d < min_dist:
                min_dist = d
                handler_node_idx = i

    att_mean = att_weights.mean(dim=1).cpu().numpy()
    edge_index = edge_index.cpu().numpy()
    
    # --- Top Interactions Analysis (Defense Focus) ---
    edge_list = []
    for k in range(len(att_mean)):
        src = edge_index[0, k] 
        tgt = edge_index[1, k] 
        w = att_mean[k]
        if src == tgt: continue
        
        tgt_role_name = "Handler" if tgt == handler_node_idx else node_roles[tgt]
        src_role_name = "Handler" if src == handler_node_idx else node_roles[src]
        
        # ディフェンスが味方以外を見ている場合
        if tgt_role_name == 'Def' and src_role_name != 'Def':
            # ★ 修正: Source(見られる側)がBallなら表記を変更
            tgt_display = f"Def({tgt})"
            
            if src_role_name == 'Ball':
                src_display = "Ball(Handler)"
            else:
                src_display = f"{src_role_name}({src})"
            
            edge_list.append({'src_display': src_display, 'tgt_display': tgt_display, 'weight': w})
    
    edge_list.sort(key=lambda x: x['weight'], reverse=True)
    top_edges = edge_list[:3]

    # --- プロット ---
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    draw_half_court(ax, color='black')
    
    for i, (px, py) in enumerate(node_positions):
        role = node_roles[i]
        if role == 'Ball': c='orange'; s=250; z=20; label='Ball'
        elif role == 'Off': c='red'; s=300; z=15; label='Offense'
        else: c='blue'; s=300; z=15; label='Defense'
        ax.scatter(px, py, c=c, s=s, zorder=z, edgecolors='white', linewidth=1.5)
        
        # ★ 追加: ノードIDを描画 (見やすく白文字・太字で)
        ax.text(px, py, str(i), fontsize=10, color='white', ha='center', va='center', fontweight='bold', zorder=25)

        if i == handler_node_idx:
            ax.scatter(px, py, s=500, facecolors='none', edgecolors='gold', linewidth=3, zorder=z+1)

    max_att = att_mean.max() if len(att_mean) > 0 else 1.0
    for k in range(len(att_mean)):
        src = edge_index[0, k]
        tgt = edge_index[1, k]
        w = att_mean[k]
        
        if w > 0.05:
            viewer_role = node_roles[tgt]
            if viewer_role == 'Def': line_color = 'blue'; alpha_base = 0.6
            elif viewer_role == 'Off' or viewer_role == 'Ball': line_color = 'red'; alpha_base = 0.4
            else: line_color = 'gray'; alpha_base = 0.3

            alpha = min(0.9, (w / max_att) * alpha_base + 0.1)
            width = (w / max_att) * 5.0 + 0.5
            p_src = node_positions[src]
            p_tgt = node_positions[tgt]
            ax.plot([p_tgt[0], p_src[0]], [p_tgt[1], p_src[1]], c=line_color, alpha=alpha, linewidth=width, zorder=10)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Defense', markerfacecolor='blue', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Offense', markerfacecolor='red', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Handler', markerfacecolor='none', markeredgecolor='gold', markeredgewidth=2, markersize=15),
        Line2D([0], [0], color='blue', lw=2, label='Def Attention'),
        Line2D([0], [0], color='red', lw=2, label='Off Connect')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=10, title="Node & Edge Types")

    top_text = "Top 3 Defensive Threats (Gravity):\n"
    for rank, e in enumerate(top_edges, 1):
        top_text += f"{rank}. {e['tgt_display']} -> {e['src_display']} : {e['weight']:.3f}\n"

    plt.text(-46, -24, top_text, fontsize=11, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))

    plt.xlim(-47.5, 0)
    plt.ylim(-25, 25)
    plt.gca().set_aspect('equal', adjustable='box')
    
    time_display = f"{int(actual_time // 60):02d}:{int(actual_time % 60):02d}"
    plt.title(f"Visualized Graph Structure (GAT)\nTime: {time_display} | Prob: {target_prob}", fontsize=14)
    plt.axis('off')
    
    save_name = f"FinalGraph_v4_{time_display.replace(':','-')}_{game_id}.png"
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"✅ Saved Final Analysis: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_id', type=str, required=True)
    parser.add_argument('--event_id', type=int, required=True)
    parser.add_argument('--target_time', type=str, required=True)
    parser.add_argument('--target_prob', type=str, required=True)
    
    args = parser.parse_args()
    main(args.game_id.zfill(10), args.event_id, args.target_time, args.target_prob)