import pandas as pd
import numpy as np
import json
import os
import argparse
from Event import Event
from Constant import Constant
import matplotlib.pyplot as plt

# --- 設定 (make_dataset_classification.py と合わせる) ---
# ★ 修正: 生データではなく、クリーニング済みデータを参照する
TARGET_CSV = 'cleaned_shots_data_v2.csv' 
TRACKING_DIR = './data/2016.NBA.Raw.SportVU.Game.Logs'
OUTPUT_DIR = './output_check_viz' # 確認用GIFの保存先

# 切り出しロジック用定数
RIM_HEIGHT = 10.0
FPS = 25
WINDOW_PRE_PEAK = 3.0
WINDOW_POST_PEAK = 1.5 
MIN_FRAMES = 25

# ==========================================
# make_dataset.py から移植したロジック関数群
# ==========================================
def build_tracking_index(moments):
    """モーメントを (Quarter, GameClock整数部) でインデックス化"""
    time_index = {}
    for i, m in enumerate(moments):
        q = m[0]
        t_int = int(m[2]) 
        key = (q, t_int)
        if key not in time_index:
            time_index[key] = []
        time_index[key].append(i)
    return time_index

def get_shot_window_moments(moments, time_index, target_q, target_sec):
    """指定時刻周辺のデータを探索し、ボールのピークを中心にウィンドウを切り出す"""
    # 1. CSVの時刻周辺(前後4秒)のインデックス候補を取得
    candidate_indices = []
    for offset in range(-4, 5): 
        key = (target_q, target_sec + offset)
        if key in time_index:
            candidate_indices.extend(time_index[key])
    
    if not candidate_indices:
        print("  -> 指定時刻周辺のトラッキングデータが見つかりませんでした。")
        return None

    candidate_indices.sort()
    start_search = candidate_indices[0]
    end_search = candidate_indices[-1]
    
    # 2. 候補区間内のボールのZ座標を取得
    raw_segment = moments[start_search : end_search+1]
    ball_z_list = []
    for m in raw_segment:
        # player_id が -1 のデータがボール
        ball = next((x for x in m[5] if x[0] == -1), None)
        ball_z_list.append(ball[4] if ball else 0) # ball[4] が Z座標
    
    ball_z_arr = np.array(ball_z_list)
    if len(ball_z_arr) == 0: 
        print("  -> 候補区間にボールデータがありません。")
        return None

    # 3. ボールの最高到達点がリング(10ft)より低い場合は除外
    max_z = np.max(ball_z_arr)
    if max_z < RIM_HEIGHT: 
        print(f"  -> ボールの最高点が低すぎます (Max Z: {max_z:.2f} < {RIM_HEIGHT})。")
        return None 

    # 4. ピーク位置を特定し、前後を切り出す
    local_peak_idx = np.argmax(ball_z_arr)
    global_peak_idx = start_search + local_peak_idx

    extract_start = max(0, global_peak_idx - int(WINDOW_PRE_PEAK * FPS))
    extract_end = min(len(moments), global_peak_idx + int(WINDOW_POST_PEAK * FPS))

    if (extract_end - extract_start) < MIN_FRAMES:
        print("  -> 切り出したフレーム数が少なすぎます。")
        return None

    print(f"  -> ウィンドウ切り出し成功: フレーム {extract_start} から {extract_end} (ピーク: {global_peak_idx})")
    # 切り出したモーメントのリストを返す
    return moments[extract_start : extract_end]


# ==========================================
# メイン可視化関数
# ==========================================
def visualize_dataset_window(game_id, event_id_target):
    # 1. CSVから該当プレーの情報を取得
    print(f"Loading CSV: {TARGET_CSV} ...")
    if not os.path.exists(TARGET_CSV):
        print(f"Error: {TARGET_CSV} not found.")
        return

    # ★ 修正: GAME_IDを文字列として読み込む (0落ち防止)
    df = pd.read_csv(TARGET_CSV, dtype={'GAME_ID': str})
    
    # URL列がない場合（cleaned_shots_data.csv）はID生成をスキップ
    if 'URL' in df.columns and 'GAME_ID' not in df.columns:
        unique_urls = df['URL'].unique()
        url_to_id = {url: f"002150{str(i + 1).zfill(4)}" for i, url in enumerate(unique_urls)}
        df['GAME_ID'] = df['URL'].map(url_to_id)
    
    if 'EVENTNUM' not in df.columns:
        df['EVENTNUM'] = range(len(df))
    
    # 該当プレーを検索
    target_play = df[(df['GAME_ID'] == game_id) & (df['EVENTNUM'] == event_id_target)]
    
    if target_play.empty:
        print(f"Error: GameID={game_id}, EventID={event_id_target} がCSVに見つかりません。")
        return
    
    target_row = target_play.iloc[0]
    target_q = int(target_row['Quarter'])
    
    # 秒数の欠損値処理
    sec_val = target_row['SecLeft']
    try:
        target_sec = int(sec_val)
    except:
        target_sec = 0
        
    outcome = target_row['ShotOutcome']
    
    print(f"Found Play in CSV: Q{target_q}, SecLeft={target_sec}, Outcome={outcome}")

    # 2. JSONデータの読み込み
    json_path = os.path.join(TRACKING_DIR, f"{game_id}.json")
    print(f"Loading Tracking Data: {json_path} ...")
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found.")
        return
    
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # 3. 全モーメントを展開し、インデックスを作成
    all_moments = []
    target_event_metadata = None

    # メタデータ取得用（cleanedデータの場合、元のeventidは失われている可能性があるので、
    # 最初のイベント情報をダミーとして使うか、あれば使う）
    
    # JSON全部読み込む
    for event in json_data['events']:
        all_moments.extend(event.get('moments', []))

    # メタデータは可視化（チーム名など）に必要なので、とりあえず最初のイベントを使う
    target_event_metadata = json_data['events'][0]

    time_index = build_tracking_index(all_moments)

    # 4. make_dataset.py と同じロジックでウィンドウを切り出す
    print("Applying window extraction logic...")
    shot_moments = get_shot_window_moments(all_moments, time_index, target_q, target_sec)

    if shot_moments is None:
        print("❌ データセット作成ロジックでの切り出しに失敗しました。このプレーはデータセットに含まれていません。")
        return

    # 5. 切り出したデータを使って Event オブジェクトを作成
    sliced_event_dict = target_event_metadata.copy()
    sliced_event_dict['moments'] = shot_moments 
    
    event_viz = Event(sliced_event_dict)

    # 6. 保存と表示
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    output_filename = os.path.join(OUTPUT_DIR, f"Check_{game_id}_Event_{event_id_target}_{outcome}.gif")
    print(f"Generating GIF animation (this takes time)...")
    
    try:
        # Event.py の show メソッドを利用
        event_viz.show(save_path=output_filename)
        print(f"✅ Animation saved: {output_filename}")
        print("作成されたGIFを確認してください。シュートの頂点がアニメーションの中盤に来ていれば成功です。")
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize the EXACT window used in dataset generation.')
    parser.add_argument('--game_id', type=str, required=True, help='Game ID (e.g., 0021500001)')
    parser.add_argument('--event_id', type=int, required=True, help='Event ID (e.g., 20)')
    
    args = parser.parse_args()

    visualize_dataset_window(args.game_id, args.event_id)