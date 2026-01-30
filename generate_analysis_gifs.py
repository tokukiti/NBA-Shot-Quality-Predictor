import os
import sys
import pandas as pd

# 既存の可視化ツールをインポート
try:
    from visualize_dataset_check import visualize_dataset_window
except ImportError:
    print("エラー: visualize_dataset_check.py が見つかりません。")
    sys.exit(1)

# --- 設定 ---
OUTPUT_DIR = './analysis_output_gifs' # 保存先

# あなたが抽出した注目プレーのリスト (GameID, EventID)
# ※頭の0が抜けていても、コード内で自動補完します
target_plays = [
    # --- Good Process, Bad Result (AI:入る -> 結果:Miss) ---
    {'game_id': 21500001, 'event_id': 30,   'prob': 0.95, 'type': 'GoodProcess_BadResult'},
    {'game_id': 21500015, 'event_id': 1944, 'prob': 0.95, 'type': 'GoodProcess_BadResult'},
    {'game_id': 21500022, 'event_id': 3133, 'prob': 0.94, 'type': 'GoodProcess_BadResult'},
    {'game_id': 21500002, 'event_id': 247,  'prob': 0.94, 'type': 'GoodProcess_BadResult'},
    {'game_id': 21500003, 'event_id': 468,  'prob': 0.93, 'type': 'GoodProcess_BadResult'},
    {'game_id': 21500005, 'event_id': 750,  'prob': 0.91, 'type': 'GoodProcess_BadResult'},
    {'game_id': 21500022, 'event_id': 3285, 'prob': 0.90, 'type': 'GoodProcess_BadResult'},
    {'game_id': 21500009, 'event_id': 1170, 'prob': 0.90, 'type': 'GoodProcess_BadResult'},
    {'game_id': 21500009, 'event_id': 1097, 'prob': 0.90, 'type': 'GoodProcess_BadResult'},
    {'game_id': 21500016, 'event_id': 2210, 'prob': 0.88, 'type': 'GoodProcess_BadResult'},
    

    # --- Bad Process, Good Result (AI:無理 -> 結果:Make) ---
    {'game_id': 21500023, 'event_id': 3394, 'prob': 0.04, 'type': 'BadProcess_GoodResult'},
    {'game_id': 21500020, 'event_id': 2909, 'prob': 0.06, 'type': 'BadProcess_GoodResult'},
    {'game_id': 21500011, 'event_id': 1463, 'prob': 0.07, 'type': 'BadProcess_GoodResult'},
    {'game_id': 21500005, 'event_id': 734,  'prob': 0.10, 'type': 'BadProcess_GoodResult'},
    {'game_id': 21500019, 'event_id': 2746, 'prob': 0.11, 'type': 'BadProcess_GoodResult'},
    {'game_id': 21500012, 'event_id': 1603, 'prob': 0.11, 'type': 'BadProcess_GoodResult'},
    {'game_id': 21500002, 'event_id': 323,  'prob': 0.12, 'type': 'BadProcess_GoodResult'},
    {'game_id': 21500022, 'event_id': 3192, 'prob': 0.12, 'type': 'BadProcess_GoodResult'},
    {'game_id': 21500018, 'event_id': 2480, 'prob': 0.13, 'type': 'BadProcess_GoodResult'},
    {'game_id': 21500011, 'event_id': 1424, 'prob': 0.15, 'type': 'BadProcess_GoodResult'},
]

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"フォルダ作成: {OUTPUT_DIR}")

    print(f"--- GIF生成開始 ({len(target_plays)}件) ---")

    for i, play in enumerate(target_plays):
        # IDの0埋め処理 (例: 21500001 -> "0021500001")
        raw_id = str(play['game_id'])
        game_id_str = "00" + raw_id if len(raw_id) == 8 else raw_id
        
        event_id = play['event_id']
        play_type = play['type']
        prob = play['prob']
        
        print(f"[{i+1}/{len(target_plays)}] Processing Game: {game_id_str}, Event: {event_id} ({play_type})")
        
        try:
            # 可視化実行 (visualize_dataset_check.py の関数を使用)
            # ※この関数は output_check_viz に保存するので、後で移動する
            visualize_dataset_window(game_id_str, event_id)
            
            # ファイルの移動とリネーム
            src_dir = './output_check_viz'
            moved = False
            for f in os.listdir(src_dir):
                # 生成されたファイルを探す
                if f.startswith(f"Check_{game_id_str}_Event_{event_id}"):
                    src_path = os.path.join(src_dir, f)
                    
                    # わかりやすい名前に変更して保存
                    # 例: GoodProcess_Prob95_Game00215...gif
                    new_name = f"{play_type}_Prob{int(prob*100)}_{game_id_str}_{event_id}.gif"
                    dst_path = os.path.join(OUTPUT_DIR, new_name)
                    
                    if os.path.exists(src_path):
                        os.replace(src_path, dst_path)
                        print(f"    --> Saved to: {dst_path}")
                        moved = True
            
            if not moved:
                print("    Warning: GIF file not found in temp folder.")

        except Exception as e:
            print(f"    Error: {e}")

    print(f"\n✅ 全て完了しました！ '{OUTPUT_DIR}' を確認してください。")

if __name__ == "__main__":
    main()