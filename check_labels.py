import torch
import random
import os
import sys
from tqdm import tqdm

# 既存の可視化ツールをインポート
# ※ visualize_dataset_check.py が同じフォルダにある必要があります
try:
    from visualize_dataset_check import visualize_dataset_window
except ImportError:
    print("エラー: visualize_dataset_check.py が見つかりません。")
    sys.exit(1)

# --- 設定 ---
DATA_PATH = 'dataset_v12_classification.pt'
OUTPUT_DIR = './output_label_check' # 確認用GIFの保存先
SAMPLES_PER_CLASS = 3 # 各クラス何個確認するか

def main():
    if not os.path.exists(DATA_PATH):
        print(f"エラー: データセット {DATA_PATH} がありません。")
        return

    print(f"Loading dataset: {DATA_PATH} ...")
    try:
        dataset = torch.load(DATA_PATH, weights_only=False)
    except:
        dataset = torch.load(DATA_PATH)

    # ラベルごとにデータを分類
    class_indices = {0: [], 1: [], 2: []}
    
    for i, data in enumerate(dataset):
        label = data['play_type_label'].item()
        if label in class_indices:
            class_indices[label].append(data)

    # 保存先作成
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    label_names = {0: "Isolation", 1: "PickRoll", 2: "PickPop"}

    print("\n--- Generating Check GIFs ---")

    for label, name in label_names.items():
        samples = class_indices[label]
        count = len(samples)
        print(f"\nChecking Class {label}: {name} (Total: {count})")
        
        if count == 0:
            print(" -> データがありません。")
            continue

        # ランダムに抽出
        selected_samples = random.sample(samples, min(count, SAMPLES_PER_CLASS))
        
        for i, sample in enumerate(selected_samples):
            game_id = sample['game_id']
            event_id = sample['event_id']
            
            print(f" -> [{i+1}/{SAMPLES_PER_CLASS}] Generating GIF for Game {game_id}, Event {event_id}...")
            
            # GIF生成関数の呼び出し
            # visualize_dataset_check.py の関数を利用
            # ※保存先を一時的に変更するか、ファイル名で区別する工夫が必要ですが、
            #   ここでは visualize_dataset_check の出力先設定に依存します。
            #   作成されたGIFを手動で確認してください。
            
            try:
                visualize_dataset_window(game_id, event_id)
                
                # 生成されたファイルをリネームして整理（オプション）
                # output_check_viz フォルダにできているはずなので、それを移動・リネーム
                src_dir = './output_check_viz'
                for f in os.listdir(src_dir):
                    if f.startswith(f"Check_{game_id}_Event_{event_id}"):
                        src_path = os.path.join(src_dir, f)
                        dst_name = f"Label{label}_{name}_{game_id}_{event_id}.gif"
                        dst_path = os.path.join(OUTPUT_DIR, dst_name)
                        
                        if os.path.exists(src_path):
                            os.replace(src_path, dst_path)
                            print(f"    Saved to: {dst_path}")
            except Exception as e:
                print(f"    Error: {e}")

    print(f"\n✅ 全確認フローが完了しました。フォルダ '{OUTPUT_DIR}' を確認してください。")

if __name__ == "__main__":
    main()