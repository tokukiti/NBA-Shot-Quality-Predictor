import pandas as pd
import numpy as np
import re
import sys

# --- 設定 ---
INPUT_CSV = 'NBA_PBP_2015-16.csv'
OUTPUT_CSV = 'cleaned_shots_data_v2.csv' 

def run_pipeline():
    print(f"Loading raw CSV: {INPUT_CSV}...")
    try:
        # DtypeWarningを防ぐため low_memory=False を指定
        df = pd.read_csv(INPUT_CSV, encoding='ISO-8859-1', low_memory=False)
    except FileNotFoundError:
        print(f"エラー: {INPUT_CSV} が見つかりません。")
        sys.exit(1)

    print(f"Original rows: {len(df)}")

    # ==========================================
    # 1. GAME_ID の生成
    # ==========================================
    # URLカラムがある場合、そこからIDを抽出するのが最も確実
    if 'URL' in df.columns:
        unique_urls = df['URL'].unique()
        # URLからIDへのマッピング辞書を作成 (例: /boxscore/0021500001 -> 0021500001)
        # ※もしURLが空の場合は連番を振るロジックになっています
        url_to_id = {url: f"002150{str(i + 1).zfill(4)}" for i, url in enumerate(unique_urls)}
        df['GAME_ID'] = df['URL'].map(url_to_id)
    elif 'GAME_ID' not in df.columns:
        print("警告: 'URL' も 'GAME_ID' も見つかりません。データが正しく処理できない可能性があります。")

    # ==========================================
    # 2. 欠損試合の除外
    # ==========================================
    missing_games = ['0021500006', '0021500008', '0021500014']
    if 'GAME_ID' in df.columns:
        df = df[~df['GAME_ID'].isin(missing_games)]

    # ==========================================
    # 3. ★重要: シュートデータの厳密な抽出★
    # ==========================================
    
    # 【優先策】EVENTMSGTYPE (公式コード) がある場合
    # NBAデータでは通常、1=Make(FG), 2=Miss(FG), 3=Free Throw と決まっているためこれが最強
    if 'EVENTMSGTYPE' in df.columns:
        print("Filtering by EVENTMSGTYPE (1=Make, 2=Miss)...")
        # 1と2だけ残すことで、フリースロー(3)やその他を自動的に除外
        df = df[df['EVENTMSGTYPE'].isin([1, 2])]
    
    # 【次善策】ShotOutcome (文字列) で判定する場合
    # EVENTMSGTYPEがない、または念のためのダブルチェック
    else:
        print("EVENTMSGTYPE not found. Using strict string filtering...")
        if 'ShotOutcome' in df.columns:
            # 大文字小文字を無視して 'make', 'miss' のみを抽出
            outcome_mask = df['ShotOutcome'].astype(str).str.lower().str.strip().isin(['make', 'miss'])
            df = df[outcome_mask]
        
        # ★★★ ノイズ除去 (pipeline3の意図 + pipeline3-2の技術) ★★★
        # "Free Throw", "Technical", "1 of 2" などの文字列が含まれる行を正規表現で除外
        noise_pattern = re.compile(r'Free Throw|1 of |2 of |3 of |Technical|Defensive 3 Seconds', re.IGNORECASE)
        
        # HomePlay と AwayPlay の内容を結合してチェック
        combined_desc = df['HomePlay'].fillna('') + " " + df['AwayPlay'].fillna('')
        
        # ノイズを含まない行だけを残す
        df = df[~combined_desc.apply(lambda x: bool(noise_pattern.search(x)))]

    # ==========================================
    # 4. データの整形
    # ==========================================

    # 選手名(Shooter)がないデータは座標を持っていても意味がないので削除
    if 'Shooter' in df.columns:
        df = df[df['Shooter'].notna()]
    
    # 時間データの整形 (数値化)
    if 'Quarter' in df.columns:
        df['Quarter'] = pd.to_numeric(df['Quarter'], errors='coerce').fillna(1).astype(int)
    if 'SecLeft' in df.columns:
        df['SecLeft'] = pd.to_numeric(df['SecLeft'], errors='coerce').fillna(0).astype(int)

    # インデックスを振り直して、新しい一意なID (EVENTNUM) を作成
    df = df.reset_index(drop=True)
    df['EVENTNUM'] = df.index  # これが後のデータセット作成で重要になります

    # ==========================================
    # 5. 列の選定と保存
    # ==========================================
    
    # 残したい列のリスト
    target_columns = [
        'GAME_ID', 'EVENTNUM', 'Quarter', 'SecLeft', 'Time', 
        'HomeTeam', 'AwayTeam', 'HomePlay', 'AwayPlay', 
        'Shooter', 'ShotType', 'ShotOutcome', 'EVENTMSGTYPE'
    ]
    
    # 実際にデータフレームに存在する列だけを選ぶ (エラー回避)
    existing_cols = [c for c in target_columns if c in df.columns]
    final_df = df[existing_cols]

    print(f"Filtered rows (Strict Valid Field Goals Only): {len(final_df)}")
    
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f" Cleaned data saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_pipeline()