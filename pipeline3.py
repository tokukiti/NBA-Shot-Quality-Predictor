import pandas as pd
import numpy as np

# --- 設定 ---
INPUT_CSV = 'NBA_PBP_2015-16.csv'
OUTPUT_CSV = 'cleaned_shots_data_v2.csv' # v2として保存

def run_pipeline():
    print(f"Loading raw CSV: {INPUT_CSV} ...")
    try:
        df = pd.read_csv(INPUT_CSV, encoding='ISO-8859-1')
    except FileNotFoundError:
        print(f"エラー: {INPUT_CSV} が見つかりません。")
        exit()

    print(f"Original rows: {len(df)}")

    # 1. URLからGAME_IDを生成
    unique_urls = df['URL'].unique()
    url_to_id = {url: f"002150{str(i + 1).zfill(4)}" for i, url in enumerate(unique_urls)}
    df['GAME_ID'] = df['URL'].map(url_to_id)

    # 2. 欠損している試合を除外
    missing_games = ['0021500006', '0021500008', '0021500014']
    df = df[~df['GAME_ID'].isin(missing_games)]

    # ==========================================
    # ★★★ 修正点: フリースローの除外 ★★★
    # ==========================================
    
    # ShotOutcome が 'make' か 'miss' の行だけを残す
    if 'ShotOutcome' in df.columns:
        df['ShotOutcome'] = df['ShotOutcome'].astype(str).str.lower().str.strip()
        df = df[df['ShotOutcome'].isin(['make', 'miss'])]
    
    # 【追加】Play内容に "Free Throw" が含まれる行を削除
    # HomePlay または AwayPlay に "Free Throw" という文字列があるかチェック
    is_ft_home = df['HomePlay'].astype(str).str.contains('Free Throw', case=False, na=False)
    is_ft_away = df['AwayPlay'].astype(str).str.contains('Free Throw', case=False, na=False)
    
    # どちらにも含まれていない行だけを残す
    df = df[~(is_ft_home | is_ft_away)]

    # Shooter(選手名)が入っていない行を削除
    if 'Shooter' in df.columns:
        df = df[df['Shooter'].notna()]
    
    # 時間の整形
    df['SecLeft'] = pd.to_numeric(df['SecLeft'], errors='coerce').fillna(0).astype(int)
    df['Quarter'] = pd.to_numeric(df['Quarter'], errors='coerce').fillna(1).astype(int)

    # ID振り直し
    df = df.reset_index(drop=True)
    df['EVENTNUM'] = df.index

    # 必要な列だけ保存
    columns_to_keep = [
        'GAME_ID', 'EVENTNUM', 'Quarter', 'SecLeft', 
        'AwayTeam', 'AwayPlay', 'HomeTeam', 'HomePlay', 
        'Shooter', 'ShotOutcome'
    ]
    existing_cols = [c for c in columns_to_keep if c in df.columns]
    final_df = df[existing_cols]

    print(f"Filtered rows (Valid Field Goals Only): {len(final_df)}")
    
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Cleaned data saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_pipeline()