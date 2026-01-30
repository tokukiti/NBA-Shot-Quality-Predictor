import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
import os

# --- 設定 ---
RESULT_CSV = 'evaluation_results_v11.csv'
DATA_CSV = 'cleaned_shots_data_v2.csv'
OUTPUT_DIR = './analysis_graphs'

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. データの読み込みと結合
    print("Loading data...")
    df_res = pd.read_csv(RESULT_CSV)
    df_data = pd.read_csv(DATA_CSV)
    
    # IDの型を合わせる (念のため)
    df_res['game_id'] = df_res['game_id'].astype(str).str.zfill(10)
    df_data['GAME_ID'] = df_data['GAME_ID'].astype(str).str.zfill(10)
    
    # 結合 (GameIDとEventIDで紐付け)
    # df_resの event_id は df_data の EVENTNUM に対応
    merged = pd.merge(
        df_res, 
        df_data[['GAME_ID', 'EVENTNUM', 'Shooter', 'ShotType']], 
        left_on=['game_id', 'event_id'], 
        right_on=['GAME_ID', 'EVENTNUM'],
        how='inner'
    )
    
    print(f"Merged Data: {len(merged)} rows")

    # ==========================================
    # A. キャリブレーションカーブ (信頼性曲線)
    # ==========================================
    print("Generating Calibration Curve...")
    prob_true, prob_pred = calibration_curve(merged['actual'], merged['prob_make'], n_bins=10)
    
    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, marker='o', label='Your Model', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated', color='gray')
    plt.xlabel('Predicted Probability (AI)')
    plt.ylabel('Actual Fraction of Positives (Reality)')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/calibration_curve.png")
    plt.close()
    print(f" -> Saved: {OUTPUT_DIR}/calibration_curve.png")

    # ==========================================
    # B. 選手別分析 (Shot Quality vs FG%)
    # ==========================================
    print("Generating Player Analysis...")
    
    # 選手ごとの集計
    # - Count: シュート本数
    # - FG%: 実際の成功率
    # - xFG% (Expected FG%): AIが予測した確率の平均（＝シュート難易度の逆数。高いほど簡単）
    # - Diff: 実力 - 期待値 (プラスなら「上手い」、マイナスなら「下手」または「不運」)
    
    player_stats = merged.groupby('Shooter').agg({
        'actual': ['count', 'mean'],
        'prob_make': 'mean'
    }).reset_index()
    
    player_stats.columns = ['Shooter', 'Attempts', 'Actual_FG%', 'Expected_FG%']
    
    # シュート本数が少ない選手は除外 (ノイズになるため。例: 20本以上)
    player_stats = player_stats[player_stats['Attempts'] >= 15]
    
    # Diff (Shooting Ability Indicator)
    player_stats['Diff'] = player_stats['Actual_FG%'] - player_stats['Expected_FG%']
    
    # 散布図作成
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=player_stats, 
        x='Expected_FG%', 
        y='Actual_FG%', 
        size='Attempts', 
        sizes=(50, 400),
        hue='Diff',
        palette='coolwarm',
        alpha=0.8
    )
    
    # 対角線 (期待通り)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    
    # 有名選手や外れ値を注釈
    # Diffの上位/下位、Attemps上位などをラベル付け
    top_diff = player_stats.nlargest(3, 'Diff')
    bot_diff = player_stats.nsmallest(3, 'Diff')
    most_att = player_stats.nlargest(3, 'Attempts')
    
    labels_to_show = pd.concat([top_diff, bot_diff, most_att]).drop_duplicates()
    
    for _, row in labels_to_show.iterrows():
        plt.text(
            row['Expected_FG%']+0.01, 
            row['Actual_FG%'], 
            row['Shooter'], 
            fontsize=9,
            weight='bold'
        )

    plt.title('Player Performance: Actual FG% vs AI Expected FG% (Process Quality)')
    plt.xlabel('Average Shot Quality (Expected FG% by AI)\n<-- Tough Shots       Easy Shots -->')
    plt.ylabel('Actual FG% (Result)')
    plt.xlim(0.2, 0.8) # 範囲は見やすさのために調整してください
    plt.ylim(0.2, 0.8)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"{OUTPUT_DIR}/player_analysis.png")
    plt.close()
    print(f" -> Saved: {OUTPUT_DIR}/player_analysis.png")
    
    # 上位/下位ランキングのCSV保存
    player_stats.sort_values('Diff', ascending=False).to_csv(f"{OUTPUT_DIR}/player_ranking.csv", index=False)
    print(f" -> Saved: {OUTPUT_DIR}/player_ranking.csv")

if __name__ == "__main__":
    main()