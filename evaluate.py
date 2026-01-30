import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import calibration_curve
import os

# --- 設定 ---
RESULT_CSV = 'evaluation_results_v11.csv'
DATA_CSV = 'cleaned_shots_data_v2.csv'
OUTPUT_DIR = './final_analysis_graphs'

def get_dist(x, y):
    # コート上の距離計算 (ゴール: 88.75, 25.0 と仮定)
    return np.sqrt((x - 88.75)**2 + (y - 25.0)**2)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. データの読み込みと結合
    print("Loading data...")
    df_res = pd.read_csv(RESULT_CSV)
    df_data = pd.read_csv(DATA_CSV) # ここには座標データがないため、距離計算には工夫が必要
    # ※ clean_data_v2には座標がないため、ShotTypeや距離情報があればそれを使う
    # 今回は簡易的に ShotType で代用するか、もし距離データがない場合はラベルで代用
    
    # ID結合
    df_res['game_id'] = df_res['game_id'].astype(str).str.zfill(10)
    df_data['GAME_ID'] = df_data['GAME_ID'].astype(str).str.zfill(10)
    
    merged = pd.merge(
        df_res, 
        df_data, 
        left_on=['game_id', 'event_id'], 
        right_on=['GAME_ID', 'EVENTNUM'],
        how='inner'
    )
    print(f"Merged Data: {len(merged)} rows")

    # ==========================================
    # 1. Calibration Curve (再掲)
    # ==========================================
    plt.figure(figsize=(6, 6))
    prob_true, prob_pred = calibration_curve(merged['actual'], merged['prob_make'], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label='ST-GAT Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Probability')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/1_calibration_curve.png")
    plt.close()

    # ==========================================
    # 2. Player Analysis (条件緩和版)
    # ==========================================
    player_stats = merged.groupby('Shooter').agg({
        'actual': ['count', 'mean'],
        'prob_make': 'mean'
    }).reset_index()
    player_stats.columns = ['Shooter', 'Attempts', 'Actual_FG%', 'Expected_FG%']
    
    # ★ 緩和: 5本以上でOKとする
    player_stats = player_stats[player_stats['Attempts'] >= 5]
    player_stats['Diff'] = player_stats['Actual_FG%'] - player_stats['Expected_FG%']
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=player_stats, x='Expected_FG%', y='Actual_FG%', 
        size='Attempts', hue='Diff', palette='coolwarm', sizes=(20, 200)
    )
    plt.plot([0, 1], [0, 1], '--', color='gray')
    
    # 上位3名を表示
    for _, row in player_stats.nlargest(3, 'Diff').iterrows():
        plt.text(row['Expected_FG%'], row['Actual_FG%'], row['Shooter'], fontsize=8)
        
    plt.title('Player Performance (Attempts >= 5)')
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.savefig(f"{OUTPUT_DIR}/2_player_analysis.png")
    plt.close()

    # ==========================================
    # 3. Shot Quality Histogram (新規)
    # ==========================================
    plt.figure(figsize=(8, 5))
    sns.histplot(data=merged, x='prob_make', bins=20, kde=True, hue='actual', element="step")
    plt.title('Distribution of Shot Quality (AI Prediction)')
    plt.xlabel('Predicted Probability (Shot Quality)')
    plt.savefig(f"{OUTPUT_DIR}/3_shot_quality_dist.png")
    plt.close()

    # ==========================================
    # 4. Shot Zone Analysis (簡易版)
    # ==========================================
    # データに距離情報がない場合、ShotTypeなどの文字列から推測
    # ここでは ShotType があると仮定して集計
    if 'ShotType' in merged.columns:
        zone_stats = merged.groupby('ShotType').agg({
            'actual': 'count',
            'prob_make': 'mean' # AIの自信度
        }).rename(columns={'actual': 'Count', 'prob_make': 'AvgConf'})
        
        # 精度計算
        accuracies = []
        aucs = []
        zones = []
        
        for zone in merged['ShotType'].unique():
            subset = merged[merged['ShotType'] == zone]
            if len(subset) > 10:
                zones.append(zone)
                accuracies.append(accuracy_score(subset['actual'], subset['predicted']))
                try:
                    aucs.append(roc_auc_score(subset['actual'], subset['prob_make']))
                except:
                    aucs.append(0.5)
        
        zone_df = pd.DataFrame({'Zone': zones, 'Accuracy': accuracies, 'AUC': aucs})
        
        plt.figure(figsize=(10, 5))
        sns.barplot(data=zone_df, x='Zone', y='Accuracy')
        plt.title('Model Accuracy by Shot Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/4_zone_accuracy.png")
        plt.close()

    print(f"✅ 全てのグラフを {OUTPUT_DIR} に保存しました。")

if __name__ == "__main__":
    main()