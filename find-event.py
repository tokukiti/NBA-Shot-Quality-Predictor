import pandas as pd

# CSV読み込み
df = pd.read_csv('evaluation_results_v11.csv')

# パターン1: 良いプロセス・悪い結果 (AI「入る！」 -> 結果「Miss」)
good_process = df[(df['actual'] == 0) & (df['prob_make'] > 0.80)]
print("--- Good Process, Bad Result (Top 10) ---")
print(good_process.sort_values('prob_make', ascending=False).head(10))

# パターン2: 悪いプロセス・良い結果 (AI「無理！」 -> 結果「Make」)
bad_process = df[(df['actual'] == 1) & (df['prob_make'] < 0.20)]
print("\n--- Bad Process, Good Result (Top 10) ---")
print(bad_process.sort_values('prob_make', ascending=True).head(10))