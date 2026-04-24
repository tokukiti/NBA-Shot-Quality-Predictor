import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_model_diagram_compact_horizontal():
    # 図の初期化 (横幅を抑え、高さを少し確保してアスペクト比を改善: 10x4.5インチ)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    # 描画範囲を要素ギリギリに設定
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(0, 4.5)
    ax.axis('off')

    # --- 共通設定 (フォントサイズを大幅に拡大) ---
    arrow_props = dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=2.0) # 矢印も太く
    font_main = 14    # 基本フォントサイズ拡大
    font_math = 16    # 数式フォントサイズ拡大
    font_bold = 14    # ボールド体のサイズ

    # Y座標の基準線
    base_y = 2.2

    # ==========================================
    # 1. Input Sequence (間隔を詰める)
    # ==========================================
    # X座標: 1.0 ~ 4.0
    times = ['$t-N$', '...', '$t$']
    x_starts = [1.0, 2.2, 3.2] # 間隔を狭く設定
    
    for i, (x, t_label) in enumerate(zip(x_starts, times)):
        if t_label == '...':
            ax.text(x + 0.2, base_y, '...', fontsize=24, ha='center', va='center')
            continue

        # ノードを描画
        center_x, center_y = x + 0.4, base_y
        radius = 0.5 # 半径を少し小さくして密集感
        for angle in np.linspace(0, 2*np.pi, 10, endpoint=False): 
            nx = center_x + radius * np.cos(angle)
            ny = center_y + radius * np.sin(angle)
            node_circle = patches.Circle((nx, ny), 0.08, facecolor='#dddddd', edgecolor='black')
            ax.add_patch(node_circle)
            ax.plot([nx, center_x], [ny, center_y], color='gray', lw=0.5, zorder=0)

        # ラベル (位置を調整)
        ax.text(center_x, base_y - 1.0, f'Graph', ha='center', fontsize=font_main)
        ax.text(center_x, base_y + 0.8, r'$G_{' + t_label.replace('$', '') + r'}$', ha='center', fontsize=font_math)

    # セクション枠
    ax.text(2.5, 4.0, '1. Input Sequence (4.0s)', ha='center', fontsize=font_bold, fontweight='bold')
    
    # 矢印 (Input -> GAT) 短く
    ax.annotate('', xy=(4.8, base_y), xytext=(4.0, base_y), arrowprops=arrow_props)

    # ==========================================
    # 2. Spatial GAT (コンパクトに)
    # ==========================================
    # X座標: 4.8 ~ 6.8
    gat_x = 4.8
    gat_w = 2.0
    gat_h = 2.2
    rect_gat = patches.FancyBboxPatch((gat_x, base_y - gat_h/2), gat_w, gat_h, boxstyle="round,pad=0.1", 
                                      linewidth=2, edgecolor='black', facecolor='#f0f0f0')
    ax.add_patch(rect_gat)
    
    ax.text(gat_x + gat_w/2, base_y + 0.4, 'Spatial\nGATv2', ha='center', va='center', fontsize=font_bold, fontweight='bold')
    ax.text(gat_x + gat_w/2, base_y - 0.5, r'$\alpha_{ij}$ (Attn)', ha='center', va='center', fontsize=font_main, color='#333333')

    # 矢印 (GAT -> Pooling) 短く
    ax.annotate('', xy=(7.3, base_y), xytext=(6.8, base_y), arrowprops=arrow_props)

    # ==========================================
    # 3. Global Mean Pooling (コンパクトに)
    # ==========================================
    # X座標: 7.3 ~ 8.3
    pool_x = 7.3
    # 台形
    polygon = patches.Polygon([[pool_x, base_y + 0.5], [pool_x, base_y - 0.5], 
                               [pool_x + 1.0, base_y - 0.3], [pool_x + 1.0, base_y + 0.3]], 
                              closed=True, edgecolor='black', facecolor='white', lw=1.5)
    ax.add_patch(polygon)
    
    ax.text(pool_x + 0.5, base_y - 1.0, 'Mean\nPooling', ha='center', fontsize=font_main)
    ax.text(pool_x + 0.5, base_y + 0.8, r'Avg($\cdot$)', ha='center', fontsize=font_math)

    # 矢印 (Pooling -> LSTM) 短く
    ax.annotate('', xy=(8.8, base_y), xytext=(8.3, base_y), arrowprops=arrow_props)

    # ==========================================
    # 4. Temporal LSTM (コンパクトに)
    # ==========================================
    # X座標: 8.8 ~ 10.6
    lstm_x = 8.8
    lstm_w = 1.8
    lstm_h = 1.8
    rect_lstm = patches.FancyBboxPatch((lstm_x, base_y - lstm_h/2), lstm_w, lstm_h, boxstyle="round,pad=0.1", 
                                       linewidth=2, edgecolor='black', facecolor='#e6e6e6')
    ax.add_patch(rect_lstm)
    
    ax.text(lstm_x + lstm_w/2, base_y + 0.3, 'Temporal\nLSTM', ha='center', va='center', fontsize=font_bold, fontweight='bold')
    ax.text(lstm_x + lstm_w/2, base_y - 0.5, r'$h_t$ (Context)', ha='center', fontsize=font_main)

    # 矢印 (LSTM -> Output) 短く
    ax.annotate('', xy=(11.2, base_y), xytext=(10.6, base_y), arrowprops=arrow_props)

    # ==========================================
    # 5. Output (コンパクトに)
    # ==========================================
    # X座標: 11.2 ~
    # FC層 (円)
    circle_fc = patches.Circle((11.5, base_y), 0.4, facecolor='white', edgecolor='black', lw=2)
    ax.add_patch(circle_fc)
    ax.text(11.5, base_y, 'FC', ha='center', va='center', fontsize=font_bold, fontweight='bold')

    # 最終出力矢印 (極短)
    ax.annotate('', xy=(12.3, base_y), xytext=(11.9, base_y), arrowprops=arrow_props)
    
    # 出力ラベル (矢印のすぐ右に大きく配置)
    ax.text(12.4, base_y + 0.1, r'$\hat{y}$', ha='left', va='center', fontsize=20, fontweight='bold')
    ax.text(12.4, base_y - 0.4, 'Prob.', ha='left', va='top', fontsize=12)

    plt.tight_layout()
    # 余白をギリギリまで削って保存
    plt.savefig('stgat_model_compact_horizontal.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.show()

# 実行
draw_model_diagram_compact_horizontal()