# 🏀 NBA Shot Quality Predictor (ST-GAT)

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-3C2179?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

## 📌 概要 (Overview)
NBAのトラッキングデータ（選手とボールの時空間座標データ）とプレイバイプレイデータを用い、**「シュートの質（成功確率：Expected FG%）」を高精度に予測するAIモデル**です。

バスケットボールにおける複雑な選手間の連携や駆け引きをモデリングするため、**時空間グラフニューラルネットワーク（ST-GAT: Spatial-Temporal Graph Attention Network）**を独自に実装しました。単なる結果予測に留まらず、Attention（注意機構）の重みを解析することで「ディフェンスの意識（Gravity）」を可視化し、モデルの解釈性を持たせている点が最大の特徴です。

## 🔥 本プロジェクトのこだわりと特徴

### 1. ドメイン知識に基づく独自の特徴量設計と正規化 (`make_dataset.py`)
生のトラッキングデータをそのまま入力するのではなく、競技特性を考慮した前処理（特徴量エンジニアリング）を泥臭く実装しました。
* コートの反転処理（常に同じハーフコートで攻撃している状態に正規化）
* 選手の速度(v)・加速度(a)の算出
* 選手間距離、リングからの距離などの空間的特徴量の算出とスケーリング

### 2. 時空間の複雑な関係性を捉えるモデル設計 (`model.py`)
空間的な関係性（選手間の位置や距離）と、時間的な変化（選手の動きの軌跡）の双方を捉えるハイブリッドなアーキテクチャをPyTorchで構築しました。
* **Spatial Feature:** `GATv2Conv` を用いて、各フレームにおける選手間の関係性（誰が誰に影響を与えているか）を学習。
* **Temporal Feature:** `Global Mean Pooling` でグラフ全体の特徴を圧縮後、`LSTM` に入力してシュートに至るまでの数秒間の時系列変化を学習。

### 3. 不均衡データへの対応と厳密な評価指標 (`train.py`, `evaluate.py`)
* シュートの成否という不均衡なラベルに対して、BCEWithLogitsLossに `pos_weight` を導入。
* 単なるAccuracyではなく、F1-ScoreやROC-AUCを重視してモデルの汎化性能を評価。
* 予測確率と実際の成功率の乖離を見る Calibration Curve（信頼性曲線）を出力し、確率の妥当性を検証。

### 4. Attentionによる「Gravity」の可視化と定性評価 (`visualize_graph_structure_final.py`)
AIが「なぜそのシュート確率を弾き出したのか」を説明可能にするため、GATのAttention Weightを可視化しました。これにより、「ディフェンスがボールマンにどれだけ引きつけられているか（Gravity）」や、「ノーマークの選手が生まれるプロセス」を定量・定性の両面から分析可能です。

## 📊 分析から得られたインサイト (Insights)
本モデルの出力（Expected FG%）と、実際の選手の成功率（Actual FG%）を比較することで、選手の「シュート力（Shot Making Ability）」を定量化しました。

* **Good Process, Bad Result:** AIの予測確率は高い（良い崩しができている）が、結果的に外れたシュート。
* **Bad Process, Good Result:** AIの予測確率は低い（タフショット）が、個人のスキルでねじ込んだシュート。
* `Diff = Actual FG% - Expected FG%` を算出し、期待値以上にタフショットを決める優秀なシューターを可視化。

## 📁 ディレクトリ構成 (Directory Structure)
主要なコードのみ抜粋しています。

```text
.
├── pipeline.py                # 生データからシュートイベントのみを厳密に抽出・クレンジング
├── make_dataset.py            # トラッキングデータから時系列グラフデータ(PyG Data)を生成
├── model.py                   # ST-GAT + LSTM のアーキテクチャ定義
├── train.py                   # 学習パイプライン (ロス履歴の保存、評価指標の算出)
├── evaluate*.py               # キャリブレーションカーブや選手別分析グラフの生成
└── visualize_*.py             # GIFアニメーション生成、Attention(Gravity)の可視化