import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Batch
from torch.nn import LSTM, Linear, Dropout

class STGAT(torch.nn.Module):
    def __init__(self, node_features, edge_features, hidden_channels, out_channels=1):
        super(STGAT, self).__init__()
        
        # heads=3 なので、Attentionは3つ分の平均などを後で見ることになります
        self.gat1 = GATv2Conv(node_features, hidden_channels, heads=3, edge_dim=edge_features)
        
        # GAT2への入力は heads * hidden_channels
        self.gat2 = GATv2Conv(hidden_channels * 3, hidden_channels, heads=1, edge_dim=edge_features)

        self.lstm = LSTM(input_size=hidden_channels, 
                         hidden_size=hidden_channels, 
                         batch_first=True)

        self.dropout = Dropout(p=0.5)
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, out_channels)

    def forward(self, data_list, return_attention=False):
 
        device = next(self.parameters()).device
        
        # バッチ化 (リストが来たらBatchにする、すでにBatchならそのまま)
        if isinstance(data_list, Batch):
            batch_data = data_list.to(device)
        else:
            # data_listがリストの場合
            batch_data = Batch.from_data_list(data_list).to(device)
        
        x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr
        
        # --- 1. GAT1 (ここでAttentionを取得する分岐を追加) ---
        if return_attention:
            # GATv2Conv は return_attention_weights=True で (out, (edge_index, alpha)) を返す仕様
            x, (att_edge_index, att_weights) = self.gat1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        else:
            x = self.gat1(x, edge_index, edge_attr=edge_attr)
            
        x = F.elu(x)
        
        # --- 2. GAT2 ---
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        # --- 3. Pooling (グラフ単位の特徴量へ) ---
        frame_features = global_mean_pool(x, batch_data.batch)
        
        # --- 4. LSTM (時系列処理) ---
        # data_list の長さ = シーケンス長 (フレーム数)
        # バッチサイズ1の分析前提、あるいは data_list 全体を1シーケンスとして扱う
        seq_len = len(data_list) if isinstance(data_list, list) else batch_data.num_graphs
        
        # [seq_len, hidden] -> [1, seq_len, hidden]
        # view()を使って安全にreshape
        seq_tensor = frame_features.view(1, seq_len, -1)
        
        lstm_out, (h_n, c_n) = self.lstm(seq_tensor)
        
        # 最後の時刻の隠れ層状態を取得
        last_hidden = h_n[-1] 

        # --- 5. Classifier ---
        out = self.dropout(last_hidden)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        
        # 分析モードならAttentionも返す
        if return_attention:
            return out, (att_edge_index, att_weights)
        else:
            return out