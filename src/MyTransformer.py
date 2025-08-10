import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, num_layers, heads, dim_feedforward):
        super(TimeSeriesTransformer, self).__init__()
        self.linear_in = nn.Linear(input_size, dim_feedforward)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.linear_out = nn.Linear(dim_feedforward, 1)

    def forward(self, src):
        src = self.linear_in(src.unsqueeze(0))
        output = self.transformer_encoder(src)
        output = self.linear_out(output)
        return output.squeeze()

# 示例：初始化模型
# model = TimeSeriesTransformer(input_size=input_window * k, num_layers=2, heads=4, dim_feedforward=512)
