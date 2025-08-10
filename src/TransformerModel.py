import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, batch_first=True)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, src, diff_src):
        src = self.embedding(src)
        tgt = torch.zeros_like(src)  # 使用与src形状相同的空序列作为tgt
        output = self.transformer(src, tgt)
        output = self.fc(output)  # 输出的维度可能是 [batch_size, seq_len, 1]
        output = output.mean(dim=1)  # 对时间步长求平均，结果维度为 [batch_size, 1]
        return output.squeeze()  # 删除最后一个维度，结果维度为 [batch_size]