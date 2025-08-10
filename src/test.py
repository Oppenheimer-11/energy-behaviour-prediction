import os 
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from TransformerModel import TransformerModel
from TimeSeriesDataset import TimeSeriesDataset
from tqdm import tqdm

def main():
    # 读取CSV文件
    data = pd.read_csv('differenced.csv')

    # 检测GPU可用性并选择设备
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("使用GPU" if device.type == "cuda" else "使用CPU")

    # 提取目标变量和数值特征
    target = data['target']
    features = data.drop(columns=['target'])
    features = features.select_dtypes(include=[np.number])

    # 使用StandardScaler进行特征缩放
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)


    # 创建模型实例
    input_dim = X_train.shape[1]
    d_model = 64  # Transformer模型的维度
    nhead = 2    # 注意力头的数量
    num_layers = 4  # Transformer层的数量
    EXPERIMENT_ID = 3
    model = TransformerModel(input_dim, d_model, nhead, num_layers).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # 将数据转换为PyTorch张量并移到GPU
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).to(device)

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    # 创建DataLoader
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    path_checkpoint = f'../model/checkpoint/ckpt_expr_{EXPERIMENT_ID}.pth'
    start_epoch = 1
    if os.path.exists(path_checkpoint):
        print('load model...')
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch

    # 训练模型
    num_epochs = 10
    epoch_loss_lst = []
    model.train()
    for iepoch, epoch in enumerate(range(start_epoch + 1, num_epochs)):
        total_loss = 0
        for count, (inputs, labels) in tqdm(enumerate(train_loader, start=1)):
            # print(f'batch {count}/{batch_size}')
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss}')
        epoch_loss_lst.append(average_loss)
    print(f'Loss record: {epoch_loss_lst}')
        
    # save model
    checkpoint = {
            "net": model.state_dict(),
            'optimizer':optimizer.state_dict(),
            "epoch": iepoch
        }
    if not os.path.isdir("../model/checkpoint"):
        os.mkdir("../model/checkpoint")
    torch.save(checkpoint, f'../model/checkpoint/ckpt_expr_{EXPERIMENT_ID}.pth')
        
    # 在测试集上评估模型
    print('test...')
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            test_outputs = model(inputs, labels)
            test_loss = criterion(test_outputs, labels)
            total_test_loss += test_loss.item()
            print(f'batch loss: {test_loss.item()}')
    average_test_loss = total_test_loss / len(test_loader)
    print(f'Test Loss: {average_test_loss}')


if __name__ == "__main__":
    main()
