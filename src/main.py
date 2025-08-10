import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import InformerConfig, InformerModel
from accelerate import Accelerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load the CSV data
df = pd.read_csv("adding_previous15_data.csv")
df = df[df['is_consumption'] == 1]
df = df.dropna()

# Extract features, target, and past values
feature = df.drop(columns=["Unnamed: 0", 'target', 'row_id', 'datetime', 'data_block_id', 'prediction_unit_id',
                           'date', 'date_client', 'forecast_date_electricity', 'origin_date_electricity',
                           'forecast_date_gas', 'origin_date_gas', 'target_2_days_ago', 'target_3_days_ago',
                           'target_4_days_ago', 'target_5_days_ago', 'target_6_days_ago', 'target_7_days_ago',
                           'target_8_days_ago', 'target_9_days_ago', 'target_10_days_ago', 'target_11_days_ago',
                           'target_12_days_ago', 'target_13_days_ago', 'target_14_days_ago', 'target_15_days_ago']
                  ).values
target_array = df['target'].values
past_value_array = df[['target_2_days_ago', 'target_3_days_ago',
                       'target_4_days_ago', 'target_5_days_ago', 'target_6_days_ago', 'target_7_days_ago',
                       'target_8_days_ago', 'target_9_days_ago', 'target_10_days_ago', 'target_11_days_ago',
                       'target_12_days_ago', 'target_13_days_ago', 'target_14_days_ago', 'target_15_days_ago']].values

# Prepare features and tensors
features = feature
past_time_features_list = []
future_time_features_list = []

for i in range(len(features) - 14):
    past_time_features_list.append(features[i:i+14])

past_time_features = np.array(past_time_features_list)

for i in range(14, len(features) - 7):
    future_time_features_list.append(features[i:i+7])

future_time_features = np.array(future_time_features_list)

past_time_features_tensor = torch.tensor(past_time_features, dtype=torch.float)
future_time_features_tensor = torch.tensor(future_time_features, dtype=torch.float)

past_values_adjusted = np.array([target_array[i:i+14] for i in range(len(target_array) - 14)])
future_values_adjusted = np.array([target_array[i:i+7] for i in range(14, len(target_array) - 7)])

past_time_features_tensor = torch.tensor(past_time_features, dtype=torch.float)[:-(14+7)]
future_time_features_tensor = torch.tensor(future_time_features, dtype=torch.float)

past_values_tensor = torch.tensor(past_values_adjusted, dtype=torch.float)[:-(14+7)]
future_values_tensor = torch.tensor(future_values_adjusted, dtype=torch.float)

# Split the dataset into train and test (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(past_time_features_tensor, past_values_tensor, test_size=0.2, random_state=42)

# Create PyTorch Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, past_time_features, past_values, future_time_features, future_values):
        self.past_time_features = past_time_features
        self.past_values = past_values
        self.future_time_features = future_time_features
        self.future_values = future_values
        self.past_observed_mask = torch.ones_like(self.past_values)

    def __len__(self):
        return len(self.past_time_features)

    def __getitem__(self, idx):
        return {
            'past_time_features': self.past_time_features[idx],
            'past_values': self.past_values[idx],
            'past_observed_mask': self.past_observed_mask[idx],
            'future_time_features': self.future_time_features[idx],
            'future_values': self.future_values[idx]
        }

# Create datasets and dataloaders for train and test
train_dataset = TimeSeriesDataset(X_train, y_train, future_time_features_tensor, future_values_tensor)
test_dataset = TimeSeriesDataset(X_test, y_test, future_time_features_tensor, future_values_tensor)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Configure model and optimizer
config = InformerConfig(
    input_size=1,
    prediction_length=7,
    lags_sequence=[1, 2, 3, 4, 5, 6, 7],
    num_time_features=46,
    dropout=0.1,
    encoder_layers=2,
    decoder_layers=2,
    d_model=64,
)

model = InformerModel(config)

accelerator = Accelerator()
device = accelerator.device
model.to(device)
optimizer = AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=1e-1)

# Training loop (adjust the number of epochs as needed)
model.train()
for epoch in tqdm(range(10)):
    for idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            future_values=batch["future_values"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device)
        )
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()

        if idx % 100 == 0:
            print(loss.item())

# Evaluation loop
model.eval()
forecasts = []

for batch in test_dataloader:
    outputs = model.generate(
        past_time_features=batch["past_time_features"].to(device),
        past_values=batch["past_values"].to(device),
        future_time_features=batch["future_time_features"].to(device),
        past_observed_mask=batch["past_observed_mask"].to(device),
    )
    forecasts.append(outputs.sequences.cpu().numpy())