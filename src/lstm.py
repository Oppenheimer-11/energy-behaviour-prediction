import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 检查是否有可用的GPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用GPU
    print("GPU可用")
else:
    device = torch.device("cpu")   # 使用CPU
    print("GPU不可用")
    
data = pd.read_csv('adding_previous15_data.csv')

columns_to_difference = [
    'temperature_h_mean', 'dewpoint_h_mean', 'rain_h_mean', 'snowfall_h_mean', 'surface_pressure_h_mean',
    'cloudcover_total_h_mean', 'cloudcover_low_h_mean', 'cloudcover_mid_h_mean', 'cloudcover_high_h_mean',
    'windspeed_10m_h_mean', 'winddirection_10m_h_mean', 'shortwave_radiation_h_mean',
    'direct_solar_radiation_h_mean', 'diffuse_radiation_h_mean','temperature_f_mean', 'dewpoint_f_mean',
    'cloudcover_high_f_mean', 'cloudcover_low_f_mean', 'cloudcover_mid_f_mean', 'cloudcover_total_f_mean',
    '10_metre_u_wind_component_f_mean', '10_metre_v_wind_component_f_mean', 'direct_solar_radiation_f_mean',
    'surface_solar_radiation_downwards_f_mean', 'snowfall_f_mean','target'
    # Add the names of the columns you want to difference here
]
# Define the window size for differencing (e.g., 90 days)
window_size = 90

# Calculate the rolling mean for each specified column
for column in columns_to_difference:
    data[f'{column}_rolling_mean'] = data.groupby('prediction_unit_id')[column].transform(lambda x: x.rolling(window=window_size).mean())

# Calculate the differences and store them in new columns
for column in columns_to_difference:
    data[f'{column}_differenced'] = data[column] - data[f'{column}_rolling_mean']

# Drop the intermediate rolling mean columns if not needed
for column in columns_to_difference:
    data.drop(columns=[f'{column}_rolling_mean'], inplace=True)
    
columns_to_remove = [
    'temperature_h_mean', 'dewpoint_h_mean', 'rain_h_mean', 'snowfall_h_mean', 'surface_pressure_h_mean',
    'cloudcover_total_h_mean', 'cloudcover_low_h_mean', 'cloudcover_mid_h_mean', 'cloudcover_high_h_mean',
    'windspeed_10m_h_mean', 'winddirection_10m_h_mean', 'shortwave_radiation_h_mean',
    'direct_solar_radiation_h_mean', 'diffuse_radiation_h_mean','temperature_f_mean', 'dewpoint_f_mean',
    'cloudcover_high_f_mean', 'cloudcover_low_f_mean', 'cloudcover_mid_f_mean', 'cloudcover_total_f_mean',
    '10_metre_u_wind_component_f_mean', '10_metre_v_wind_component_f_mean', 'direct_solar_radiation_f_mean',
    'surface_solar_radiation_downwards_f_mean', 'snowfall_f_mean','target','date', 'year', 'quarter', 'month',
    'week', 'hour', 'day_of_year','day_of_month', 'day_of_week', 'data_block_id', 'row_id','hour_h',
    'hours_ahead_f_mean'
]

# Remove the columns from the DataFrame
data = data.drop(columns=columns_to_remove)

# Define hyperparameters
input_dim = 10  # Replace with your actual input dimension
hidden_dim = 64  # Replace with your chosen hidden dimension
num_layers = 2  # Replace with your chosen number of layers
output_dim = 1  # For regression task

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Initialize the model, loss function, and optimizer
model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#data = data[0:2000]
# Split the data into features (X) and target (y)
X = data.drop(columns=['target_differenced'])  # Features
y = data['target_differenced']  # Target

# Split the data into training and testing sets (adjust test_size as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

non_numeric_columns = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

# Handle non-numeric columns based on their data type
for column in non_numeric_columns:
    if X_train[column].dtype == 'object':
        # Handle categorical data (e.g., one-hot encoding or label encoding)
        X_train = pd.get_dummies(X_train, columns=[column])
    elif X_train[column].dtype == 'datetime64[ns]':
        # Handle datetime data (e.g., feature extraction or timestamp conversion)
        X_train[column + '_year'] = X_train[column].dt.year
        X_train[column + '_month'] = X_train[column].dt.month
        # Add more features as needed
        X_train.drop(columns=[column], inplace=True)
    # Add more handling for other data types as needed

# After preprocessing, convert X_train to a NumPy array and then to a PyTorch tensor
X_train_np = X_train.values
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train.values, dtype=torch.float32).to(device)

# Training loop
num_epochs = 10  # Replace with your desired number of epochs
model.train()  # Set the model to training mode
for epoch in tqdm(range(num_epochs)):
    optimizer.zero_grad()  # Clear gradients
    outputs = model(X_train)  # Forward pass
    loss = criterion(outputs, y_train)  # Compute loss
    loss.backward()  # Backpropagation
    # Apply gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimizer.step()  # Update weights
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# Testing loop
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    test_outputs = model(X_test.to(device))  # Forward pass on test data
    test_loss = criterion(test_outputs, y_test.to(device))  # Compute test loss
    print(f'Test Loss: {test_loss.item()}')

# Calculate RMSE (Root Mean Squared Error) for evaluation
rmse = torch.sqrt(test_loss).item()
print(f'RMSE: {rmse}')