import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from model import TrafficLSTM
import joblib

def train_model():
    # 1. Load data
    df = pd.read_csv('traffic_data_simulated.csv')
    
    # We'll train one model for all directions for simplicity, 
    # or we could include direction as a feature. 
    # Let's just use the vehicle_count series.
    data = df['vehicle_count'].values.astype(float).reshape(-1, 1)
    
    # 2. Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Save the scaler for inference
    joblib.dump(scaler, 'traffic_scaler.gz')
    
    # 3. Create sequences
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    SEQ_LENGTH = 24 # 24 hours of history
    X, y = create_sequences(scaled_data, SEQ_LENGTH)
    
    # Convert to PyTorch tensors
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    
    # Split into train/test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 4. Initialize model
    model = TrafficLSTM(input_size=1, hidden_size=64, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 5. Training loop
    epochs = 20
    batch_size = 32
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            outputs = model(X_batch)
            optimizer.zero_grad()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
    # 6. Save model
    torch.save(model.state_dict(), 'traffic_lstm.pt')
    print("Model saved to 'traffic_lstm.pt'")

if __name__ == "__main__":
    train_model()
