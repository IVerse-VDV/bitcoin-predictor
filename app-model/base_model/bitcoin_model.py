import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import random

# random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)




class BitcoinPredictor(nn.Module):
    """
    neural Network for Bitcoin price movement prediction
    Binary classification: 1 (Up), 0 (Down)
    """
    def __init__(self, input_size=3):
        super(BitcoinPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)




def generate_dummy_data(n_samples=10000):
    print("Generating dummy Bitcoin price data...")
    
    # start with base price around 50000
    base_price = 50000
    prices = [base_price]
    
    # generate realistic price movements
    for i in range(n_samples * 4):  # generate more data for better sequences
        # random walk with trend and volatility
        trend = np.random.normal(0, 0.02)  # small trend
        volatility = np.random.normal(0, 0.05)  # higher volatility
        change = trend + volatility
        
        new_price = prices[-1] * (1 + change)
        # keep prices in reasonable range
        new_price = max(20000, min(100000, new_price))
        prices.append(new_price)
    
    # create sequences of 3 days + target (next day movement)
    sequences = []
    targets = []
    
    for i in range(len(prices) - 4):
        seq = prices[i:i+3]  # 3 days of prices
        next_price = prices[i+3]  # next day price
        current_price = prices[i+2]  # last day in sequence
        
        # binary target: 1 if price goes up, 0 if down
        target = 1 if next_price > current_price else 0
        
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences[:n_samples]), np.array(targets[:n_samples])




def train_model():
    print("Starting model training...")
    
    # generate training data
    X, y = generate_dummy_data(10000)
    
    # scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    
    # split data
    split_idx = int(0.8 * len(X_tensor))
    X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
    y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
    
    # init model
    model = BitcoinPredictor(input_size=3)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # training loop
    model.train()
    best_val_acc = 0
    
    for epoch in range(200):
        # forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # validation
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_predictions = (val_outputs > 0.5).float()
                val_accuracy = (val_predictions == y_val).float().mean().item()
                
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    #save model
                    torch.save(model.state_dict(), 'bitcoin_model.pth')
                    joblib.dump(scaler, 'scaler.pkl')
                
                print(f'Epoch [{epoch+1}/200], Loss: {loss.item():.4f}, Val Acc: {val_accuracy:.4f}')
            
            model.train()
    
    print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
    print("Model saved as 'bitcoin_model.pth'")
    print("Scaler saved as 'scaler.pkl'")




def load_model():
    try:
        model = BitcoinPredictor(input_size=3)
        model.load_state_dict(torch.load('bitcoin_model.pth', map_location='cpu'))
        model.eval()
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        print("Model files not found. Please run train_model() first.")
        return None, None




def predict_price_movement(day1, day2, day3):
    model, scaler = load_model()
    if model is None:
        return None, None
    
    # prepare input
    input_data = np.array([[day1, day2, day3]])
    input_scaled = scaler.transform(input_data)
    input_tensor = torch.FloatTensor(input_scaled)
    
    # make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probability = output.item()
        prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability






if __name__ == "__main__":
    # train the model
    train_model()
    
    # test prediction
    print("\nTesting prediction...")
    pred, prob = predict_price_movement(50000, 51000, 52000)
    if pred is not None:
        direction = "UP" if pred == 1 else "DOWN"
        print(f"Prediction: {direction} ({prob:.2%})")

