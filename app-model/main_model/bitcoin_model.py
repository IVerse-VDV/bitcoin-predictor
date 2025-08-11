import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import random
import matplotlib.pyplot as plt

# random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)




class BitcoinPredictor(nn.Module):
    """
    neural Network for Bitcoin price movement prediction
    Uses 7day input sequence for higher accuracy
    """
    def __init__(self, input_size=7, hidden_layers=[512, 256, 128, 64, 32]):
        super(BitcoinPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        #architecture
        for i, hidden_size in enumerate(hidden_layers):
            # linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            
            # bach normalization for better training stability
            layers.append(nn.BatchNorm1d(hidden_size))
            # dropout with decreasing rates
            dropout_rate = 0.5 - (i * 0.1)  # start high, decrease gradually
            dropout_rate = max(0.1, dropout_rate)
            layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)




def generate_dummy_data(n_samples=15000):
    print("Generating Bitcoin price data...")
    
    # start with base price
    base_price = 45000
    prices = [base_price]
    
    # mrket cycle parameters
    trend_cycle_length = 200  # long term trend cycle
    short_cycle_length = 50   # short term cycle
    
    for i in range(n_samples * 8):
        # long term trend (bull/bear market simulation)
        long_trend = 0.3 * np.sin(2 * np.pi * i / trend_cycle_length)
        
        # short term cycle
        short_trend = 0.1 * np.sin(2 * np.pi * i / short_cycle_length)
        
        # mrket momentum (autocorrelation)
        if len(prices) >= 3:
            momentum = 0.05 * ((prices[-1] - prices[-3]) / prices[-3])
        else:
            momentum = 0
        
        # random volatility with varying intnsity
        base_volatility = 0.03
        volatility_multiplier = 1 + 0.5 * np.sin(2 * np.pi * i / 100)  # Varying volatility
        volatility = np.random.normal(0, base_volatility * volatility_multiplier)
        
        # weekend effect (lower volatility on weekends)
        day_of_week = i % 7
        weekend_factor = 0.7 if day_of_week >= 5 else 1.0
        
        # combine all factors
        total_change = (long_trend + short_trend + momentum + volatility) * weekend_factor
        
        # apply change
        new_price = prices[-1] * (1 + total_change)
        
        # keep prices in realistic range with dynamic buonds
        min_price = base_price * 0.3  # can drop to 30% of base
        max_price = base_price * 3.0  # can rise to 300% of base
        new_price = max(min_price, min(max_price, new_price))
        
        prices.append(new_price)
    
    # sequences of 7 days + target
    sequences = []
    targets = []
    
    for i in range(len(prices) - 8):  # need 7 + 1 for prediction
        seq = prices[i:i+7]  # 7 days of prices
        next_price = prices[i+7]  #next day price
        current_price = prices[i+6]  # last day in sequence
        
        # target logic with threshold
        price_change_pct = (next_price - current_price) / current_price
        
        # use small threshold to avoid noise (only predict significant moves)
        threshold = 0.005  # 0.5% threshold
        if price_change_pct > threshold:
            target = 1  # Up
        elif price_change_pct < -threshold:
            target = 0  # Down
        else:
            continue  # skip small movements to focus on clear trends
        
        sequences.append(seq)
        targets.append(target)
    
    # balance the dataset
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    # balanced dataset
    up_indices = np.where(targets == 1)[0]
    down_indices = np.where(targets == 0)[0]
    
    # take equal samples from both clases
    min_samples = min(len(up_indices), len(down_indices), n_samples // 2)
    
    balanced_indices = np.concatenate([
        np.random.choice(up_indices, min_samples, replace=False),
        np.random.choice(down_indices, min_samples, replace=False)
    ])
    
    balanced_sequences = sequences[balanced_indices]
    balanced_targets = targets[balanced_indices]
    
    # shuffle
    shuffle_indices = np.random.permutation(len(balanced_sequences))
    balanced_sequences = balanced_sequences[shuffle_indices]
    balanced_targets = balanced_targets[shuffle_indices]
    
    print(f"Generated {len(balanced_sequences)} balanced samples")
    print(f"Up movements: {np.sum(balanced_targets)}, Down movements: {len(balanced_targets) - np.sum(balanced_targets)}")
    
    return balanced_sequences, balanced_targets




def create_features(sequences):
    features_list = []
    
    for seq in sequences:
        # prices (normalized to relative changes)
        relative_changes = []
        for i in range(1, len(seq)):
            change = (seq[i] - seq[i-1]) / seq[i-1]
            relative_changes.append(change)
        
        # === Technical indicators ===
        # 1. simple moving averages
        sma_3 = np.mean(seq[-3:]) / seq[-1]  # 3 day SMA ratio
        sma_7 = np.mean(seq) / seq[-1]       # 7 day SMA ratio
        
        # 2. volatility (standard deviation of returns)
        volatility = np.std(relative_changes)
        
        # 3. momentum indicators
        momentum_3 = (seq[-1] - seq[-4]) / seq[-4] if len(seq) >= 4 else 0  # 3 day momentum
        momentum_7 = (seq[-1] - seq[0]) / seq[0]    # 7 day momentum
        
        # 4. price position in range
        price_max = np.max(seq)
        price_min = np.min(seq)
        price_position = (seq[-1] - price_min) / (price_max - price_min) if price_max != price_min else 0.5
        
        # 5. trend strength
        price_slope = np.polyfit(range(len(seq)), seq, 1)[0] / seq[-1]  # normalized slope
        
        # combine all faetures
        features = relative_changes + [sma_3, sma_7, volatility, momentum_3, momentum_7, price_position, price_slope]
        features_list.append(features)
    
    return np.array(features_list)




def train_model():
    print("Starting advanced model training...")
    
    # generate training data
    X_sequences, y = generate_dummy_data(12000)
    
    # create enhanced features
    X_features = create_features(X_sequences)
    
    print(f"Feature dimensions: {X_features.shape}")
    
    # scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    # convert to pytrch tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    
    # split data with stratificatio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_tensor, y_tensor, test_size=0.4, random_state=42, 
        stratify=y_tensor.numpy().ravel()
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42,
        stratify=y_temp.numpy().ravel()
    )
    
    # init model
    input_size = X_features.shape[1]
    model = BitcoinPredictor(input_size=input_size)
    
    # training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15
    )
    
    # training loop with early stopping
    best_val_acc = 0
    patience = 25
    patience_counter = 0
    train_losses = []
    val_accuracies = []
    
    print("Starting training...")
    for epoch in range(300):
        # training phase
        model.train()
        train_loss = 0
        
        # forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        train_loss = loss.item()
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_predictions = (val_outputs > 0.5).float()
            val_accuracy = (val_predictions == y_val).float().mean().item()
            
            # test accuracy
            test_outputs = model(X_test)
            test_predictions = (test_outputs > 0.5).float()
            test_accuracy = (test_predictions == y_test).float().mean().item()
        
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        
        # learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_accuracy)
        new_lr = optimizer.param_groups[0]['lr']
        
        # print lr change if it occurred
        if old_lr != new_lr:
            print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        # early stopping and model saving
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': input_size,
                'best_val_acc': best_val_acc,
                'test_accuracy': test_accuracy
            }, 'bitcoin_model.pth') # bitcoin_model.pth
            joblib.dump(scaler, 'scaler.pkl') # scaler.pkl
        else:
            patience_counter += 1
        
        # print progress
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/300], Loss: {train_loss:.4f}, '
                  f'Val Acc: {val_accuracy:.4f}, Test Acc: {test_accuracy:.4f}, LR: {current_lr:.6f}')
        
        # early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print("Model saved as 'bitcoin_model.pth'")
    
    return model, scaler, best_val_acc




def load_model():
    try:
        checkpoint = torch.load('bitcoin_model.pth', map_location='cpu')
        
        input_size = checkpoint['input_size']
        model = BitcoinPredictor(input_size=input_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        scaler = joblib.load('scaler.pkl')
        
        print(f"Model loaded successfully!")
        print(f"Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
        
        return model, scaler
    except FileNotFoundError:
        print("Model files not found. Please run train_model() first.")
        return None, None




def ai_predict(prices_7_days):
    if len(prices_7_days) != 7:
        raise ValueError("Exactly 7 days of prices required")
    
    model, scaler = load_model()
    if model is None:
        return None, None
    
    # create features from the 7 day sequence
    features = create_features(np.array([prices_7_days]))
    features_scaled = scaler.transform(features)
    input_tensor = torch.FloatTensor(features_scaled)
    
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
    test_prices = [45000, 46000, 45500, 47000, 46800, 48000, 47500]
    pred, prob = ai_predict(test_prices)
    if pred is not None:
        direction = "UP" if pred == 1 else "DOWN"
        print(f"Prediction: {direction} ({prob:.2%})")