import yfinance as yf
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        """
        X : numpy array de forme (num_samples, seq_length, num_features)
        y : numpy array de forme (num_samples,)
        """
        self.X = torch.tensor(X, dtype=torch.float64)
        self.y = torch.tensor(y, dtype=torch.float64)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sequences(X, y, seq_length=10):
    """
    Construit des séquences glissantes pour le modèle.
    Chaque séquence de 'seq_length' jours sert à prédire la valeur du jour suivant.
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

def extract_data(action, period):
    
    ticker = yf.Ticker(action)
    df = ticker.history(period=period)

    #INPUT
    # Moyenne mobile sur 20 et 50 jours
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    # Calcul du RSI sur 14 jours
    df['RSI'] = compute_rsi(df['Close'], window=14)
    # Calcul de la volatilité (écart-type des rendements sur 20 jours)
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=20).std()
    df = df.dropna() # Supprimer les lignes avec des valeurs manquantes (par exemple dues aux fenêtres mobiles)

    # Pour chaque jour Sentiment Analysis
    # sentiment_scores = df.index.to_series().apply(get_daily_sentiment)
    # sentiment_df = pd.DataFrame(list(sentiment_scores), index=df.index)
    # df = pd.concat([df, sentiment_df], axis=1)
    
    features = ['Open', 'High', 'Low', 'Volume', 'MA20', 'MA50', 'RSI', 'Volatility']#,
                #'sentiment_positive', 'sentiment_neutral', 'sentiment_negative']

    # OUTPUT
    target = 'Close'

    X = df[features]
    y = df[target]

    return X, y

def load_data(X, y, batch_size = 32, train_prctg = 0.8, seq_length=10) :
    seq_length = 10
    X_seq, y_seq = create_sequences(X, y, seq_length=seq_length)

    split_index = int(train_prctg * len(X_seq))
    X_train, y_train = X_seq[:split_index], y_seq[:split_index]
    X_val, y_val = X_seq[split_index:], y_seq[split_index:]

    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

#---Fonction secondaires-----------------------------------------------------------------------------------------------

def compute_rsi(series, window=14):
    delta = series.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_daily_sentiment(date):
    return {'sentiment_positive': 0.5, 'sentiment_neutral': 0.3, 'sentiment_negative': 0.2}