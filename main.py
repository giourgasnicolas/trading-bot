import torch
import torch.nn as nn
import torch.optim as optim
from market_forecasting_transformer import TransformerMarketPredictor

from data import extract_data, load_data
from train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#--Parameter----------------------------------------
train_prctg = 0.8
batch_size = 32
epochs = 100
learning_rate = 0.001

input_size = 8            # Dimension des features d'entrée
d_model = 64               # Dimension des embeddings internes du Transformer
nhead = 4                  # Nombre de têtes d'attention
num_layers = 2             # Nombre de couches de Transformer Encoder
output_size = 1            # Prédiction d'une valeur continue

#--DATA---------------------------------------------
X,y = extract_data("MSFT", "10y")
train_loader, val_loader = load_data(X,y, train_prctg=train_prctg, batch_size=batch_size)

#--Modèle-------------------------------------------

architecture_details = {
    "d_model": d_model,
    "nhead": nhead,
    "num_layers": num_layers,
    "learning_rate": learning_rate,
    "epochs": epochs,
    "loss" : "MSE", 
    "batch_size": batch_size,
    "train_prctg": train_prctg,
}

model = TransformerMarketPredictor(input_size, d_model, nhead, num_layers, output_size)
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#--Train---------------------------------------------

train(epochs, model, optimizer, criterion, train_loader, val_loader, architecture_details=architecture_details)

#--Evaluate------------------------------------------




