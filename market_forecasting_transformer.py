import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Module d'encodage positionnel permettant d'injecter l'information temporelle dans les embeddings.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Création d'une matrice (max_len, d_model) pour stocker les encodages positionnels
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # forme : (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Ajoute l'encodage positionnel à l'input.
        x : Tensor de forme (batch_size, seq_length, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerMarketPredictor(nn.Module):
    """
    Modèle de prédiction de marché basé sur un Transformer.
    Il utilise une projection d'entrée, un encodage positionnel et un Transformer Encoder.
    """
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dropout=0.1):
        super(TransformerMarketPredictor, self).__init__()
        # Projection des caractéristiques d'entrée dans un espace de dimension d_model
        self.input_projection = nn.Linear(input_size, d_model)
        # Encodage positionnel pour ajouter l'information temporelle
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        # Définition d'une couche encoder Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Couche de sortie pour la prédiction finale
        self.fc_out = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        """
        x : Tensor de forme (batch_size, seq_length, input_size)
        """
        # Projection de l'input
        x = self.input_projection(x)  # (batch_size, seq_length, d_model)
        # Ajout de l'encodage positionnel
        x = self.positional_encoding(x)
        # Transformer attend une entrée de forme (seq_length, batch_size, d_model)
        x = x.transpose(0, 1)
        # Passage dans le Transformer encoder
        transformer_output = self.transformer_encoder(x)
        # On utilise la dernière sortie temporelle pour la prédiction
        last_output = transformer_output[-1]  # (batch_size, d_model)
        output = self.fc_out(last_output)
        return output

