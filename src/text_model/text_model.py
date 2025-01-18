import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomTextModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, max_seq_len=128):
        super(CustomTextModel, self).__init__()
        
        # Embedding avec dimensions influencées par \u03c6
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.phi_ratio = 1.618  # Nombre d'or
        self.embed_transform = nn.Linear(embed_dim, int(embed_dim * self.phi_ratio))

        # Blocs convolutifs 1D
        self.conv1 = nn.Conv1d(in_channels=int(embed_dim * self.phi_ratio), out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # Attention fait maison
        self.attention = nn.Linear(256, 256)
        self.attention_weight = nn.Parameter(torch.tensor(3.14159))  # Constante \u03c0

        # Couches résiduelles
        self.residual = nn.Linear(256, 256)

        # Tête de classification
        self.fc = nn.Sequential(
            nn.Linear(256 * max_seq_len, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        # Régularisation et constantes
        self.inv_12_reg = -1 / 12
        self.inv_137_adjust = 1 / 137

    def forward(self, input_text):
        # Embedding des mots
        embedded = self.embedding(input_text)  # [batch_size, seq_len, embed_dim]
        embedded = self.embed_transform(embedded)  # [batch_size, seq_len, embed_dim * \u03c6]

        # Convolutions 1D
        x = embedded.permute(0, 2, 1)  # [batch_size, embed_dim * \u03c6, seq_len]
        x = F.relu(self.conv1(x))  # [batch_size, 128, seq_len]
        x = F.relu(self.conv2(x))  # [batch_size, 256, seq_len]

        # Attention fait maison
        attention_scores = torch.tanh(self.attention(x.permute(0, 2, 1))) * self.attention_weight
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len, 256]
        x = x * attention_weights.permute(0, 2, 1)  # Pondération par attention

        # Couches résiduelles
        residual = self.residual(x.permute(0, 2, 1))
        x = x + residual.permute(0, 2, 1)  # [batch_size, 256, seq_len]

        # Flatten et classification
        x = x.view(x.size(0), -1)  # [batch_size, 256 * seq_len]
        output = self.fc(x)  # [batch_size, num_classes]

        return output
