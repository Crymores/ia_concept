import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedVideoModel(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, frame_size=(112, 112), sequence_length=16):
        super(AdvancedVideoModel, self).__init__()
        
        # Extraction spatiale (CNN 2D)
        self.spatial_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Extraction temporelle (CNN 3D)
        self.temporal_extractor = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        # Attention avancée
        self.attention = nn.Linear(64, 64)
        self.attention_weight = nn.Parameter(torch.tensor(3.14159))  # Constante π

        # Tête de classification
        self.fc = nn.Sequential(
            nn.Linear(64 * sequence_length, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        # Module générateur vidéo
        self.generator = nn.Sequential(
            nn.ConvTranspose3d(64, 128, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 256, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.ConvTranspose3d(256, input_channels, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.Tanh()  # Normaliser entre -1 et 1
        )

        # Intégration des constantes
        self.inv_12_reg = -1 / 12
        self.inv_137_adjust = 1 / 137

    def forward(self, x, generate=False):
        if generate:
            # Génération vidéo à partir d'un vecteur latent
            latent = torch.randn(x.size(0), 64, 2, 2, 2, device=x.device)  # Vecteur latent 3D
            generated_video = self.generator(latent)
            return generated_video

        batch_size, seq_len, c, h, w = x.size()
        
        # Appliquer la spatial extractor frame par frame
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.spatial_extractor(x)  # [batch_size * seq_len, 256, h', w']
        _, c, h, w = x.size()
        x = x.view(batch_size, seq_len, c, h, w)

        # Appliquer la temporal extractor
        x = self.temporal_extractor(x)  # [batch_size, 64, seq_len, h', w']

        # Attention avancée
        batch_size, channels, seq_len, h, w = x.size()
        x_flat = x.view(batch_size, channels, -1)  # [batch_size, channels, seq_len * h * w]
        attention_scores = torch.tanh(self.attention(x_flat.mean(dim=2))) * self.attention_weight
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, channels]
        x = x * attention_weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # Pondérer chaque canal

        # Aplatir pour la classification
        x = x.view(batch_size, -1)  # [batch_size, 64 * seq_len]
        output = self.fc(x)

        return output
