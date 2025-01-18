import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedAudioModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, feature_type="spectrogram"):
        super(AdvancedAudioModel, self).__init__()

        # Paramètre pour le type d'entrée
        self.feature_type = feature_type

        # Bloc d'encodage universel
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Attention avancée
        self.attention = nn.Linear(128, 128)
        self.attention_weight = nn.Parameter(torch.tensor(3.14159))  # Influence de π

        # Couches convolutives multi-échelle
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.Conv2d(128, 128, kernel_size=(5, 5), padding=(2, 2)),
            nn.Conv2d(128, 128, kernel_size=(7, 7), padding=(3, 3))
        ])

        # Couches résiduelles
        self.residual = nn.Linear(128, 128)

        # Tête de classification
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        # Module générateur audio
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Tanh()  # Pour normaliser la sortie entre -1 et 1
        )

        # Intégration des constantes
        self.inv_12_reg = -1 / 12
        self.inv_137_adjust = 1 / 137

    def forward(self, x, generate=False):
        if generate:
            # Génération audio à partir d'un vecteur latent
            latent = torch.randn(x.size(0), 128, 1, 1, device=x.device)
            generated_audio = self.generator(latent)
            return generated_audio

        # Encodage universel
        x = self.encoder(x)

        # Convolutions multi-échelle
        multi_scale_features = [conv(x) for conv in self.multi_scale_conv]
        x = torch.stack(multi_scale_features).sum(dim=0)

        # Attention avancée
        batch_size, channels, height, width = x.size()
        x_flat = x.view(batch_size, channels, -1)
        attention_scores = torch.tanh(self.attention(x_flat.mean(dim=2))) * self.attention_weight
        attention_weights = F.softmax(attention_scores, dim=1)
        x = x * attention_weights.unsqueeze(2).unsqueeze(3)

        # Couches résiduelles
        residual = self.residual(x_flat.mean(dim=2))
        x = x_flat.mean(dim=2) + residual

        # Classification
        x = x.view(batch_size, -1)
        output = self.fc(x)

        return output
