import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from src.utils_model.constants import PHI, PI, INV_12, INV_137


class ConceptualImageModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ConceptualImageModel, self).__init__()

        # Prétraitement dynamique
        self.preprocessor = transforms.Compose([
            transforms.Resize((224, 224)),  # Redimensionne toutes les images
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Modèle de base : ResNet-50 pré-entraîné
        self.base_model = models.resnet50(pretrained=pretrained)
        
        # Personnalisation de la tête finale
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, int(self.base_model.fc.in_features * PHI)),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(int(self.base_model.fc.in_features * PHI), num_classes)
        )
        
        # Générateur d'images synthétiques conditionnelles
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
        # Branche conceptuelle pour interpréter les idées
        self.conceptual_branch = nn.Sequential(
            nn.Linear(512, int(512 * PI)),  # Transformation des concepts en vecteurs visuels
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(int(512 * PI), 512)
        )
        
        # Intégration des constantes mathématiques
        self.pi_weight = nn.Parameter(torch.tensor(PI))
        self.inv_12_reg = INV_12
        self.inv_137_adjust = INV_137
        
        # Mécanisme d’auto-évaluation
        self.auto_eval_loss = nn.MSELoss()
    
    def preprocess(self, x):
        # Applique les prétraitements dynamiques aux images d’entrée
        return self.preprocessor(x)

    def forward(self, x, concept=None):
        # Prétraitement initial
        x = self.preprocess(x)
        
        # Passage dans le modèle principal pour extraire des caractéristiques
        features = self.base_model(x)
        
        # Génération conditionnelle d’images synthétiques
        z = torch.randn(features.size(0), 128, 1, 1, device=features.device)
        if concept is not None:
            concept_vector = self.conceptual_branch(concept)  # Transforme le concept
            z += concept_vector.view(z.size(0), 128, 1, 1)  # Ajoute le concept comme condition
        
        synthetic_images = self.generator(z)
        
        # Auto-évaluation : évaluer la similarité entre caractéristiques et données synthétiques
        eval_score = self.auto_eval_loss(features, features.detach() * self.inv_137_adjust)
        
        # Rétroaction : ajuster les caractéristiques en fonction de l’évaluation
        features = features - eval_score * self.inv_12_reg
        
        return features, synthetic_images, eval_score
