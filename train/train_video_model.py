import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from src.video_model.advanced_video_model import AdvancedVideoModel
import cv2
from pytube import YouTube

# ================================
# Configuration des paramètres
# ================================
CONFIG = {
    "data_dir": "./data/processed/video/",
    "save_dir": "./final_model/",
    "batch_size": 8,
    "epochs": 20,
    "learning_rate": 1e-4,
    "num_classes": 10,
    "frame_size": (112, 112),
    "sequence_length": 16,
    "use_synthetic_generation": True,  # Générer des vidéos synthétiques pendant l'entraînement
    "synthetic_samples_per_epoch": 100,  # Nombre de vidéos synthétiques à générer par époque
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "youtube_download_dir": "./data/youtube/"
}

# ================================
# Dataset vidéo personnalisé
# ================================
class VideoDataset(Dataset):
    def __init__(self, data_dir, labels_file, frame_size=(112, 112), sequence_length=16):
        self.data_dir = data_dir
        self.labels = []
        self.files = []
        self.frame_size = frame_size
        self.sequence_length = sequence_length

        with open(labels_file, "r") as f:
            for line in f:
                file_name, label = line.strip().split("\t")
                self.files.append(file_name)
                self.labels.append(int(label))

        self.transform = Compose([
            Resize(frame_size),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.data_dir, self.files[idx])
        frames = torch.load(video_path)
        frames = torch.stack([self.transform(frame) for frame in frames[:self.sequence_length]])
        label = self.labels[idx]

        return frames, label

# ================================
# Fonction de génération de vidéos synthétiques
# ================================
def generate_synthetic_videos(model, config):
    synthetic_videos = []
    synthetic_labels = []

    model.eval()
    with torch.no_grad():
        for _ in range(config["synthetic_samples_per_epoch"]):
            latent = torch.randn(config["batch_size"], 64, 2, 2, 2, device=config["device"])
            generated_video = model(latent, generate=True).cpu()
            synthetic_videos.append(generated_video)
            synthetic_labels.extend([config["num_classes"] - 1] * generated_video.size(0))  # Etiquette par défaut

    return torch.cat(synthetic_videos), torch.tensor(synthetic_labels)

# ================================
# Fonction d'entraînement
# ================================
def train_model(config, model, train_loader, val_loader):
    model.to(config["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print("-" * 10)

        # Génération de vidéos synthétiques si activé
        if config["use_synthetic_generation"]:
            synthetic_videos, synthetic_labels = generate_synthetic_videos(model, config)
            synthetic_dataset = torch.utils.data.TensorDataset(synthetic_videos, synthetic_labels)
            synthetic_loader = DataLoader(synthetic_dataset, batch_size=config["batch_size"], shuffle=True)
        else:
            synthetic_loader = None

        # Phase d'entraînement
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(config["device"]), labels.to(config["device"])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * len(inputs)
            running_corrects += torch.sum(preds == labels)

        if synthetic_loader:
            for inputs, labels in synthetic_loader:
                inputs, labels = inputs.to(config["device"]), labels.to(config["device"])

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * len(inputs)
                running_corrects += torch.sum(preds == labels)

        epoch_loss = running_loss / (len(train_loader.dataset) + (len(synthetic_loader.dataset) if synthetic_loader else 0))
        epoch_acc = running_corrects.double() / (len(train_loader.dataset) + (len(synthetic_loader.dataset) if synthetic_loader else 0))
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * len(inputs)
                val_corrects += torch.sum(preds == labels)

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Sauvegarde du meilleur modèle
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config["save_dir"], "best_video_model.pth"))
            print("Model improved, saving checkpoint.")

        scheduler.step()

# ================================
# Main
# ================================
def main():
    train_dataset = VideoDataset(
        data_dir=os.path.join(CONFIG["data_dir"], "train"),
        labels_file=os.path.join(CONFIG["data_dir"], "train_labels.txt"),
        frame_size=CONFIG["frame_size"],
        sequence_length=CONFIG["sequence_length"]
    )
    val_dataset = VideoDataset(
        data_dir=os.path.join(CONFIG["data_dir"], "val"),
        labels_file=os.path.join(CONFIG["data_dir"], "val_labels.txt"),
        frame_size=CONFIG["frame_size"],
        sequence_length=CONFIG["sequence_length"]
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    model = AdvancedVideoModel(
        num_classes=CONFIG["num_classes"],
        frame_size=CONFIG["frame_size"],
        sequence_length=CONFIG["sequence_length"]
    )

    train_model(CONFIG, model, train_loader, val_loader)

if __name__ == "__main__":
    main()
