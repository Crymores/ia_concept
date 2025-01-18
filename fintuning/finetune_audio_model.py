import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, MFCC
from src.audio_model import AdvancedAudioModel

# ================================
# Configuration des paramètres
# ================================
CONFIG = {
    "data_dir": "./data/processed/audio/",
    "save_dir": "./final_model/",
    "model_path": "./final_model/best_audio_model.pth",  # Chemin du modèle pré-entraîné
    "batch_size": 16,
    "epochs": 20,
    "learning_rate": 1e-4,
    "freeze_encoder": False,  # Option pour geler l'encodeur
    "finetuning_type": "full",  # Type de finetuning : "full" ou "head"
    "num_classes": 10,
    "feature_type": "spectrogram",  # Choix : "spectrogram", "mfcc", "raw"
    "scheduler_step_size": 5,
    "scheduler_gamma": 0.5,
    "synthetic_data_dir": "./data/synthetic/audio/",  # Répertoire des données synthétiques
    "use_synthetic_data": False,  # Inclure les données synthétiques pour le finetuning
    "synthetic_weight": 0.3,  # Pondération des données synthétiques dans l'entraînement
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# ================================
# Dataset audio personnalisé
# ================================
class AdvancedAudioDataset(Dataset):
    def __init__(self, data_dir, labels_file, feature_type="spectrogram"):
        self.data_dir = data_dir
        self.labels = []
        self.files = []
        self.feature_type = feature_type

        # Charger les fichiers et labels
        with open(labels_file, "r") as f:
            for line in f:
                file_name, label = line.strip().split("\t")
                self.files.append(file_name)
                self.labels.append(int(label))

        # Préparer les transformations
        if feature_type == "spectrogram":
            self.transform = nn.Sequential(MelSpectrogram(), AmplitudeToDB())
        elif feature_type == "mfcc":
            self.transform = MFCC()
        else:
            self.transform = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.data_dir, self.files[idx])
        waveform, _ = torch.load(audio_path)  # Charger l'audio traité
        label = self.labels[idx]

        if self.transform:
            waveform = self.transform(waveform)

        return waveform.unsqueeze(0), label  # Ajouter une dimension de canal

# ================================
# Chargement des données synthétiques
# ================================
def load_synthetic_data(config):
    synthetic_data = []
    synthetic_labels = []

    if not os.path.exists(config["synthetic_data_dir"]):
        return None, None

    for file in os.listdir(config["synthetic_data_dir"]):
        if file.endswith(".pt"):
            data = torch.load(os.path.join(config["synthetic_data_dir"], file))
            synthetic_data.append(data["audio"])
            synthetic_labels.append(data["label"])

    return torch.stack(synthetic_data), torch.tensor(synthetic_labels)

# ================================
# Fonction de finetuning
# ================================
def finetune_model(config, model, train_loader, val_loader, synthetic_data=None, synthetic_labels=None):
    model.to(config["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler_step_size"], gamma=config["scheduler_gamma"])

    best_acc = 0.0

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print("-" * 10)

        # Phase d'entraînement
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(config["device"]), labels.to(config["device"])

            # Ajouter des données synthétiques si activé
            if config["use_synthetic_data"] and synthetic_data is not None:
                synthetic_inputs = synthetic_data[:len(inputs)].to(config["device"])
                synthetic_labels_batch = synthetic_labels[:len(inputs)].to(config["device"])

                # Combiner vraies et synthétiques
                combined_inputs = torch.cat([inputs, synthetic_inputs])
                combined_labels = torch.cat([labels, synthetic_labels_batch])
            else:
                combined_inputs = inputs
                combined_labels = labels

            optimizer.zero_grad()
            outputs = model(combined_inputs)
            loss = criterion(outputs, combined_labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * combined_inputs.size(0)
            running_corrects += torch.sum(preds == combined_labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Phase de validation
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
            torch.save(model.state_dict(), os.path.join(config["save_dir"], "finetuned_audio_model.pth"))
            print("Model improved, saving checkpoint.")

        scheduler.step()

# ================================
# Main
# ================================
def main():
    train_dataset = AdvancedAudioDataset(
        data_dir=os.path.join(CONFIG["data_dir"], "train"),
        labels_file=os.path.join(CONFIG["data_dir"], "train_labels.txt"),
        feature_type=CONFIG["feature_type"]
    )
    val_dataset = AdvancedAudioDataset(
        data_dir=os.path.join(CONFIG["data_dir"], "val"),
        labels_file=os.path.join(CONFIG["data_dir"], "val_labels.txt"),
        feature_type=CONFIG["feature_type"]
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    synthetic_data, synthetic_labels = None, None
    if CONFIG["use_synthetic_data"]:
        synthetic_data, synthetic_labels = load_synthetic_data(CONFIG)

    model = AdvancedAudioModel(num_classes=CONFIG["num_classes"], feature_type=CONFIG["feature_type"])
    model.load_state_dict(torch.load(CONFIG["model_path"]))

    # Configurer les paramètres pour le finetuning
    if CONFIG["freeze_encoder"]:
        for param in model.encoder.parameters():
            param.requires_grad = False
    if CONFIG["finetuning_type"] == "head":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    finetune_model(CONFIG, model, train_loader, val_loader, synthetic_data, synthetic_labels)

if __name__ == "__main__":
    main()
