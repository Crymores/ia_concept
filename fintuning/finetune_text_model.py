import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from src.text_model.text_model import CustomTextModel

# ================================
# Configuration des paramètres
# ================================
CONFIG = {
    "data_dir": "./data/processed/text/",  # Répertoire des données
    "save_dir": "./final_model/",  # Répertoire pour sauvegarder les modèles
    "model_path": "./final_model/best_custom_text_model.pth",  # Modèle pré-entraîné
    "batch_size": 16,  # Taille des lots
    "epochs": 10,  # Nombre d'époques pour le finetuning
    "learning_rate": 1e-4,  # Taux d'apprentissage initial
    "freeze_embedding": False,  # Gèle les embeddings si nécessaire
    "finetuning_type": "full",  # Type de finetuning : "full" ou "head"
    "scheduler_step_size": 5,  # Réduction du taux d'apprentissage tous les n epochs
    "scheduler_gamma": 0.5,  # Facteur de réduction du taux d'apprentissage
    "vocab_size": 5000,  # Taille du vocabulaire
    "embed_dim": 128,  # Dimension des embeddings
    "num_classes": 2,  # Nombre de classes de sortie
    "max_seq_len": 128,  # Longueur maximale des séquences
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # GPU ou CPU
    "synthetic_data_dir": "./data/synthetic/text/",  # Répertoire des données synthétiques (si applicable)
    "use_synthetic_data": False,  # Inclure les données synthétiques pour le finetuning
    "synthetic_weight": 0.3,  # Pondération des données synthétiques dans l'entraînement
}

# ================================
# Dataset personnalisé pour le texte
# ================================
class TextDataset(Dataset):
    def __init__(self, file_path, max_seq_len=128):
        self.data = []
        self.labels = []
        self.max_seq_len = max_seq_len

        with open(file_path, "r") as f:
            for line in f:
                label, text = line.strip().split("\t")
                self.labels.append(int(label))
                tokens = [int(t) for t in text.split()[:max_seq_len]]
                tokens += [0] * (max_seq_len - len(tokens))  # Padding
                self.data.append(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

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
            synthetic_data.append(data["text"])
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
        print(f"Epoch {epoch+1}/{config['epochs']}")
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
                synthetic_inputs_labels = synthetic_labels[:len(inputs)].to(config["device"])

                # Combiner vraies et synthétiques
                combined_inputs = torch.cat([inputs, synthetic_inputs])
                combined_labels = torch.cat([labels, synthetic_inputs_labels])
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
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Sauvegarde du meilleur modèle
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config["save_dir"], "finetuned_text_model.pth"))
            print("Model improved, saving checkpoint.")

        scheduler.step()

# ================================
# Main
# ================================
def main():
    # Charger les données
    train_dataset = TextDataset(os.path.join(CONFIG["data_dir"], "finetune_train.txt"), CONFIG["max_seq_len"])
    val_dataset = TextDataset(os.path.join(CONFIG["data_dir"], "finetune_val.txt"), CONFIG["max_seq_len"])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # Charger les données synthétiques si activé
    synthetic_data, synthetic_labels = None, None
    if CONFIG["use_synthetic_data"]:
        synthetic_data, synthetic_labels = load_synthetic_data(CONFIG)

    # Charger le modèle pré-entraîné
    model = CustomTextModel(
        vocab_size=CONFIG["vocab_size"],
        embed_dim=CONFIG["embed_dim"],
        num_classes=CONFIG["num_classes"]
    )
    model.load_state_dict(torch.load(CONFIG["model_path"]))

    # Ajuster les paramètres pour le finetuning
    if CONFIG["freeze_embedding"]:
        for param in model.embedding.parameters():
            param.requires_grad = False
    if CONFIG["finetuning_type"] == "head":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    # Lancer le finetuning
    finetune_model(CONFIG, model, train_loader, val_loader, synthetic_data, synthetic_labels)

if __name__ == "__main__":
    main()
