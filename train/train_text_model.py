import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from src.text_model.text_model import CustomTextModel

# Configuration
CONFIG = {
    "data_dir": "./data/processed/text/",
    "save_dir": "./final_model/",
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 1e-3,
    "vocab_size": 5000,
    "embed_dim": 128,
    "num_classes": 2,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# Dataset personnalisé
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

# Fonction d'entraînement
def train_model(config, model, train_loader, val_loader):
    model.to(config["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print("-" * 10)

        # Entraînement
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
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
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
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Sauvegarde du meilleur modèle
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config["save_dir"], "best_custom_text_model.pth"))
            print("Model improved, saving checkpoint.")

        scheduler.step()

# Main
def main():
    train_dataset = TextDataset(os.path.join(CONFIG["data_dir"], "train.txt"))
    val_dataset = TextDataset(os.path.join(CONFIG["data_dir"], "val.txt"))
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    model = CustomTextModel(
        vocab_size=CONFIG["vocab_size"],
        embed_dim=CONFIG["embed_dim"],
        num_classes=CONFIG["num_classes"]
    )

    train_model(CONFIG, model, train_loader, val_loader)

if __name__ == "__main__":
    main()
