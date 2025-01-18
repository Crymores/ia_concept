import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.image_model.image_model import ConceptualImageModel

# ================================
# Configuration des paramètres
# ================================
CONFIG = {
    "data_dir": "./data/processed/images/",  # Répertoire des données
    "synthetic_dir": "./data/synthetic/images/",  # Répertoire des données générées
    "save_dir": "./final_model/",  # Répertoire pour sauvegarder les modèles
    "model_path": "./final_model/best_image_model.pth",  # Chemin du modèle pré-entraîné
    "batch_size": 16,  # Taille des lots
    "synthetic_batch_size": 32,  # Taille des lots pour la génération
    "epochs": 10,  # Nombre d'époques pour le finetuning
    "learning_rate": 5e-5,  # Taux d'apprentissage
    "freeze_base_model": True,  # Gèle les couches de base pour le finetuning
    "finetuning_type": "full",  # Type de finetuning : "full" ou "head" (seule la tête est ajustée)
    "num_synthetic_images": 1000,  # Nombre total d'images synthétiques à générer
    "scheduler_step_size": 5,  # Intervalle de réduction du taux d'apprentissage
    "scheduler_gamma": 0.1,  # Facteur de réduction du taux d'apprentissage
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # GPU ou CPU
}

# ================================
# Préparation des données
# ================================
def get_dataloaders(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "finetune_train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "finetune_val"), transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# ================================
# Génération de données synthétiques
# ================================
def generate_synthetic_images(model, config):
    os.makedirs(config["synthetic_dir"], exist_ok=True)
    model.to(config["device"])
    model.eval()
    
    print(f"Generating {config['num_synthetic_images']} synthetic images...")
    generated_images = []

    with torch.no_grad():
        for _ in range(0, config["num_synthetic_images"], config["synthetic_batch_size"]):
            z = torch.randn(config["synthetic_batch_size"], 128, 1, 1, device=config["device"])
            synthetic_images = model.generator(z).cpu()
            generated_images.extend(synthetic_images)

    # Sauvegarde des images générées
    for idx, img in enumerate(generated_images):
        save_path = os.path.join(config["synthetic_dir"], f"synthetic_{idx}.pt")
        torch.save(img, save_path)

    print(f"Images synthétiques sauvegardées dans {config['synthetic_dir']}")

# ================================
# Auto-évaluation des données synthétiques
# ================================
def evaluate_synthetic_images(model, config):
    synthetic_files = [os.path.join(config["synthetic_dir"], f) for f in os.listdir(config["synthetic_dir"]) if f.endswith(".pt")]
    similarity_scores = []

    print(f"Evaluating {len(synthetic_files)} synthetic images...")
    with torch.no_grad():
        for file in synthetic_files:
            img = torch.load(file).to(config["device"])
            img = img.unsqueeze(0)  # Ajouter la dimension batch
            
            # Extraire des caractéristiques avec le modèle
            features, _, _ = model(img)
            
            # Comparer les caractéristiques à des images aléatoires
            random_z = torch.randn(1, 128, 1, 1, device=config["device"])
            random_img = model.generator(random_z)
            random_features, _, _ = model(random_img)
            
            similarity = nn.functional.cosine_similarity(features, random_features).item()
            similarity_scores.append(similarity)
    
    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    print(f"Score moyen de similarité des données synthétiques : {avg_similarity:.4f}")
    return avg_similarity

# ================================
# Finetuning avec données synthétiques
# ================================
def finetune_model_with_synthetic(model, config, train_loader, val_loader, synthetic_files):
    model.to(config["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
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

            # Ajouter des données synthétiques
            synthetic_data = [torch.load(f).to(config["device"]) for f in synthetic_files[:len(inputs)]]
            synthetic_inputs = torch.stack(synthetic_data)
            synthetic_labels = torch.zeros(synthetic_inputs.size(0), dtype=torch.long).to(config["device"])  # Labels neutres

            # Combiner vraies et synthétiques
            combined_inputs = torch.cat([inputs, synthetic_inputs])
            combined_labels = torch.cat([labels, synthetic_labels])

            optimizer.zero_grad()
            outputs, _, _ = model(combined_inputs)
            loss = criterion(outputs, combined_labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * combined_inputs.size(0)
            running_corrects += torch.sum(preds == combined_labels.data)

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
                outputs, _, _ = model(inputs)
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
            torch.save(model.state_dict(), os.path.join(config["save_dir"], "finetuned_with_synthetic.pth"))
            print("Model improved with synthetic data, saving checkpoint.")

        scheduler.step()

# ================================
# Main
# ================================
def main():
    # Charger le modèle
    model = ConceptualImageModel(num_classes=10, pretrained=False)
    model.load_state_dict(torch.load(CONFIG["model_path"]))

    # Générer et évaluer des données synthétiques
    generate_synthetic_images(model, CONFIG)
    evaluate_synthetic_images(model, CONFIG)

    # Charger les données pour finetuning
    train_loader, val_loader = get_dataloaders(CONFIG["data_dir"], CONFIG["batch_size"])
    synthetic_files = [os.path.join(CONFIG["synthetic_dir"], f) for f in os.listdir(CONFIG["synthetic_dir"]) if f.endswith(".pt")]

    # Finetuning avec données synthétiques
    finetune_model_with_synthetic(model, CONFIG, train_loader, val_loader, synthetic_files)

if __name__ == "__main__":
    main()
