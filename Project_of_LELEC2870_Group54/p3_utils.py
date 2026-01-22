import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ============================================================
# 0) Gestion du device
# ============================================================

def get_device():
    """Retourne 'cuda' si disponible, sinon 'cpu'."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1) Dataset Images
# ============================================================

class HeartImageDataset(Dataset):
    """
    Dataset PyTorch pour charger les images PNG du projet.
    
    Paramètres :
    -----------
    image_filenames : list[str]
        Liste des noms de fichiers, ex : ["heart_0.png", ...]
    images_dir : str
        Chemin du dossier contenant les images (Img_train ou Img_test)
    """

    def __init__(self, image_filenames, images_dir):
        self.image_filenames = list(image_filenames)
        self.images_dir = images_dir

        # CNN simple → images normalisées [0.5, 0.5]
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((48, 48)),              # taille cohérente pour CNN simple
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        fname = self.image_filenames[idx]
        path = os.path.join(self.images_dir, fname)
        img = Image.open(path).convert("L")
        img = self.transform(img)
        return img


#=========================================================


#========================================================

from torch.utils.data import DataLoader

# ============================================================
# Fonction DataLoader 
# ============================================================

def build_image_dataloader(dataset, batch_size=32, shuffle=False, num_workers=0):
    """
    Construit un DataLoader PyTorch pour les images.

    Paramètres :
    -----------
    dataset : HeartImageDataset
    batch_size : int
    shuffle : bool
    num_workers : int

    Retour :
    --------
    DataLoader prêt à l’emploi
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


# ============================================================
# 2) CNN simple
# ============================================================

class SimpleCNNExtractor(nn.Module):
    """
    CNN léger que nous allons utiliser pour extraire des features (dimension 64).
    Conformé au niveau attendu dans un TP universitaire.
    """

    def __init__(self):
        super().__init__()

        self.features_block = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),    # (48 → 24)

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),    # (24 → 12)
        )

        # Feature map → densité finale
        self.fc = nn.Linear(32 * 12 * 12, 64)  # vecteur final = 64-dim

    def forward(self, x):
        x = self.features_block(x)
        x = x.view(x.size(0), -1)  # flatten
        features = self.fc(x)
        return features


# ============================================================
# 3) Extraction des features images
# ============================================================

def extract_image_features(model, dataloader):
    """
    Extrait un vecteur de dimension 64 pour chaque image.

    Retour :
    -------
    numpy array de forme (n_samples, 64)
    """

    device = get_device()
    model = model.to(device)
    model.eval()

    feats_list = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            feats = model(batch)          # shape (B, 64)
            feats_list.append(feats.cpu().numpy())

    return np.vstack(feats_list)


# ============================================================
# 4) Fusion Tabulaire + Images
# ============================================================

def combine_tabular_and_image_features(X_train_tab, X_test_tab,
                                       X_train_img, X_test_img,
                                       img_feature_prefix="img_feat_"):
    """
    Concatène les variables tabulaires (Partie 2) et les features images (Partie 3).

    Retour :
    -------
    X_train_combined : (n_train, p_tab + 64)
    X_test_combined  : (n_test, p_tab + 64)
    feature_names    : liste de noms des colonnes (tab + img)
    """

    X_train_tab = np.asarray(X_train_tab)
    X_test_tab = np.asarray(X_test_tab)
    X_train_img = np.asarray(X_train_img)
    X_test_img = np.asarray(X_test_img)

    # sécurité
    assert len(X_train_tab) == len(X_train_img)
    assert len(X_test_tab) == len(X_test_img)

    # fusion horizontale
    X_train_comb = np.hstack([X_train_tab, X_train_img])
    X_test_comb = np.hstack([X_test_tab, X_test_img])

    # noms des features images
    img_cols = [f"{img_feature_prefix}{i}" for i in range(X_train_img.shape[1])]
    tab_cols = [f"tab_{i}" for i in range(X_train_tab.shape[1])]
    feature_names = tab_cols + img_cols

    print("\n=== FUSION TAB + IMG ===")
    print("X_train_tab :", X_train_tab.shape)
    print("X_train_img :", X_train_img.shape)
    print("X_train_comb:", X_train_comb.shape)
    print("X_test_comb :", X_test_comb.shape)

    return X_train_comb, X_test_comb, feature_names


# ============================================================
# 5) VISUALISATION OPTIONNELLE
# ============================================================

def show_image_grid(image_filenames, images_dir, n=12):
    """Affiche un grid d’images pour le rapport."""
    plt.figure(figsize=(12, 6))

    for i in range(min(n, len(image_filenames))):
        path = os.path.join(images_dir, image_filenames[i])
        img = Image.open(path).convert("L")

        plt.subplot(3, 4, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")

    plt.suptitle("Échantillon d’images — Dataset cardiaque")
    plt.tight_layout()
    plt.show()


def tsne_plot(features, y, title="t-SNE des features images"):
    """Représentation 2D par t-SNE."""
    features = np.asarray(features)
    y = np.asarray(y).ravel()

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb = tsne.fit_transform(features)

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=y, cmap="viridis", alpha=0.7)
    plt.colorbar(sc, label="Risk")
    plt.title(title)
    plt.show()
