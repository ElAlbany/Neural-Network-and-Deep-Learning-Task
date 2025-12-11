import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from helper import create_optimal_vocabulary, preprocess_text_for_small_data
from models import TextCNNGLU


def load_data(csv_path: str = "train_augmented.csv") -> pd.DataFrame:
    """
    Load augmented data if available, otherwise fall back to original train.csv.
    """
    if os.path.exists(csv_path):
        print(f"Loading augmented data from {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
        print(f"{csv_path} not found. Falling back to train.csv...")
        df = pd.read_csv("train.csv")
    return df[["text", "review"]].dropna()


def encode_dataset(df: pd.DataFrame, vocab: Dict[str, int], max_len: int = 150):
    """
    Encode all texts in the dataframe to fixed-length integer sequences.
    """
    print(f"Encoding {len(df):,} texts with max_len={max_len}...")
    encoded = [preprocess_text_for_small_data(t, vocab, max_len=max_len) for t in df["text"].tolist()]
    X = torch.tensor(encoded, dtype=torch.long)

    label_map = {"Very bad": 0, "Bad": 1, "Good": 2, "Very good": 3, "Excellent": 4}
    y = torch.tensor(df["review"].map(label_map).values, dtype=torch.long)
    return X, y, label_map


def make_loaders(X: torch.Tensor, y: torch.Tensor, batch_size: int = 64):
    """
    Create train/validation/test DataLoaders.
    """
    train_size, val_size, test_size = 0.7, 0.15, 0.15

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    val_size_adjusted = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    print("✅ DataLoaders created:")
    print(f"  Train: {len(X_train):,} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(X_val):,} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(X_test):,} samples, {len(test_loader)} batches")
    return train_loader, val_loader, test_loader


def train_textcnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
):
    """
    Train TextCNNGLU on the provided loaders and return the best model (in memory).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_state = None

    print(f"Starting training for {num_epochs} epochs on {device}...")
    print("=" * 60)

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, _ = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_total += yb.size(0)
                val_correct += (preds == yb).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {avg_val_loss:.4f}, Val   Acc: {val_acc:.2f}%")
        print("-" * 60)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"✅ Loaded best validation state (Val Acc: {best_val_acc:.2f}%)")
    return model


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, label_map: Dict[str, int]):
    """
    Evaluate the model on a DataLoader and print metrics.
    """
    inv_label_map = {v: k for k, v in label_map.items()}
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds, all_labels = [], []
    test_loss = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            test_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    avg_loss = test_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds) * 100

    print("\n" + "=" * 60)
    print("TEST SET RESULTS (TextCNNGLU)")
    print("=" * 60)
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {acc:.2f}%")
    print("\nClassification Report:")
    target_names = [inv_label_map[i] for i in range(len(inv_label_map))]
    print(classification_report(all_labels, all_preds, target_names=target_names))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


def main():
    df = load_data("train_augmented.csv")

    # Create vocabulary from augmented texts
    vocab = create_optimal_vocabulary(df["text"].tolist(), target_size=8000)
    print(f"Vocabulary size: {len(vocab):,}")

    # Encode data
    X, y, label_map = encode_dataset(df, vocab, max_len=150)

    # Make loaders
    train_loader, val_loader, test_loader = make_loaders(X, y, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TextCNNGLU(vocab_size=len(vocab), embed_dim=128, num_classes=len(label_map)).to(device)

    # Train (in-memory best model)
    model = train_textcnn(model, train_loader, val_loader, device, num_epochs=10, lr=1e-3, weight_decay=1e-2)

    # Evaluate on test set (no saved model reload)
    evaluate(model, test_loader, device, label_map)


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    main()
