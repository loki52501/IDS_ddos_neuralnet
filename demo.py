"""
IDS Demo -- End-to-end inference pipeline for DDoS detection.

Usage:
    python demo.py --mode inference   # Run inference (requires trained models)
    python demo.py --mode train       # Train a tiny demo model on a subset
    python demo.py --mode pipeline    # Show full preprocessing pipeline only

This script demonstrates the complete flow from raw network flows
to Attack / Benign / Suspicious classification.
"""

import argparse
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_PATH = "dataset/merged_CSVs.csv"
CLEAN_PATH = "ddos_clean.parquet"
MODELS_DIR = "models"

DROP_COLS = ["flow_id", "src_ip", "dst_ip", "timestamp"]
STRING_SENTINEL = "not a complete handshake"
PARSE_COLS = ["delta_start", "handshake_duration"]

WINDOW_SIZE = 30
STRIDE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model architectures (must match training code)
# ---------------------------------------------------------------------------


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM + soft attention."""

    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=3, dropout=0.4):
        super().__init__()
        self.proj_linear = nn.Linear(input_size, 128)
        self.proj_bn = nn.BatchNorm1d(128)
        self.proj_drop = nn.Dropout(dropout * 0.5)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        feat = hidden_size * 2
        self.attn_score = nn.Linear(feat, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(feat),
            nn.Linear(feat, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        B, T, n_feat = x.shape
        x_flat = self.proj_linear(x.reshape(B * T, n_feat))
        x_flat = self.proj_bn(x_flat)
        x_flat = torch.relu(x_flat)
        x_flat = self.proj_drop(x_flat)
        x = x_flat.reshape(B, T, 128)
        out, _ = self.lstm(x)
        scores = self.attn_score(out)
        weights = torch.softmax(scores, dim=1)
        context = (weights * out).sum(dim=1)
        return self.head(context)


class CNN1DClassifier(nn.Module):
    """1D Residual CNN."""

    def __init__(self, input_size, num_classes=3, dropout=0.4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_size, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
        )
        self.proj2 = nn.Conv1d(128, 256, kernel_size=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.relu(self.block1(x) + x)
        x = self.relu(self.block2(x) + self.proj2(x))
        x = self.pool(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------


def clean_dataset(df):
    """Clean and preprocess raw DataFrame."""
    print("\n[1] Cleaning dataset...")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"    [OK] Sorted chronologically")

    for col in PARSE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace(STRING_SENTINEL, np.nan), errors="coerce")
    print(f"    [OK] Fixed string sentinel columns")

    if "protocol" in df.columns:
        le_proto = LabelEncoder()
        df["protocol"] = le_proto.fit_transform(df["protocol"].astype(str))

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    label_cols = ["label", "activity"]
    feat_cols = [c for c in df.columns if c not in label_cols]

    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    nan_before = df[feat_cols].isna().sum().sum()
    df[feat_cols] = df[feat_cols].fillna(0)
    print(f"    [OK] Filled {nan_before:,} NaN/Inf values")

    for col in feat_cols:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype == np.int64:
            df[col] = pd.to_numeric(df[col], downcast="integer")

    print(f"    [OK] Shape: {df.shape}, Features: {len(feat_cols)}")
    return df, feat_cols


def build_windows(X, y, window_size=30, stride=15):
    """Build sliding windows over time-sorted data."""
    print(f"\n[3] Building windows (size={window_size}, stride={stride})...")
    wins, lbls = [], []
    for s in range(0, len(X) - window_size + 1, stride):
        wins.append(X[s : s + window_size])
        lbls.append(int(y[s + window_size - 1]))
    print(f"    [OK] Total windows: {len(lbls):,}")
    return np.array(wins, dtype=np.float32), np.array(lbls, dtype=np.int64)


# ---------------------------------------------------------------------------
# Demo modes
# ---------------------------------------------------------------------------


def run_pipeline():
    """Show full preprocessing pipeline without model inference."""
    print("=" * 60)
    print("  IDS Demo -- Full Preprocessing Pipeline")
    print("=" * 60)

    if not os.path.exists(CSV_PATH):
        print(f"\n[ERROR] Dataset not found: {CSV_PATH}")
        print("    Please ensure 'dataset/merged_CSVs.csv' exists.")
        sys.exit(1)

    print(f"\n[0] Loading raw CSV...")
    raw = pd.read_csv(CSV_PATH, low_memory=False, nrows=50000)
    print(f"    Loaded {len(raw):,} rows × {len(raw.columns)} cols")

    df_clean, feat_cols = clean_dataset(raw)

    le_label = LabelEncoder()
    df_clean["label_enc"] = le_label.fit_transform(df_clean["label"])
    print(f"\n[2] Labels: {list(le_label.classes_)}")

    X_all = df_clean[feat_cols].values.astype(np.float32)
    y_all = df_clean["label_enc"].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X_all).astype(np.float32)
    print(f"    [OK] Scaled to [0, 1]")

    sel_var = VarianceThreshold(threshold=0.001)
    X_sel = sel_var.fit_transform(X_scaled)
    print(f"    [OK] Variance threshold: {X_scaled.shape[1]} -> {X_sel.shape[1]} features")

    wins, lbls = build_windows(X_sel, y_all, WINDOW_SIZE, STRIDE)
    print(f"\n[4] Sample window shape: {wins[0].shape}")
    print(f"    Sample label: {le_label.classes_[lbls[0]]}")
    print("\n[SUCCESS] Pipeline complete. Ready for model inference.")
    print("=" * 60)


def run_inference():
    """Run inference with saved models."""
    print("=" * 60)
    print("  IDS Demo -- Inference Mode")
    print("=" * 60)

    required = [
        f"{MODELS_DIR}/bilstm_ids.pt",
        f"{MODELS_DIR}/cnn1d_ids.pt",
        f"{MODELS_DIR}/temporal_scaler.pkl",
        f"{MODELS_DIR}/variance_selector.pkl",
        f"{MODELS_DIR}/label_encoder.pkl",
        f"{MODELS_DIR}/corr_keep_idx.npy",
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print(f"\n[ERROR] Missing model artifacts:")
        for p in missing:
            print(f"    - {p}")
        print(f"\n   Run the training notebook (ids.ipynb) first,")
        print(f"   or run: python demo.py --mode train")
        sys.exit(1)

    scaler = joblib.load(f"{MODELS_DIR}/temporal_scaler.pkl")
    sel_var = joblib.load(f"{MODELS_DIR}/variance_selector.pkl")
    le_label = joblib.load(f"{MODELS_DIR}/label_encoder.pkl")
    corr_keep = np.load(f"{MODELS_DIR}/corr_keep_idx.npy")

    n_features = len(corr_keep)
    bilstm = BiLSTMClassifier(n_features).to(DEVICE)
    cnn1d = CNN1DClassifier(n_features).to(DEVICE)
    bilstm.load_state_dict(torch.load(f"{MODELS_DIR}/bilstm_ids.pt", map_location=DEVICE))
    cnn1d.load_state_dict(torch.load(f"{MODELS_DIR}/cnn1d_ids.pt", map_location=DEVICE))
    bilstm.eval()
    cnn1d.eval()

    print(f"\n[OK] Models loaded on {DEVICE}")
    print(f"[OK] Input features: {n_features}")
    print(f"[OK] Classes: {list(le_label.classes_)}")

    # Simulate a window for demo
    demo_window = torch.randn(1, WINDOW_SIZE, n_features).to(DEVICE)
    with torch.no_grad():
        p_bilstm = torch.softmax(bilstm(demo_window), dim=1)
        p_cnn = torch.softmax(cnn1d(demo_window), dim=1)
        p_ensemble = (p_bilstm + p_cnn) / 2

    pred = p_ensemble.argmax(1).item()
    conf = p_ensemble[0, pred].item()
    print(f"\n{'='*60}")
    print(f"  Demo Prediction")
    print(f"{'='*60}")
    print(f"  BiLSTM   : {dict(zip(le_label.classes_, [f'{v:.3f}' for v in p_bilstm[0].cpu().numpy()]))}")
    print(f"  1D CNN   : {dict(zip(le_label.classes_, [f'{v:.3f}' for v in p_cnn[0].cpu().numpy()]))}")
    print(f"  Ensemble : {le_label.classes_[pred]} (confidence: {conf:.3f})")
    print(f"{'='*60}")


def run_train_demo():
    """Train tiny models on a subset for demonstration."""
    print("=" * 60)
    print("  IDS Demo -- Train Tiny Demo Models")
    print("=" * 60)
    print("\n  This trains small models on 20,000 rows for demo purposes.")
    print("  For full training, run ids.ipynb.\n")

    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Dataset not found: {CSV_PATH}")
        sys.exit(1)

    raw = pd.read_csv(CSV_PATH, low_memory=False, nrows=20000)
    df_clean, feat_cols = clean_dataset(raw)

    le_label = LabelEncoder()
    df_clean["label_enc"] = le_label.fit_transform(df_clean["label"])
    X_all = df_clean[feat_cols].values.astype(np.float32)
    y_all = df_clean["label_enc"].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X_all).astype(np.float32)

    sel_var = VarianceThreshold(threshold=0.001)
    X_sel = sel_var.fit_transform(X_scaled)

    corr = np.corrcoef(X_sel[:5000].T)
    upper = np.triu(np.abs(corr), k=1)
    drop = set(int(j) for j in np.where(upper > 0.97)[1])
    keep = [i for i in range(X_sel.shape[1]) if i not in drop]
    X_sel = X_sel[:, keep]
    n_features = X_sel.shape[1]

    wins, lbls = build_windows(X_sel, y_all, WINDOW_SIZE, STRIDE)
    split = int(0.8 * len(wins))
    X_tr, y_tr = wins[:split], lbls[:split]
    X_te, y_te = wins[split:], lbls[split:]

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, f"{MODELS_DIR}/temporal_scaler.pkl")
    joblib.dump(sel_var, f"{MODELS_DIR}/variance_selector.pkl")
    joblib.dump(le_label, f"{MODELS_DIR}/label_encoder.pkl")
    np.save(f"{MODELS_DIR}/corr_keep_idx.npy", np.array(keep, dtype=np.int32))

    # Train tiny BiLSTM
    model = BiLSTMClassifier(n_features, hidden_size=64, num_layers=1, dropout=0.3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    X_tr_t = torch.FloatTensor(X_tr)
    y_tr_t = torch.LongTensor(y_tr)
    dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    print("\n[5] Training demo BiLSTM (5 epochs)...")
    for epoch in range(1, 6):
        model.train()
        total_loss = 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
        print(f"    Epoch {epoch} | loss: {total_loss / len(dataset):.4f}")

    torch.save(model.state_dict(), f"{MODELS_DIR}/bilstm_ids.pt")
    print(f"    [OK] Saved -> {MODELS_DIR}/bilstm_ids.pt")

    # Quick eval
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_te).to(DEVICE)).argmax(1).cpu().numpy()
    acc = (preds == y_te).mean()
    print(f"\n[6] Demo test accuracy: {acc:.2%}")
    print("    (Full training yields ~94% -- see ids.ipynb)")
    print("\n[OK] Demo training complete.")
    print(f"   Run 'python demo.py --mode inference' to test.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDS Demo -- DDoS Detection Pipeline")
    parser.add_argument(
        "--mode",
        choices=["pipeline", "inference", "train"],
        default="pipeline",
        help="pipeline: show preprocessing only | inference: run with saved models | train: train demo models",
    )
    args = parser.parse_args()

    t0 = time.time()
    if args.mode == "pipeline":
        run_pipeline()
    elif args.mode == "inference":
        run_inference()
    elif args.mode == "train":
        run_train_demo()
    print(f"\n[TIME]  Total time: {time.time() - t0:.1f}s\n")
