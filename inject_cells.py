"""
Injects Week 5 PoC cells into ids.ipynb.
Run once: python inject_cells.py
"""
import json, uuid

NB = "F:/capstone_ids/ids.ipynb"

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

def cid():  return uuid.uuid4().hex[:12]
def code(s): return {"cell_type":"code","id":cid(),"metadata":{},"outputs":[],"source":s}
def md(s):   return {"cell_type":"markdown","id":cid(),"metadata":{},"source":s}

cells = []

# ── SECTION HEADER ─────────────────────────────────────────────────────────────
cells.append(md(
"---\n"
"# Week 5 PoC — Temporal IDS Pipeline + Models\n\n"
"**Data → Clean → Sort by time → Window → BiLSTM + 1D CNN**\n\n"
"| Step | What happens |\n"
"|---|---|\n"
"| 1. Load | `merged_CSVs.csv` (324 cols, 540k rows) |\n"
"| 2. Clean | Parse timestamp, fix string cols, encode protocol, fill NaN/Inf |\n"
"| 3. Sort | Chronological order by `timestamp` (real temporal order) |\n"
"| 4. Save | `ddos_clean.parquet` (fast reload for future runs) |\n"
"| 5. Window | Sliding window of 20 time-sorted flows → one sequence |\n"
"| 6. Models | Bidirectional LSTM **and** 1D CNN trained and compared |\n"
))

# ── CELL 1: IMPORTS ─────────────────────────────────────────────────────────────
cells.append(code(
"""import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os, joblib, warnings, time
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch  : {torch.__version__}")
print(f"Device   : {device}")
if device.type == 'cuda':
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
print("\\nAll imports OK")"""
))

# ── CELL 2: LOAD + INSPECT ──────────────────────────────────────────────────────
cells.append(md("## Step 1 — Load & Inspect\n\nLoad the raw CSV and understand what needs cleaning."))

cells.append(code(
"""CSV_PATH    = 'dataset/merged_CSVs.csv'
CLEAN_PATH  = 'ddos_clean.parquet'

# ── columns we will NOT use as model features ──────────────────────────────────
DROP_COLS = ['flow_id', 'src_ip', 'dst_ip', 'timestamp']
# These are identifiers or will be used for sorting only.
# Keeping IPs would cause the model to memorise addresses, not traffic patterns.

# ── string columns that need special parsing ───────────────────────────────────
# delta_start / handshake_duration contain "not a complete handshake"
STRING_SENTINEL = 'not a complete handshake'
PARSE_COLS = ['delta_start', 'handshake_duration']

print(f"Loading {CSV_PATH} ...")
t0 = time.time()
raw = pd.read_csv(CSV_PATH, low_memory=False)
print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  Shape  : {raw.shape}")
print(f"  Memory : {raw.memory_usage(deep=True).sum()/1e6:.0f} MB")

print(f"\\nTimestamp range:")
print(f"  min: {raw['timestamp'].min()}")
print(f"  max: {raw['timestamp'].max()}")

print(f"\\nLabel distribution:")
print(raw['label'].value_counts())

print(f"\\nProblem column preview (delta_start):")
print(raw['delta_start'].value_counts().head())"""
))

# ── CELL 3: CLEANING ────────────────────────────────────────────────────────────
cells.append(md("## Step 2 — Clean\n\n- Parse timestamp for sorting\n- Fix string columns\n- Encode `protocol`\n- Fill NaN / Inf\n- Drop identifier columns"))

cells.append(code(
"""def clean_dataset(df):
    print("Cleaning dataset...")

    # 1. Parse timestamp → sort by it → drop it from features
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"  ✓ Sorted chronologically by timestamp")

    # 2. Fix string sentinel columns: "not a complete handshake" → NaN → 0
    for col in PARSE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].replace(STRING_SENTINEL, np.nan), errors='coerce'
            )
    print(f"  ✓ Converted {PARSE_COLS} to numeric (sentinel → NaN → 0)")

    # 3. Encode protocol (object → integer)
    if 'protocol' in df.columns:
        le_proto = LabelEncoder()
        df['protocol'] = le_proto.fit_transform(df['protocol'].astype(str))
        print(f"  ✓ Encoded protocol: {dict(zip(le_proto.classes_, le_proto.transform(le_proto.classes_)))}")

    # 4. Drop identifier / non-feature columns (keep labels)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"  ✓ Dropped {cols_to_drop}")

    # 5. Separate feature columns from label columns
    label_cols = ['label', 'activity']
    feat_cols  = [c for c in df.columns if c not in label_cols]

    # 6. Replace Inf with NaN, then fill all NaN with 0
    #    (NaN arises from skewness/cov on single-value distributions, and from the
    #    string-sentinel columns; 0 is safe because all features are mean/std/count
    #    type statistics where 0 = "no data / undefined".)
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    nan_before = df[feat_cols].isna().sum().sum()
    df[feat_cols] = df[feat_cols].fillna(0)
    print(f"  ✓ Filled {nan_before:,} NaN/Inf values with 0")

    # 7. Downcast numeric types to save memory
    for col in feat_cols:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype == np.int64:
            df[col] = pd.to_numeric(df[col], downcast='integer')

    print(f"  ✓ Downcast to float32/int where possible")
    print(f"\\nCleaned shape : {df.shape}")
    print(f"Feature cols  : {len(feat_cols)}")
    print(f"Memory now    : {df.memory_usage(deep=True).sum()/1e6:.0f} MB")
    return df, feat_cols

df_clean, feature_cols = clean_dataset(raw)"""
))

# ── CELL 4: ENCODE LABELS + SAVE ────────────────────────────────────────────────
cells.append(code(
"""# Encode labels
le_label    = LabelEncoder()
le_activity = LabelEncoder()
df_clean['label_enc']    = le_label.fit_transform(df_clean['label'])
df_clean['activity_enc'] = le_activity.fit_transform(df_clean['activity'])

print("Label encoding:")
for cls, enc in zip(le_label.classes_, range(len(le_label.classes_))):
    count = (df_clean['label'] == cls).sum()
    print(f"  {enc} = {cls:<15}  ({count:,} rows)")

# Save cleaned + sorted parquet for fast future reloads
df_clean.to_parquet(CLEAN_PATH, index=False)
print(f"\\n✓ Saved cleaned data → {CLEAN_PATH}")
print(f"  Shape : {df_clean.shape}")"""
))

# ── FEATURE / LABEL SPLIT ───────────────────────────────────────────────────────
cells.append(code(
"""# Extract arrays
X_all = df_clean[feature_cols].values.astype(np.float32)  # (N, F)
y_all = df_clean['label_enc'].values                        # (N,) – 3-class

N_FEATURES = X_all.shape[1]
N_CLASSES  = len(le_label.classes_)

print(f"Feature matrix : {X_all.shape}")
print(f"Labels         : {y_all.shape}  — classes: {le_label.classes_}")
print(f"\\nClass counts after time-sort:")
for cls, enc in zip(le_label.classes_, range(N_CLASSES)):
    print(f"  {enc} {cls}: {(y_all==enc).sum():,}")"""
))

# ── CELL 5: SCALE + SPLIT ───────────────────────────────────────────────────────
cells.append(md("## Step 3 — Scale & Split (no data leakage)\n\n"
"Split **before** scaling so the scaler is fit only on training rows.  \n"
"Split **before** windowing so windows never bridge train/val/test boundaries."))

cells.append(code(
"""# ── temporal split (no shuffle — preserve time order) ─────────────────────────
n = len(X_all)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_tr_raw, y_tr = X_all[:train_end],        y_all[:train_end]
X_va_raw, y_va = X_all[train_end:val_end], y_all[train_end:val_end]
X_te_raw, y_te = X_all[val_end:],          y_all[val_end:]

# ── fit scaler on TRAIN only ───────────────────────────────────────────────────
scaler = MinMaxScaler(feature_range=(0, 1))
X_tr = scaler.fit_transform(X_tr_raw).astype(np.float32)
X_va = scaler.transform(X_va_raw).astype(np.float32)
X_te = scaler.transform(X_te_raw).astype(np.float32)

# Save scaler for inference
os.makedirs('models', exist_ok=True)
joblib.dump(scaler,     'models/temporal_scaler.pkl')
joblib.dump(le_label,   'models/label_encoder.pkl')
joblib.dump(le_activity,'models/activity_encoder.pkl')

print(f"Train rows : {len(X_tr):,}  ({100*len(X_tr)/n:.0f}%)")
print(f"Val   rows : {len(X_va):,}  ({100*len(X_va)/n:.0f}%)")
print(f"Test  rows : {len(X_te):,}  ({100*len(X_te)/n:.0f}%)")
print(f"\\nFeature range after scaling: [{X_tr.min():.2f}, {X_tr.max():.2f}]")"""
))

# ── CELL 6: WINDOWING ────────────────────────────────────────────────────────────
cells.append(md("## Step 4 — Temporal Windowing\n\n"
"Each sample = **W consecutive flows in time order** = one sequence for the model.\n\n"
"- `WINDOW_SIZE = 20` → each sequence covers ~20 chronologically adjacent flows  \n"
"- `STRIDE = 10` → 50% overlap between consecutive windows  \n"
"- Label = **majority vote** across all flows in the window"))

cells.append(code(
"""WINDOW_SIZE = 20   # flows per sequence
STRIDE      = 10   # step between windows (50% overlap)
BATCH_SIZE  = 256

class FlowWindowDataset(Dataset):
    \"\"\"
    Wraps a (N, F) scaled feature matrix into overlapping fixed-length windows.
    Each window is shape (WINDOW_SIZE, N_FEATURES) — ready for LSTM / 1D CNN.
    Label = majority class in the window.
    \"\"\"
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 window_size: int = 20, stride: int = 10):
        assert len(X) == len(y)
        windows, labels = [], []
        for s in range(0, len(X) - window_size + 1, stride):
            windows.append(X[s : s + window_size])
            labels.append(int(np.bincount(y[s : s + window_size]).argmax()))

        # (N_windows, window_size, N_features)
        self.X = torch.from_numpy(np.array(windows, dtype=np.float32))
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_ds = FlowWindowDataset(X_tr, y_tr, WINDOW_SIZE, STRIDE)
val_ds   = FlowWindowDataset(X_va, y_va, WINDOW_SIZE, STRIDE)
test_ds  = FlowWindowDataset(X_te, y_te, WINDOW_SIZE, STRIDE)

kw = dict(pin_memory=(device.type=='cuda'), num_workers=0)
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  **kw)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, **kw)
test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, **kw)

print(f"Window size     : {WINDOW_SIZE} flows")
print(f"Stride          : {STRIDE}")
print(f"Input tensor    : (batch, {WINDOW_SIZE}, {N_FEATURES})")
print(f"\\nWindows — train : {len(train_ds):,}  | val : {len(val_ds):,}  | test : {len(test_ds):,}")"""
))

# ── MODEL A: BiLSTM ─────────────────────────────────────────────────────────────
cells.append(md("## Model A — Bidirectional LSTM\n\n"
"Reads the 20-flow window **forward and backward**, capturing long-range temporal dependencies.\n\n"
"```\n"
"Input (B, 20, F) → BiLSTM layers → last hidden state (B, H×2) → classifier head → (B, 3)\n"
"```"))

cells.append(code(
"""class BiLSTMClassifier(nn.Module):
    \"\"\"
    Bidirectional LSTM for temporal flow-sequence classification.
    Input  : (batch, seq_len, n_features)
    Output : (batch, n_classes)
    \"\"\"
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 num_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
        )
        feat = hidden_size * 2          # ×2 for bidirectional
        self.head = nn.Sequential(
            nn.LayerNorm(feat),
            nn.Linear(feat, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)           # (B, T, H×2)
        return self.head(out[:, -1, :]) # classify from last timestep

lstm_model = BiLSTMClassifier(
    input_size  = N_FEATURES,
    hidden_size = 128,
    num_layers  = 2,
    num_classes = N_CLASSES,
    dropout     = 0.3,
).to(device)

n_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
print(f"BiLSTM — {n_params:,} trainable parameters")
print(lstm_model)"""
))

# ── MODEL B: 1D CNN ──────────────────────────────────────────────────────────────
cells.append(md("## Model B — 1D CNN\n\n"
"Treats each feature as a **channel** and convolves across the 20 time steps,  \n"
"detecting local temporal patterns (e.g., repeated SYN floods in consecutive flows).\n\n"
"```\n"
"Input (B, 20, F) → transpose → (B, F, 20) → Conv1d blocks → GlobalAvgPool → classifier\n"
"```"))

cells.append(code(
"""class CNN1DClassifier(nn.Module):
    \"\"\"
    1D CNN for temporal flow-sequence classification.
    Input  : (batch, seq_len, n_features)
    Output : (batch, n_classes)
    \"\"\"
    def __init__(self, input_size, num_classes=3, dropout=0.3):
        super().__init__()
        # Conv1d expects (B, C, L): features=channels, time=length
        self.conv = nn.Sequential(
            # Block 1 — local pattern detection
            nn.Conv1d(input_size, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256,        256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(dropout * 0.5),

            # Block 2 — higher-level patterns
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(dropout * 0.5),

            # Global average pool over time → (B, 512, 1)
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),               # (B, 512)
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)          # (B, T, F) → (B, F, T)
        return self.head(self.conv(x))

cnn1d_model = CNN1DClassifier(
    input_size  = N_FEATURES,
    num_classes = N_CLASSES,
    dropout     = 0.3,
).to(device)

n_params = sum(p.numel() for p in cnn1d_model.parameters() if p.requires_grad)
print(f"1D CNN — {n_params:,} trainable parameters")
print(cnn1d_model)"""
))

# ── TRAINING UTILITIES ──────────────────────────────────────────────────────────
cells.append(md("## Training Utilities"))

cells.append(code(
"""def _make_criterion(dataset, n_classes):
    \"\"\"Class-weighted CrossEntropyLoss to handle imbalanced labels.\"\"\"
    y_np = dataset.y.numpy()
    w    = compute_class_weight('balanced', classes=np.arange(n_classes), y=y_np)
    return nn.CrossEntropyLoss(weight=torch.FloatTensor(w).to(device))

def _train_epoch(model, loader, optimizer, criterion):
    model.train()
    loss_sum = correct = total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_sum += loss.item() * len(y)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += len(y)
    return loss_sum / total, correct / total

@torch.no_grad()
def _eval(model, loader, criterion):
    model.eval()
    loss_sum = correct = total = 0
    all_p, all_t = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits    = model(X)
        loss_sum += criterion(logits, y).item() * len(y)
        preds     = logits.argmax(1)
        correct  += (preds == y).sum().item()
        total    += len(y)
        all_p.extend(preds.cpu().numpy())
        all_t.extend(y.cpu().numpy())
    return loss_sum / total, correct / total, np.array(all_p), np.array(all_t)

def train_model(model, name, train_loader, val_loader,
                n_classes, epochs=25, lr=3e-4, patience=6):
    criterion = _make_criterion(train_loader.dataset, n_classes)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    best_val, best_state, no_improve = float('inf'), None, 0
    history = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[])

    print(f"\\n{'='*58}")
    print(f"  Training : {name}")
    print(f"{'='*58}")

    for ep in range(1, epochs + 1):
        tr_l, tr_a       = _train_epoch(model, train_loader, optimizer, criterion)
        va_l, va_a, _, _ = _eval(model, val_loader, criterion)
        scheduler.step(va_l)

        history['train_loss'].append(tr_l); history['val_loss'].append(va_l)
        history['train_acc'].append(tr_a);  history['val_acc'].append(va_a)

        is_best = va_l < best_val
        if is_best:
            best_val   = va_l
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        tag = " ←" if is_best else ""
        if ep % 5 == 0 or is_best or no_improve >= patience:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  ep {ep:3d} | train {tr_l:.4f}/{tr_a:.3f}"
                  f" | val {va_l:.4f}/{va_a:.3f} | lr={lr_now:.2e}{tag}")

        if no_improve >= patience:
            print(f"  Early stop at epoch {ep}")
            break

    model.load_state_dict(best_state)
    print(f"  Best val loss: {best_val:.4f}")
    return history

print("Training utilities ready.")"""
))

# ── TRAIN BiLSTM ────────────────────────────────────────────────────────────────
cells.append(md("## Train Model A — BiLSTM"))

cells.append(code(
"""lstm_history = train_model(
    lstm_model, "Bidirectional LSTM",
    train_loader, val_loader,
    n_classes = N_CLASSES,
    epochs    = 25,
    lr        = 3e-4,
    patience  = 6,
)"""
))

# ── TRAIN 1D CNN ─────────────────────────────────────────────────────────────────
cells.append(md("## Train Model B — 1D CNN"))

cells.append(code(
"""cnn1d_history = train_model(
    cnn1d_model, "1D CNN",
    train_loader, val_loader,
    n_classes = N_CLASSES,
    epochs    = 25,
    lr        = 3e-4,
    patience  = 6,
)"""
))

# ── EVALUATE ─────────────────────────────────────────────────────────────────────
cells.append(md("## Evaluation — Test Set"))

cells.append(code(
"""def full_eval(model, loader, name):
    \"\"\"Run inference on test set, print report, return (acc, confusion_matrix).\"\"\"
    criterion = nn.CrossEntropyLoss()       # unweighted for fair comparison
    _, acc, preds, targets = _eval(model, loader, criterion)
    print(f"\\n{'='*58}")
    print(f"  {name}")
    print(f"{'='*58}")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print()
    print(classification_report(targets, preds,
                                target_names=le_label.classes_,
                                digits=4))
    return acc, confusion_matrix(targets, preds)

lstm_acc, lstm_cm = full_eval(lstm_model,  test_loader, "Bidirectional LSTM — Test")
cnn_acc,  cnn_cm  = full_eval(cnn1d_model, test_loader, "1D CNN — Test")"""
))

# ── VISUALISE ─────────────────────────────────────────────────────────────────────
cells.append(code(
"""fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    f"Week 5 PoC — BiLSTM vs 1D CNN  "
    f"| window={WINDOW_SIZE} flows, stride={STRIDE} | {N_FEATURES} features",
    fontsize=13, fontweight='bold')

for row, (hist, cm, acc, name) in enumerate([
    (lstm_history,  lstm_cm,  lstm_acc,  "BiLSTM"),
    (cnn1d_history, cnn_cm,   cnn_acc,   "1D CNN"),
]):
    # Loss
    ax = axes[row, 0]
    ax.plot(hist['train_loss'], lw=2, label='Train')
    ax.plot(hist['val_loss'],   lw=2, label='Val',   linestyle='--')
    ax.set_title(f"{name} — Loss");  ax.set_xlabel("Epoch")
    ax.legend(); ax.grid(alpha=0.3)

    # Accuracy
    ax = axes[row, 1]
    ax.plot(hist['train_acc'], lw=2, label='Train')
    ax.plot(hist['val_acc'],   lw=2, label='Val',   linestyle='--')
    ax.set_title(f"{name} — Accuracy"); ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)

    # Confusion matrix
    ax = axes[row, 2]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=le_label.classes_,
                yticklabels=le_label.classes_)
    ax.set_title(f"{name} — test acc={acc:.3f}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig('temporal_comparison.png', dpi=120, bbox_inches='tight')
plt.show()
print("Saved: temporal_comparison.png")"""
))

# ── SAVE + SUMMARY ───────────────────────────────────────────────────────────────
cells.append(code(
"""torch.save(lstm_model.state_dict(),  'models/bilstm_ids.pt')
torch.save(cnn1d_model.state_dict(), 'models/cnn1d_ids.pt')

print("=" * 62)
print("WEEK 5 POC COMPLETE")
print("=" * 62)
print(f"\\nDataset   : BCCC-Cloud-DDoS-2024 (merged_CSVs.csv)")
print(f"Features  : {N_FEATURES}  (all numeric after cleaning)")
print(f"Ordering  : chronological by timestamp (real temporal order)")
print(f"\\nTemporal window:")
print(f"  Size    : {WINDOW_SIZE} consecutive flows")
print(f"  Stride  : {STRIDE}  (50% overlap)")
print(f"  Tensor  : (batch, {WINDOW_SIZE}, {N_FEATURES})")
print(f"\\nWindows — train:{len(train_ds):,}  val:{len(val_ds):,}  test:{len(test_ds):,}")
print(f"\\nModel A  Bidirectional LSTM")
print(f"  Params  : {sum(p.numel() for p in lstm_model.parameters()):,}")
print(f"  Test    : {lstm_acc:.4f}")
print(f"\\nModel B  1D CNN")
print(f"  Params  : {sum(p.numel() for p in cnn1d_model.parameters()):,}")
print(f"  Test    : {cnn_acc:.4f}")
print(f"\\nSaved:")
print(f"  ddos_clean.parquet         (cleaned, time-sorted dataset)")
print(f"  models/bilstm_ids.pt")
print(f"  models/cnn1d_ids.pt")
print(f"  models/temporal_scaler.pkl")
print("=" * 62)"""
))

# ── INJECT ───────────────────────────────────────────────────────────────────────
nb["cells"].extend(cells)

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✓ Injected {len(cells)} cells into {NB}")
print(f"  Total cells now: {len(nb['cells'])}")
