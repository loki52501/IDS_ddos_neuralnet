"""
fix_overfit.py -- Patch ids.ipynb to fix class imbalance + overfitting.

Root cause: Suspicious flows are concentrated at the END of the time-sorted
dataset, so a simple 70/15/15 temporal split leaves 0 % Suspicious in training.

Fix strategy:
  1. Fit scaler + feature-selectors on training-time rows (no leakage).
  2. Scale / select the FULL dataset.
  3. Create ALL windows over the full time-sorted dataset.
  4. Stratified split at the window level -- every split gets all 3 classes.
  5. Stronger regularisation: dropout 0.3->0.4, weight_decay->1e-3,
     label_smoothing=0.1, CosineAnnealingLR, BatchNorm in BiLSTM proj.
"""

import json, sys

NB_PATH = 'ids.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def set_cell(cell_id, src):
    lines  = src.lstrip('\n').split('\n')
    source = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] != '' else [])
    for cell in nb['cells']:
        if cell.get('id') == cell_id:
            cell['source']          = source
            cell['outputs']         = []
            cell['execution_count'] = None
            print(f'[OK] Cell {cell_id}')
            return
    print(f'[WARN] Cell {cell_id} not found', file=sys.stderr)


# =============================================================================
# Cell sources  (all docstrings use single-quote triple to avoid nesting issues)
# =============================================================================

# --- 1. Scale & Split --------------------------------------------------------
SRC_SCALE = (
    "# -- Fit scaler on first 70 % (train-time rows) -- no data leakage ------\n"
    "# We scale ALL rows so windows can be created over the full dataset.\n"
    "# Stratified window-level split (next cell) ensures class balance.\n"
    "n          = len(X_all)\n"
    "scaler_end = int(n * 0.70)   # row index where training time ends\n"
    "\n"
    "scaler = MinMaxScaler(feature_range=(0, 1))\n"
    "scaler.fit(X_all[:scaler_end])           # fit on training-time rows only\n"
    "X_scaled = scaler.transform(X_all).astype(np.float32)\n"
    "\n"
    "os.makedirs('models', exist_ok=True)\n"
    "joblib.dump(scaler,      'models/temporal_scaler.pkl')\n"
    "joblib.dump(le_label,    'models/label_encoder.pkl')\n"
    "joblib.dump(le_activity, 'models/activity_encoder.pkl')\n"
    "\n"
    "print(f'Scaler fit on first {scaler_end:,} rows  ({100*scaler_end/n:.0f} % of data)')\n"
    "print(f'Full scaled matrix : {X_scaled.shape}')\n"
    "print(f'Value range        : [{X_scaled.min():.2f}, {X_scaled.max():.2f}]')\n"
)

# --- 2. Feature selection ----------------------------------------------------
SRC_FEATSEL = (
    "from sklearn.feature_selection import VarianceThreshold\n"
    "\n"
    "print(f'Feature selection  (start: {X_scaled.shape[1]} features)')\n"
    "\n"
    "# VarianceThreshold fitted on training-time rows only\n"
    "sel_var = VarianceThreshold(threshold=0.001)\n"
    "sel_var.fit(X_scaled[:scaler_end])\n"
    "X_sel = sel_var.transform(X_scaled).astype(np.float32)\n"
    "print(f'  After variance threshold  : {X_sel.shape[1]} features')\n"
    "\n"
    "# Correlation pruning -- sample from training-time rows\n"
    "rng    = np.random.default_rng(42)\n"
    "sample = X_sel[:scaler_end][rng.choice(scaler_end, min(10_000, scaler_end), replace=False)]\n"
    "corr   = np.corrcoef(sample.T)\n"
    "upper  = np.triu(np.abs(corr), k=1)\n"
    "drop   = set(int(j) for j in np.where(upper > 0.97)[1])\n"
    "keep   = [i for i in range(X_sel.shape[1]) if i not in drop]\n"
    "\n"
    "X_sel = X_sel[:, keep]\n"
    "print(f'  After correlation pruning : {X_sel.shape[1]} features  (removed {len(drop)} redundant)')\n"
    "\n"
    "N_FEATURES = X_sel.shape[1]\n"
    "print(f'\\n  Final feature count : {N_FEATURES}  (was {X_all.shape[1]})')\n"
    "\n"
    "joblib.dump(sel_var, 'models/variance_selector.pkl')\n"
    "np.save('models/corr_keep_idx.npy', np.array(keep, dtype=np.int32))\n"
    "print('  Saved selectors -> models/')\n"
)

# --- 3. Windowing + stratified split -----------------------------------------
SRC_WINDOW = (
    "from collections import Counter\n"
    "from sklearn.model_selection import train_test_split as _tts\n"
    "\n"
    "WINDOW_SIZE = 30   # consecutive flows per sequence\n"
    "STRIDE      = 15   # 50 % overlap\n"
    "BATCH_SIZE  = 256\n"
    "\n"
    "class FlowWindowDataset(Dataset):\n"
    "    '''Wraps pre-built window arrays into a PyTorch Dataset.\n"
    "    Each window = WINDOW_SIZE time-consecutive flows (30 x N_FEATURES).\n"
    "    Built over the full time-sorted dataset; the stratified split ensures\n"
    "    every class (including Suspicious) is in all splits.\n"
    "    '''\n"
    "    def __init__(self, windows: np.ndarray, labels: np.ndarray):\n"
    "        self.X = torch.from_numpy(np.array(windows, dtype=np.float32))\n"
    "        self.y = torch.tensor(labels, dtype=torch.long)\n"
    "    def __len__(self):        return len(self.y)\n"
    "    def __getitem__(self, i): return self.X[i], self.y[i]\n"
    "\n"
    "\n"
    "# Step 1 -- build ALL windows over the full time-sorted dataset\n"
    "print('Building windows over full time-sorted dataset ...')\n"
    "all_wins, all_lbls = [], []\n"
    "for s in range(0, len(X_sel) - WINDOW_SIZE + 1, STRIDE):\n"
    "    all_wins.append(X_sel[s : s + WINDOW_SIZE])\n"
    "    all_lbls.append(int(y_all[s + WINDOW_SIZE - 1]))   # label = last flow\n"
    "\n"
    "all_wins = np.array(all_wins, dtype=np.float32)\n"
    "all_lbls = np.array(all_lbls, dtype=np.int64)\n"
    "\n"
    "print(f'\\nTotal windows : {len(all_lbls):,}')\n"
    "for ci in range(N_CLASSES):\n"
    "    cnt = (all_lbls == ci).sum()\n"
    "    print(f'  class {ci} ({le_label.classes_[ci]}): {cnt:,}  ({100*cnt/len(all_lbls):.1f} %)')\n"
    "\n"
    "# Step 2 -- stratified split at the WINDOW level\n"
    "# Suspicious flows are at the END of the time-sorted data.\n"
    "# Plain 70/15/15 temporal cut puts 0 Suspicious in training.\n"
    "# Stratifying by window label fixes this -- all 3 classes in every split.\n"
    "idx = np.arange(len(all_lbls))\n"
    "tr_idx, tmp_idx = _tts(idx, test_size=0.30, stratify=all_lbls, random_state=42)\n"
    "va_idx, te_idx  = _tts(tmp_idx, test_size=0.50, stratify=all_lbls[tmp_idx], random_state=42)\n"
    "\n"
    "train_ds = FlowWindowDataset(all_wins[tr_idx], all_lbls[tr_idx])\n"
    "val_ds   = FlowWindowDataset(all_wins[va_idx], all_lbls[va_idx])\n"
    "test_ds  = FlowWindowDataset(all_wins[te_idx], all_lbls[te_idx])\n"
    "\n"
    "kw = dict(pin_memory=(device.type == 'cuda'), num_workers=0)\n"
    "train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  **kw)\n"
    "val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, **kw)\n"
    "test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, **kw)\n"
    "\n"
    "print(f'\\nStratified split (70 / 15 / 15):')\n"
    "for split_name, ds in [('train', train_ds), ('val', val_ds), ('test', test_ds)]:\n"
    "    dist  = Counter(ds.y.tolist())\n"
    "    parts = ',  '.join(f'{le_label.classes_[k]}: {dist.get(k, 0):,}' for k in range(N_CLASSES))\n"
    "    print(f'  {split_name} ({len(ds):,}): {parts}')\n"
)

# --- 4. BiLSTM model ---------------------------------------------------------
SRC_BILSTM = (
    "import torch.nn.functional as F\n"
    "\n"
    "class BiLSTMClassifier(nn.Module):\n"
    "    '''Bidirectional LSTM + soft attention.\n"
    "    Anti-overfitting: BatchNorm1d after input projection, dropout=0.4,\n"
    "    simplified 2-layer head (256->64->n_classes).\n"
    "    '''\n"
    "    def __init__(self, input_size, hidden_size=128, num_layers=2,\n"
    "                 num_classes=3, dropout=0.4):\n"
    "        super().__init__()\n"
    "        # Input projection -- BatchNorm1d requires (N, C) so we reshape in forward\n"
    "        self.proj_linear = nn.Linear(input_size, 128)\n"
    "        self.proj_bn     = nn.BatchNorm1d(128)\n"
    "        self.proj_drop   = nn.Dropout(dropout * 0.5)\n"
    "\n"
    "        self.lstm = nn.LSTM(\n"
    "            input_size    = 128,\n"
    "            hidden_size   = hidden_size,\n"
    "            num_layers    = num_layers,\n"
    "            batch_first   = True,\n"
    "            dropout       = dropout if num_layers > 1 else 0.0,\n"
    "            bidirectional = True,\n"
    "        )\n"
    "        feat = hidden_size * 2          # 256 for bidirectional\n"
    "        self.attn_score = nn.Linear(feat, 1)\n"
    "\n"
    "        self.head = nn.Sequential(\n"
    "            nn.LayerNorm(feat),\n"
    "            nn.Linear(feat, 64),\n"
    "            nn.ReLU(),\n"
    "            nn.Dropout(dropout),\n"
    "            nn.Linear(64, num_classes),\n"
    "        )\n"
    "\n"
    "    def forward(self, x):\n"
    "        B, T, n_feat = x.shape\n"
    "        # Flatten time steps to run BatchNorm1d, then restore sequence\n"
    "        x_flat = self.proj_linear(x.reshape(B * T, n_feat))  # (B*T, 128)\n"
    "        x_flat = self.proj_bn(x_flat)\n"
    "        x_flat = F.relu(x_flat)\n"
    "        x_flat = self.proj_drop(x_flat)\n"
    "        x = x_flat.reshape(B, T, 128)         # (B, T, 128)\n"
    "\n"
    "        out, _  = self.lstm(x)                 # (B, T, feat)\n"
    "        scores  = self.attn_score(out)         # (B, T, 1)\n"
    "        weights = torch.softmax(scores, dim=1) # sum over T = 1\n"
    "        context = (weights * out).sum(dim=1)   # (B, feat)\n"
    "        return self.head(context)\n"
    "\n"
    "lstm_model = BiLSTMClassifier(\n"
    "    input_size  = N_FEATURES,\n"
    "    hidden_size = 128,\n"
    "    num_layers  = 2,\n"
    "    num_classes = N_CLASSES,\n"
    "    dropout     = 0.4,\n"
    ").to(device)\n"
    "\n"
    "n_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)\n"
    "print(f'BiLSTM + Attention -- {n_params:,} trainable parameters')\n"
)

# --- 5. CNN1D model ----------------------------------------------------------
SRC_CNN = (
    "class CNN1DClassifier(nn.Module):\n"
    "    '''1D Residual CNN.\n"
    "    Anti-overfitting: Dropout(0.2) inside residual blocks, head dropout=0.4.\n"
    "    '''\n"
    "    def __init__(self, input_size, num_classes=3, dropout=0.4):\n"
    "        super().__init__()\n"
    "        self.input_proj = nn.Sequential(\n"
    "            nn.Conv1d(input_size, 128, kernel_size=1),\n"
    "            nn.BatchNorm1d(128), nn.ReLU(),\n"
    "        )\n"
    "        # Residual block 1 -- with intra-block dropout\n"
    "        self.block1 = nn.Sequential(\n"
    "            nn.Conv1d(128, 128, kernel_size=3, padding=1),\n"
    "            nn.BatchNorm1d(128), nn.ReLU(),\n"
    "            nn.Dropout(dropout * 0.5),\n"
    "            nn.Conv1d(128, 128, kernel_size=3, padding=1),\n"
    "            nn.BatchNorm1d(128),\n"
    "        )\n"
    "        # Residual block 2\n"
    "        self.block2 = nn.Sequential(\n"
    "            nn.Conv1d(128, 256, kernel_size=3, padding=1),\n"
    "            nn.BatchNorm1d(256), nn.ReLU(),\n"
    "            nn.Dropout(dropout * 0.5),\n"
    "            nn.Conv1d(256, 256, kernel_size=3, padding=1),\n"
    "            nn.BatchNorm1d(256),\n"
    "        )\n"
    "        self.proj2 = nn.Conv1d(128, 256, kernel_size=1)\n"
    "        self.relu  = nn.ReLU()\n"
    "        self.pool  = nn.AdaptiveAvgPool1d(1)\n"
    "\n"
    "        self.head = nn.Sequential(\n"
    "            nn.Flatten(),\n"
    "            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),\n"
    "            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(dropout * 0.5),\n"
    "            nn.Linear(64, num_classes),\n"
    "        )\n"
    "\n"
    "    def forward(self, x):\n"
    "        x = x.transpose(1, 2)                         # (B, F, T)\n"
    "        x = self.input_proj(x)                        # (B, 128, T)\n"
    "        x = self.relu(self.block1(x) + x)             # residual block 1\n"
    "        x = self.relu(self.block2(x) + self.proj2(x)) # residual block 2\n"
    "        x = self.pool(x)                               # (B, 256, 1)\n"
    "        return self.head(x)\n"
    "\n"
    "cnn1d_model = CNN1DClassifier(\n"
    "    input_size  = N_FEATURES,\n"
    "    num_classes = N_CLASSES,\n"
    "    dropout     = 0.4,\n"
    ").to(device)\n"
    "\n"
    "n_params = sum(p.numel() for p in cnn1d_model.parameters() if p.requires_grad)\n"
    "print(f'1D CNN (residual) -- {n_params:,} trainable parameters')\n"
)

# --- 6. Training utilities ---------------------------------------------------
SRC_TRAIN_UTILS = (
    "def _make_criterion(dataset, n_classes, label_smoothing=0.1):\n"
    "    '''Class-weighted CrossEntropyLoss + label smoothing.\n"
    "    Weights handle imbalance; smoothing prevents overconfident predictions.\n"
    "    '''\n"
    "    y_np      = dataset.y.numpy()\n"
    "    present   = np.unique(y_np)\n"
    "    w_present = compute_class_weight('balanced', classes=present, y=y_np)\n"
    "    w         = np.ones(n_classes, dtype=np.float32)\n"
    "    w[present] = w_present\n"
    "    return nn.CrossEntropyLoss(\n"
    "        weight          = torch.FloatTensor(w).to(device),\n"
    "        label_smoothing = label_smoothing,\n"
    "    )\n"
    "\n"
    "def _train_epoch(model, loader, optimizer, criterion):\n"
    "    model.train()\n"
    "    loss_sum = correct = total = 0\n"
    "    for X, y in loader:\n"
    "        X, y = X.to(device), y.to(device)\n"
    "        optimizer.zero_grad()\n"
    "        logits = model(X)\n"
    "        loss   = criterion(logits, y)\n"
    "        loss.backward()\n"
    "        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n"
    "        optimizer.step()\n"
    "        loss_sum += loss.item() * len(y)\n"
    "        correct  += (logits.argmax(1) == y).sum().item()\n"
    "        total    += len(y)\n"
    "    return loss_sum / total, correct / total\n"
    "\n"
    "@torch.no_grad()\n"
    "def _eval(model, loader, criterion):\n"
    "    model.eval()\n"
    "    loss_sum = correct = total = 0\n"
    "    all_p, all_t = [], []\n"
    "    for X, y in loader:\n"
    "        X, y = X.to(device), y.to(device)\n"
    "        logits    = model(X)\n"
    "        loss_sum += criterion(logits, y).item() * len(y)\n"
    "        preds     = logits.argmax(1)\n"
    "        correct  += (preds == y).sum().item()\n"
    "        total    += len(y)\n"
    "        all_p.extend(preds.cpu().numpy())\n"
    "        all_t.extend(y.cpu().numpy())\n"
    "    return loss_sum / total, correct / total, np.array(all_p), np.array(all_t)\n"
    "\n"
    "def train_model(model, name, train_loader, val_loader,\n"
    "                n_classes, epochs=30, lr=3e-4, patience=7):\n"
    "    criterion = _make_criterion(train_loader.dataset, n_classes)\n"
    "    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)\n"
    "    scheduler = optim.lr_scheduler.CosineAnnealingLR(\n"
    "        optimizer, T_max=epochs, eta_min=1e-6)\n"
    "\n"
    "    best_val, best_state, no_improve = float('inf'), None, 0\n"
    "    history = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[])\n"
    "\n"
    "    print(f\"\\n{'='*58}\")\n"
    "    print(f'  Training : {name}')\n"
    "    print(f\"{'='*58}\")\n"
    "\n"
    "    for ep in range(1, epochs + 1):\n"
    "        tr_l, tr_a       = _train_epoch(model, train_loader, optimizer, criterion)\n"
    "        va_l, va_a, _, _ = _eval(model, val_loader, criterion)\n"
    "        scheduler.step()\n"
    "\n"
    "        history['train_loss'].append(tr_l); history['val_loss'].append(va_l)\n"
    "        history['train_acc'].append(tr_a);  history['val_acc'].append(va_a)\n"
    "\n"
    "        is_best = va_l < best_val\n"
    "        if is_best:\n"
    "            best_val   = va_l\n"
    "            best_state = {k: v.clone() for k, v in model.state_dict().items()}\n"
    "            no_improve = 0\n"
    "        else:\n"
    "            no_improve += 1\n"
    "\n"
    "        tag = ' <-' if is_best else ''\n"
    "        if ep % 5 == 0 or is_best or no_improve >= patience:\n"
    "            lr_now = optimizer.param_groups[0]['lr']\n"
    "            print(f'  ep {ep:3d} | train {tr_l:.4f}/{tr_a:.3f}'\n"
    "                  f' | val {va_l:.4f}/{va_a:.3f} | lr={lr_now:.2e}{tag}')\n"
    "\n"
    "        if no_improve >= patience:\n"
    "            print(f'  Early stop at epoch {ep}')\n"
    "            break\n"
    "\n"
    "    model.load_state_dict(best_state)\n"
    "    print(f'  Best val loss: {best_val:.4f}')\n"
    "    return history\n"
    "\n"
    "print('Training utilities ready.')\n"
)

# --- 7. Train BiLSTM call ----------------------------------------------------
SRC_TRAIN_LSTM = (
    "lstm_history = train_model(\n"
    "    lstm_model, 'Bidirectional LSTM + Attention',\n"
    "    train_loader, val_loader,\n"
    "    n_classes = N_CLASSES,\n"
    "    epochs    = 30,\n"
    "    lr        = 3e-4,\n"
    "    patience  = 7,\n"
    ")\n"
)

# --- 8. Train CNN call -------------------------------------------------------
SRC_TRAIN_CNN = (
    "cnn1d_history = train_model(\n"
    "    cnn1d_model, '1D CNN (residual)',\n"
    "    train_loader, val_loader,\n"
    "    n_classes = N_CLASSES,\n"
    "    epochs    = 30,\n"
    "    lr        = 3e-4,\n"
    "    patience  = 7,\n"
    ")\n"
)

# =============================================================================
# Apply patches
# =============================================================================
set_cell('8e8f8c92b69f', SRC_SCALE)
set_cell('8894ae0eb948', SRC_FEATSEL)
set_cell('f413c9be7f05', SRC_WINDOW)
set_cell('8b0e21c89b10', SRC_BILSTM)
set_cell('ace0ca7ad5a1', SRC_CNN)
set_cell('7fdfc30194da', SRC_TRAIN_UTILS)
set_cell('e79afb587315', SRC_TRAIN_LSTM)
set_cell('02dcbca77737', SRC_TRAIN_CNN)

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('\n[OK] ids.ipynb patched successfully.')
