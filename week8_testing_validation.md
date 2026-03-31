# Week 8: Testing and Validation Strategies
## Temporal IDS — BiLSTM + 1D CNN Ensemble

**Project**: Network Intrusion Detection System (IDS) using Temporal Deep Learning
**Dataset**: BCCC-Cloud-DDoS-2024 · 700,774 flows · 86 features · 46,717 temporal windows
**Task**: 3-class classification — Attack / Benign / Suspicious

---

## 1. Overview

This document defines the testing and validation plan for the temporal IDS prototype developed in Week 6. The prototype consists of two deep learning models — a Bidirectional LSTM with soft attention and a 1D Residual CNN — combined into a soft-vote ensemble. Both models classify sliding windows of 30 time-ordered network flows into three categories: **Attack**, **Benign**, and **Suspicious**.

The Week 6 prototype established:
- A fully reproducible data pipeline (`merged_CSVs.csv` → `ddos_clean.parquet`)
- Feature reduction from 318 → 86 features via variance threshold + correlation pruning
- Temporal windowing: 30-flow windows, stride 15 (50% overlap)
- Stratified 70/15/15 train/val/test split by window label
- A soft-vote ensemble combining BiLSTM and 1D CNN outputs
- A confidence-based escalation mechanism (threshold = 0.75)

Week 8 builds on this by formalizing the testing approach, defining success metrics, and planning security assessment.

---

## 2. Test Scenarios

### 2.1 AI Model Performance

| Scenario ID | Description | Input | Expected Outcome |
|---|---|---|---|
| T-01 | Benign traffic classification | Normal flows from `merged_CSVs.csv` | Predicted: Benign, confidence ≥ 0.75 |
| T-02 | DDoS attack detection | Attack-labeled flows | Predicted: Attack, F1 ≥ 0.94 |
| T-03 | Suspicious traffic detection | Suspicious-labeled flows | Predicted: Suspicious, recall ≥ 0.90 |
| T-04 | Mixed-window classification | Windows bridging Benign → Attack | Majority-vote label correct |
| T-05 | Low-confidence escalation | Ambiguous or novel traffic | Confidence < 0.75 → flagged for human review |
| T-06 | Temporal ordering integrity | Flows sorted by timestamp | No future-data leakage into past windows |
| T-07 | Ensemble vs. individual model | Test set, all three predictors | Ensemble macro F1 ≥ both individual models |
| T-08 | Class imbalance robustness | Suspicious class (8.4% of windows) | Recall ≥ 0.90 despite minority class size |

### 2.2 Data Pipeline

| Scenario ID | Description | Expected Outcome |
|---|---|---|
| P-01 | Re-run cleaning on raw CSV | `ddos_clean.parquet` matches original shape (700,774 × 322) |
| P-02 | Scaler fit boundary | Scaler fit only on rows 0–490,541 (70%), no leakage |
| P-03 | Feature selector reproducibility | `variance_selector.pkl` + `corr_keep_idx.npy` produce same 86 columns |
| P-04 | Sentinel value handling | "not a complete handshake" in `delta_start`/`handshake_duration` → 0.0 |
| P-05 | Label encoding consistency | `le_label.classes_` always: `['Attack', 'Benign', 'Suspicious']` |

### 2.3 Cybersecurity / Adversarial

| Scenario ID | Description | Expected Outcome |
|---|---|---|
| S-01 | Evasion — low-rate DDoS | Slow SYN flood (< threshold rates) | Detect within 2–3 windows or escalate |
| S-02 | Evasion — mimicry attack | Attack flows crafted to resemble Benign | Confidence drop flagged for escalation |
| S-03 | Data poisoning simulation | Corrupt 5% of training labels | Accuracy degradation < 3% absolute |
| S-04 | Model extraction resistance | Repeated black-box queries | System does not expose raw logits |
| S-05 | Input validation | NaN/Inf injected into feature vector | `nan_to_num` sanitisation prevents crash |

---

## 3. Success Metrics

### 3.1 Primary Metrics (from prototype test set)

| Metric | BiLSTM | 1D CNN | Ensemble | Target |
|---|---|---|---|---|
| Overall accuracy | 93.84% | 94.11% | — | ≥ 94% |
| Attack F1 | 0.9451 | 0.9429 | — | ≥ 0.94 |
| Benign F1 | 0.9593 | 0.9626 | — | ≥ 0.95 |
| Suspicious F1 | 0.7945 | 0.8051 | — | ≥ 0.80 |
| Macro F1 | 0.8996 | 0.9035 | target ≥ 0.91 | ≥ 0.91 |

> Baseline established in `ids.ipynb` — all future model changes must maintain or exceed these numbers.

### 3.2 Confidence / Escalation Metrics

| Metric | Current Value | Target |
|---|---|---|
| Auto-classify coverage @ thr=0.75 | TBD (see `confidence_analysis.png`) | ≥ 85% |
| Accuracy on auto-classified subset | TBD | ≥ 97% |
| False escalation rate | TBD | ≤ 15% |

### 3.3 Security Metrics

| Metric | Target |
|---|---|
| Detection rate (Attack class recall) | ≥ 93% |
| False positive rate (Benign misclassified as Attack) | ≤ 5% |
| Suspicious class recall | ≥ 90% |
| System crash rate under malformed input | 0 |

---

## 4. Tools and Frameworks

### 4.1 AI Validation

| Tool | Purpose |
|---|---|
| `sklearn.metrics.classification_report` | Per-class precision, recall, F1 |
| `sklearn.metrics.confusion_matrix` + seaborn | Visual confusion matrix |
| `sklearn.metrics.accuracy_score` | Overall accuracy |
| PyTorch `DataLoader` (test split) | Reproducible held-out evaluation |
| `numpy.nan_to_num` | Input sanitisation testing |
| `joblib` | Scaler/encoder serialization and reload testing |

### 4.2 Model Robustness

| Tool | Purpose |
|---|---|
| Custom confidence threshold sweep | Coverage-accuracy trade-off curves |
| Label corruption script (to be written) | Data poisoning simulation (Scenario S-03) |
| Adversarial window generator (to be written) | Evasion testing (Scenarios S-01, S-02) |

### 4.3 Security Assessment

| Tool | Purpose |
|---|---|
| Manual code review | Input validation, serialisation (pickle) safety |
| `bandit` (Python SAST) | Static analysis of pipeline scripts |
| Wireshark / `tcpreplay` | Replay captured DDoS pcap for live-ish testing |
| Scapy | Craft synthetic attack flows for evasion tests |

---

## 5. Test Data Strategy

### 5.1 Existing Test Split

The test set is already isolated: **7,008 windows** (15% of 46,717), stratified by class:

| Class | Count | % |
|---|---|---|
| Attack | 2,281 | 32.5% |
| Benign | 4,136 | 59.0% |
| Suspicious | 591 | 8.4% |

This split is fixed by `random_state=42` and must not be touched during hyperparameter tuning.

### 5.2 Additional Test Data

| Source | How to Collect | Purpose |
|---|---|---|
| Held-out CSV days | Use `Thursday_14_Dec_2023.csv` only as a zero-shot test | Generalisation to unseen dates |
| Synthetic DDoS flows (Scapy) | Craft SYN-flood, UDP-flood windows | Adversarial / evasion testing |
| Benign background traffic | CICFlowMeter on lab machine | False positive rate on real benign traffic |

### 5.3 No-Leakage Guarantee

The following constraints must hold across all test runs:
1. `MinMaxScaler` is fit only on rows `0:490541` (the first 70% by timestamp).
2. `VarianceThreshold` and correlation pruning are fit only on the same training slice.
3. Test windows must not contain flows from before their true chronological position (temporal ordering enforced by `sort_values('timestamp')`).

---

## 6. AI Validation Strategy

### 6.1 Cross-Validation Plan

Because the data is time-ordered, standard k-fold is inappropriate. The plan:

1. **Walk-forward validation**: Train on days 1–3, validate on day 4; train on days 1–4, validate on day 5. Compare F1 across folds.
2. **Stratified hold-out** (already done in prototype): confirms class balance in every split.

### 6.2 Ablation Studies

| Ablation | What to Remove | Expected Impact |
|---|---|---|
| No attention | Remove `attn_score` from BiLSTM | Accuracy drop ≤ 1% expected |
| No feature selection | Use all 318 features | Memory increase; check if F1 changes |
| Window size sensitivity | Test W=10, 20, 30, 50 | Suspicious recall most sensitive |
| No ensemble | Single model only | Macro F1 drops by ~0.5% |

### 6.3 Suspicious Class Deep Dive

The Suspicious class has precision ≈ 70% (highest false positive source). Planned analysis:
- Inspect misclassified Suspicious windows: are they early-stage attacks or genuinely ambiguous?
- Test oversampling (SMOTE on window level) vs. current class-weighted loss.
- Document whether collapsing Suspicious → Attack improves or harms overall detection.

---

## 7. Cybersecurity Audit Strategy

### 7.1 Threat Model

The IDS pipeline processes untrusted network data. Relevant threats:

| Threat | Vector | Mitigation in Prototype |
|---|---|---|
| Adversarial evasion | Attacker crafts flows below detection threshold | Confidence escalation at 0.75 |
| Data poisoning | Corrupt labels in training CSV | Class-weighted loss limits influence |
| Model inversion | Extract training data from model weights | Model served offline; no API exposure |
| Pickle injection | Load malicious `.pkl` scaler file | Validate file hash before loading |
| Input overflow | NaN/Inf in feature vector crashes model | `nan_to_num` applied in pipeline |

### 7.2 Penetration Testing Plan

| Test | Method | Pass Criterion |
|---|---|---|
| Evasion — SYN flood slow-rate | Replay at 10% normal rate via `tcpreplay` | Detection within 3 windows |
| Evasion — protocol mimicry | Craft attack flows matching Benign feature ranges | Confidence < 0.75 escalates |
| Pickle injection | Load a crafted `.pkl` with malicious `__reduce__` | Pipeline rejects / crashes safely |
| Input fuzzing | Feed random float32 arrays as feature vectors | No crash; output is valid 3-class prob |

### 7.3 Vulnerability Scanning

- Run `bandit -r .` on all Python source to identify: unsafe `pickle.load`, shell injection in subprocess calls, hardcoded credentials.
- Review `models/` directory: confirm `.pt` files (safe) are used instead of `.pkl` where possible. The `temporal_scaler.pkl` should be integrity-checked with SHA-256 on load.

---

## 8. Documented Test Cases

### TC-001: Attack Detection on Test Split

**Input**: 2,281 Attack-class windows from held-out test set
**Steps**:
1. Load `models/bilstm_ids.pt`, `models/cnn1d_ids.pt`, `models/temporal_scaler.pkl`
2. Run ensemble inference on test loader
3. Record precision, recall, F1 for Attack class

**Expected**: Recall ≥ 0.93, F1 ≥ 0.94
**Tool**: `classification_report` in `ids.ipynb`

---

### TC-002: Suspicious Class Recall

**Input**: 591 Suspicious-class windows
**Steps**:
1. Run ensemble on test loader
2. Extract per-class metrics for class index 2

**Expected**: Recall ≥ 0.90, F1 ≥ 0.80
**Current result**: BiLSTM recall = 0.9357, CNN recall = 0.9154

---

### TC-003: Confidence Escalation at thr=0.75

**Input**: Full 7,008-window test set
**Steps**:
1. Compute `ens_probs.max(axis=1)` for all windows
2. Count windows with max-prob < 0.75
3. Verify these are predominantly Suspicious-class or near-decision-boundary cases

**Expected**: ≥ 85% windows auto-classified; escalated subset has lower accuracy than auto-classified subset
**Tool**: `confidence_analysis.png` code block in `ids.ipynb`

---

### TC-004: Pipeline Re-run Reproducibility

**Input**: `dataset/merged_CSVs.csv`
**Steps**:
1. Delete `ddos_clean.parquet` and all `models/` files
2. Re-run all cells in `ids.ipynb` from top
3. Compare final test accuracy to baseline

**Expected**: Accuracy within ±0.1% of 94.11% (CNN) and 93.84% (LSTM)
**Pass criterion**: Deterministic results with `random_state=42`

---

### TC-005: Malformed Input Handling

**Input**: Feature vector with `np.nan`, `np.inf`, and `-np.inf` values
**Steps**:
1. Construct a (1, 30, 86) tensor with injected NaN/Inf
2. Apply `np.nan_to_num` (as done in pipeline)
3. Run through model

**Expected**: No exception raised; output is a valid 3-element probability vector summing to 1.0

---

## 9. Limitations and Risks

| Limitation | Impact | Mitigation |
|---|---|---|
| Suspicious class precision ≈ 70% | ~30% of Suspicious alerts are false positives | Escalation threshold; human review queue |
| Single dataset (BCCC-Cloud-DDoS-2024) | Unknown generalisation | Test on held-out dates; plan for cross-dataset evaluation |
| Fixed window size (30 flows) | May miss very long or very short attack sequences | Ablation over W=10,20,30,50 |
| Offline pipeline only | No real-time deployment tested | Out of scope; models are structured for future streaming extension |
| Pickle serialisation for scaler | Potential code injection if file tampered | SHA-256 integrity check on load |

---

## 10. Milestone Summary

| Deliverable | Status | Target Date |
|---|---|---|
| Baseline test metrics (from prototype) | Done — in `ids.ipynb` | Week 6 |
| Formal test case documentation | This document | Week 8 |
| Adversarial evasion test implementation | Planned | Week 9 |
| Walk-forward cross-validation | Planned | Week 9 |
| Ablation study results | Planned | Week 9 |
| Security audit (`bandit`, hash check) | Planned | Week 9 |
| Final validation report | Planned | Week 10 |
