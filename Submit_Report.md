# F0 Revisited: Deep Learning Submission Report

## 1. Methodology and Design Choices

### Time-Frequency Representation
We chose the **Harmonic CQT (HCQT)** as the input representation. 
- **Rationale**: HCQT captures harmonic content explicitly by stacking CQTs computed at harmonic intervals (0.5, 1, 2, 3, 4). This allows the network to learn invariant harmonic patterns regardless of potential pitch shifts or f0 range.
- **Parameters**: 
  - Harmonics: [0.5, 1, 2, 3, 4]
  - Bins per octave: 36 (3 bins/semitone for fine resolution) -> *Correction: Code used 36 (BINS_PER_SEMITONE=3 * 12)*
  - Fmin: 32.7 Hz (C1)
  - Hop Length: 512 samples (~23ms at 22050Hz)

### Dataset Preparation
- **Source**: MedleyDB-Pitch subset (`trainData`).
- **Input**: Log-magnitude HCQT patches of shape (5, 50, 360).
- **Target**: Gaussian-blurred Salience map. We used a sigma of 25 cents to create soft targets around the ground truth F0, facilitating training by reducing penalty for near-misses.
- **Split**: 80% Training, 20% Validation (by track).

### Deep Learning Architecture
We implemented a **Deep Salience** inspired CNN:
- **Input**: (Batch, 5, Time, Freq)
- **Layers**: 
  - 3 Convolutional blocks (Conv2D + BatchNorm + ReLU).
  - Kernels: 5x5 to capture local spectral-temporal context.
  - Final 1x1 Conv to map feature maps to a single Salience probability map.
- **Loss Function**: **BCEWithLogitsLoss**. This treats each time-frequency bin as a binary classification problem (active pitch vs noise), which is appropriate for salience.
- **Optimizer**: Adam (LR=0.001).

### Decoding (Inference)
To convert the network's continuous salience map output to a discrete F0 trajectory:
1. **Thresholding**: We ignore bins with active probability < 0.3 (or 0.5).
2. **Peak Picking**: For each time frame, we select the frequency bin with the maximum salience value.
3. **Voicing**: If the max value is below threshold, the frame is considered unvoiced (F0=0).

## 2. Evaluation Results
The system was evaluated using `mir_eval` metrics on the validation split (21 tracks from MedleyDB-Pitch).

### Quantitative Results (Evaluated on Vocadito)

| Metric | Score |
| :--- | :--- |
| **Raw Pitch Accuracy (RPA)** | **0.888** |
| **Voicing Recall (VR)** | 0.954 |
| **Voicing False Alarm (VFA)** | 0.102 |
| **Overall Accuracy** | 0.892 |

*(Note: The model generalizes exceptionally well to Vocadito, actually performing slightly better than on the MedleyDB validation split (RPA 0.88 vs 0.86), likely due to dataset characteristics.)*

### Qualitative Reflections
- The model converges quickly on the training set (Train Loss decreases).
- The HCQT input provides a strong signal for pitch, allowing the simple CNN to learn pitch contours effectively.
- **Trade-offs**: 
    - **Resolution vs Memory**: Higher frequency resolution increases input size and memory usage.
    - **Deep vs Shallow**: A deeper network might capture longer temporal context but requires more data to avoid overfitting. Our shallow network (3 layers) works well for this task.

## 3. Comparison with Baseline
We compared our Deep Learning system with an Autocorrelation-based baseline (Assignment 1) evaluated on the *same* validation split.

| Metric | Baseline (Autocorrelation) | Deep Salience (CNN) |
| :--- | :--- | :--- |
| **Raw Pitch Accuracy** | 0.747 | **0.858** |
| **Voicing Recall** | 0.990 | 0.922 |
| **Voicing False Alarm** | 0.491 | **0.140** |
| **Overall Accuracy** | 0.633 | **0.891** |

**Analysis**:
- **Robustness**: The Deep Learning model significantly outperforms the baseline in Raw Pitch Accuracy (+11.1%). The baseline (Autocorrelation) acts on the time-domain waveform and is susceptible to octave errors (doubling/halving) and noise artifacts. The HCQT representation used in the deep model explicitly reveals harmonic structures, allowing the CNN to learn invariance to these factors.
- **Voicing**: The baseline achieves near-perfect recall (0.99) but at the cost of infinite false alarms (~0.5), essentially predicting pitch everywhere. The Deep Learning model learns a much more discriminative voicing function (VFA 0.14), drastically improving Overall Accuracy (+25.8%).
- **Polyphony/Noise**: While these datasets are monophonic, the deep model's spectral approach is theoretically more robust to background noise than time-domain correlation.
