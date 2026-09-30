# Architectural & Engineering Decisions (Technical Viva Defense)

This document records every architectural, mathematical, and algorithmic decision made during this project, explaining **why** each technique was selected over practical alternatives.

---

## 1. Data Engineering & Preprocessing Decisions

### Decision 1.1: Piecewise-Linear RUL Capping at 125 Cycles
- **Chosen Approach**: Capped ground truth RUL at $RUL_{capped} = \min(RUL_{true}, 125)$.
- **Alternative Considered**: Uncapped linear degradation ($RUL = \max(\text{cycle}) - \text{cycle}$).
- **Why Chosen?**
  In the early operational life of an engine (cycles 1 to $\sim 80$), components operate within nominal tolerances. Material degradation is microstructural and sensor signals exhibit zero measurable drift. 
  If an uncapped target is used, a model must predict $RUL=310$ on Engine A at cycle 10 and $RUL=160$ on Engine B at cycle 10, despite both engines outputting identical healthy sensor readings. This forces the model to fit random sensor noise. Capping at 125 cycles aligns mathematical loss strictly with the physical regime where degradation is observable.

### Decision 1.2: Grouped Split by Engine ID (Zero Cross-Engine Leakage)
- **Chosen Approach**: Partitioned 80 engines for training and 20 engines for validation strictly by `unit_number`:
  $$\text{Train Engines} \cap \text{Validation Engines} = \emptyset$$
- **Alternative Considered**: Standard random row-level train/test split (e.g. `train_test_split(df, test_size=0.2)`).
- **Why Chosen?**
  In time-series sensor telemetry, consecutive cycles from the same physical machine are auto-correlated and share engine-specific baseline manufacturing offsets. Splitting randomly by row places cycle $t$ in train and cycle $t+1$ of the *same* engine in validation. This constitutes catastrophic data leakage, producing deceptively high validation scores ($R^2 > 0.98$) that collapse on real unseen test engines.

### Decision 1.3: Dropping Invariant / Zero-Variance Channels
- **Chosen Approach**: Pruned 7 channels (`s_1, s_5, s_6, s_10, s_16, s_18, s_19`) and flight settings (`setting_1, setting_2, setting_3`), leaving the canonical 14 degradation sensors.
- **Alternative Considered**: Retaining all 26 columns and relying on model regularization (e.g., Ridge / Lasso).
- **Why Chosen?**
  FD001 simulates a single flight condition (sea-level ambient pressure and temperature). These 7 channels have zero or near-zero variance ($< 10^{-4}$). Retaining them risks rank deficiency in ordinary least squares, injects numerical noise into tree splits, and wastes weights in neural networks without providing any degradation signal.

### Decision 1.4: MinMaxScaler Over StandardScaler
- **Chosen Approach**: Normalized all active channels using `MinMaxScaler(feature_range=(0, 1))` fitted exclusively on the 80 training engines.
- **Alternative Considered**: `StandardScaler(mean=0, std=1)`.
- **Why Chosen?**
  1. Turbofan sensors measure bounded physical thermodynamic properties (rotational speeds, absolute pressures, temperatures in °R). `MinMaxScaler` preserves the strictly non-negative nature of these values.
  2. Bounding features strictly within $[0, 1]$ prevents gradient saturation in neural network activation functions (tanh, sigmoid, relu).
  3. Standard deviation scaling can be sensitive to extreme sensor spikes during initial wear-in.
  4. Crucially, the scaler was fitted **only on training data** to avoid leakage into validation and test sets.

### Decision 1.5: Window Size of 30 Operational Cycles
- **Chosen Approach**: Sliding sequence window length $W = 30$ cycles.
- **Alternative Considered**: Window sizes of 10, 15, or 50 cycles.
- **Why Chosen?**
  - A window of 10 cycles is too brief to distinguish true degradation drift from high-frequency sensor noise.
  - In the test dataset (`test_FD001.txt`), the shortest recorded engine trajectory is 31 cycles. A window size of $W = 30$ ensures that **every single test engine in the fleet** possesses at least one complete, unpadded sequence window, eliminating the need for artificial extrapolation.

---

## 2. Model Selection & Modeling Decisions

### Decision 2.1: Benchmarking from Linear Baseline to Deep Learning
- **Chosen Approach**: Structured progression from Linear Regression $\rightarrow$ Random Forest $\rightarrow$ XGBoost $\rightarrow$ 1D-CNN $\rightarrow$ PyTorch LSTM.
- **Why Chosen?**
  In academic research, deep learning complexity must be empirically justified against simpler benchmarks. By establishing a Linear Regression baseline ($R^2 = 0.80$, RMSE = 18.60), we proved that XGBoost (RMSE = 17.63) and PyTorch LSTM (RMSE = 14.72) provide measurable value rather than arbitrary complexity.

### Decision 2.2: PyTorch LSTM as the Primary Sequence Model
- **Chosen Approach**: Two-layer LSTM network (`hidden_dim=64/32`, `dropout=0.2`) taking 3D tensors `(N, 30, 14)`.
- **Alternative Considered**: 1D-CNN, Multi-Layer Perceptron (MLP), GRU.
- **Why Chosen?**
  - Classical tree models and MLPs treat sequence features as static snapshot summaries (mean and std), ignoring the chronological order and rate of acceleration.
  - 1D-CNN applies fixed localized filters without temporal recurrence.
  - The LSTM's recurrent memory cell ($C_t$) and hidden state ($h_t$) maintain a persistent running trajectory of cumulative component damage, capturing the acceleration of wear as the engine nears end-of-life.

### Decision 2.3: Smooth L1 (Huber) Loss Over Mean Squared Error
- **Chosen Approach**: Trained PyTorch models using `nn.SmoothL1Loss()`.
- **Alternative Considered**: Standard Mean Squared Error (`nn.MSELoss()`).
- **Why Chosen?**
  $$\mathcal{L}(y, \hat{y}) = \begin{cases} 0.5(y - \hat{y})^2 & \text{if } |y - \hat{y}| \le 1.0 \\ |y - \hat{y}| - 0.5 & \text{otherwise} \end{cases}$$
  Turbofan sensor readings occasionally exhibit noisy spikes during transient engine control adjustments. MSE squares these errors, causing massive gradient spikes that destabilize recurrent weights. Smooth L1 is quadratic for small errors and linear for large errors, providing smooth and robust optimization.

### Decision 2.4: Adoption of the NASA Asymmetric Scoring Function
- **Chosen Approach**: Primary evaluation guided by the official PHM 2008 asymmetric penalty:
  $$S = \sum_{d_i < 0} \left(e^{-d_i / 13} - 1\right) + \sum_{d_i \ge 0} \left(e^{d_i / 10} - 1\right)$$
- **Why Chosen?**
  Symmetric metrics like RMSE and MAE assume that an error of $+10$ cycles (predicting the engine will last 10 cycles longer than it can) is equal to an error of $-10$ cycles (predicting the engine will fail 10 cycles earlier). 
  In aviation, an early prediction costs minor downtime, while a late prediction causes an engine failure during flight. Penalizing late errors exponentially steeper ($e^{d/10}$ vs $e^{-d/13}$) reflects operational reality.

---

## 3. UI/UX Design Decisions

The Streamlit maintenance console was built following the philosophy of an internal industrial tool:

### Decision 3.1: Restrained Industrial Color Palette
- **Palette**: Off-white background (`#f8f9fa`), slate-grey structural borders (`#e2e8f0`), and slate-blue accent (`#2b5c8f`).
- **Eliminated Tropes**: No purple/blue neon gradients, glassmorphism, or dark neon glows.
- **Why?** Control-room operators and maintenance engineers work long shifts. Bright gradients cause visual fatigue. Color is reserved strictly for operational state:
  - Green (`#166534`): Healthy ($RUL > 75$)
  - Amber (`#9a3412`): Maintenance Due ($25 \le RUL \le 75$)
  - Red (`#991b1b`): Critical Wear ($RUL < 25$)

### Decision 3.2: Monospace Telemetry Typography
- **Typography**: System sans-serif for UI labels, monospace (`ui-monospace, Menlo, Consolas`) for telemetry values, cycle counters, and error numbers.
- **Why?** In tabular engineering displays, proportional fonts cause numbers to shift horizontally when values change. Monospace fonts ensure fixed-width alignment and readability.

### Decision 3.3: Information Hierarchy ("Lead with the Answer")
- **Layout**: Top section immediately presents Predicted RUL, Ground Truth RUL, Error Margin, and Health Status Badge in three prominent cards before detailing historical sensor trends.
- **Why?** An engineer triaging a fleet needs the maintenance decision immediately. Raw sensor waveforms are secondary diagnostic evidence.

### Decision 3.4: Multi-Model Consensus Transparency
- **Feature**: Displayed predictions from Linear Regression, Random Forest, XGBoost, and LSTM side-by-side.
- **Why?** In real industrial operations, trusting a single "black box" deep learning model is risky. Showing agreement across independent mathematical paradigms (linear, tree ensemble, and recurrent network) gives engineers confidence in the prognosis.

### Decision 3.5: Fleet Triage Filtering (Risk-Based Navigation)
- **Feature**: Summary pill badges (Healthy: 56, Maintenance Due: 33, Critical Wear: 11) with a radio filter to narrow down the 100-engine dropdown.
- **Why?** Maintenance engineers do not inspect engines randomly; they prioritize machinery in the "Critical Wear" or "Maintenance Due" categories. Allowing one-click filtering by operational status turns the tool from a demonstration viewer into a triage console.

### Decision 3.6: Flight Scheduling Simulation (Operational "What-If" Planning)
- **Feature**: An interactive cycle slider projecting remaining RUL and warning when an engine will cross the 25-cycle critical wear threshold.
- **Why?** Maintenance planners need to answer: *"Can Engine 47 fly 20 more flight legs before being grounded for overhaul?"* Simulating additional operational cycles provides immediate operational decision support.

### Decision 3.7: Diagnostic Dispatch Export (One-Click CSV Work Order)
- **Feature**: One-click download button generating a maintenance dispatch report for the selected engine.
- **Why?** Bridges the software prototype to the physical factory floor, allowing engineers to export the telemetric snapshot directly into maintenance documentation.

