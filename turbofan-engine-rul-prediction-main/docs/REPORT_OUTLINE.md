# College Project Report Outline: Remaining Useful Life Prediction for Turbofan Machinery

This document provides a complete chapter-wise blueprint and drafting guide for your final-year Bachelor's project report or thesis.

---

## Front Matter
- **Title Page**: Remaining Useful Life (RUL) Prediction for Turbofan Machinery Using Deep Recurrent Neural Networks on NASA C-MAPSS Telemetry
- **Certificate of Authenticity**: Departmental certificate signed by Project Supervisor and Head of Department.
- **Declaration by Student**: Formal declaration of original work.
- **Acknowledgements**: Academic gratitude to college, supervisor, and department faculty.
- **Abstract** (Summary: 250–300 words):
  - *Context*: Predictive maintenance and condition-based monitoring in aerospace.
  - *Problem*: Traditional fixed-interval maintenance causes wasteful early replacement or catastrophic in-flight failure.
  - *Methodology*: NASA C-MAPSS dataset (FD001), piecewise-linear RUL capping (125 cycles), leak-free engine-level grouping, MinMax scaling, and comparative benchmarking across Linear Regression, Random Forest, XGBoost, 1D-CNN, and PyTorch LSTM.
  - *Key Results*: PyTorch LSTM achieved the best performance with Test RMSE = 14.72 cycles, MAE = 11.19 cycles, $R^2 = 0.8745$, and a 43.3% reduction in NASA Asymmetric Score (361.4 vs 637.4 for XGBoost).
  - *Deliverable*: Interactive industrial maintenance console built with Streamlit.
- **Table of Contents**
- **List of Figures** (11 generated figures)
- **List of Tables** (Dataset audit table, feature summary, master benchmark results)
- **List of Abbreviations** (RUL, C-MAPSS, HPC, LPC, HPT, LPT, FADEC, RMSE, MAE, LSTM, CNN)

---

## Chapter 1: Introduction
- **1.1 Background & Motivation**: The economic and safety stakes of commercial aircraft propulsion maintenance.
- **1.2 Evolution of Maintenance Paradigms**: Reactive (run-to-failure) $\rightarrow$ Preventive (fixed-interval time limits) $\rightarrow$ Predictive (Condition-Based Maintenance).
- **1.3 Problem Statement**: Estimating remaining operational cycles from noisy, multi-sensor degradation time-series under manufacturing variations.
- **1.4 Project Objectives**:
  1. Profile thermodynamic sensor trajectories and identify invariant telemetry channels.
  2. Implement a zero-leakage preprocessing pipeline with piecewise-linear RUL capping and engine-level grouping.
  3. Develop and benchmark a spectrum of models from simple linear baselines to deep recurrent networks.
  4. Evaluate models using both standard symmetric metrics and safety-critical asymmetric risk metrics.
  5. Deploy a practical, industrial-grade maintenance-engineer console in Streamlit.
- **1.5 Report Organization**: Chapter-by-chapter roadmap.

---

## Chapter 2: Literature Review
- **2.1 Prognostics and Health Management (PHM)**: Conceptual framework of diagnostic vs prognostic systems.
- **2.2 Physics-Based vs Data-Driven Prognostics**:
  - Physics-based (damage mechanics, Paris-Erdogan crack propagation models) and why they fail for complex systems.
  - Data-driven machine learning methods (data abundance, sensor fusion).
- **2.3 The NASA C-MAPSS Benchmark**: Review of PHM 2008 competition papers (Saxena, Goebel, Simon, Eklund).
- **2.4 Classical Machine Learning for RUL**: Support Vector Regression (SVR), Random Forest, and Gradient Boosted Decision Trees on engineered statistical features.
- **2.5 Deep Learning in Prognostics**:
  - Recurrent Neural Networks (RNN) and the vanishing gradient problem.
  - Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRU).
  - Convolutional Neural Networks (1D-CNN) for temporal pattern extraction.
- **2.6 Research Gaps Identified**: Inadequate reporting of directional error bias (late vs early prediction risk), over-optimistic evaluation due to row-level data leakage, and lack of practical operational deployment interfaces.

---

## Chapter 3: Dataset Architecture & Degradation Physics
- **3.1 Simulation Architecture**: The Modular Aero-Propulsion System Simulation (C-MAPSS) engine model.
- **3.2 Sub-Dataset Topologies**: Comparison of FD001, FD002, FD003, and FD004 (operating conditions, fault modes, engine counts).
- **3.3 Telemetry Schema & Channel Description**: Detailed audit of the 26 columns, physical units, and sensor measurement stations ($T_0, T_{24}, T_{30}, T_{50}, P_0, P_{15}, P_{30}, N_f, N_c, P_{s30}, W_f$).
- **3.4 Thermodynamic Damage Mechanics**: High Pressure Compressor (HPC) erosion, blade surface roughness, stage pressure degradation, and closed-loop FADEC controller fuel compensation.
- **3.5 Statistical Variance Profile**: Empirical identification and physical justification of the 7 invariant channels ($s_1, s_5, s_6, s_{10}, s_{16}, s_{18}, s_{19}$) in FD001.

---

## Chapter 4: Proposed Methodology & Feature Pipeline
- **4.1 System Architecture Overview**: End-to-end block diagram from raw telemetry to decision support.
- **4.2 Ground-Truth RUL Target Formulation**: Mathematical definition of run-to-failure linear targets.
- **4.3 Piecewise-Linear RUL Capping**: Mathematical formulation of $RUL_{capped} = \min(RUL, 125)$ and justification based on the initial healthy plateau.
- **4.4 Zero-Leakage Grouped Partitioning**: Strict engine-level splitting (80 train engines / 20 validation engines).
- **4.5 Bounded MinMax Normalization**: Mathematical scaling to $[0, 1]$ and why it was fitted strictly on training engines.
- **4.6 Tabular Feature Engineering (Classical ML)**: Rolling mean, rolling standard deviation, and instantaneous values over a 30-cycle window (42 features total).
- **4.7 3D Sliding Sequence Window Generation (Deep Learning)**: Tensor construction `(N, 30, 14)` for recurrent sequence encoding.

---

## Chapter 5: Predictive Models & Architecture Design
- **5.1 Baseline Model**: Ordinary Least Squares (OLS) Linear Regression benchmark.
- **5.2 Bagging Ensemble**: Random Forest Regressor (hyperparameters: 100 estimators, max depth 12).
- **5.3 Gradient Boosting**: XGBoost Regressor (hyperparameters: 100 estimators, max depth 5, learning rate 0.05, $L_1/L_2$ regularization).
- **5.4 Temporal Convolutional Network (1D-CNN)**: 1D convolution layers, batch normalization, max pooling, adaptive pooling, and fully connected regressor.
- **5.5 Deep Recurrent Network (PyTorch TurbofanLSTM)**:
  - Mathematical formulation of LSTM memory cells and gating equations (forget, input, candidate, output).
  - Network architecture: 2-layer LSTM (`hidden_dim=64/32`, `dropout=0.2`), linear regressor head.
- **5.6 Training Optimization Strategy**: Smooth L1 (Huber) loss, Adam optimizer, learning rate reduction on plateau, and early stopping on validation engines.

---

## Chapter 6: Experimental Results & Benchmarking
- **6.1 Evaluation Metrics**:
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - Coefficient of Determination ($R^2$)
  - NASA PHM 2008 Asymmetric Scoring Function
- **6.2 Master Benchmark Comparison**: Empirical performance table across all 5 models evaluated on 100 test engines.
- **6.3 Learning Dynamics**: Analysis of training and validation loss curves (overfitting mitigation).
- **6.4 Feature Importance Analysis**: Random Forest MDI and XGBoost Gain rankings (dominance of $T_{30}$ and $P_{s30}$).
- **6.5 Directional Asymmetric Risk Analysis**: Safe early maintenance ($d_i < 0$) vs dangerous late predictions ($d_i \ge 0$), and why Random Forest failed on the NASA score despite low MAE.
- **6.6 Individual Engine Case Studies**: Trajectory tracking and multi-model consensus across test engines 24, 31, 47, 68, and 99.

---

## Chapter 7: Maintenance Console Application
- **7.1 Industrial UI/UX Philosophy**: Calm, information-first, non-decorative design for operational engineers.
- **7.2 Software Architecture**: Streamlit framework, CSS styling, model weight serialization (`.joblib`, `.pt`).
- **7.3 Operational Workflow**:
  - Engine selection and real-time RUL estimation with error bounds.
  - Multi-model fleet consensus evaluation.
  - Critical telemetry snapshot with nominal baseline comparison.
  - Interactive sensor degradation plotting with 10-cycle smoothed trend.
  - Model benchmarking dashboard and technical methodology reference.

---

## Chapter 8: Conclusion, Limitations & Future Work
- **8.1 Summary of Contributions**: Validation of end-to-end prognostic pipeline with zero data leakage, proof of LSTM superiority on temporal telemetry, and deployment of industrial console.
- **8.2 Critical Project Limitations**: Single flight regime (FD001 sea level), single failure mode, deterministic point predictions.
- **8.3 Future Work**:
  - Extension to multi-condition datasets (FD002/FD004) using flight condition clustering.
  - Bayesian Neural Networks and Monte Carlo Dropout for probabilistic uncertainty intervals.
  - Engineered safety offsets for zero-tolerance aerospace deployment.

---

## References
- Key academic citations formatted in IEEE style (NASA PHM papers, deep learning for RUL, gradient boosting literature).
