# 25 High-Frequency Viva Questions & Model Answers

This document contains 25 questions frequently asked by external examiners in college project vivas on Predictive Maintenance and Machine Learning, paired with concise, confident answers based strictly on the codebase.

---

### Category 1: Problem Formulation & Dataset Mechanics

#### Q1: What is Remaining Useful Life (RUL), and how does it differ from fault detection?
**Answer**: Fault detection is a binary or multi-class classification problem that answers *"Is the engine faulty right now?"* In contrast, Remaining Useful Life (RUL) is a continuous regression problem that answers *"How many operational cycles does this machinery have left before catastrophic failure occurs?"* RUL enables condition-based predictive maintenance, allowing maintenance crews to schedule depot overhauls before breakdown while avoiding premature replacements.

#### Q2: What is the NASA C-MAPSS dataset?
**Answer**: C-MAPSS stands for *Commercial Modular Aero-Propulsion System Simulation*. It is a high-fidelity thermodynamic turbofan simulation developed by NASA. It simulates run-to-failure degradation trajectories for a fleet of identical engines operating under normal wear-and-tear, contaminated with realistic operational sensor noise and initial manufacturing variations.

#### Q3: What is the difference between FD001, FD002, FD003, and FD004?
**Answer**:
- **FD001**: 1 operating flight condition (Sea Level), 1 failure mode (High Pressure Compressor degradation).
- **FD002**: 6 operating flight conditions (variable altitude, Mach, and thrust), 1 failure mode (HPC degradation).
- **FD003**: 1 operating flight condition (Sea Level), 2 failure modes (HPC degradation + Fan degradation).
- **FD004**: 6 operating flight conditions, 2 failure modes (HPC + Fan degradation).
Our core project implements FD001, where flight parameters are held constant so degradation dynamics can be isolated and modeled cleanly.

#### Q4: How did you compute the target RUL labels for the training set?
**Answer**: In the training set, each engine is run until total mechanical failure. For each operational cycle $t$ of an engine that lasted $T_{max}$ cycles, the ground-truth linear RUL is:
$$RUL_t = T_{max} - t$$
At the final cycle, $RUL = 0$.

#### Q5: Why did you apply piecewise-linear RUL capping at 125 cycles?
**Answer**: In early engine life (cycles 1 to $\sim 80$), components operate in a healthy state with no measurable wear. Sensors exhibit flat, nominal baselines. If we do not cap RUL, the model is forced to predict $RUL=300$ for Engine A and $RUL=180$ for Engine B when both engines exhibit identical sensor readings. This forces the model to fit random sensor jitter. Capping RUL at 125 cycles aligns the mathematical target with the physical onset of observable degradation.

---

### Category 2: Preprocessing, Data Leakage & Feature Engineering

#### Q6: Why did you split the dataset by engine ID (`unit_number`) rather than random row splitting?
**Answer**: Telemetry rows from the same engine share persistent physical manufacturing characteristics and degradation history. If we split randomly by row (e.g. standard `train_test_split`), cycle $t$ might be in the training set while cycle $t+1$ of the *same* engine is in the validation set. This causes catastrophic temporal data leakage, artificially inflating validation accuracy while failing completely on unseen engines. Splitting strictly by engine ID ensures true generalization.

#### Q7: Why did you drop sensors 1, 5, 6, 10, 16, 18, 19 and setting 3 in FD001?
**Answer**: Because FD001 simulates a single flight condition (sea-level ambient temperature and pressure). These 7 sensors and setting 3 have standard deviations of zero or near-zero ($< 10^{-4}$). Retaining them injects numerical noise, risks collinearity in linear regression, and wastes parameters in neural networks without providing any degradation signal.

#### Q8: Why did you choose MinMaxScaler over StandardScaler?
**Answer**: Turbofan sensors measure bounded physical quantities (pressures, temperatures, shaft speeds) that are strictly non-negative. `MinMaxScaler` maps these signals onto $[0, 1]$ without distorting the zero baseline or creating artificial negative physical values. Furthermore, bounded $[0, 1]$ inputs prevent gradient saturation in neural network activations (tanh, sigmoid, relu).

#### Q9: Why did you fit the scaler strictly on the training engines?
**Answer**: To prevent data leakage. If we fit the scaler across both training and test data, the test set's minimum and maximum values would influence the scaling parameters. In real-world predictive maintenance, future test telemetry is unknown at training time.

#### Q10: Why did you engineer rolling mean and standard deviation features for classical models?
**Answer**: Instantaneous sensor snapshots contain high-frequency sensor noise. 
- **Rolling Mean (over 30 cycles)**: Smooths out high-frequency noise and captures the underlying thermodynamic drift.
- **Rolling Std (over 30 cycles)**: Captures variance expansion—as mechanical components erode, aerodynamic flutter and vibrational fluctuations increase.

#### Q11: Why did you choose a window length of 30 cycles?
**Answer**: A window of 10 cycles is too short to separate true degradation slope from transient sensor noise. Crucially, in the official test dataset (`test_FD001.txt`), the shortest engine run prior to cutoff is 31 cycles. A window size of 30 ensures that every single test engine has at least one complete sequence window without requiring artificial padding.

---

### Category 3: Machine Learning & Deep Learning Architectures

#### Q12: Why start with Linear Regression as a baseline?
**Answer**: In applied machine learning and project defense, model complexity must be justified. A Linear Regression baseline establishes the minimum benchmark performance ($R^2 = 0.80$, RMSE = 18.60). If a complex deep learning model cannot decisively beat a simple linear model, the complexity is unwarranted.

#### Q13: How does Random Forest work, and how did it perform?
**Answer**: Random Forest is a bagging ensemble of decision trees. It trains hundreds of trees on bootstrapped subsets of data with random feature subsampling to reduce variance. On our engineered rolling features, it achieved an MAE of 13.54 cycles, but suffered an elevated NASA score (755.9) due to several late prediction outliers.

#### Q14: How does XGBoost work, and why did it beat Random Forest?
**Answer**: XGBoost uses gradient boosting: instead of training trees independently in parallel like Random Forest, it trains shallow decision trees sequentially. Each new tree fits to the negative gradient (residuals) of the preceding trees using 2nd-order Taylor expansions with $L_1$ and $L_2$ regularization. This gave XGBoost the best performance among classical models (RMSE = 17.63, MAE = 12.90, $R^2$ = 0.8200).

#### Q15: Which features had the highest importance in your tree models?
**Answer**: In both Random Forest (61.2% importance) and XGBoost (24.0% importance), the #1 feature was **`s_3_roll_mean_30`** (High Pressure Compressor outlet temperature, $T_{30}$). This directly matches thermodynamic theory: as compressor blades erode and lose efficiency, the engine controller burns more fuel to maintain thrust, causing compressor exit temperature to climb steadily.

#### Q16: What is the architecture of your PyTorch LSTM model?
**Answer**: Our `TurbofanLSTM` consists of:
1. Input layer taking 3D sequence tensors of shape `(batch_size, 30 cycles, 14 sensors)`.
2. Two sequential LSTM layers (`hidden_dim=64`, `dropout=0.2`).
3. Fully connected regressor head (`Linear(64 -> 32) -> ReLU -> Dropout(0.2) -> Linear(32 -> 1)`).
4. Output: a single scalar representing estimated remaining cycles.

#### Q17: How do LSTM gates help in predicting RUL?
**Answer**: Unlike static models, an LSTM maintains a cell state ($C_t$) and hidden state ($h_t$). Its **Forget Gate** discards noise from ancient cycles, its **Input Gate** incorporates new sensor shifts from the current cycle, and its **Output Gate** extracts the updated degradation state. This allows the network to learn the **rate of acceleration** of wear across time.

#### Q18: Why did the LSTM outperform XGBoost and Random Forest?
**Answer**: Classical models collapse the 30-cycle trajectory into single summary numbers (mean and std), discarding the chronological order of readings. The LSTM processes the sequence step-by-step, preserving the precise temporal trajectory and curvature of degradation, resulting in a 16.5% lower RMSE (14.72 vs 17.63) and a 43.3% lower NASA score (361.4 vs 637.4).

#### Q19: Why did the 1D-CNN underperform compared to the LSTM?
**Answer**: 1D Convolutions apply static sliding kernel filters across local 3-cycle patches without persistent recurrent memory. C-MAPSS degradation is a slow, monotonic thermodynamic trend spanning dozens of cycles, not a high-frequency acoustic burst. Recurrent memory is mathematically superior to feed-forward convolutions for this task.

#### Q20: Why did you use Smooth L1 (Huber) Loss instead of standard MSE?
**Answer**: Mean Squared Error (MSE) squares errors. In turbofan telemetry, transient flight adjustments cause occasional sensor spikes. Squaring these errors creates massive gradient spikes that destabilize LSTM weights. Smooth L1 loss is quadratic for small errors ($|e| \le 1$) and linear for large errors ($|e| > 1$), making training smooth and robust against outliers.

---

### Category 4: Evaluation, Metrics & Real-World Deployment

#### Q21: What is the NASA Asymmetric Scoring Function, and why is it used?
**Answer**: Standard metrics (RMSE, MAE) treat early and late predictions equally. In aviation:
- An **early prediction** ($d_i < 0$) causes slightly premature maintenance, costing minor downtime.
- A **late prediction** ($d_i \ge 0$) means the engine is assumed healthy when it is failing, risking catastrophic in-flight engine shutdown.
The NASA score penalizes late predictions exponentially steeper:
$$S = \sum_{d_i < 0} \left(e^{-d_i / 13} - 1\right) + \sum_{d_i \ge 0} \left(e^{d_i / 10} - 1\right)$$

#### Q22: Why did Random Forest have a worse NASA score than Linear Regression despite having a lower MAE?
**Answer**: Random Forest achieved lower MAE (13.54 vs 15.00), but produced multiple late prediction outliers above $+25$ cycles. In the exponential formula, $e^{40/10} = 54.6$. A few large late errors explode the NASA score exponentially. This proves that standard MAE is insufficient for safety-critical maintenance auditing.

#### Q23: How did you validate your models without looking at the test set?
**Answer**: We performed an engine-grouped validation split during training (80 engines train, 20 engines validation). We monitored validation loss for early stopping, learning rate scheduling, and hyperparameter tuning. The test set (`test_FD001.txt` and `RUL_FD001.txt`) was touched strictly once after training completed for final benchmarking.

#### Q24: How does your Streamlit application handle model consensus?
**Answer**: The application provides a side-by-side consensus display comparing Linear Regression, Random Forest, XGBoost, and LSTM for any selected engine. If all four independent models predict low RUL, maintenance engineers have high confidence to pull the engine for depot overhaul.

#### Q25: What are the main limitations of your project, and how would you extend it?
**Answer**:
1. **Single Flight Condition**: FD001 operates only at sea level. In commercial aviation, engines climb, cruise at 35,000 ft, and descend. Future work would extend the model to FD002/FD004 using operating condition clustering (k-means) or flight condition normalization.
2. **Deterministic RUL**: Our model outputs a point estimate. In real operations, Bayesian Neural Networks or Monte Carlo Dropout should be used to output probabilistic confidence intervals (e.g. 95% survival bounds).
3. **Safety Offset**: To eliminate the risk of late predictions, production systems apply an engineered safety buffer (e.g. subtracting 10 cycles from predicted RUL).
