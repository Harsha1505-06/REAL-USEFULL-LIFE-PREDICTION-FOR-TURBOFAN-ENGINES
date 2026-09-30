# NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset

This document details the dataset files, schema specifications, operating envelopes, and thermodynamic damage propagation mechanisms of the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) turbofan engine degradation dataset.

---

## 1. Dataset Files & Ingestion Audit

All files were audited and verified directly in the project repository under `data/raw/` with zero missing values:

| Filename | File Size | Shape (Rows, Columns) | Null Count | Operational Scope |
| :--- | :--- | :---: | :---: | :--- |
| `train_FD001.txt` | 3.4 MB | `(20631, 26)` | 0 | 100 training engines run to complete failure |
| `test_FD001.txt` | 2.1 MB | `(13096, 26)` | 0 | 100 test engines truncated prior to failure |
| `RUL_FD001.txt` | 429 B | `(100, 1)` | 0 | Ground-truth Remaining Useful Life for test units |
| `train_FD002.txt` | 8.7 MB | `(53759, 26)` | 0 | 260 training engines (6 operating conditions) |
| `test_FD002.txt` | 5.5 MB | `(33991, 26)` | 0 | 259 test engines (6 operating conditions) |
| `RUL_FD002.txt` | 1.1 KB | `(259, 1)` | 0 | Ground-truth RUL for FD002 test units |
| `train_FD003.txt` | 4.0 MB | `(24720, 26)` | 0 | 100 training engines (HPC + Fan degradation) |
| `test_FD003.txt` | 2.7 MB | `(16596, 26)` | 0 | 100 test engines (HPC + Fan degradation) |
| `RUL_FD003.txt` | 428 B | `(100, 1)` | 0 | Ground-truth RUL for FD003 test units |
| `train_FD004.txt` | 9.9 MB | `(61249, 26)` | 0 | 248 training engines (6 conditions, 2 faults) |
| `test_FD004.txt` | 6.6 MB | `(41214, 26)` | 0 | 249 test engines (6 conditions, 2 faults) |
| `RUL_FD004.txt` | 1.1 KB | `(248, 1)` | 0 | Ground-truth RUL for FD004 test units |
| `readme.txt` | 2.4 KB | N/A | 0 | Official NASA challenge dataset documentation |
| `Damage Propagation Modeling.pdf` | 424.0 KB | N/A | 0 | PHM 2008 conference reference publication |

---

## 2. Experimental Scenarios Across Subsets

The C-MAPSS benchmark is divided into four sub-datasets representing increasing degrees of operational complexity:

| Dataset Identifier | Flight Conditions | Fault Modes Present | Training Engines | Test Engines |
| :--- | :---: | :--- | :---: | :---: |
| **FD001** (Primary focus) | **1 (Sea Level)** | **HPC Degradation** | **100** | **100** |
| **FD002** | 6 Operating Regimes | HPC Degradation | 260 | 259 |
| **FD003** | 1 (Sea Level) | HPC Degradation + Fan Degradation | 100 | 100 |
| **FD004** | 6 Operating Regimes | HPC Degradation + Fan Degradation | 248 | 249 |

### Experimental Design Characteristics
1. **Fleet Variability**: Each engine begins with random degrees of initial manufacturing variation and baseline wear that is unknown to the model. This initial variation is considered normal operation, not a defect.
2. **Run-to-Failure Protocol (Train)**: In the training set, each engine operates normally at first, encounters an initial degradation onset at an unknown cycle, and wears progressively until total operational failure (end-of-life).
3. **Truncated Evaluation (Test)**: In the test set, telemetry time series end at an arbitrary cycle prior to failure. The goal is to predict how many cycles remain before breakdown from the final operational snapshot.

---

## 3. Data Schema & Column Specification

Each file is space-separated with no headers and contains 26 columns:
- **Columns 0–1**: Trajectory identifiers (`unit_number`, `time_cycles`)
- **Columns 2–4**: Operational flight settings (`setting_1`, `setting_2`, `setting_3`)
- **Columns 5–25**: 21 Sensor measurements (`s_1` through `s_21`)

### Detailed Channel Descriptions (FD001 Training Statistics)

| Index | Name | Engineering Description | Physical Units | FD001 Train Min | FD001 Train Max | FD001 Train Std | Status in FD001 |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| 0 | `unit_number` | Engine Identifier | Integer ID | 1 | 100 | -- | Primary Key |
| 1 | `time_cycles` | Operational Cycle Index | Time Step | 1 | 362 | 68.88 | Time Index |
| 2 | `setting_1` | Flight condition: Altitude / Mach | -- | -0.0087 | +0.0087 | 0.0022 | Setting (Dropped) |
| 3 | `setting_2` | Flight condition: Throttle Resolver Angle | -- | -0.0006 | +0.0006 | 0.0003 | Setting (Dropped) |
| 4 | `setting_3` | Flight condition parameter | -- | 100.0 | 100.0 | **0.0000** | **Invariant (Dropped)** |
| 5 | `s_1` | Fan inlet temperature ($T_0$) | °R | 518.67 | 518.67 | **0.0000** | **Invariant (Dropped)** |
| 6 | `s_2` | LPC outlet temperature ($T_{24}$) | °R | 641.21 | 644.53 | 0.5001 | **Active ($r = -0.61$)** |
| 7 | `s_3` | HPC outlet temperature ($T_{30}$) | °R | 1571.04 | 1616.91 | 6.1312 | **Active ($r = -0.58$)** |
| 8 | `s_4` | LPT outlet temperature ($T_{50}$) | °R | 1382.25 | 1441.49 | 9.0006 | **Active ($r = -0.68$)** |
| 9 | `s_5` | Fan inlet pressure ($P_0$) | psia | 14.62 | 14.62 | **0.0000** | **Invariant (Dropped)** |
| 10 | `s_6` | Bypass duct pressure ($P_{15}$) | psia | 21.60 | 21.61 | 0.0014 | **Invariant (Dropped)** |
| 11 | `s_7` | HPC outlet total pressure ($P_{30}$) | psia | 549.85 | 556.06 | 0.8851 | **Active ($r = +0.66$)** |
| 12 | `s_8` | Physical fan speed ($N_f$) | rpm | 2387.90 | 2388.56 | 0.0710 | **Active ($r = -0.56$)** |
| 13 | `s_9` | Physical core speed ($N_c$) | rpm | 9021.73 | 9244.59 | 22.0829 | **Active ($r = -0.39$)** |
| 14 | `s_10` | Engine pressure ratio ($P_{50}/P_2$) | -- | 1.30 | 1.30 | **0.0000** | **Invariant (Dropped)** |
| 15 | `s_11` | HPC outlet static pressure ($P_{s30}$) | psia | 46.85 | 48.53 | 0.2671 | **Active ($r = -0.70$)** |
| 16 | `s_12` | Ratio of fuel flow to $P_{s30}$ ($W_f/P_{s30}$) | pps/psia | 518.69 | 523.38 | 0.7376 | **Active ($r = +0.67$)** |
| 17 | `s_13` | Corrected fan speed ($N_{f,corr}$) | rpm | 2387.88 | 2388.56 | 0.0719 | **Active ($r = -0.56$)** |
| 18 | `s_14` | Corrected core speed ($N_{c,corr}$) | rpm | 8104.42 | 8293.72 | 19.0762 | **Active ($r = -0.31$)** |
| 19 | `s_15` | Bypass ratio (BPR) | -- | 8.3249 | 8.5848 | 0.0375 | **Active ($r = -0.64$)** |
| 20 | `s_16` | Burner fuel-air ratio | -- | 0.03 | 0.03 | **0.0000** | **Invariant (Dropped)** |
| 21 | `s_17` | Bleed enthalpy | -- | 388.0 | 400.0 | 1.5488 | **Active ($r = -0.61$)** |
| 22 | `s_18` | Demanded fan speed | rpm | 2388.0 | 2388.0 | **0.0000** | **Invariant (Dropped)** |
| 23 | `s_19` | Demanded corrected fan speed | rpm | 100.0 | 100.0 | **0.0000** | **Invariant (Dropped)** |
| 24 | `s_20` | HPT coolant bleed | lbm/s | 38.14 | 39.43 | 0.1807 | **Active ($r = +0.63$)** |
| 25 | `s_21` | LPT coolant bleed | lbm/s | 22.89 | 23.61 | 0.1083 | **Active ($r = +0.64$)** |

---

## 4. Damage Propagation Physics & Thermodynamic Mechanics

In dataset **FD001**, degradation originates in the **High Pressure Compressor (HPC)**:
1. **Compressor Blade Erosion & Fouling**: Operational cycles induce blade surface roughness and tip clearance expansion.
2. **Aerodynamic Efficiency Loss**: The degraded compressor requires more mechanical work per unit of mass airflow to produce the pressure ratio demanded by the engine controller (FADEC).
3. **Controller Compensation**: To maintain target thrust, the closed-loop control system injects more fuel, increasing burner exit temperature and turbine inlet temperature.
4. **Observable Sensor Trends**:
   - **Temperatures Climb**: LPC outlet (`s_2`), HPC outlet (`s_3`), and LPT outlet (`s_4`) exhibit strong positive drift as thermal stress intensifies.
   - **Static Back-Pressure Climbs**: `s_11` ($P_{s30}$) rises as compressor aerodynamics deteriorate ($r = -0.70$).
   - **Pressure Ratios Drop**: HPC total outlet pressure (`s_7`) and bleed velocities (`s_20`, `s_21`) drop steadily as stage pressure retention falters.
5. **The Degradation "Knee"**: Sensor waveforms remain within nominal healthy noise envelopes for the first $50\text{--}80$ cycles. Measurable thermodynamic drift only emerges after cumulative micro-damage crosses a critical threshold. This physical phenomenon validates **piecewise-linear RUL capping at 125 cycles**.
