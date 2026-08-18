# SHMOO ML Optimization System
# Shmoo_VL

An automated, local Machine Learning system for semiconductor Shmoo plot characterization, arbitrary multi-device ($1 \dots N$) performance ranking, Fmax pass/fail boundary extraction, guardband optimization, and executive PDF report generation.

---

## 🌟 Key Features

- **100% Offline & Local**: Fully self-contained pipeline; zero cloud services or external API dependencies.
- **Machine Learning Classification**: LightGBM / Gradient Boosting classifier achieving $\ge 95\%$ cross-validated accuracy on tabular VDD vs Frequency test points.
- **RANSAC Boundary Extraction**: Robust linear regression extracting the physical $F_{\text{max}}(\text{GHz}) = a \cdot \text{VDD} + b$ boundary with guardband calculations.
- **Arbitrary Multi-Device ($1 \dots N$) Analysis**: Automatically fits per-device boundaries, ranks devices from High Performer to Low Performer, and sets overall population guardbands governed by the speed-limiting corner device.
- **Universal Schema Support**: Works seamlessly across ATPG Scan Datasets, M-BIST Memory Datasets, and general silicon characterization logs.
- **Interactive Web Dashboard**: Modern glassmorphism dark-theme web interface with drag-and-drop file upload.
- **Executive PDF Reports**: Generates professional ReportLab PDF reports containing embedded Shmoo plots, yield tables, failure mode breakdown, multi-device ranking tables, and AI recommendations.

---

## 📁 Repository Structure

```
shmoo_ml_system/
├── run.py                          # Top-level application entry point
├── requirements.txt                # System dependencies
├── README.md                       # Project documentation
├── models/                         # Local LLM model storage (phi3-mini-q4.gguf)
├── uploads/                        # Session file upload cache
├── reports/                        # PDF report output directory
├── shmoo_dataset/                  # Sample Scan and Memory datasets
├── src/
│   ├── app.py                      # Flask API server & web routes
│   ├── data/
│   │   └── preprocessor.py         # Data validation, normalization & feature engineering
│   ├── ml/
│   │   └── model.py                # LightGBM Classifier + RANSAC boundary extractor
│   ├── report/
│   │   ├── generator.py            # ReportLab PDF report builder
│   │   └── plot_builder.py         # Matplotlib offline Shmoo plot generator
│   ├── text/
│   │   ├── template_engine.py      # Rule-based narrative engine
│   │   └── llm_engine.py           # Offline Phi-3 Mini LLM narrative engine
│   └── templates/
│       └── index.html              # Interactive Web UI template
```

---

## 🚀 Quickstart

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/naveensreekanth/shmoo_vl.git
cd shmoo_vl
pip install -r requirements.txt
```

### 2. Running the Web Application

Start the server:

```bash
python run.py
```

Open your browser and navigate to:
[http://localhost:5000](http://localhost:5000)

### 3. Workflow
1. **Upload Dataset**: Drag & drop any `.csv` or `.xlsx` Shmoo dataset (e.g. from `shmoo_dataset/`).
2. **View Analysis**: Inspect the cross-validation score, recommended operating point ($VDD_{\text{rec}}, F_{\text{rec}}$), guardbands, multi-device ranking, and interactive Shmoo plot.
3. **Generate Report**: Choose between **Option B (Template)** or **Option A (Local LLM)** narrative mode, and click **Download Executive Report (PDF)**.

---

## ⚙️ Dataset Schema Requirements

The input dataset should contain the following core columns:

| Column | Type | Description |
| :--- | :--- | :--- |
| `Point_ID` | Integer / String | Unique identifier per test point |
| `VDD_V` | Float | Voltage supply level (V) |
| `Frequency_GHz` | Float | Test frequency (GHz) |
| `Test_Result` | String | `PASS` or `FAIL` |
| `Failure_Code` | String | Failure classification code (`FREQ_MARGIN`, `TIMING`, `STUCK_AT`, `COUPLING_FAULT`, `RETENTION_FAULT`, `ADDRESS_DECODE_FAULT`, `NA`) |

*Optional columns*: `Lot_ID`, `Wafer_ID`, `Die_ID`, `Pattern_ID`, `March_Algorithm`, `Memory_Instance`, `Temperature_C`, `Current_mA`, `Timing_ns`, `Leakage_mA`, `Margin_GHz`.

---

## 📄 License

MIT License. Developed for Semiconductor Testing Optimization.
