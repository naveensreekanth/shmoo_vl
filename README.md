# SHMOO ML Optimization System
# Shmoo_VL

An automated Machine Learning system for semiconductor Shmoo plot characterization, automated Shmoo pattern classification, arbitrary multi-device ($1 \dots N$) performance ranking, Fmax pass/fail boundary extraction, guardband optimization, and executive PDF report generation.

---

## 🌟 Key Features

- **Automated Shmoo Classification**: Automatically classifies 10+ Shmoo plot patterns (Normal, Wall, Brick Wall, Reverse Speedpath, Floor, Finger, Marginality, Power IR Drop, Hold Time Race) and provides root-cause diagnostic analysis.
- **Dynamic Methodology Engine**: Customized analysis for **MBIST**, **LBIST**, **ATPG**, and **Scan Chain Testing**.
- **Machine Learning Classification**: LightGBM / Gradient Boosting classifier achieving $\ge 95\%$ cross-validated accuracy on tabular VDD vs Frequency test points.
- **RANSAC Boundary Extraction**: Robust linear regression extracting the physical $F_{\text{max}}(\text{GHz}) = a \cdot \text{VDD} + b$ boundary with guardband calculations.
- **Arbitrary Multi-Device ($1 \dots N$) Analysis**: Automatically fits per-device boundaries, ranks devices from High Performer to Low Performer, and sets overall population guardbands governed by the speed-limiting corner device.
- **Interactive Web Dashboard**: Modern glassmorphism dark-theme web interface with drag-and-drop file upload, interactive point tooltips, and view-mode filters.
- **Executive PDF Reports**: Generates professional ReportLab PDF reports containing embedded Shmoo plots, yield tables, failure mode breakdown, multi-device ranking tables, and AI recommendations.

---

## 📁 Repository Structure

```
shmoo_ml_system/
├── run.py                          # Top-level local application entry point
├── Procfile                        # Production web deployment entry point (Gunicorn)
├── Dockerfile                      # Docker container deployment configuration
├── .dockerignore                   # Docker build exclusions
├── requirements.txt                # System dependencies
├── README.md                       # Project documentation
├── models/                         # Local LLM model storage (phi3-mini-q4.gguf)
├── uploads/                        # Session file upload cache
├── reports/                        # PDF report output directory
├── scripts/                        # Dataset generation & verification scripts
├── shmoo_dataset/                  # 10+ Sample specialized Shmoo datasets
├── src/
│   ├── app.py                      # Flask API server & web routes
│   ├── data/
│   │   └── preprocessor.py         # Data validation, normalization & feature engineering
│   ├── ml/
│   │   └── model.py                # LightGBM Classifier, RANSAC extractor & Shmoo Classifier
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

## 🚀 Quickstart (Local)

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

---

## 🌐 Production Cloud Deployment

### Option 1: Deploy on Render / Railway (Recommended)

1. Connect your GitHub repository (`https://github.com/naveensreekanth/shmoo_vl.git`) to **Render** or **Railway**.
2. Select **Web Service**.
3. Set the build and start commands:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --chdir src app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
4. Deploy! Render/Railway automatically detects `Procfile` and serves the app online.

### Option 2: Deploy with Docker

Build and run the container locally or on any cloud server (AWS, GCP, Azure, DigitalOcean):

```bash
# Build the Docker image
docker build -t shmoo-ml-system .

# Run the container on port 5000
docker run -p 5000:5000 shmoo-ml-system
```

---

## ⚙️ Dataset Schema Requirements

The input dataset should contain the following core columns:

| Column | Type | Description |
| :--- | :--- | :--- |
| `Point_ID` | Integer / String | Unique identifier per test point |
| `VDD_V` | Float | Voltage supply level (V) |
| `Frequency_GHz` | Float | Test frequency (GHz) |
| `Test_Result` | String | `PASS` or `FAIL` |
| `Failure_Code` | String | Failure classification code (`FREQ_MARGIN`, `TIMING`, `STUCK_AT`, `POWER_IR_DROP_FAIL`, `HOLD_TIME_VIOLATION`, `LOW_VDD_WALL`, `BRICK_WALL_INIT`, `REVERSE_SPEEDPATH_LEAKAGE`, `FLOOR_LEAKAGE_FAIL`, `FINGER_RESONANCE_COUPLING`, `MARGINALITY_VDDL_FAIL`) |

---

## 📄 License

MIT License. Developed for Semiconductor Testing Optimization.
