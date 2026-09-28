# MAORS: Maternal Adverse Outcome Risk Score System

MAORS is an industry-grade Clinical Decision Support System designed for the early identification of pregnant women at risk of adverse pregnancy outcomes (Preterm birth, Low Birth Weight, and Preeclampsia).

## 🌟 Key Features

1. **Clinical Triage Workstation**: An interactive, dynamic UI for assessing patient risk across 26 clinical factors spanning Demographics, Obstetric History, Vitals, Nutrition, and Socio-Environmental barriers.
2. **Machine Learning Engine**: Hybrid intelligence utilizing deterministic clinical rules combined with an advanced `GradientBoostingClassifier` trained via SMOTE class-balancing and GridSearchCV hyperparameter tuning.
3. **Dynamic Driver Attribution**: Instead of static rules, the ML engine computes real-time patient-specific risk drivers by analyzing feature deviations against population statistics, producing highly personalized "Top Driving Factors".
4. **Cohort Batch Screening**: Process hundreds of patients instantly by uploading an Excel/CSV dataset.
5. **What-If Clinical Simulator**: Model targeted clinical interventions (e.g., controlling blood pressure, correcting anemia) and observe risk reduction probabilistically in real-time.

## 🚀 How to Run (Docker / Industry Standard)

The application is fully containerized for seamless deployment.

### Prerequisites
- [Docker](https://www.docker.com/products/docker-desktop/) installed on your machine.

### Windows
Double-click the `run.bat` file, or open a terminal and run:
```bat
run.bat
```

### macOS / Linux
Open your terminal and execute:
```bash
chmod +x run.sh
./run.sh
```

**Once booted, open your browser and navigate to:** [http://localhost:8080](http://localhost:8080)

## 🧠 System Architecture

### The Frontend (UI/UX)
- A highly polished, single-page application built with modern HTML5, Tailwind CSS, and Lucide icons.
- Features dynamic paging, smooth transitions, real-time SVG medical telemetry gauges, and micro-animations to deliver a premium "industry-grade" feel.

### The Backend (FastAPI)
- **`app.py`**: High-performance RESTful API powering the frontend.
- **`maors_cli.py`**: A robust command-line interface for running audits and processing batch files via terminal.
- **`build_models.py`**: The model training pipeline that builds the Gradient Boosting models and feature statistics.

### The ML Engine (`maors/ml_engine.py`)
- Evaluates clinical risk tiers.
- Computes `Outcome_Preterm`, `Outcome_LBW`, and `Outcome_Preeclampsia` probabilities.
- Identifies features with standard deviation shifts triggering adverse risks dynamically.

## 🛑 Stopping the Server
To shut down the Docker containers, run:
```bash
docker-compose down
```
