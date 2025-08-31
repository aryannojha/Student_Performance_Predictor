
# 🎓 Student Performance Predictor

A clean, production-ready template to **train** a machine learning model and **predict** student performance.
Includes a FastAPI service, a Streamlit UI, tests, Dockerfile, CI, and a small sample dataset so it runs out-of-the-box.

## Features
- End-to-end `scikit-learn` pipeline with preprocessing.
- Regression target: `final_grade` (0–100).
- REST API (`FastAPI`) and a simple UI (Streamlit).
- Unit tests with `pytest`.
- Docker support and GitHub Actions CI.

## Project Structure
```
student-performance-predictor/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── Dockerfile
├── Makefile
├── .github/workflows/ci.yml
├── src/
│   ├── data/students.csv
│   └── student_perf/
│       ├── __init__.py
│       ├── features.py
│       ├── model.py
│       ├── train.py
│       ├── predict.py
│       ├── api.py
│       ├── app_streamlit.py
│       ├── schemas.py
│       └── utils.py
└── tests/
    ├── test_features.py
    └── test_model.py
```

## Quickstart
```bash
# 1) Create & activate a virtualenv (optional but recommended)
python -m venv .venv && source .venv/bin/activate  # (Linux/Mac)
# Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Train
python -m src.student_perf.train --data src/data/students.csv --out artifacts/model.joblib

# 4) Predict (CLI)
python -m src.student_perf.predict --model artifacts/model.joblib --json example.json

# 5) Run API
uvicorn src.student_perf.api:app --reload --port 8000

# 6) Run Streamlit UI
streamlit run src/student_perf/app_streamlit.py
```

## Example Input (JSON)
```json
{
  "hours_studied": 3.5,
  "attendance_rate": 0.95,
  "past_grades_avg": 78,
  "extracurricular": "yes",
  "parental_involvement": "medium",
  "sleep_hours": 7.0,
  "internet_access": "yes",
  "test_prep_course": "completed",
  "gender": "F",
  "age": 16,
  "school_support": "no",
  "absences": 2
}
```

## Docker
```bash
# Build
docker build -t student-perf .
# Run
docker run -p 8000:8000 student-perf
```

## Tests
```bash
pytest -q
```

## Notes
- The sample dataset is synthetic and small, but the pipeline is real. Replace `src/data/students.csv` with your own data to improve performance.
- For classification (pass/fail), you can adapt the pipeline in `model.py` easily.
