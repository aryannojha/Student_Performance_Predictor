
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-train model at build (optional). Comment if you prefer runtime training.
RUN mkdir -p artifacts &&     python -m src.student_perf.train --data src/data/students.csv --out artifacts/model.joblib || true

EXPOSE 8000
CMD ["uvicorn", "src.student_perf.api:app", "--host", "0.0.0.0", "--port", "8000"]
