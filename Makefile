
.PHONY: train api ui test

train:
	python -m src.student_perf.train --data src/data/students.csv --out artifacts/model.joblib

api:
	uvicorn src.student_perf.api:app --reload --port 8000

ui:
	streamlit run src/student_perf/app_streamlit.py

test:
	pytest -q
