
import pandas as pd
from src.student_perf.model import train_eval

def test_train_eval_runs():
    df = pd.read_csv("src/data/students.csv")
    model, r2 = train_eval(df)
    assert hasattr(model, "predict")
