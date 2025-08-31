
from typing import List, Tuple
import pandas as pd

NUMERICAL = [
    "hours_studied",
    "attendance_rate",
    "past_grades_avg",
    "sleep_hours",
    "age",
    "absences",
]

CATEGORICAL = [
    "extracurricular",
    "parental_involvement",
    "internet_access",
    "test_prep_course",
    "gender",
    "school_support",
]

TARGET = "final_grade"

ALL_COLUMNS: List[str] = NUMERICAL + CATEGORICAL + [TARGET]

def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[NUMERICAL + CATEGORICAL].copy()
    y = df[TARGET].copy()
    return X, y
