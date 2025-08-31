
import pandas as pd
from src.student_perf.features import split_xy, NUMERICAL, CATEGORICAL, TARGET

def test_split_xy():
    df = pd.DataFrame({
        "hours_studied": [2.0],
        "attendance_rate": [0.9],
        "past_grades_avg": [75],
        "sleep_hours": [7.0],
        "age": [16],
        "absences": [1],
        "extracurricular": ["yes"],
        "parental_involvement": ["medium"],
        "internet_access": ["yes"],
        "test_prep_course": ["completed"],
        "gender": ["F"],
        "school_support": ["no"],
        "final_grade": [80.0],
    })
    X, y = split_xy(df)
    assert list(X.columns) == NUMERICAL + CATEGORICAL
    assert y.name == TARGET
    assert y.iloc[0] == 80.0
