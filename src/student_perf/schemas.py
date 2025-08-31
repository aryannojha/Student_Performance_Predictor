
from pydantic import BaseModel, Field
from typing import Literal

class Features(BaseModel):
    hours_studied: float = Field(..., ge=0, le=24)
    attendance_rate: float = Field(..., ge=0, le=1)
    past_grades_avg: float = Field(..., ge=0, le=100)
    extracurricular: Literal["yes", "no"]
    parental_involvement: Literal["low", "medium", "high"]
    sleep_hours: float = Field(..., ge=0, le=24)
    internet_access: Literal["yes", "no"]
    test_prep_course: Literal["none", "completed", "ongoing"]
    gender: Literal["M", "F"]
    age: int = Field(..., ge=5, le=100)
    school_support: Literal["yes", "no"]
    absences: int = Field(..., ge=0, le=365)
