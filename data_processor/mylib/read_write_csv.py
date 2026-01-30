from datetime import datetime
from pathlib import Path
import pandas as pd

DATA_DIR = Path('..') / 'data'
RAW_DIR = Path('raw')
CLEANED_DIR = Path('cleaned')


BASIC_FILE = 'basic'
INTAKE_FORM_FILE = 'intake_form'
OVERALL_FILE = 'grades'
FINAL_ASSESSMENT_FILE = 'final_assessment'
STATUS_FILE = 'status'


def read_csv(filename: str) -> pd.DataFrame | None:
    dir_path = DATA_DIR / RAW_DIR

    return pd.read_csv(dir_path / f'{filename}.csv')


def write_csv(df: pd.DataFrame, filename: str):
    dir_path = DATA_DIR / CLEANED_DIR
    dir_path.mkdir(parents=True, exist_ok=True)

    df.to_csv(dir_path / f'{filename}.csv', index=False)

    return filename
