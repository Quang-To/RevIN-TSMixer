import pandas as pd
from pathlib import Path


def data_loader(filename: str = "data_TSI_v2.csv") -> pd.DataFrame:

    project_root = Path(__file__).resolve().parents[2]

    file_location = project_root / "data" / "raw_data" / filename

    df = pd.read_csv(file_location)

    return df