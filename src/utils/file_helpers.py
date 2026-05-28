import json
import csv
import pickle
import pandas as pd
from pathlib import Path
from utils import get_logger


logger = get_logger('util')


def pickle_writer(file_path: str, data: list):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def csv_writer(file_path: str, data: list):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(data).to_csv(file_path, index=False, encoding="utf-8")


def file_writer(file_path: str, data: dict):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        string_data = [str(item) for item in data]
        f.write("\n".join(string_data))


def load_csv(file_path: str) -> list:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except FileNotFoundError:
        logger(f"The file doesn't be found.: {file_path}")
        return []
    except Exception as e:
        logger(f"error: {e}")
        return []


def load_json(file_path: str) -> dict:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger(f"The file doesn't be found.: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        logger(f"JSON parse failed.: {e}")
        return {}
