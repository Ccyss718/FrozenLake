import numpy as np
from pathlib import Path
import json

def save_q_table(q_table, file_path):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.save(path, q_table)

def load_q_table(file_path):
    path = Path(file_path)

    q_table = np.load(path, allow_pickle = False)

    return q_table

def save_json(data, file_path):
    path = Path(file_path)
    path.parent.mkdir(parents=True,exist_ok=True)

    with path.open(mode='w',encoding="utf-8") as file:
        json.dump(data,file,indent=4,ensure_ascii=False)

def load_json(file_path):
    path = Path(file_path)

    with path.open(mode="r", encoding="utf-8") as file:
        return json.load(file)
