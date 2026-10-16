import json

import numpy as np
import pytest

from artifacts import load_json, load_q_table, save_json, save_q_table


def test_q_table_round_trip_preserves_shape_dtype_and_values(tmp_path):
    original = np.arange(64, dtype=np.float32).reshape(16, 4)
    file_path = tmp_path / "nested" / "checkpoints" / "q_table.npy"

    save_q_table(original, file_path)
    loaded = load_q_table(file_path)

    assert file_path.exists()
    assert loaded.shape == original.shape
    assert loaded.dtype == original.dtype
    np.testing.assert_array_equal(loaded, original)


def test_save_q_table_uses_the_requested_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    q_table = np.zeros((16, 4), dtype=float)
    requested_path = tmp_path / "outputs" / "checkpoints" / "custom.npy"

    save_q_table(q_table, requested_path)

    assert requested_path.exists()
    assert not (tmp_path / "q_table.npy").exists()


def test_load_q_table_raises_for_a_missing_file(tmp_path):
    missing_path = tmp_path / "missing.npy"

    with pytest.raises(FileNotFoundError):
        load_q_table(missing_path)


def test_save_json_creates_readable_utf8_file_and_preserves_data(tmp_path):
    data = {
        "environment": {
            "name": "冰湖",
            "is_slippery": False,
        },
        "training": {
            "episodes": 5000,
            "alpha": 0.1,
        },
    }
    file_path = tmp_path / "nested" / "config.json"

    save_json(data, file_path)

    text = file_path.read_text(encoding="utf-8")
    loaded = json.loads(text)
    assert file_path.exists()
    assert loaded == data
    assert "冰湖" in text
    assert "\n    " in text

    assert load_json(file_path) == data
