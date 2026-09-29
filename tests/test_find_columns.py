"""Filtering uses the same metadata columns as the notebook summary."""

import importlib
import pickle
from unittest.mock import patch

import pandas as pd

from mmpp.core.mmpp import MMPP


def test_find_restores_path_parameters_from_older_database(tmp_path):
    paths = []
    for period in (10, 20):
        path = tmp_path / "sincamplitude_0.05" / f"tperiod_{period}.zarr"
        path.mkdir(parents=True)
        paths.append(str(path))

    # The old cache has only attributes read from the Zarr files.
    with (tmp_path / "mmpy_database.pkl").open("wb") as stream:
        pickle.dump(
            pd.DataFrame({"path": paths, "SincAmplitude": [0.05, 0.05]}), stream
        )

    jobs = MMPP(str(tmp_path))

    assert "tperiod" in jobs.columns
    assert jobs.df["tperiod"].tolist() == [10, 20]
    assert jobs._get_parameter_stats()["tperiod"]["unique"] == 2
    assert jobs.find_paths(tperiod=20) == [paths[1]]
    assert jobs.find_paths(SINCAMPLITUDE=0.05) == paths
    assert jobs.find_paths(sincamplitude=0.05) == paths
    assert jobs.zarr_results[0].attributes["tperiod"] in (10, 20)


def test_find_column_names_ignore_case_and_missing_columns_are_sorted(tmp_path):
    paths = []
    for period in (10, 20):
        path = tmp_path / f"tperiod_{period}.zarr"
        path.mkdir()
        paths.append(str(path))

    with (tmp_path / "mmpy_database.pkl").open("wb") as stream:
        pickle.dump(pd.DataFrame({"path": paths, "Zeta": [1, 2]}), stream)

    jobs = MMPP(str(tmp_path))
    assert jobs.find_paths(TPERIOD=20) == [paths[1]]
    assert jobs.find_paths(zeta=2) == [paths[1]]

    mmpp_module = importlib.import_module("mmpp.core.mmpp")
    with patch.object(mmpp_module.log, "error") as error:
        assert jobs.find_paths(missing=1) == []
    message = error.call_args.args[0]
    assert "Available columns: ['path', 'tperiod', 'Zeta']" in message
