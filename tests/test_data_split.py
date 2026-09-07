"""Official fold split: train=1-8, val=9, test=10, with no leakage across splits."""
import numpy as np
import pandas as pd

from igraphecg.data.dataset import split_indices
from igraphecg.data.ptbxl_loader import assign_official_splits


def test_assign_official_splits():
    df = pd.DataFrame({"strat_fold": [1, 5, 8, 9, 10]})
    out = assign_official_splits(df)
    assert list(out["split"]) == ["train", "train", "train", "val", "test"]


def test_split_indices_disjoint():
    fold = np.array([1, 2, 8, 9, 10, 10, 9, 3])
    idx = split_indices(fold)
    all_idx = np.concatenate([idx["train"], idx["val"], idx["test"]])
    # mutually disjoint, and their union is the whole set
    assert len(np.unique(all_idx)) == len(all_idx) == len(fold)
    # test comes only from fold 10
    assert set(fold[idx["test"]].tolist()) == {10}
    assert set(fold[idx["val"]].tolist()) == {9}
    assert set(fold[idx["train"]].tolist()) <= set(range(1, 9))


def test_fold10_not_in_train():
    fold = np.array([10, 9, 8, 7, 1, 10])
    idx = split_indices(fold)
    assert not set(idx["test"]).intersection(idx["train"])
    assert not set(idx["test"]).intersection(idx["val"])
