"""scp_codes -> superclass mapping and the single-label clean criterion."""
import pandas as pd

from igraphecg.data import label_utils as lu


def _fake_scp():
    return pd.DataFrame(
        {"diagnostic": [1, 1, 1, 1, 1, 0],
         "diagnostic_class": ["NORM", "MI", "STTC", "CD", "HYP", None]},
        index=["NORM", "IMI", "NDT", "IRBBB", "LVH", "SR"],
    )


def test_code_to_superclass():
    m = lu.build_code_to_superclass(_fake_scp())
    assert m["NORM"] == "NORM" and m["IMI"] == "MI" and m["IRBBB"] == "CD"
    assert "SR" not in m  # non-diagnostic codes are dropped


def test_parse_scp_codes():
    d = lu.parse_scp_codes("{'NORM': 100.0, 'SR': 0.0}")
    assert d == {"NORM": 100.0, "SR": 0.0}
    assert lu.parse_scp_codes("") == {}


def test_single_label_clean():
    code2super = lu.build_code_to_superclass(_fake_scp())
    db = pd.DataFrame({
        "scp_codes": [
            "{'NORM': 100.0, 'SR': 0.0}",   # NORM only -> clean
            "{'IMI': 100.0, 'NDT': 80.0}",  # MI + STTC -> multi-target, dropped
            "{'IRBBB': 100.0}",             # CD -> clean
            "{'IMI': 100.0, 'LVH': 50.0}",  # MI + HYP -> excluded by default
            "{'SR': 0.0}",                  # no target superclass -> dropped
        ],
    }, index=[1, 2, 3, 4, 5])
    out = lu.assign_single_label(db, code2super, exclude_hyp_overlap=True)
    assert out.loc[1, "is_clean"] and out.loc[1, "label_name"] == "NORM"
    assert not out.loc[2, "is_clean"]
    assert out.loc[3, "is_clean"] and out.loc[3, "label_name"] == "CD"
    assert not out.loc[4, "is_clean"]   # excluded by the HYP overlap
    assert not out.loc[5, "is_clean"]

    # without the HYP-overlap exclusion, record 4 becomes clean(MI)
    out2 = lu.assign_single_label(db, code2super, exclude_hyp_overlap=False)
    assert out2.loc[4, "is_clean"] and out2.loc[4, "label_name"] == "MI"
