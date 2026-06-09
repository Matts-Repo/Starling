import numpy as np
import pytest
import yaml

from starling.batch.recipe import Recipe
from starling.io._output import load_maps, save_maps


def make_recipe_dict(tmp_path):
    return {
        "output_dir": str(tmp_path / "out"),
        "device": "cpu",
        "preprocess": {"background": {"method": "mean", "n": 5}, "roi": "auto"},
        "fits": ["moments", "gauss1d"],
        "scans": [
            {"file": "/data/a.h5", "scan_id": "1.1", "alias": "t00"},
            {"file": "/data/a.h5", "scan_id": "2.1", "alias": "t01"},
        ],
    }


def test_recipe_roundtrip(tmp_path):
    d = make_recipe_dict(tmp_path)
    p = tmp_path / "recipe.yaml"
    p.write_text(yaml.safe_dump(d))
    r = Recipe.load(p)
    assert [s.alias for s in r.scans] == ["t00", "t01"]
    assert r.fits == ["moments", "gauss1d"]
    assert r.output_path("t00").endswith("t00.h5")


def test_recipe_hash_ignores_scan_list(tmp_path):
    d1 = make_recipe_dict(tmp_path)
    d2 = make_recipe_dict(tmp_path)
    d2["scans"] = d2["scans"][:1]
    assert Recipe.from_dict(d1).recipe_hash() == Recipe.from_dict(d2).recipe_hash()
    d2["fits"] = ["moments"]
    assert Recipe.from_dict(d1).recipe_hash() != Recipe.from_dict(d2).recipe_hash()


def test_recipe_validation_errors(tmp_path):
    d = make_recipe_dict(tmp_path)
    d["fits"] = ["bogus"]
    with pytest.raises(ValueError, match="bogus"):
        Recipe.from_dict(d)
    d = make_recipe_dict(tmp_path)
    d["scans"] = []
    with pytest.raises(ValueError, match="empty"):
        Recipe.from_dict(d)
    d = make_recipe_dict(tmp_path)
    d["scans"][1]["alias"] = "t00"
    with pytest.raises(ValueError, match="unique"):
        Recipe.from_dict(d)


def test_save_load_maps_roundtrip(tmp_path):
    path = str(tmp_path / "maps.h5")
    maps = {
        "mean": np.random.rand(8, 8, 2),
        "gauss1d": {"params": np.random.rand(8, 8, 6)},
    }
    save_maps(path, maps, scan_params={"scan_shape": [4, 5]}, extra_attrs={"alias": "t00"})
    loaded, attrs = load_maps(path)
    np.testing.assert_array_equal(loaded["mean"], maps["mean"])
    np.testing.assert_array_equal(loaded["gauss1d"]["params"], maps["gauss1d"]["params"])
    assert attrs["alias"] == "t00"
    assert attrs["scan_params"]["scan_shape"] == [4, 5]
