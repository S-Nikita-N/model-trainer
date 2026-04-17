from omegaconf import OmegaConf

from model_trainer.hydra_builder import build_item, build_items_dict, build_items_list


class _Echo:
    def __init__(self, value: int = 0, extra: str = ""):
        self.value = value
        self.extra = extra


_TARGET = f"{__name__}._Echo"


def test_build_item_returns_default_when_cfg_is_none():
    assert build_item(None, default="sentinel") == "sentinel"


def test_build_item_returns_default_when_no_target():
    cfg = OmegaConf.create({"value": 1})
    assert build_item(cfg, default=42) == 42


def test_build_item_instantiates_and_forwards_kwargs():
    cfg = OmegaConf.create({"_target_": _TARGET, "value": 7})
    obj = build_item(cfg, extra="hi")
    assert isinstance(obj, _Echo)
    assert obj.value == 7
    assert obj.extra == "hi"


def test_build_items_list_skips_untargeted():
    cfg = OmegaConf.create(
        {
            "a": {"_target_": _TARGET, "value": 1},
            "b": {"just": "metadata"},
            "c": {"_target_": _TARGET, "value": 2},
        }
    )
    items = build_items_list(cfg)
    assert [i.value for i in items] == [1, 2]


def test_build_items_dict_preserves_keys():
    cfg = OmegaConf.create(
        {
            "x": {"_target_": _TARGET, "value": 10},
            "y": {"_target_": _TARGET, "value": 20},
        }
    )
    items = build_items_dict(cfg)
    assert set(items.keys()) == {"x", "y"}
    assert items["x"].value == 10
    assert items["y"].value == 20


def test_build_items_list_returns_default_on_none():
    assert build_items_list(None) == []
    assert build_items_list(None, default=["sentinel"]) == ["sentinel"]
