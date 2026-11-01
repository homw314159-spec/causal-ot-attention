from pathlib import Path

import pytest

from geoattn_train.config import load_config

CONFIGS = Path(__file__).resolve().parents[3] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIGS.rglob("*.yaml")), ids=lambda p: p.name)
def test_all_shipped_configs_load(path: Path) -> None:
    """Every YAML under configs/ is an experiment config, or an arm table (top-level
    ``arms``, e.g. configs/exp0002/arms.yaml) whose every arm loads on the stage's base
    config(s) next to it, or a table of sections that each name their base config
    (configs/exp0003/arms.yaml), so a mistyped override fails here rather than mid-experiment."""
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw and all(isinstance(s, dict) and {"config", "arms"} <= set(s) for s in raw.values()):
        _check_sections(raw)
        return
    if "arms" not in raw:
        load_config(path)
        return
    bases = [
        b for b in (path.with_name("screen.yaml"), path.with_name("confirm.yaml")) if b.exists()
    ]
    assert bases, "an arm table needs a stage config next to it"
    for base in bases:
        for name, overrides in raw["arms"].items():
            cfg = load_config(base, overrides)
            assert cfg.name, (base.name, name)
        load_config(base, raw["ot"])


def _check_sections(raw: dict) -> None:
    """Each arm on its section's base config, with every part's overrides (if any)."""
    for section in raw.values():
        base = CONFIGS.parent / section["config"]
        parts = [p["overrides"] for p in section.get("parts", {}).values()] or [[]]
        for name, overrides in section["arms"].items():
            for extra in parts:
                cfg = load_config(base, [*overrides, *extra])
                assert cfg.name, (base.name, name)


def test_overrides_and_float_coercion() -> None:
    cfg = load_config(
        overrides=["train.lr=3e-4", "attention.normalizer.name=sinkhorn", "train.grad_clip=null"]
    )
    assert cfg.train.lr == pytest.approx(3e-4) and isinstance(cfg.train.lr, float)
    assert cfg.attention.normalizer.name == "sinkhorn"
    assert cfg.train.grad_clip is None


def test_list_fields() -> None:
    cfg = load_config(overrides=["train.lr_milestones=[35, 41]", "tracking.tags=[a, b]"])
    assert cfg.train.lr_milestones == [35, 41]
    assert cfg.tracking.tags == ["a", "b"]
    with pytest.raises(TypeError, match=r"train\.lr_milestones\[1\]"):
        load_config(overrides=["train.lr_milestones=[35, x]"])


def test_unknown_key_is_an_error() -> None:
    with pytest.raises(KeyError, match="n_iter"):
        load_config(overrides=["attention.normalizer.n_iter=5"])


def test_type_errors_are_reported_with_path() -> None:
    with pytest.raises(TypeError, match=r"model\.depth"):
        load_config(overrides=["model.depth=deep"])


def test_rope_scaling_defaults_to_the_lm_training_length() -> None:
    from geoattn_core import RoPE
    from geoattn_train.config import AttentionConfig
    from geoattn_train.factory import build_pe

    pe = build_pe(AttentionConfig(pe="rope", rope_scaling="yarn"), 16, train_len=512)
    assert isinstance(pe, RoPE) and pe.scaling == "yarn" and pe.train_len == 512
    pe = build_pe(AttentionConfig(pe="rope", rope_scaling="pi", rope_train_len=256), 16, 4, 512)
    assert isinstance(pe, RoPE) and pe.train_len == 256


def test_pe_bias_composes_with_rope() -> None:
    from geoattn_core import RoPE
    from geoattn_core.pe import ComposedPE, LogDistanceBias
    from geoattn_train.config import AttentionConfig
    from geoattn_train.factory import build_pe

    pe = build_pe(AttentionConfig(pe="rope", pe_bias="logbias"), 16, num_heads=4)
    assert isinstance(pe, ComposedPE)
    assert isinstance(pe.rotary, RoPE) and isinstance(pe.additive, LogDistanceBias)
    assert isinstance(build_pe(AttentionConfig(pe="rope"), 16, num_heads=4), RoPE)


def test_content_biases_need_a_causal_model_and_the_width() -> None:
    from geoattn_core.pe import ForgetGate, GrapeAP
    from geoattn_train.config import AttentionConfig
    from geoattn_train.factory import build_pe

    with pytest.raises(ValueError, match="causal"):
        build_pe(AttentionConfig(pe="grape_a"), 16, num_heads=4)  # e.g. a ViT
    with pytest.raises(ValueError, match="dim"):
        build_pe(AttentionConfig(pe="fox"), 16, num_heads=4, causal=True)
    ap = build_pe(AttentionConfig(pe="grape_ap", grape_p_dim=4), 16, 4, dim=64, causal=True)
    assert isinstance(ap, GrapeAP) and ap.p_dim == 4
    assert isinstance(build_pe(AttentionConfig(pe="fox"), 16, 4, dim=64, causal=True), ForgetGate)


def test_gated_logbias_composes_with_rope() -> None:
    from geoattn_core import RoPE
    from geoattn_core.pe import ComposedPE, GatedLogDistanceBias
    from geoattn_train.config import AttentionConfig
    from geoattn_train.factory import build_pe

    cfg = AttentionConfig(pe="rope", pe_bias="gated_logbias")
    pe = build_pe(cfg, 16, num_heads=4, dim=64)
    assert isinstance(pe, ComposedPE) and isinstance(pe.rotary, RoPE)
    assert isinstance(pe.additive, GatedLogDistanceBias)
    with pytest.raises(ValueError, match="dim"):
        build_pe(cfg, 16, num_heads=4)


def test_clock_logbias_is_causal_only() -> None:
    from geoattn_core.pe import ClockLogDistanceBias, ComposedPE
    from geoattn_train.config import AttentionConfig
    from geoattn_train.factory import build_pe

    cfg = AttentionConfig(pe="rope", pe_bias="clock_logbias")
    with pytest.raises(ValueError, match="causal"):
        build_pe(cfg, 16, num_heads=4, dim=64)
    pe = build_pe(cfg, 16, num_heads=4, dim=64, causal=True)
    assert isinstance(pe, ComposedPE) and isinstance(pe.additive, ClockLogDistanceBias)
