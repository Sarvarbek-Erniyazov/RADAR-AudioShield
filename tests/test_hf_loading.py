import pytest
from audioshield.utils.hf_loading import (
    validate_load_report, get_pinned_revision, BackboneLoadError, KNOWN_PRETRAIN_HEAD_KEYS)

def test_known_pretrain_heads_pass():
    validate_load_report(list(KNOWN_PRETRAIN_HEAD_KEYS), [])

def test_unknown_unexpected_key_raises():
    with pytest.raises(BackboneLoadError):
        validate_load_report(["encoder.layers.0.attention.q_proj.weight"], [])

def test_missing_keys_raise():
    with pytest.raises(BackboneLoadError):
        validate_load_report([], ["encoder.layers.3.final_layer_norm.weight"])

def test_unpinned_revision_refused(tmp_path):
    with pytest.raises(BackboneLoadError):
        get_pinned_revision("facebook/wav2vec2-xls-r-300m", str(tmp_path / "nope.yaml"))

@pytest.mark.network
def test_real_backbone_loads_clean():
    """Live-Hub load for both pinned backbones. Run #2 (2026-07-18) found
    load_backbone() forced use_safetensors=True while neither
    microsoft/wavlm-large nor facebook/wav2vec2-xls-r-300m publishes
    safetensors -- this test only covered one backbone and, being
    network-marked, had never actually been executed on any machine to date,
    so it caught nothing. Covers both pinned backbones now."""
    from audioshield.utils.hf_loading import load_backbone, get_pinned_revision
    for model_name in ("facebook/wav2vec2-xls-r-300m", "microsoft/wavlm-large"):
        model = load_backbone(model_name)
        assert model is not None
        assert model.config._audioshield_pinned_revision == get_pinned_revision(model_name)
