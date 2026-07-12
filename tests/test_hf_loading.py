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
    from audioshield.utils.hf_loading import load_backbone
    load_backbone("facebook/wav2vec2-xls-r-300m")
