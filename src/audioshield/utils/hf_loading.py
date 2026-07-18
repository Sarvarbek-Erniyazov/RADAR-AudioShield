"""Strict, revision-pinned backbone loading.
Audit ref: §5 ("WavLM load warnings unexamined", unpinned revisions).
Integration: the existing backbone module must route from_pretrained through
load_backbone(); tests/test_hf_loading.py enforces the pure-logic parts offline.
"""
from __future__ import annotations
from pathlib import Path
import yaml

# Keys that are EXPECTED to be unexpected when loading a pretraining checkpoint
# into Wav2Vec2Model/WavLMModel (pretraining heads absent from the encoder class).
KNOWN_PRETRAIN_HEAD_KEYS = {
    "quantizer.codevectors", "quantizer.weight_proj.weight", "quantizer.weight_proj.bias",
    "project_q.weight", "project_q.bias", "project_hid.weight", "project_hid.bias",
}

class BackboneLoadError(RuntimeError):
    pass

def validate_load_report(unexpected_keys, missing_keys) -> None:
    """Pure function: raise unless deviations are exactly the known-benign set."""
    unknown = set(unexpected_keys) - KNOWN_PRETRAIN_HEAD_KEYS
    if unknown:
        raise BackboneLoadError(f"Unexpected keys outside known pretrain-head set: {sorted(unknown)}")
    if missing_keys:
        raise BackboneLoadError(f"Missing keys (checkpoint/architecture mismatch): {sorted(missing_keys)[:10]}")

def get_pinned_revision(model_name: str, revisions_path: str = "configs/backbone_revisions.yaml") -> str:
    p = Path(revisions_path)
    if not p.exists():
        raise BackboneLoadError(
            f"{revisions_path} missing — run scripts/pin_backbone_revisions.py once (online) and commit it. "
            "Floating (unpinned) backbone revisions are forbidden (audit §5)."
        )
    revs = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if model_name not in revs:
        raise BackboneLoadError(f"No pinned revision for {model_name} in {revisions_path}.")
    return revs[model_name]

def load_backbone(model_name: str, torch_dtype=None, revisions_path: str = "configs/backbone_revisions.yaml",
                  local_files_only: bool = False):
    """Load with pinned revision + strict load-report validation.
    local_files_only defaults to False (unchanged prior behavior, needed by the
    @pytest.mark.network real-download test); callers running fully offline
    (e.g. ssl_backbone.py, which sets HF_HUB_OFFLINE=1) pass local_files_only=True
    explicitly."""
    from transformers import AutoModel
    revision = get_pinned_revision(model_name, revisions_path)
    # No use_safetensors=True: microsoft/wavlm-large and facebook/wav2vec2-xls-r-300m
    # publish pytorch_model.bin only, so forcing safetensors can never succeed for
    # these backbones (gate run #2, 2026-07-18). Let transformers' default apply
    # (prefer safetensors, fall back to .bin) -- the revision pin above already fixes
    # the exact bytes loaded, and modern transformers loads .bin via weights_only
    # torch loading, so this isn't an integrity/safety regression.
    model, loading_info = AutoModel.from_pretrained(
        model_name, revision=revision,
        torch_dtype=torch_dtype, output_loading_info=True,
        local_files_only=local_files_only,
    )
    validate_load_report(loading_info.get("unexpected_keys", []),
                         loading_info.get("missing_keys", []))
    model.config._audioshield_pinned_revision = revision
    return model
