"""Pin exact HF commit SHAs for both backbones into configs/backbone_revisions.yaml.
Audit ref: §5 (unpinned HF revisions, WavLM load warnings); v3 delta 10 (two backbones).
Run once online; the file is then committed and loaders REQUIRE it (no floating revisions).
"""
import sys, yaml
from pathlib import Path
from huggingface_hub import HfApi

BACKBONES = ["facebook/wav2vec2-xls-r-300m", "microsoft/wavlm-large"]

def main():
    api = HfApi()
    out = {}
    for name in BACKBONES:
        sha = api.model_info(name).sha
        out[name] = sha
        print(f"{name} -> {sha}")
    p = Path("configs/backbone_revisions.yaml")
    p.write_text(yaml.safe_dump(out, sort_keys=True), encoding="utf-8")
    print(f"wrote {p}")

if __name__ == "__main__":
    sys.exit(main())
