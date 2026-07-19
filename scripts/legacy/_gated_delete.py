import os, shutil, sys
from pathlib import Path
import numpy as np

BASE = Path(".").resolve()
TARGETS = ["01_ASVspoof5", "03_DiffSSD", "09_VCTK"]
AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}
CONFIRM = os.environ.get("CONFIRM", "") == "yes"

def gate(name):
    c, cache = BASE / name, BASE / "_embcache_xlsr300m" / name
    if not (c / "_SHA256.txt").exists() or (c / "_SHA256.txt").stat().st_size == 0:
        return None, "GATE1 FAIL: no _SHA256.txt"
    inv = (c / "_INVENTORY.tsv").read_text(encoding="utf-8").splitlines()
    n_audio = sum(1 for l in inv if Path(l.split("\t")[0]).suffix.lower() in AUDIO_EXT)
    done_p, skip_p = cache / "_done.txt", cache / "_skipped.txt"
    if not done_p.exists():
        return None, "GATE2 FAIL: no embedding cache"
    n_done = len(set(done_p.read_text().splitlines()))
    n_skip = len(skip_p.read_text().splitlines()) if skip_p.exists() else 0
    n_emb = sum(len(np.load(s)["paths"]) for s in sorted(cache.glob("shard_*.npz")))
    if n_done < n_audio:            return None, f"GATE2 FAIL: done {n_done} < audio {n_audio}"
    if n_emb != n_done - n_skip:    return None, f"GATE2 FAIL: shards {n_emb} != done-skip {n_done-n_skip}"
    gb = sum(int(l.split("\t")[1]) for l in inv) / 2**30
    return gb, f"gates OK (audio={n_audio}, embedded={n_emb}, skipped={n_skip}, {gb:.1f} GiB)"

total = 0.0
for name in TARGETS:
    gb, msg = gate(name)
    print(f"[{name}] {msg}")
    if gb is None:
        print("  -> NOT deleting this corpus."); continue
    total += gb
    if CONFIRM:
        keep = BASE / "_manifests_preserved" / name
        keep.mkdir(parents=True, exist_ok=True)
        for f in ("_SHA256.txt", "_INVENTORY.tsv"):
            shutil.copy2(BASE / name / f, keep / f)
        shutil.rmtree(BASE / name)
        print(f"  -> DELETED (manifests preserved in _manifests_preserved/{name}/)")
    else:
        print("  -> dry-run: would delete")
print(f"\n{'freed' if CONFIRM else 'would free'}: ~{total:.1f} GiB")
if not CONFIRM: print("to execute for real:  CONFIRM=yes python _gated_delete.py")
