import os, sys, time, json
import numpy as np, soundfile as sf, torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample_poly
from math import gcd

BASE = Path(".").resolve()
OUT = BASE / "_embcache_xlsr300m"
SR = 16000
MAX_SEC = float(os.environ.get("MAX_SEC", 12))
BATCH = int(os.environ.get("EXTRACT_BATCH", 12))
WORKERS = int(os.environ.get("EXTRACT_WORKERS", 2))
SHARD = 20000
AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}
ORDER = ["01_ASVspoof5","03_DiffSSD","09_VCTK","02_In-the-Wild","04_ReplayDF","05_AI4T","07_FakeOrReal"]

class AudioDS(Dataset):
    def __init__(self, root, rels):
        self.root, self.rels = str(root), rels
    def __len__(self): return len(self.rels)
    def __getitem__(self, i):
        rel = self.rels[i]
        try:
            x, sr = sf.read(os.path.join(self.root, rel), dtype="float32", always_2d=True)
            x = x.mean(axis=1)
            if sr != SR:
                g = gcd(sr, SR); x = resample_poly(x, SR // g, sr // g).astype(np.float32)
            n = len(x); dur = n / SR
            cap = int(MAX_SEC * SR)
            if n > cap:
                s = (n - cap) // 2; x = x[s:s + cap]
            if len(x) < 800: x = np.pad(x, (0, 800 - len(x)))
            return rel, x, dur
        except Exception as e:
            return rel, None, str(e)

def collate(items): return items

def forward_chunked(model, dev, batch):
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            L = max(len(x) for _, x, _ in batch)
            wav = torch.zeros(len(batch), L)
            mask = torch.zeros(len(batch), L)
            for i, (_, x, _) in enumerate(batch):
                wav[i, :len(x)] = torch.from_numpy(x); mask[i, :len(x)] = 1
            out = model(wav.to(dev), attention_mask=mask.to(dev), output_hidden_states=True)
            hs = torch.stack(out.hidden_states, 1)          # B,25,T,1024
            fl = model._get_feat_extract_output_lengths(mask.sum(-1).to(dev)).long()
            emb = torch.stack([hs[i, :, :fl[i]].mean(1) for i in range(len(batch))])
            return emb.half().cpu().numpy()                  # B,25,1024
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(batch) == 1: raise
        h = len(batch) // 2
        return np.concatenate([forward_chunked(model, dev, batch[:h]),
                               forward_chunked(model, dev, batch[h:])])

def make_loader(ds, workers):
    return DataLoader(ds, batch_size=BATCH, num_workers=workers, collate_fn=collate,
                      pin_memory=True, persistent_workers=False,
                      prefetch_factor=(2 if workers > 0 else None))

def process(model, dev, corpus, stats):
    cdir = OUT / corpus.name; cdir.mkdir(parents=True, exist_ok=True)
    done_p, skip_p = cdir / "_done.txt", cdir / "_skipped.txt"
    done = set(done_p.read_text().splitlines()) if done_p.exists() else set()
    inv = corpus / "_INVENTORY.tsv"
    rels = [l.split("\t")[0] for l in inv.read_text(encoding="utf-8").splitlines()
            if Path(l.split("\t")[0]).suffix.lower() in AUDIO_EXT]
    todo = [r for r in rels if r not in done]
    print(f"[{corpus.name}] audio={len(rels)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo: print(f"[{corpus.name}] EMBEDDINGS COMPLETE"); return
    ds = AudioDS(corpus, todo)
    try:
        loader = make_loader(ds, WORKERS); it = iter(loader); first = next(it)
    except Exception as e:
        print(f"  workers={WORKERS} failed ({type(e).__name__}: {e}) -> falling back to num_workers=0", flush=True)
        loader = make_loader(ds, 0); it = iter(loader); first = next(it)
    shard_i = len(list(cdir.glob("shard_*.npz")))
    P, E, D = [], [], []
    def flush():
        nonlocal shard_i, P, E, D
        if not P: return
        np.savez(cdir / f"shard_{shard_i:04d}.npz", paths=np.array(P),
                 emb=np.stack(E), dur=np.array(D, dtype=np.float32))
        with open(done_p, "a", encoding="utf-8") as f: f.write("\n".join(P) + "\n")
        shard_i += 1; P, E, D = [], [], []
    batch = first
    while True:
        good = [(r, x, d) for r, x, d in batch if x is not None]
        bad = [(r, d) for r, x, d in batch if x is None]
        if bad:
            with open(skip_p, "a", encoding="utf-8") as f:
                f.writelines(f"{r}\t{d}\n" for r, d in bad)
            with open(done_p, "a", encoding="utf-8") as f:
                f.writelines(r + "\n" for r, _ in bad)
        if good:
            emb = forward_chunked(model, dev, good)
            for j, (r, _, d) in enumerate(good):
                P.append(r); E.append(emb[j]); D.append(d)
            stats["n"] += len(good); stats["sec"] += sum(d for _, _, d in good)
        if len(P) >= SHARD: flush()
        if stats["n"] >= 2000 and not stats["cal"]:
            stats["cal"] = True
            xrt = stats["sec"] / max(time.time() - stats["t0"], 1)
            print(f"  CALIBRATION: {xrt:.0f}x realtime, avg {stats['sec']/stats['n']:.1f}s/file "
                  f"-> this corpus ETA ~{(len(todo)-stats['n'])*(stats['sec']/stats['n'])/xrt/3600:.1f} h", flush=True)
        if stats["n"] % 5000 < BATCH:
            el = time.time() - stats["t0"]
            print(f"  ... {stats['n']} files, {stats['sec']/3600:.1f} audio-h, "
                  f"{stats['sec']/max(el,1):.0f}x RT, elapsed {el/3600:.2f} h", flush=True)
        try: batch = next(it)
        except StopIteration: break
    flush()
    print(f"[{corpus.name}] EMBEDDINGS COMPLETE (skipped: "
          f"{len(skip_p.read_text().splitlines()) if skip_p.exists() else 0})", flush=True)

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA not available"
    free, _ = torch.cuda.mem_get_info()
    assert free > 3.5e9, f"only {free/1e9:.1f} GB VRAM free — close the other GPU job first"
    dev = "cuda"
    from transformers import Wav2Vec2Model
    print("loading facebook/wav2vec2-xls-r-300m (first run downloads ~1.2 GB)...", flush=True)
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m", use_safetensors=True,
                                          torch_dtype=torch.float16).to(dev).eval()
    torch.set_num_threads(4)
    targets = sys.argv[1:] or ORDER
    targets = ORDER if targets == ["all"] else targets
    for name in targets:
        c = BASE / name
        if not (c / "_INVENTORY.tsv").exists():
            print(f"[{name}] no _INVENTORY.tsv — run checksums first; skip"); continue
        stats = {"n": 0, "sec": 0.0, "t0": time.time(), "cal": False}
        process(model, dev, c, stats)
    print("ALL REQUESTED CORPORA DONE")
