import hashlib, json, os, re, subprocess, sys, time
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

BASE = Path(".").resolve(); DEST = BASE / "10_MLAAD"; REPO = "mueller91/MLAAD"
AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}
PRIORITY = ["en","de","fr","es","it","pl","ru","uk"]
MIN_FREE_GB = 30

def free_gb():
    import shutil; return shutil.disk_usage(BASE).free / 2**30

def sha256f(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def robust_download(rev, prefix):
    for att in range(1, 9):
        try:
            snapshot_download(REPO, repo_type="dataset", revision=rev, local_dir=str(DEST),
                              allow_patterns=[prefix + "**"], max_workers=3)
            return
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"    download attempt {att}/8 failed: {type(e).__name__}: {str(e)[:140]}", flush=True)
            if att == 8: raise
            time.sleep(min(60 * att, 300))   # collisions/net blips clear in seconds; resume skips done files

def main():
    assert os.environ.get("HF_TOKEN"), "export HF_TOKEN first"
    DEST.mkdir(exist_ok=True)
    api = HfApi(); rev = api.dataset_info(REPO).sha
    (DEST / "_SOURCE_REVISION.json").write_text(json.dumps({"repo": REPO, "revision": rev,
        "pinned": time.strftime("%F %T")}, indent=2))
    print(f"repo revision pinned: {rev}", flush=True)
    files = api.list_repo_files(REPO, repo_type="dataset", revision=rev)
    langs = {}
    for f in files:
        m = re.search(r"(?:^|/)fake/([^/]+)/", f)
        if m: langs.setdefault(m.group(1), []).append(f)
    order = [l for l in PRIORITY if l in langs] + sorted(l for l in langs if l not in PRIORITY)
    master = DEST / "_MASTER_MANIFEST.tsv"
    seen = set(l.split("\t")[0] for l in master.read_text(encoding="utf-8").splitlines()) if master.exists() else set()
    marker_dir = DEST / "_lang_done"; marker_dir.mkdir(exist_ok=True)

    for lang in order:
        if (marker_dir / lang).exists():
            print(f"[{lang}] done — skip", flush=True); continue
        if free_gb() < MIN_FREE_GB:
            print(f"STOP: {free_gb():.0f} GB free < {MIN_FREE_GB}"); sys.exit(1)
        print(f"\n=== [{lang}] {len(langs[lang])} repo files | free {free_gb():.0f} GB ===", flush=True)
        prefix = re.match(r"(.*?fake/" + re.escape(lang) + r"/)", langs[lang][0]).group(1)
        robust_download(rev, prefix)
        new_audio = [p for p in DEST.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT
                     and f"fake/{lang}/" in p.as_posix() and ".cache" not in p.as_posix()]
        print(f"[{lang}] audio on disk: {len(new_audio)}; hashing new entries...", flush=True)
        inv, man = [], []
        for p in sorted(new_audio):
            rel = p.relative_to(DEST).as_posix(); st = p.stat()
            inv.append(f"{rel}\t{st.st_size}\t{int(st.st_mtime)}")
            if rel not in seen:
                man.append(f"{rel}\t{st.st_size}\t{sha256f(p)}\t{rev}"); seen.add(rel)
        (DEST / "_INVENTORY.tsv").write_text("\n".join(inv) + "\n", encoding="utf-8", newline="\n")
        with open(master, "a", encoding="utf-8", newline="\n") as f:
            f.write("".join(l + "\n" for l in man))
        print(f"[{lang}] manifest += {len(man)}; embedding...", flush=True)
        r = subprocess.run([sys.executable, "_extract_xlsr.py", "10_MLAAD"], cwd=str(BASE))
        if r.returncode != 0:
            print(f"[{lang}] EXTRACTION FAILED — audio kept."); sys.exit(1)
        done = set((BASE / "_embcache_xlsr300m/10_MLAAD/_done.txt").read_text().splitlines())
        missing = [p for p in new_audio if p.relative_to(DEST).as_posix() not in done]
        if missing:
            print(f"[{lang}] GATE FAIL: {len(missing)} not embedded — audio kept."); sys.exit(1)
        locked = 0
        for p in new_audio:
            for _ in range(4):
                try: p.unlink(); break
                except PermissionError: time.sleep(5)
            else: locked += 1
        for d in sorted({p.parent for p in new_audio}, key=lambda x: -len(x.parts)):
            try: d.rmdir()
            except OSError: pass
        if locked: print(f"[{lang}] note: {locked} locked files left; harmless, cleaned next run", flush=True)
        (marker_dir / lang).write_text(time.strftime("%F %T"))
        print(f"[{lang}] DONE: embedded, manifested, released. free {free_gb():.0f} GB", flush=True)
    print("\nALL LANGUAGES DONE")

if __name__ == "__main__":
    main()
