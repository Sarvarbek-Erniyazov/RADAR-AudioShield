import hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path(".").resolve()
SKIP_DIRS = {".git", "__pycache__"}

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def process(corpus, summary):
    cache_p = corpus / "_HASH_CACHE.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    files, total = [], 0
    for root, dirs, fs in os.walk(corpus):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in fs:
            if not f.startswith("_"):
                files.append(Path(root) / f)
    entries, todo = [], []
    for p in sorted(files):
        st = p.stat(); rel = p.relative_to(corpus).as_posix(); total += st.st_size
        key = f"{rel}|{st.st_size}|{int(st.st_mtime)}"
        (entries.append((rel, st.st_size, cache[key])) if key in cache
         else todo.append((p, rel, st.st_size, key)))
    print(f"[{corpus.name}] files={len(files)} size={total/2**30:.1f}GiB cached={len(entries)} to_hash={len(todo)}", flush=True)
    t0, n = time.time(), 0
    def work(it):
        nonlocal n
        p, rel, size, key = it
        sha = sha256_file(p); n += 1
        if n % 2000 == 0:
            print(f"  ... {n}/{len(todo)} ({n/max(time.time()-t0,1):.0f} files/s)", flush=True)
        return rel, size, key, sha
    if todo:
        with ThreadPoolExecutor(max_workers=4) as ex:
            for i, (rel, size, key, sha) in enumerate(ex.map(work, todo)):
                entries.append((rel, size, sha)); cache[key] = sha
                if (i + 1) % 5000 == 0:
                    cache_p.write_text(json.dumps(cache))
    cache_p.write_text(json.dumps(cache))
    entries.sort()
    (corpus / "_INVENTORY.tsv").write_text("".join(f"{r}\t{s}\n" for r, s, _ in entries), newline="\n")
    (corpus / "_SHA256.txt").write_text("".join(f"{h}  {r}\n" for r, _, h in entries), newline="\n")
    summary.append(f"{corpus.name}: {len(entries)} files, {total/2**30:.1f} GiB")
    print(f"[{corpus.name}] DONE", flush=True)

summary = [time.strftime("%F %T"), str(BASE), ""]
for c in sorted(BASE.iterdir()):
    if c.is_dir() and c.name not in SKIP_DIRS and not c.name.startswith("_"):
        try:
            process(c, summary)
        except KeyboardInterrupt:
            print("interrupted — cached; re-run to resume"); sys.exit(1)
        except Exception as e:
            summary.append(f"{c.name}: FAILED {e}"); print(f"[{c.name}] FAILED: {e}")
(BASE / "_CHECKSUM_SUMMARY.txt").write_text("\n".join(summary) + "\n")
print("\nwrote _CHECKSUM_SUMMARY.txt")
