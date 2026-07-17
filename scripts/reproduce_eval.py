"""Commit-7 reproduction gate. Audit ref: v3 Step 2a closing requirement ("reproduce
before refactor"). Loads e007 checkpoints, verifies hashes, re-runs cross_test on the
``manifests/v2`` pin, and asserts EERs match the committed experiments/e007/*.json
within tolerance. Proves the 2a repairs changed instruments, not the phenomenon.

Manifest-pin evidence: ``README.md:45`` documented the original ``manifests``
cross-test command; C0 compared the legacy and v2 files row-for-row over every shared
column and found all six identical at their full reference populations (ITW 31,779;
ReplayDF 52,320; AI4T 3,148; ASVspoof5 323,307; DiffSSD 94,226; FakeOrReal 69,300).
The gate therefore pins ``manifests/v2`` without changing the reproduced examples.

Usage: python scripts/reproduce_eval.py [--tol 0.002]
"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

CKPT_DIR = Path("/e/AI_voice_detection/checkpoint_backup")   # collaborator machine; adjust locally
FALLBACK_CKPT = Path("runs")                                  # if run on the training machine
EXPECTED = {  # from experiments/e007/*.json (committed)
    "e007_A_fresh": {"inthewild": 0.1805, "replaydf": 0.3327, "ai4t": 0.2565},
    "e007_B_fresh": {"inthewild": 0.1167, "replaydf": 0.3276, "ai4t": 0.2629},
    "e007_C_xlsr_fresh": {"inthewild": 0.2009, "replaydf": 0.4530, "ai4t": 0.3435},
}
EVAL_CORPORA = ("inthewild", "replaydf", "ai4t")
DEV_CORPORA = ("diffssd", "fakeorreal", "asvspoof5")
MANIFEST_DIR = "manifests/v2"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def load_expected_hashes():
    txt = Path("CHECKPOINTS.md").read_text(encoding="utf-8")
    out = {}
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 64:
            path = parts[1].lstrip("*")     # sha256sum binary-mode marker
            out[path] = parts[0]            # path -> sha
    return out


def build_cmd(run: str, ckpt: Path, data_root: str) -> list[str]:
    """Build one cross-test child command without touching the filesystem."""
    cmd = [
        sys.executable,
        "-m",
        "audioshield.evaluation.cross_test",
        "--checkpoint",
        str(ckpt),
        "--corpora",
        *EVAL_CORPORA,
        "--manifest-dir",
        MANIFEST_DIR,
        "--dev-corpora",
        *DEV_CORPORA,
    ]
    if data_root:
        cmd += ["--data-root", data_root]
    cmd += ["--out", str(Path(f"repro_{run}.json")), "--force"]
    return cmd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.002)
    ap.add_argument("--data-root", default="")
    args = ap.parse_args()

    expected_hashes = load_expected_hashes()
    if not expected_hashes:
        print("[reproduce_eval][warn] CHECKPOINTS.md yielded no parsable hash lines", flush=True)

    results, failures = {}, []
    for run, expected_eers in EXPECTED.items():
        # locate checkpoint -- try the nested "runs/<run>/best.pt" form (matches
        # CHECKPOINTS.md's keys) under CKPT_DIR first, then the flattened legacy
        # candidate, then the local training-machine layout.
        ckpt = None
        for cand in [CKPT_DIR / "runs" / run / "best.pt",
                    CKPT_DIR / f"runs_{run}_best.pt",
                    FALLBACK_CKPT / run / "best.pt"]:
            if cand.exists(): ckpt = cand; break
        if ckpt is None:
            failures.append(f"{run}: checkpoint not found"); continue
        print(f"\n=== {run} :: {ckpt} ===", flush=True)
        got_sha = sha256(ckpt)
        print(f"  sha256={got_sha[:16]}...", flush=True)
        expected_key = f"runs/{run}/best.pt"
        expected_sha = expected_hashes.get(expected_key)
        if expected_sha is None:
            failures.append(f"{run}: no expected hash for {expected_key} in CHECKPOINTS.md"); continue
        if got_sha != expected_sha:
            failures.append(
                f"{run}: checkpoint hash mismatch -- expected {expected_sha[:16]}..., "
                f"got {got_sha[:16]}... ({ckpt}); refusing to trust this checkpoint's EERs")
            continue
        print(f"  hash OK (matches CHECKPOINTS.md)", flush=True)
        cmd = build_cmd(run, ckpt, args.data_root)
        out_json = Path(f"repro_{run}.json")
        # --force: this script intentionally overwrites its own scratch output on every
        # re-run of the gate (report finding 3.5 -- cross_test.py now refuses to overwrite
        # by default; this is the one caller that legitimately wants repeat-overwrite).
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-1500:] if r.stdout else "(no stdout)")
        if r.returncode != 0:
            failures.append(f"{run}: cross_test exit {r.returncode}\n{r.stderr[-500:]}"); continue
        # parse produced json (path per cross_test's writer)
        prod = json.loads(out_json.read_text()) if out_json.exists() else None
        if prod is None:
            failures.append(f"{run}: no output json produced"); continue
        per = prod.get("per_corpus", {})
        for corpus, exp in expected_eers.items():
            got = per.get(corpus, {}).get("eer") if isinstance(per.get(corpus), dict) else per.get(corpus)
            if got is None:
                failures.append(f"{run}/{corpus}: missing in output"); continue
            d = abs(got - exp)
            status = "OK" if d <= args.tol else "DRIFT"
            print(f"  {corpus:10s} expected={exp:.4f} got={got:.4f} delta={d:.4f} [{status}]")
            if d > args.tol:
                failures.append(f"{run}/{corpus}: |delta|={d:.4f} > tol={args.tol}")
        results[run] = per

    print("\n" + "="*50)
    if failures:
        print("REPRODUCTION GATE: FAIL"); [print("  -", f) for f in failures]
        sys.exit(1)
    print("REPRODUCTION GATE: PASS — 2a repairs preserved the eval phenomenon.")

if __name__ == "__main__":
    main()
