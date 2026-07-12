"""Generate the authoritative per-host lockfile + environment record.
Audit ref: §5 "dependencies/revisions unpinned"; Roadmap v3 Step 2a, Commit 1.
Usage: python scripts/freeze_environment.py <host_tag>   e.g. local4060 | gpu24
"""
import importlib, json, platform, subprocess, sys
from pathlib import Path

def main(tag: str):
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True, check=True).stdout
    Path(f"requirements.lock.{tag}").write_text(freeze, encoding="utf-8", newline="\n")
    info = {"host_tag": tag, "python": sys.version, "platform": platform.platform()}
    for mod in ("torch", "transformers", "numpy", "scipy", "sklearn", "soundfile"):
        try:
            m = importlib.import_module(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            info[mod] = "MISSING"
    try:
        import torch
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        pass
    Path(f"environment.{tag}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    missing = [k for k, v in info.items() if v == "MISSING"]
    if missing:
        sys.exit(f"MISSING core dependencies: {missing} — install before locking.")
    print(f"wrote requirements.lock.{tag} and environment.{tag}.json")
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "local")
