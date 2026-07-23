"""Step 4 gate consumer (docs/gate_prereg.md).

Evaluates the 8 pre-registered Step 4 criteria against whatever real inputs
are actually present on disk, and emits a single verdict JSON. This script
never runs the gate itself (no training, no GPU) -- it is a read-only
analysis over already-produced files:

  - Phase A battery JSON, produced by scripts/run_reliance_battery.py.
    Two real on-disk shapes, both handled (verified against
    tests/fixtures/step3/*.json, the real fixtures this module is built
    against):
      * manifest-shaped (the file passed to --out): top-level `batteries`
        (list) + `prereg_candidates` (list) + `checkpoints` + `w_metrics`
        + `join_stats`, optionally `battery_files` (name -> path of a
        standalone per-battery copy of the same content).
      * standalone per-battery file (one entry of `battery_files`):
        top-level singular `battery` + `prereg_candidate`, same shared
        `checkpoints`/`w_metrics`/`join_stats` keys.

  - Phase B model-space embedding caches, produced by
    scripts/extract_model_embeddings.py: <out-root>/<checkpoint-stem>/
    <corpus-dir>/shard_*.npz, each shard holding (paths, emb, meta) per
    that script's own docstring and `_write_shard_atomic`. This script only
    checks presence + structural schema (dir/shard exist, keys present,
    embedding_dim readable) -- no Phase B cache exists on this machine yet,
    so every check of it is exercised by a synthetic stand-in in tests and
    reported `pending_input` for real runs until the collaborator machine
    produces it.

    <checkpoint-stem>/<corpus-dir> above are the REAL on-disk names, not
    Phase A's own checkpoint/corpus identifiers verbatim (docs/gate_prereg.md's
    2026-07-23 Phase B verification amendment): the extractor's canonical
    flat-file convention writes to runs_<checkpoint-key>_best/<CORPUS_DIR[
    corpus-key]>, e.g. runs_e007_A_fresh_best/04_ReplayDF for checkpoint key
    "e007_A_fresh" and corpus key "replaydf" -- NOT e007_A_fresh/replaydf,
    which is what both `analysis/step4/gate_verdict.json` and
    `analysis/step4/gate_verdict_prereg.json` looked for before this fix
    (vacuously; that path can never exist). `_real_phase_b_cache_names`
    below does this translation before every `check_phase_b_cache` call.

  - Per-checkpoint EER files, produced by cross_test.py (directly, or via
    scripts/reproduce_eval.py's `repro_<run>.json`): top-level `checkpoint`
    (the .pt path, whose parent dir name is the run name, e.g.
    "e007_A_fresh") + `per_corpus.<corpus>.eer`. `experiments/e007/
    e007_{A,B,C}_fresh_crosstest.json` already exist in this repository
    today and are real, not synthetic.

  - Seeded head-replicate results, produced by scripts/head_replicates.py
    (this session's Task 3 deliverable): per-seed effect direction, for
    criterion 8.

Every criterion that needs data not present is reported as
status="pending_input" -- never crashes, never silently promoted to
pass/fail. A criterion whose underlying Phase A field is ITSELF the
`not_estimable` sentinel (e.g. the w-dim mismatch already recorded in every
real per_checkpoint block today) is reported as status="not_estimable",
distinct from `pending_input`: `not_estimable` means "Phase A already
determined this is structurally impossible with the current embedding
cache"; `pending_input` means "this script doesn't have the upstream file
yet." An overall three-outcome classification (see docs/gate_prereg.md §3)
is emitted only when every criterion has resolved to pass/fail -- i.e. none
are pending_input or not_estimable.

A fourth, PER-BATTERY-only status, "excluded", covers real Phase A output
that is itself an OPERATIONAL failure (a bootstrap that timed out, an
estimator that never produced, a battery whose whole prereg_candidate
summary is the `{"name":..., "skipped": "..."}` shape
`summarize_prereg_candidate` returns for a skipped/failed battery) --
never a scientific fact about reliance. FAIL is reserved for a genuine
decided negative on data that DID complete; an excluded battery is dropped
from a criterion's evidence entirely (neither pass nor fail) rather than
silently defaulting to a fail-shaped value (see `_aggregate_battery_statuses`).
A criterion with at least one genuinely decided (pass/fail) battery
resolves from those alone; one with only excluded/pending/not_estimable
batteries resolves to not_estimable or pending_input, never fail.

Follows the heartbeat/flush/incremental-output/exit-0 conventions already
established in scripts/run_reliance_battery.py: every stage is timestamped
and printed with flush=True; the verdict file is always written (temp file
+ os.replace) and the process always exits 0 -- this is a reporting tool,
not a build gate that should break a CI run while collaborator-machine work
is still in flight.

Usage:
    python scripts/run_gate.py \\
        --phase-a tests/fixtures/step3/reliance_layer9_boot.json \\
        --eer experiments/e007/e007_A_fresh_crosstest.json \\
              experiments/e007/e007_B_fresh_crosstest.json \\
              experiments/e007/e007_C_xlsr_fresh_crosstest.json \\
        --out analysis/step4/gate_verdict.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_PENDING = "pending_input"
STATUS_NOT_ESTIMABLE = "not_estimable"
STATUS_EXCLUDED = "excluded"

DEFAULT_PHASE_B_OUT_ROOT = Path("analysis/step3/_embcache_modelspace")
DEFAULT_VERDICT_OUT = Path("analysis/step4/gate_verdict.json")


# ---------------------------------------------------------------------------
# Heartbeat logging -- same shape as scripts/run_reliance_battery.py's
# _timestamp/_log, so gate runs read consistently with Phase A runs in a
# combined log.
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _log(msg: str) -> None:
    print(f"[{_timestamp()}] {msg}", flush=True)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    tmp_path.replace(path)


def _criterion(status: str, evidence: str, numbers: dict | None = None) -> dict:
    return dict(status=status, evidence=evidence, numbers=numbers or {})


def _is_candidate_skipped(cand: dict) -> bool:
    """True for the shape scripts/run_reliance_battery.py's own
    summarize_prereg_candidate returns for a battery that was itself
    skipped or failed at the whole-battery level: {"name": ..., "skipped":
    "<reason>"} -- none of the normal fields (stable_rank_window,
    estimators_agree_sign, n_groups, ...) are present at all in that shape,
    so treating a missing field as False/[] would silently manufacture a
    fail out of an operational gap."""
    return bool(cand.get("skipped"))


def _aggregate_battery_statuses(per_battery_status: dict[str, str]) -> str:
    """Combine each battery's own status into one overall criterion
    status. A battery marked STATUS_EXCLUDED or STATUS_PENDING never turns
    a criterion FAIL by itself -- FAIL is reserved for at least one
    battery whose OWN result is a genuinely decided (pass/fail) negative.
    Excluded/pending batteries are simply dropped from the evidence tally;
    they don't count as passes either. Only when NO battery reached a
    decided pass/fail does the overall status fall back to not_estimable
    (if any battery is not_estimable) or pending_input (otherwise) --
    matching this module's existing not_estimable/pending_input
    distinction one level up."""
    decided = {k: v for k, v in per_battery_status.items() if v in (STATUS_PASS, STATUS_FAIL)}
    if decided:
        return STATUS_FAIL if any(v == STATUS_FAIL for v in decided.values()) else STATUS_PASS
    statuses = set(per_battery_status.values())
    if not statuses:
        return STATUS_PENDING
    if STATUS_NOT_ESTIMABLE in statuses:
        return STATUS_NOT_ESTIMABLE
    return STATUS_PENDING


def _lookup_factor_corpus(factor_corpus_map: dict, corpus: str | None, factor: str | None,
                           used_keys: set | None = None) -> str | None:
    """Resolves an eval corpus for a battery's (corpus, factor) pair.
    Supports three key shapes in `factor_corpus_map`, checked in order:
    (1) an exact (corpus, factor) tuple key -- for two batteries that
    share a factor but need DIFFERENT scored corpora (e.g. diffssd's and
    replaydf's generator_id batteries); (2) a (None, factor) wildcard-corpus
    tuple key -- one eval corpus for this factor regardless of battery
    corpus; (3) a bare factor string key -- the original, single-corpus-map
    convenience shape, kept for backward compatibility. If `used_keys` is
    passed, the matched key is added to it -- callers use this to warn
    about factor_corpus_map entries that never matched any battery."""
    for key in ((corpus, factor), (None, factor), factor):
        if key in factor_corpus_map:
            if used_keys is not None:
                used_keys.add(key)
            return factor_corpus_map[key]
    return None


# ---------------------------------------------------------------------------
# Phase A loading -- normalizes both real on-disk shapes to one internal
# representation: a list of battery records.
# ---------------------------------------------------------------------------


def load_phase_a_file(path: Path) -> list[dict]:
    """Load one Phase A JSON file (manifest-shaped or standalone
    per-battery-shaped -- both real, observed shapes) and return a list of
    normalized battery records: {battery, prereg_candidate, checkpoints,
    w_metrics, join_stats, source}. Raises only on a file that parses but
    matches neither known shape -- callers are expected to catch this and
    degrade to pending_input, per this script's never-crash contract for
    missing/malformed *optional* inputs; a --phase-a path the caller
    explicitly supplied but that doesn't parse is a configuration error
    worth surfacing, not silently swallowing.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    shared = dict(
        checkpoints=raw.get("checkpoints", {}),
        w_metrics=raw.get("w_metrics", {}),
        join_stats=raw.get("join_stats", {}),
        schema_version=raw.get("schema_version"),
        git_sha=raw.get("git_sha"),
    )
    records = []
    if "battery" in raw and "prereg_candidate" in raw:
        records.append(dict(battery=raw["battery"], prereg_candidate=raw["prereg_candidate"],
                             source=str(path), source_shape="standalone", **shared))
    elif "batteries" in raw and "prereg_candidates" in raw:
        cand_by_name = {c["name"]: c for c in raw["prereg_candidates"] if "name" in c}
        for b in raw["batteries"]:
            records.append(dict(battery=b, prereg_candidate=cand_by_name.get(b.get("name"), {}),
                                 source=str(path), source_shape="manifest", **shared))
    else:
        raise ValueError(f"{path}: unrecognized Phase A schema -- no 'battery'/'prereg_candidate' "
                          "or 'batteries'/'prereg_candidates' keys found")
    return records


def _dedupe_battery_records(records: list[dict]) -> tuple[list[dict], list[str]]:
    """A user may pass both a manifest (--out file, batteries inline) and
    one of its own standalone per-battery files (battery_files entries) in
    the same --phase-a invocation -- without deduping, that battery would
    be counted twice by every criterion's per-battery tally. Keeps the
    standalone per-battery copy over the manifest's inline copy (the more
    specific source); warns, rather than silently overwriting, when the
    two copies actually disagree on content (never expected in practice --
    both are supposed to be the same battery result -- but a silent
    overwrite on a genuine conflict is exactly the kind of thing this
    project's discipline exists to avoid)."""
    by_name: dict[str, dict] = {}
    warnings: list[str] = []
    for rec in records:
        name = rec["battery"].get("name")
        if name is None:
            warnings.append(f"a battery record from {rec.get('source')} has no 'name' -- skipped")
            continue
        if name not in by_name:
            by_name[name] = rec
            continue
        existing = by_name[name]
        if existing["battery"] != rec["battery"] or existing["prereg_candidate"] != rec["prereg_candidate"]:
            warnings.append(
                f"battery {name!r} appears more than once across --phase-a inputs with DIFFERING "
                f"content ({existing['source']} vs {rec['source']}) -- keeping the standalone "
                "per-battery copy if one of the two is standalone, else the first one seen"
            )
        else:
            warnings.append(f"battery {name!r} appears more than once across --phase-a inputs "
                             f"(identical content, {existing['source']} vs {rec['source']}) -- deduped")
        if rec.get("source_shape") == "standalone" and existing.get("source_shape") != "standalone":
            by_name[name] = rec
    return list(by_name.values()), warnings


def load_phase_a_inputs(paths: list[Path]) -> tuple[list[dict], list[str]]:
    """Best-effort load of every --phase-a path. A path that doesn't exist
    or fails to parse is recorded as a warning and skipped, never crashes
    the run. Battery records are deduped by name across every path
    combined (see _dedupe_battery_records) -- a duplicate never inflates a
    criterion's per-battery tally."""
    records, warnings = [], []
    for p in paths:
        p = Path(p)
        if not p.exists():
            warnings.append(f"phase-a input not found: {p}")
            continue
        try:
            records.extend(load_phase_a_file(p))
        except Exception as exc:
            warnings.append(f"phase-a input {p} failed to load: {exc}")
    deduped, dedupe_warnings = _dedupe_battery_records(records)
    return deduped, warnings + dedupe_warnings


# ---------------------------------------------------------------------------
# EER loading (cross_test.py / reproduce_eval.py output shape)
# ---------------------------------------------------------------------------


def load_eer_file(path: Path) -> tuple[str, dict[str, float]]:
    """One cross_test.py-shaped result file -> (run_name, {corpus: eer}).
    run_name is the parent-directory name of the checkpoint path
    (result['checkpoint'], e.g. "runs/e007_A_fresh/best.pt" -> "e007_A_fresh"),
    which is exactly how checkpoints are named/keyed in Phase A battery
    output -- verified against experiments/e007/*_crosstest.json (real,
    committed files) and scripts/run_reliance_battery.py's checkpoint dict
    keys."""
    d = json.loads(path.read_text(encoding="utf-8"))
    ckpt_path = Path(d.get("checkpoint", ""))
    run_name = ckpt_path.parent.name or path.stem
    per_corpus = d.get("per_corpus", {})
    eers = {c: float(v["eer"]) for c, v in per_corpus.items() if isinstance(v, dict) and "eer" in v}
    return run_name, eers


def load_eer_inputs(paths: list[Path]) -> tuple[dict[str, dict[str, float]], list[str]]:
    out, warnings = {}, []
    for p in paths:
        p = Path(p)
        if not p.exists():
            warnings.append(f"eer input not found: {p}")
            continue
        try:
            run_name, eers = load_eer_file(p)
            out[run_name] = eers
        except Exception as exc:
            warnings.append(f"eer input {p} failed to load: {exc}")
    return out, warnings


# ---------------------------------------------------------------------------
# Phase B cache presence/schema check (extract_model_embeddings.py output)
# ---------------------------------------------------------------------------

# Corpus-key -> on-disk dataset-directory-name table, DUPLICATED (not
# imported) from scripts/run_reliance_battery.py's own CORPUS_DIR.
# run_reliance_battery.py imports torch at module level (to load a
# checkpoint's classifier weight); importing it here just to reuse this
# 3-entry table would make this CPU-only, "no GPU" reporting tool (see this
# module's own docstring) contingent on torch importing cleanly -- exactly
# the GPU-stack-contamination problem step3_modelspace_hardening_addendum.md's
# Finding 2 fixed for run_reliance_modelspace.py's _sha256_file reuse.
# Relocating this table to a shared, stdlib-only module (the Finding-2-style
# proper fix) would require editing run_reliance_battery.py, out of scope
# for this change -- keep these two tables in sync by hand if a new corpus
# is added to either.
CORPUS_DIR = {
    "diffssd": "03_DiffSSD",
    "replaydf": "04_ReplayDF",
    "vctk": "09_VCTK",
}


def _real_phase_b_cache_names(checkpoint_key: str, corpus_key: str) -> tuple[str, str] | None:
    """Translates Phase A's own identifiers (checkpoint_key like
    "e007_A_fresh", corpus_key like "replaydf") into the REAL on-disk names
    scripts/extract_model_embeddings.py actually writes under, per its
    canonical flat-file naming convention (no --run-name):
    f"runs_{checkpoint_key}_best" for the checkpoint directory
    (scripts/run_reliance_modelspace.py's own construction at its
    ckpt_path = ckpt_dir / f"runs_{run}_best.pt", confirmed by direct read)
    and CORPUS_DIR[corpus_key] for the corpus directory. Returns None if
    corpus_key isn't in CORPUS_DIR -- the caller must report a decided,
    legible failure rather than silently guessing a directory name for an
    unknown corpus."""
    corpus_dir = CORPUS_DIR.get(corpus_key)
    if corpus_dir is None:
        return None
    return f"runs_{checkpoint_key}_best", corpus_dir


def check_phase_b_cache(out_root: Path, checkpoint_stem: str, corpus_dir: str) -> dict:
    """Presence + structural check only, for the REAL shard layout written
    by scripts/extract_model_embeddings.py: <out_root>/<checkpoint_stem>/
    <corpus_dir>/shard_*.npz, each an atomically-written npz with
    (paths, emb, meta) keys (meta a JSON-encoded 0-d string array). Never
    raises: any read failure degrades to status="fail" with the exception
    message, an absent directory/shard set degrades to
    status="pending_input" (this is the expected, current state on this
    machine -- no Phase B cache exists here yet)."""
    cache_dir = out_root / checkpoint_stem / corpus_dir
    if not cache_dir.is_dir():
        return dict(status=STATUS_PENDING, path=str(cache_dir),
                     reason=f"no Phase B embedding cache directory at {cache_dir} -- run "
                            "scripts/extract_model_embeddings.py for this checkpoint/corpus "
                            "(collaborator GPU machine)")
    shards = sorted(cache_dir.glob("shard_*.npz"))
    if not shards:
        return dict(status=STATUS_PENDING, path=str(cache_dir),
                     reason=f"{cache_dir} exists but contains no shard_*.npz files yet")
    try:
        n_rows_total, embedding_dim, dtype = 0, None, None
        for shard_path in shards:
            with np.load(shard_path, allow_pickle=True) as npz:
                missing = {"paths", "emb", "meta"} - set(npz.files)
                if missing:
                    return dict(status=STATUS_FAIL, path=str(shard_path),
                                 reason=f"{shard_path} missing expected keys {sorted(missing)}, "
                                        f"found {sorted(npz.files)}")
                emb = npz["emb"]
                n_rows_total += int(emb.shape[0])
                embedding_dim = int(emb.shape[1])
                dtype = str(emb.dtype)
        return dict(status=STATUS_PASS, path=str(cache_dir), n_shards=len(shards),
                     n_rows=n_rows_total, embedding_dim=embedding_dim, dtype=dtype)
    except Exception as exc:
        return dict(status=STATUS_FAIL, path=str(cache_dir),
                     reason=f"failed to read shard(s) under {cache_dir}: {exc}")


# ---------------------------------------------------------------------------
# Head-replicate loading (Task 3 deliverable, scripts/head_replicates.py)
# ---------------------------------------------------------------------------


def load_head_replicates(path: Path | None) -> tuple[list[dict] | None, list[str]]:
    if path is None:
        return None, []
    path = Path(path)
    if not path.exists():
        return None, [f"head-replicate input not found: {path}"]
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        replicates = d.get("replicates")
        if not isinstance(replicates, list):
            return None, [f"{path}: expected a 'replicates' list, got {type(replicates)}"]
        return replicates, []
    except Exception as exc:
        return None, [f"head-replicate input {path} failed to load: {exc}"]


# ---------------------------------------------------------------------------
# Per-checkpoint reliance extraction shared by C2/C7 -- reads whatever
# per_checkpoint alignment/r_var values are present, honoring Phase A's own
# not_estimable sentinel rather than treating it as zero/missing.
# ---------------------------------------------------------------------------


def _per_checkpoint_reliance(battery: dict, metric: str = "alignment") -> dict[str, dict]:
    """checkpoint_name -> {status, value} aggregated (mean of finite fold
    values) across every fold_result's effect.per_checkpoint[name][metric].
    status is 'not_estimable' if every fold reports Phase A's own
    not_estimable sentinel for this checkpoint/metric, 'ok' if at least one
    finite value exists."""
    out: dict[str, dict] = {}
    for estimator in battery.get("estimators", {}).values():
        for fold in estimator.get("fold_results", []):
            per_ckpt = fold.get("effect", {}).get("per_checkpoint", {})
            for ckpt_name, ckpt_metrics in per_ckpt.items():
                entry = ckpt_metrics.get(metric)
                bucket = out.setdefault(ckpt_name, dict(values=[], any_not_estimable=False))
                if isinstance(entry, dict):
                    if entry.get("status") == STATUS_NOT_ESTIMABLE:
                        bucket["any_not_estimable"] = True
                    elif entry.get("value") is not None and np.isfinite(entry["value"]):
                        bucket["values"].append(float(entry["value"]))
                elif entry is not None and np.isfinite(entry):
                    bucket["values"].append(float(entry))
    result = {}
    for ckpt_name, bucket in out.items():
        if bucket["values"]:
            result[ckpt_name] = dict(status="ok", value=float(np.mean(bucket["values"])))
        elif bucket["any_not_estimable"]:
            result[ckpt_name] = dict(status=STATUS_NOT_ESTIMABLE, value=None)
        else:
            result[ckpt_name] = dict(status=STATUS_PENDING, value=None)
    return result


# ---------------------------------------------------------------------------
# C1 -- replication across >=2 backbones
# ---------------------------------------------------------------------------


def criterion_1_replication(primary: list[dict], secondary: list[dict]) -> dict:
    if not secondary:
        return _criterion(
            STATUS_PENDING,
            "no second-backbone (WavLM-Large) battery input provided -- pass "
            "--phase-a-secondary-backbone once the collaborator machine's WavLM-Large "
            "embedding pass produces it",
        )
    primary_by_name = {r["battery"]["name"]: r for r in primary}
    secondary_by_name = {r["battery"]["name"]: r for r in secondary}
    shared_names = sorted(set(primary_by_name) & set(secondary_by_name))
    if not shared_names:
        return _criterion(
            STATUS_PENDING,
            "primary and secondary-backbone battery sets share no battery name to compare",
            dict(primary_names=sorted(primary_by_name), secondary_names=sorted(secondary_by_name)),
        )
    per_battery = {}
    for name in shared_names:
        p_boot = primary_by_name[name]["battery"]["headline_bootstrap"]
        s_boot = secondary_by_name[name]["battery"]["headline_bootstrap"]
        if p_boot.get("status") != "ok" or s_boot.get("status") != "ok":
            per_battery[name] = dict(agree=None, reason="headline_bootstrap not ok on one or both backbones")
            continue
        same_sign = np.sign(p_boot["mean"]) == np.sign(s_boot["mean"])
        disjoint_opposite = (p_boot["hi"] < 0 < s_boot["lo"]) or (s_boot["hi"] < 0 < p_boot["lo"])
        per_battery[name] = dict(agree=bool(same_sign and not disjoint_opposite),
                                  primary_mean=p_boot["mean"], secondary_mean=s_boot["mean"])
    decided = {k: v for k, v in per_battery.items() if v["agree"] is not None}
    if not decided:
        return _criterion(STATUS_PENDING, "no shared battery had a usable headline_bootstrap on both backbones",
                           dict(per_battery=per_battery))
    all_agree = all(v["agree"] for v in decided.values())
    return _criterion(
        STATUS_PASS if all_agree else STATUS_FAIL,
        f"{sum(v['agree'] for v in decided.values())}/{len(decided)} shared batteries replicate "
        "(sign agreement, no disjoint-opposite CIs) across the two backbones",
        dict(per_battery=per_battery),
    )


# ---------------------------------------------------------------------------
# C2 -- positive reliance <-> matching-factor-degradation association
# ---------------------------------------------------------------------------


def criterion_2_association(records: list[dict], eers: dict[str, dict[str, float]],
                             factor_corpus_map: dict) -> dict:
    """factor_corpus_map is keyed as described by _lookup_factor_corpus:
    (corpus, factor) tuples for a corpus-specific mapping (needed once
    e.g. diffssd's and replaydf's generator_id batteries need DIFFERENT
    scored corpora), (None, factor) tuples for a corpus-agnostic
    fallback, or bare factor strings (legacy, single-corpus convenience).
    Every map key actually used is tracked so unused keys (a likely typo)
    are reported in `numbers["unmatched_factor_corpus_map_keys"]`."""
    if not records:
        return _criterion(STATUS_PENDING, "no Phase A battery input provided")
    per_battery = {}
    used_keys = set()
    for rec in records:
        battery = rec["battery"]
        name = battery["name"]
        cand = rec.get("prereg_candidate", {})
        if _is_candidate_skipped(cand):
            per_battery[name] = dict(status=STATUS_EXCLUDED, reason=f"battery skipped: {cand.get('skipped')}")
            continue
        factor = battery.get("factor")
        corpus = battery.get("corpus")
        eval_corpus = _lookup_factor_corpus(factor_corpus_map, corpus, factor, used_keys)
        if eval_corpus is None:
            per_battery[name] = dict(
                status=STATUS_PENDING,
                reason=f"no factor-corpus mapping supplied for corpus={corpus!r}/factor={factor!r} -- see "
                       "docs/gate_prereg.md §4 ambiguity 2 (diffssd is dev-tier-for-thresholding "
                       "only, never itself EER-scored by cross_test.py)",
            )
            continue
        reliance = _per_checkpoint_reliance(battery, metric="alignment")
        pairs = []
        any_not_estimable = False
        for ckpt_name, rel in reliance.items():
            if rel["status"] == STATUS_NOT_ESTIMABLE:
                any_not_estimable = True
                continue
            if rel["status"] != "ok":
                continue
            eer = eers.get(ckpt_name, {}).get(eval_corpus)
            if eer is None:
                continue
            pairs.append((ckpt_name, rel["value"], eer))
        if len(pairs) < 2:
            per_battery[name] = dict(
                status=STATUS_NOT_ESTIMABLE if any_not_estimable else STATUS_PENDING,
                reason="fewer than 2 checkpoints have both an estimable reliance value and a "
                       f"matching EER on {eval_corpus!r}" + (" (reliance not_estimable: w-dim mismatch, "
                       "pending Phase B model-space embeddings)" if any_not_estimable else ""),
                pairs=pairs,
            )
            continue
        reliances = np.array([p[1] for p in pairs])
        target_eers = np.array([p[2] for p in pairs])
        corr = (float(np.corrcoef(reliances, target_eers)[0, 1])
                if np.std(reliances) > 0 and np.std(target_eers) > 0 else float("nan"))
        positive = np.isfinite(corr) and corr > 0
        per_battery[name] = dict(
            status=STATUS_PASS if positive else (STATUS_FAIL if np.isfinite(corr) else STATUS_PENDING),
            correlation=corr, pairs=pairs, eval_corpus=eval_corpus,
        )
    overall = _aggregate_battery_statuses({k: v["status"] for k, v in per_battery.items()})
    unmatched = sorted(str(k) for k in factor_corpus_map if k not in used_keys)
    return _criterion(
        overall, f"per-battery association results: { {k: v['status'] for k, v in per_battery.items()} }",
        dict(per_battery=per_battery, unmatched_factor_corpus_map_keys=unmatched),
    )


# ---------------------------------------------------------------------------
# C3 -- grouped-bootstrap CIs (methodological precondition)
# ---------------------------------------------------------------------------


def criterion_3_grouped_bootstrap(records: list[dict], min_n_boot: int = 1000, min_n_groups: int = 8) -> dict:
    """A battery whose bootstrap task itself failed/timed out
    (`headline_bootstrap.status != "ok"`, e.g. the real
    {status:"failed", error:..., timed_out:true, stage:"bootstrap"} shape
    scripts/run_reliance_battery.py's worker-timeout path produces -- no
    lo/hi/mean/n_boot/n_groups keys at all in that shape) is an OPERATIONAL
    gap, not a scientific fact about that battery's CIs -- excluded from
    this criterion's evidence, never counted as a fail. Same for a battery
    whose whole prereg_candidate is the skipped/failed-battery shape."""
    if not records:
        return _criterion(STATUS_PENDING, "no Phase A battery input provided")
    per_battery = {}
    for rec in records:
        battery = rec["battery"]
        name = battery["name"]
        boot = battery.get("headline_bootstrap", {})
        cand = rec.get("prereg_candidate", {})
        if _is_candidate_skipped(cand) or "n_groups" not in cand:
            per_battery[name] = dict(status=STATUS_EXCLUDED,
                                       reason=f"prereg_candidate missing/skipped: {cand.get('skipped', 'no n_groups key')}")
            continue
        if boot.get("status") != "ok":
            per_battery[name] = dict(status=STATUS_EXCLUDED,
                                       reason=f"headline_bootstrap not ok (status={boot.get('status')!r}, "
                                              f"error={boot.get('error')!r}) -- operational, not a CI failure")
            continue
        ok = (
            boot.get("n_boot", 0) >= min_n_boot
            and cand.get("n_groups", 0) >= min_n_groups
            and cand.get("grouping_degenerate") is False
        )
        per_battery[name] = dict(
            status=STATUS_PASS if ok else STATUS_FAIL, n_boot=boot.get("n_boot"),
            n_groups=cand.get("n_groups"), grouping_degenerate=cand.get("grouping_degenerate"),
        )
    overall = _aggregate_battery_statuses({k: v["status"] for k, v in per_battery.items()})
    n_pass = sum(v["status"] == STATUS_PASS for v in per_battery.values())
    n_decided = sum(v["status"] in (STATUS_PASS, STATUS_FAIL) for v in per_battery.values())
    return _criterion(
        overall,
        f"{n_pass}/{n_decided} decided batteries have a usable grouped bootstrap "
        f"(status=ok, n_boot>={min_n_boot}, n_groups>={min_n_groups}, non-degenerate grouping); "
        f"{len(per_battery) - n_decided} excluded/operational-gap",
        dict(per_battery=per_battery),
    )


# ---------------------------------------------------------------------------
# C4 -- intervention effects exceed equal-norm random controls; task-
# direction removal as positive control
# ---------------------------------------------------------------------------


def criterion_4_intervention_vs_random(records: list[dict], min_fraction: float = 0.5) -> dict:
    """`projection_removal_control` reports the main factor-projection
    effect (`true_effect`, `random_effects`, `exceeds_random` -- already
    `true_effect > random_mean + 2*random_std`) alongside a SIBLING
    `task_direction_effect` field, the positive control. Verified against
    the real fixtures: both `exceeds_random` and `task_direction_effect`
    are bare floats when estimable (smoke.json) and `task_direction_effect`
    degrades to the `not_estimable` sentinel dict when w is unavailable
    (the real accent battery) -- `task_direction_effect` never carries its
    own `exceeds_random` key, so the positive control's own exceeds-random
    flag has to be computed here from the same random_mean/random_std the
    main effect uses.

    AMENDED (docs/gate_prereg.md's 2026-07-22 C4 amendment): the verdict is
    UNANIMOUS across checkpoints -- every checkpoint's own
    per_checkpoint[ckpt].projection_removal_control (populated by
    run_reliance_modelspace.py's merge_checkpoint_estimator_results for
    every checkpoint, the same field _per_checkpoint_reliance already
    reads for C2/C7) must independently clear the same main-effect-majority/
    positive-control-unanimity bars documented above, scoped to that
    checkpoint's own folds; one checkpoint failing fails the whole battery,
    regardless of how strongly the others pass. A checkpoint whose own
    per_checkpoint entry has no projection_removal_control key at all (the
    cache-space regime, where the control was checkpoint-independent by
    construction) falls back to the fold-level
    fold['effect']['projection_removal_control'] instead -- every
    checkpoint then shares that one value, which reproduces the pre-
    amendment fold-level-only computation exactly. Which path was used is
    recorded per checkpoint (and, if uniform across the battery, once at
    the battery level too) as `control_source`, so the regime is never
    ambiguous in a results file. `per_checkpoint[ckpt]` verdicts are
    reported individually alongside the aggregate so a failure reads as
    e.g. "e007_A pass, e007_B pass, e007_C fail", never a bare boolean.
    The battery-level `main_exceeds_random_fraction`/`n_main_folds`/
    `n_control_folds_estimable` fields are the UNCHANGED pre-amendment
    (fold-level-only) computation, kept only for schema continuity -- they
    no longer decide `status`; `per_checkpoint` is authoritative.
    Cross-checkpoint aggregation reuses `_aggregate_battery_statuses`
    (unanimous: any decided FAIL among checkpoints fails the battery; a
    checkpoint that's `not_estimable`/never decided is dropped from the
    tally rather than counted either way, the same rule this module
    already uses one level up to combine battery-level verdicts into an
    overall criterion status)."""
    if not records:
        return _criterion(STATUS_PENDING, "no Phase A battery input provided")
    per_battery = {}
    for rec in records:
        battery = rec["battery"]
        name = battery["name"]
        cand = rec.get("prereg_candidate", {})
        if _is_candidate_skipped(cand):
            per_battery[name] = dict(status=STATUS_EXCLUDED, reason=f"battery skipped: {cand.get('skipped')}")
            continue

        # Legacy, fold-level-only figures -- UNCHANGED computation (reads
        # only the fold-level field, exactly as before the amendment),
        # kept for schema continuity. No longer used to decide `status`.
        main_flags, control_flags, control_any_not_estimable = [], [], False
        for estimator in battery.get("estimators", {}).values():
            if estimator.get("status") != "ok":
                continue  # a failed estimator contributes no fold_results anyway; explicit for clarity
            for fold in estimator.get("fold_results", []):
                prc = fold.get("effect", {}).get("projection_removal_control", {})
                if "exceeds_random" in prc:
                    main_flags.append(bool(prc["exceeds_random"]))
                tde = prc.get("task_direction_effect")
                random_mean, random_std = prc.get("random_mean"), prc.get("random_std")
                if isinstance(tde, dict) and tde.get("status") == STATUS_NOT_ESTIMABLE:
                    control_any_not_estimable = True
                elif (tde is not None and not isinstance(tde, dict)
                      and random_mean is not None and random_std is not None):
                    control_flags.append(bool(tde > random_mean + 2 * random_std))
        main_fraction = float(np.mean(main_flags)) if main_flags else None

        # Per-checkpoint figures -- THE unanimous-rule verdict.
        per_ckpt_acc: dict[str, dict] = {}
        battery_sources: set[str] = set()
        for estimator in battery.get("estimators", {}).values():
            if estimator.get("status") != "ok":
                continue
            for fold in estimator.get("fold_results", []):
                eff = fold.get("effect", {})
                fold_level_prc = eff.get("projection_removal_control", {})
                for ckpt_name, ckpt_entry in eff.get("per_checkpoint", {}).items():
                    ckpt_prc = ckpt_entry.get("projection_removal_control")
                    if ckpt_prc:
                        prc, source = ckpt_prc, "per_checkpoint"
                    else:
                        prc, source = fold_level_prc, "fold_level_fallback"
                    battery_sources.add(source)
                    acc = per_ckpt_acc.setdefault(ckpt_name, dict(
                        main_flags=[], control_flags=[], control_any_not_estimable=False, sources=set(),
                    ))
                    acc["sources"].add(source)
                    if "exceeds_random" in prc:
                        acc["main_flags"].append(bool(prc["exceeds_random"]))
                    tde = prc.get("task_direction_effect")
                    ckpt_random_mean, ckpt_random_std = prc.get("random_mean"), prc.get("random_std")
                    if isinstance(tde, dict) and tde.get("status") == STATUS_NOT_ESTIMABLE:
                        acc["control_any_not_estimable"] = True
                    elif (tde is not None and not isinstance(tde, dict)
                          and ckpt_random_mean is not None and ckpt_random_std is not None):
                        acc["control_flags"].append(bool(tde > ckpt_random_mean + 2 * ckpt_random_std))

        per_checkpoint_verdicts = {}
        for ckpt_name, acc in sorted(per_ckpt_acc.items()):
            ckpt_main_fraction = float(np.mean(acc["main_flags"])) if acc["main_flags"] else None
            ckpt_main_ok = ckpt_main_fraction is not None and ckpt_main_fraction >= min_fraction
            if acc["control_flags"]:
                ckpt_status = STATUS_PASS if (ckpt_main_ok and all(acc["control_flags"])) else STATUS_FAIL
            elif acc["control_any_not_estimable"]:
                ckpt_status = STATUS_NOT_ESTIMABLE
            else:
                ckpt_status = STATUS_PENDING
            ckpt_sources = sorted(acc["sources"])
            per_checkpoint_verdicts[ckpt_name] = dict(
                status=ckpt_status, main_exceeds_random_fraction=ckpt_main_fraction,
                n_main_folds=len(acc["main_flags"]), n_control_folds_estimable=len(acc["control_flags"]),
                control_source=ckpt_sources[0] if len(ckpt_sources) == 1 else ckpt_sources,
            )

        if per_checkpoint_verdicts:
            status = _aggregate_battery_statuses(
                {ckpt: v["status"] for ckpt, v in per_checkpoint_verdicts.items()}
            )
            verdict_summary = ", ".join(f"{ckpt} {v['status']}" for ckpt, v in sorted(per_checkpoint_verdicts.items()))
        else:
            status = STATUS_PENDING
            verdict_summary = "no per-checkpoint projection_removal_control data available"

        sorted_sources = sorted(battery_sources)
        per_battery[name] = dict(
            status=status, main_exceeds_random_fraction=main_fraction, n_main_folds=len(main_flags),
            n_control_folds_estimable=len(control_flags),
            per_checkpoint=per_checkpoint_verdicts,
            control_source=(sorted_sources[0] if len(sorted_sources) == 1
                            else sorted_sources if sorted_sources else None),
            verdict_summary=verdict_summary,
        )
    overall = _aggregate_battery_statuses({k: v["status"] for k, v in per_battery.items()})
    return _criterion(
        overall,
        "per-checkpoint main-effect-vs-random and positive-control results, UNANIMOUS across "
        "checkpoints (docs/gate_prereg.md C4 amendment, 2026-07-22) -- see per_battery[name]."
        "per_checkpoint for each checkpoint's own verdict and per_battery[name].verdict_summary "
        "for a legible pass/fail breakdown",
        dict(per_battery=per_battery),
    )


# ---------------------------------------------------------------------------
# C5 -- rank stability
# ---------------------------------------------------------------------------


def criterion_5_rank_stability(records: list[dict]) -> dict:
    """A battery whose prereg_candidate is the skipped/failed-battery shape
    (no `stable_rank_window` key at all) or whose own `rank_sensitivity`
    task didn't complete (`status != "ok"`) never gets its rank-curve
    computed -- an operational gap, excluded from this criterion, never a
    manufactured fail from treating the missing field as an empty window."""
    if not records:
        return _criterion(STATUS_PENDING, "no Phase A battery input provided")
    per_battery = {}
    for rec in records:
        battery = rec["battery"]
        name = battery["name"]
        cand = rec.get("prereg_candidate", {})
        rank_sensitivity = battery.get("rank_sensitivity", {})
        if _is_candidate_skipped(cand) or "stable_rank_window" not in cand:
            per_battery[name] = dict(status=STATUS_EXCLUDED,
                                       reason=f"prereg_candidate missing/skipped: {cand.get('skipped', 'no stable_rank_window key')}")
            continue
        if rank_sensitivity.get("status") not in (None, "ok"):
            per_battery[name] = dict(status=STATUS_EXCLUDED,
                                       reason=f"rank_sensitivity not ok (status={rank_sensitivity.get('status')!r})")
            continue
        window = cand["stable_rank_window"]
        ok = len(window) >= 2 and 1 in window
        per_battery[name] = dict(status=STATUS_PASS if ok else STATUS_FAIL, stable_rank_window=window)
    overall = _aggregate_battery_statuses({k: v["status"] for k, v in per_battery.items()})
    n_pass = sum(v["status"] == STATUS_PASS for v in per_battery.values())
    n_decided = sum(v["status"] in (STATUS_PASS, STATUS_FAIL) for v in per_battery.values())
    return _criterion(
        overall,
        f"{n_pass}/{n_decided} decided batteries have a stable rank window (>=2 consecutive ranks, "
        f"including headline rank 1); {len(per_battery) - n_decided} excluded/operational-gap",
        dict(per_battery=per_battery),
    )


# ---------------------------------------------------------------------------
# C6 -- agreement across >=2 subspace estimators
# ---------------------------------------------------------------------------


def criterion_6_estimator_agreement(records: list[dict]) -> dict:
    """`prereg_candidate.estimators_agree_sign` is `False` both when the two
    estimators genuinely disagree in sign AND when one of them never
    produced a result at all (summarize_prereg_candidate's own
    _mean_headline_metric returns nan for an empty fold_results list, and
    nan-vs-anything is unconditionally treated as disagreement upstream) --
    these are NOT the same thing, and only the first is a scientific
    negative. Checking each estimator's own `status` field directly (a real
    field on the fixtures: battery.estimators.{lda,probe}.status) is the
    only way to tell them apart; a failed estimator excludes the battery
    from this criterion instead of reporting a fabricated disagreement."""
    if not records:
        return _criterion(STATUS_PENDING, "no Phase A battery input provided")
    per_battery = {}
    for rec in records:
        battery = rec["battery"]
        name = battery["name"]
        cand = rec.get("prereg_candidate", {})
        if _is_candidate_skipped(cand) or "estimators_agree_sign" not in cand:
            per_battery[name] = dict(status=STATUS_EXCLUDED,
                                       reason=f"prereg_candidate missing/skipped: {cand.get('skipped', 'no estimators_agree_sign key')}")
            continue
        estimators = battery.get("estimators", {})
        bad_estimators = [est_name for est_name in ("lda", "probe")
                          if estimators.get(est_name, {}).get("status") != "ok"]
        if bad_estimators:
            per_battery[name] = dict(status=STATUS_EXCLUDED,
                                       reason=f"estimator(s) {bad_estimators} did not complete (status != 'ok') "
                                              "-- agreement can't be assessed, not a genuine disagreement")
            continue
        agree = bool(cand["estimators_agree_sign"])
        per_battery[name] = dict(status=STATUS_PASS if agree else STATUS_FAIL, estimators_agree_sign=agree)
    overall = _aggregate_battery_statuses({k: v["status"] for k, v in per_battery.items()})
    n_pass = sum(v["status"] == STATUS_PASS for v in per_battery.values())
    n_decided = sum(v["status"] in (STATUS_PASS, STATUS_FAIL) for v in per_battery.values())
    return _criterion(
        overall,
        f"{n_pass}/{n_decided} decided batteries show estimators_agree_sign=True "
        f"(LDA-subspace vs cross-fitted linear-probe); {len(per_battery) - n_decided} excluded/operational-gap",
        dict(per_battery=per_battery),
    )


# ---------------------------------------------------------------------------
# C7 -- no collapse after controlling for clean EER / checkpoint quality /
# training corpus
# ---------------------------------------------------------------------------


def criterion_7_no_collapse(records: list[dict], eers: dict[str, dict[str, float]],
                             factor_corpus_map: dict, clean_corpora: tuple[str, ...] = ("inthewild", "replaydf", "ai4t")) -> dict:
    """factor_corpus_map keying: see _lookup_factor_corpus / criterion_2_association."""
    if not records:
        return _criterion(STATUS_PENDING, "no Phase A battery input provided")
    per_battery = {}
    used_keys = set()
    for rec in records:
        battery = rec["battery"]
        name = battery["name"]
        cand = rec.get("prereg_candidate", {})
        if _is_candidate_skipped(cand):
            per_battery[name] = dict(status=STATUS_EXCLUDED, reason=f"battery skipped: {cand.get('skipped')}")
            continue
        factor = battery.get("factor")
        corpus = battery.get("corpus")
        eval_corpus = _lookup_factor_corpus(factor_corpus_map, corpus, factor, used_keys)
        reliance = _per_checkpoint_reliance(battery, metric="alignment")
        rows = []
        any_not_estimable = any(r["status"] == STATUS_NOT_ESTIMABLE for r in reliance.values())
        for ckpt_name, rel in reliance.items():
            if rel["status"] != "ok":
                continue
            ckpt_eers = eers.get(ckpt_name, {})
            clean_vals = [ckpt_eers[c] for c in clean_corpora if c in ckpt_eers]
            if not clean_vals or eval_corpus is None or eval_corpus not in ckpt_eers:
                continue
            rows.append(dict(ckpt=ckpt_name, reliance=rel["value"], clean_eer=float(np.mean(clean_vals)),
                              target_eer=ckpt_eers[eval_corpus]))
        if len(rows) < 3:
            per_battery[name] = dict(
                status=STATUS_NOT_ESTIMABLE if any_not_estimable else STATUS_PENDING,
                reason=f"need >=3 checkpoints with reliance + clean EER + target-corpus EER, have {len(rows)}",
                rows=rows,
            )
            continue
        reliance_arr = np.array([r["reliance"] for r in rows])
        clean_arr = np.array([r["clean_eer"] for r in rows])
        target_arr = np.array([r["target_eer"] for r in rows])
        # Both sides must vary -- np.corrcoef on a constant array raises a
        # RuntimeWarning (divide by zero in the correlation formula) and
        # returns nan; checking only reliance_arr's variance (the original
        # bug here) let a constant target_arr (e.g. every checkpoint has
        # the identical target-corpus EER) slip a warning through.
        raw_sign = (np.sign(np.corrcoef(reliance_arr, target_arr)[0, 1])
                    if np.std(reliance_arr) > 0 and np.std(target_arr) > 0 else 0.0)
        design = np.column_stack([np.ones_like(clean_arr), clean_arr])
        coef, *_ = np.linalg.lstsq(design, reliance_arr, rcond=None)
        reliance_resid = reliance_arr - design @ coef
        resid_sign = (np.sign(np.corrcoef(reliance_resid, target_arr)[0, 1])
                      if np.std(reliance_resid) > 1e-12 and np.std(target_arr) > 0 else 0.0)
        survives = bool(raw_sign != 0 and raw_sign == resid_sign)
        per_battery[name] = dict(
            status=STATUS_PASS if survives else STATUS_FAIL,
            raw_sign=float(raw_sign), residual_sign=float(resid_sign), n_checkpoints=len(rows), rows=rows,
        )
    overall = _aggregate_battery_statuses({k: v["status"] for k, v in per_battery.items()})
    unmatched = sorted(str(k) for k in factor_corpus_map if k not in used_keys)
    return _criterion(
        overall,
        "clean-EER-residualized sign-survival per battery (n=3-ish checkpoints -- descriptive, "
        "not a powered significance test; see docs/gate_prereg.md §4 ambiguity 3)",
        dict(per_battery=per_battery, unmatched_factor_corpus_map_keys=unmatched),
    )


# ---------------------------------------------------------------------------
# C8 -- consistent effect direction across >=3 independently seeded
# replicates
# ---------------------------------------------------------------------------


def criterion_8_seeded_replicates(replicates: list[dict] | None, min_replicates: int = 3) -> dict:
    if replicates is None:
        return _criterion(
            STATUS_PENDING,
            "no head-replicate result file provided -- run scripts/head_replicates.py "
            "against the real cached embeddings (collaborator machine) and pass --head-replicates",
        )
    if len(replicates) < min_replicates:
        return _criterion(
            STATUS_PENDING,
            f"only {len(replicates)} replicate(s) present, need >= {min_replicates}",
            dict(n_replicates=len(replicates)),
        )
    signs = [np.sign(r["effect"]) for r in replicates if r.get("effect") is not None]
    if len(signs) < min_replicates:
        return _criterion(STATUS_PENDING, "fewer than the minimum replicates carried a finite effect value",
                           dict(n_finite=len(signs)))
    unanimous = len(set(signs)) == 1 and signs[0] != 0
    return _criterion(
        STATUS_PASS if unanimous else STATUS_FAIL,
        f"{len(signs)} seeded replicates, signs={signs}",
        dict(seeds=[r.get("seed") for r in replicates], effects=[r.get("effect") for r in replicates]),
    )


# ---------------------------------------------------------------------------
# Overall three-outcome classification (docs/gate_prereg.md §3)
# ---------------------------------------------------------------------------


def classify_overall(criteria: dict[str, dict]) -> str | None:
    """None if any criterion is not yet a decided pass/fail (pending_input
    or not_estimable) -- an overall verdict is only ever emitted once every
    criterion has actually resolved. Otherwise: strong success if all 8
    pass; failure if any of the sign-bearing criteria (C2, C4, C6, C8) is a
    directional reversal (fail); diagnostic-only otherwise."""
    statuses = {name: c["status"] for name, c in criteria.items()}
    if any(s not in (STATUS_PASS, STATUS_FAIL) for s in statuses.values()):
        return None
    if all(s == STATUS_PASS for s in statuses.values()):
        return "strong_success"
    sign_bearing = ("C2", "C4", "C6", "C8")
    if any(statuses.get(c) == STATUS_FAIL for c in sign_bearing):
        return "failure"
    return "diagnostic_only"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_gate(
    phase_a_paths: list[Path],
    secondary_backbone_paths: list[Path] | None = None,
    eer_paths: list[Path] | None = None,
    phase_b_out_root: Path = DEFAULT_PHASE_B_OUT_ROOT,
    head_replicates_path: Path | None = None,
    factor_corpus_map: dict[str, str] | None = None,
) -> dict:
    secondary_backbone_paths = secondary_backbone_paths or []
    eer_paths = eer_paths or []
    factor_corpus_map = factor_corpus_map or {}

    _log("gate: loading Phase A battery input(s)")
    records, warnings = load_phase_a_inputs(phase_a_paths)
    _log(f"gate: loaded {len(records)} battery record(s) from {len(phase_a_paths)} file(s), "
         f"{len(warnings)} warning(s)")

    secondary_records, secondary_warnings = load_phase_a_inputs(secondary_backbone_paths)
    warnings += secondary_warnings

    _log("gate: loading EER input(s)")
    eers, eer_warnings = load_eer_inputs(eer_paths)
    warnings += eer_warnings
    _log(f"gate: loaded EERs for {len(eers)} checkpoint(s)")

    _log("gate: checking Phase B embedding-cache readiness")
    phase_b_status = {}
    seen = set()
    for rec in records:
        for ckpt_stem in rec["checkpoints"]:
            for corpus in rec["join_stats"]:
                key = (ckpt_stem, corpus)
                if key in seen:
                    continue
                seen.add(key)
                real_names = _real_phase_b_cache_names(ckpt_stem, corpus)
                if real_names is None:
                    phase_b_status[f"{ckpt_stem}/{corpus}"] = dict(
                        status=STATUS_FAIL,
                        reason=f"corpus {corpus!r} has no known on-disk dataset directory in "
                               "this module's CORPUS_DIR table -- cannot construct the real "
                               "Phase B extraction path to check (add it to CORPUS_DIR if this "
                               "is a genuinely new corpus)",
                    )
                    continue
                real_ckpt_dir, real_corpus_dir = real_names
                phase_b_status[f"{ckpt_stem}/{corpus}"] = check_phase_b_cache(
                    phase_b_out_root, real_ckpt_dir, real_corpus_dir
                )

    _log("gate: loading head-replicate input (criterion 8)")
    replicates, replicate_warnings = load_head_replicates(head_replicates_path)
    warnings += replicate_warnings

    _log("gate: computing criteria")
    criteria = dict(
        C1=criterion_1_replication(records, secondary_records),
        C2=criterion_2_association(records, eers, factor_corpus_map),
        C3=criterion_3_grouped_bootstrap(records),
        C4=criterion_4_intervention_vs_random(records),
        C5=criterion_5_rank_stability(records),
        C6=criterion_6_estimator_agreement(records),
        C7=criterion_7_no_collapse(records, eers, factor_corpus_map),
        C8=criterion_8_seeded_replicates(replicates),
    )
    for name, c in criteria.items():
        _log(f"gate: {name} -> {c['status']}")
        unmatched = c.get("numbers", {}).get("unmatched_factor_corpus_map_keys")
        if unmatched:
            warnings.append(f"{name}: factor_corpus_map entries {unmatched} matched no battery -- possible typo")

    overall = classify_overall(criteria)
    _log(f"gate: overall classification = {overall!r}")

    return dict(
        schema_version=1,
        generated_at=_timestamp(),
        inputs=dict(
            phase_a_paths=[str(p) for p in phase_a_paths],
            secondary_backbone_paths=[str(p) for p in secondary_backbone_paths],
            eer_paths=[str(p) for p in eer_paths],
            phase_b_out_root=str(phase_b_out_root),
            head_replicates_path=str(head_replicates_path) if head_replicates_path else None,
        ),
        warnings=warnings,
        phase_b_cache_status=phase_b_status,
        criteria=criteria,
        overall_classification=overall,
    )


def _parse_factor_corpus_map(raw: list[str] | None) -> dict:
    """Parses --factor-corpus-map entries into the keying
    _lookup_factor_corpus expects: "corpus:factor=eval_corpus" -> a
    (corpus, factor) tuple key (for two batteries sharing a factor across
    different corpora, needing different eval corpora); plain
    "factor=eval_corpus" -> a bare-factor-string key (legacy, corpus-
    agnostic convenience, still honored)."""
    out = {}
    for item in raw or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected factor=corpus or corpus:factor=corpus, got {item!r}")
        key, eval_corpus = item.split("=", 1)
        if ":" in key:
            corpus, factor = key.split(":", 1)
            out[(corpus, factor)] = eval_corpus
        else:
            out[key] = eval_corpus
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase-a", nargs="+", type=Path, default=[],
                    help="Phase A battery JSON file(s), primary backbone (manifest-shaped or per-battery-shaped)")
    ap.add_argument("--phase-a-secondary-backbone", nargs="+", type=Path, default=[],
                    help="Phase A battery JSON file(s) for the second backbone (WavLM-Large), for criterion 1")
    ap.add_argument("--eer", nargs="+", type=Path, default=[],
                    help="cross_test.py-shaped per-checkpoint EER JSON file(s)")
    ap.add_argument("--phase-b-out-root", type=Path, default=DEFAULT_PHASE_B_OUT_ROOT,
                    help="root of scripts/extract_model_embeddings.py's output (default: %(default)s)")
    ap.add_argument("--head-replicates", type=Path, default=None,
                    help="JSON produced by scripts/head_replicates.py, for criterion 8")
    ap.add_argument("--factor-corpus-map", nargs="+", default=None,
                    help="factor=corpus (corpus-agnostic) or corpus:factor=corpus (corpus-specific, "
                         "needed when two batteries share a factor but need different scored corpora) "
                         "pairs mapping a battery's factor to an EER-scored corpus that shares it, for "
                         "criteria 2 and 7 (e.g. language=inthewild or diffssd:generator_id=inthewild)")
    ap.add_argument("--out", type=Path, default=DEFAULT_VERDICT_OUT, help="verdict JSON output path")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    verdict = run_gate(
        phase_a_paths=args.phase_a,
        secondary_backbone_paths=args.phase_a_secondary_backbone,
        eer_paths=args.eer,
        phase_b_out_root=args.phase_b_out_root,
        head_replicates_path=args.head_replicates,
        factor_corpus_map=_parse_factor_corpus_map(args.factor_corpus_map),
    )
    _write_json_atomic(args.out, verdict)
    _log(f"gate: wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
