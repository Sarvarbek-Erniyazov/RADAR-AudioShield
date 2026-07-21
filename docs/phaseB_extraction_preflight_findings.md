# Step 3 Phase B extraction preflight — Task 1 findings

Read-only investigation, no GPU touched, `run_reliance_battery.py` /
`run_gate.py` / `gate_prereg.md` untouched. Every claim below is quoted or
line-cited from the real source, per the brief's standing rule.

## 1. Extraction point: is the saved 256-d vector exactly `binary.fc`'s input?

**Yes, confirmed exactly, by direct code reading on both sides (model
definition and extraction script) — not inferred.**

`src/audioshield/models/detector.py:61-64`:
```python
def embed(self, waveform: torch.Tensor) -> torch.Tensor:
    seq = self.ssl(waveform)        # [B, T, H]
    pooled = self.pool(seq)         # [B, 2H]
    return self.proj(pooled)        # [B, D]
```

`src/audioshield/models/detector.py:66-72` (`forward`, the training/eval
path that actually produces `spoof_logit`):
```python
def forward(self, waveform, grl_lambda=0.0) -> dict:
    z = self.embed(waveform)
    ...
    "spoof_logit": self.binary(z),
```

`src/audioshield/models/heads.py:25-31` (`BinaryHead`, no other transform
between its input and the linear layer):
```python
class BinaryHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z).squeeze(-1)   # [B] logit
```

So `logit = binary.fc(z).squeeze(-1)` where `z = embed(waveform)` exactly
— no intermediate step, no alternate hook.

`scripts/extract_model_embeddings.py:396` (the actual extraction call):
```python
emb = model.embed(batch["waveform"].to(device))
```

`scripts/extract_model_embeddings.py:131,137` (`embedding_dim_of`, used to
size every disk-space/schema check — reads the dimension from the same
layer whose weight multiplies `z`, never hardcoded):
```python
def embedding_dim_of(model) -> int:
    return int(model.binary.fc.in_features)
```

**Verdict: the extracted `emb` is exactly `z`, the tensor `binary.fc`
multiplies.** No guessed hook point — `embed()` is called directly, and
its return value is definitionally the same tensor `forward()` feeds into
`self.binary(z)`.

## 2. Output schema — cross-checked against `run_gate.py`'s `check_phase_b_cache`

**No mismatch found.** Both sides agree, verified field-by-field:

| Field | `extract_model_embeddings.py` (writer) | `run_gate.py::check_phase_b_cache` (reader) |
|---|---|---|
| Path layout | `<out-root>/<ckpt-stem>/<corpus-dir>/shard_{i:04d}.npz` (`main()`, `out_dir = out_root / ckpt_path.stem / _corpus_dir_from_rows(rows)`) | `out_root / checkpoint_stem / corpus_dir` |
| Keys | `paths`, `emb`, `meta` (`_write_shard_atomic`) | requires exactly `{"paths","emb","meta"} <= set(npz.files)` |
| `emb` shape | `(n_rows_in_shard, embedding_dim)`, 2-D, `float16`/`float32` per `--dtype` | reads `emb.shape[0]` (rows), `emb.shape[1]` (embedding_dim), `str(emb.dtype)` |
| `meta` | `np.array(json.dumps(meta))`, a 0-d string array | (checked in the preflight script below; `check_phase_b_cache` itself doesn't currently parse `meta`'s *contents*, only that the key exists — noted, not a mismatch, just an unused-but-present field on the reader side) |
| Atomicity | temp file + `os.replace` (`_write_shard_atomic`) | reader only ever sees a complete file by construction |
| Resume | shard file exists → skip, never recomputed | not applicable to the reader |

The one thing worth flagging (not a bug, a scope note): `check_phase_b_cache`
validates *presence* and *shape*, not `meta`'s field contents — it doesn't
currently check that `meta["corpus_dir"]` matches the directory it was
found in, or that `checkpoint_sha256` matches a specific checkpoint. That's
fine for its current purpose (a readiness/structural check for the gate),
but if `meta` content-validation ever matters, it isn't there yet.

## 3. THE CONSUMPTION GAP — resolved explicitly: **this path does not exist yet**

**This is a real, confirmed code gap, not a configuration/CLI question.**
Two independent, compounding reasons, both found by reading
`scripts/run_reliance_battery.py` directly:

### 3a. Shape assumption: `load_corpus_embeddings` cannot read a layer-less cache

`scripts/run_reliance_battery.py:225-252`:
```python
def load_corpus_embeddings(cache_root: Path, corpus_dir: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    ...
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as data:
            emb = data["emb"]
            paths = data["paths"]
        n_layers = emb.shape[1]
        if not (0 <= layer < n_layers):
            raise ValueError(...)
        all_paths.append(paths)
        all_emb.append(emb[:, layer, :].astype(np.float32))
```

This assumes `emb` is **3-D**: `(n, n_layers, hidden_size)` — the existing
XLS-R-300M raw-backbone cache's shape (25 hidden-state layers, pre-pooling).
Phase B's `emb` is **2-D**: `(n, 256)` (a single pooled+projected vector,
no layer axis at all — confirmed in §1/§2 above). Pointed at a Phase B
shard, `emb.shape[1]` would read as `256` (not a layer count) — `layer=9 <
256` would pass the range check without raising, and then
`emb[:, layer, :]` would raise `IndexError: too many indices for array:
array is 2-dimensional, but 3 were indexed` the moment it actually ran. A
loud crash, not silent wrong data — but a crash, confirming no code path
here tolerates Phase B's shape today.

### 3b. Architectural assumption: ONE embedding space is shared across all checkpoints

Even fixing 3a would not be enough. `scripts/run_reliance_battery.py:1699-1739`
(`main()`) loads each corpus's embedding matrix **once, per corpus,
before any checkpoint is even loaded**:
```python
corpora_needed = sorted({b["corpus"] for b in batteries})
corpus_data: dict[str, dict] = {}
...
for corpus in corpora_needed:
    ...
    cache_paths, cache_emb = load_corpus_embeddings(cache_root, corpus_dir, args.layer)
    joined_df, joined_emb, stats = join_cache_to_manifest(...)
    ...
    corpus_data[corpus] = dict(df=joined_df, emb=joined_emb, emb_full=emb_full)
```
`checkpoints = load_all_checkpoints(...)` is loaded **separately**, and
the **same** `corpus_data[corpus]["emb"]` matrix is then reused for
**every** checkpoint's `alignment`/`r_var`/`prediction_change` — only `w`
(the classifier weight) varies by checkpoint (`load_task_direction` per
checkpoint, `checkpoints[run]["w"]`). This is correct and necessary for
the *current* cache, because XLS-R-300M is a **frozen, shared backbone**
across e007_A/B/C — the raw hidden states genuinely don't depend on which
checkpoint you're evaluating.

Phase B breaks that invariant: `embed()`'s output depends on **that
checkpoint's own trained `proj`/pooling weights** (confirmed in §1 — `z =
proj(pool(ssl(waveform)))`, and `proj`/`pool` are checkpoint-specific,
trained parameters, not shared). e007_A, e007_B, and e007_C each produce a
**different 256-d space** for the same audio. There is no `Z_per_checkpoint`
concept anywhere in this file — the entire per-checkpoint metric stack
(`alignment(w, U)`, `r_var(w, U, Sigma_z)`, `prediction_change`) implicitly
assumes `U` (the factor subspace, fit once per battery) and `Z` live in the
*same* space `w` does, which is only true today because `w`'s space
(XLS-R-300M raw) is checkpoint-invariant. Once `w` moves to a
checkpoint-specific `embed()` space, `U` and `Z` would need to be
**re-fit per checkpoint**, not reused.

### Verdict

**The consumption path does not exist.** Closing it is not a CLI flag or a
`--cache-root` redirect — `--cache-root` exists (`--cache-root`, line
1607) and *could* point at `analysis/step3/_embcache_modelspace/<ckpt>/`
directory-by-directory, but even then §3a's shape assumption breaks
immediately, and §3b means a correct fix requires restructuring `main()`
to load a **separate embedding matrix per checkpoint** and **re-fit the
subspace estimators per checkpoint** rather than once per corpus — a real
architecture change to the battery loop, not a quick patch. This is
exactly the kind of change the brief and Roadmap v3 are both explicit
should NOT happen inside this preflight-only session (`run_reliance_battery.py`
is untouched here).

**No exact battery re-run command exists today.** Per the brief's own
contingency: this is surfaced as the finding, not papered over.

Confirmed via a repo-wide search that no other module attempts this
integration yet (`grep -rn "embcache_modelspace\|modelspace"` across
`scripts/`, `src/`, `tests/` returns nothing outside
`extract_model_embeddings.py`/`run_gate.py`/their own tests) — this is a
genuinely open gap, not something implemented elsewhere and missed here.
