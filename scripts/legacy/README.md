# Legacy data-pipeline scripts

Preserved for provenance. These ran on the collaborator's training machine
(outside the repo) and produced committed artifacts. Retained as documentation
of how those artifacts were made; not part of the current pipeline.

- `_extract_xlsr.py` — produced `datasets/_embcache_xlsr300m/` (XLS-R-300M,
  all 25 layers, float16, sharded .npz with `paths`/`emb`/`dur`). Reference
  implementation for future backbone extraction passes.
- `_gated_delete.py` — deleted raw audio only after verifying a complete
  embedding cache existed. This is the mechanism by which ASVspoof5, DiffSSD
  and VCTK raw audio was removed post-embedding (see
  docs/repro_2a_harness_fixes.md, run #2 deviation).
- `_build_checksums.py` — produced `manifests/checksums/*_SHA256.txt`
  (commit c528680), used to verify the DiffSSD re-acquisition (2026-07-19,
  94,601/94,604 exact).
- `_mlaad_pipeline.py` — MLAAD acquisition/batching tooling (Roadmap Step 0).

- `_kaggle_vault.sh` + `_kaggle_vault.log` — off-machine backup mechanism
  (chunked tar -> private Kaggle dataset, SHA256-gated, upload verified by
  remote file count). Attempted 2026-07-09 for 05_AI4T, 02_In-the-Wild and
  07_FakeOrReal; all three failed at the upload step with a kaggle-CLI temp
  path error (`AppData\Local\Temp\.kaggle\uploads\...` not found) and were
  never retried. The corpora were subsequently deleted post-embedding by
  `_gated_delete.py`, leaving no off-machine copy. Retained as provenance and
  as the basis for a fixed backup path (see roadmap: checkpoint/corpus
  backup).
