# Probe grouping — deferred upgrade (audit §4.7)
cross_test.py bona-source probe currently runs grouped_probe with meta=None (ungrouped,
honest-baseline). Full recording-level grouping requires threading source_id/speaker_id
into the `cache` tuple built earlier in cross_test (currently (lab, sc, emb, bsrc, clusters)).
Upgrade path: add source_id array to cache -> pass meta={"source_id": ...} to grouped_probe.
Until then, the reported bona probe is ungrouped: it may still overstate decodability vs a
fully grouped probe, so treat "probe pinned" claims as UPPER bounds on decodability.
The loop_e002.py corpus probe (has ManifestRows) IS fully groupable — wire that with meta.

## Both probe sites defer grouping (Step 2b)
loop_e002.probe_corpus_during_train: batch dict is {waveform, corpus_id} — no source_id.
Full grouping requires threading source_id through the dataset __getitem__ + collate into
the batch dict, a data-layer change deferred to 2b to keep 2a's correctness fixes isolated
from the training path validated by the reproduction gate (HV: split repair phases).
Both probes currently report ungrouped honest-baseline (balanced acc vs true majority),
which is an UPPER bound on decodability — conservative for "probe pinned" claims.
