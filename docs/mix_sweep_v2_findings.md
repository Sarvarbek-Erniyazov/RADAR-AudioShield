# Mix Sweep v2 Findings — Decision Point A (OOD Dev-Tier Corpus-Mix Selection)

Source data: `_mix_sweep_v2_results.json` / `_mix_sweep_v2.log` (all numbers below are computed
programmatically from that JSON; none are hand-transcribed). Provenance is v2, the OOD dev-tier
sweep — see §9 for why v1 is excluded from selection.

## 1. Protocol summary

63 candidate mixes over train corpora {ASVspoof5, DiffSSD, FakeOrReal, VCTK, MLAAD, ODSS_CLEAN},
scored with frozen XLS-R-300M layer-9 embeddings + `LogisticRegression` (sklearn, CPU backend,
`max_iter=1000`, 300k stratified-train cap, seed 13). Evaluation is OOD against three dev-tier
targets (ITW, ReplayDF, AI4T) via manifest-defined eval sets, with cache extras dropped at load
time (AI4T 277, ReplayDF 480, ASVspoof5 1184). Labels come from manifests/v2 plus verified
directory rules, and all `spoof_frac` assertion gates passed at load time.

## 2. Load-block provenance (from `_mix_sweep_v2.log`)

```
loading TRAIN corpora...
    01_ASVspoof5: manifest=323307 matched=323307 (100.0%) cache_extras_dropped=1184
  01_ASVspoof5: 323307 samples, spoof_frac=0.84
    03_DiffSSD: manifest=94226 matched=94226 (100.0%) cache_extras_dropped=0
  03_DiffSSD: 94226 samples, spoof_frac=0.74
    07_FakeOrReal: manifest=69300 matched=69300 (100.0%) cache_extras_dropped=0
  07_FakeOrReal: 69300 samples, spoof_frac=0.50
    09_VCTK: manifest=44242 matched=44242 (100.0%) cache_extras_dropped=0
  09_VCTK: 44242 samples, spoof_frac=0.00
  10_MLAAD: 456000 samples, spoof_frac=1.00
  12_ODSS_CLEAN: 26954 samples, spoof_frac=0.70
loading TARGET corpora...
    02_In-the-Wild: manifest=31779 matched=31779 (100.0%) cache_extras_dropped=0
  02_In-the-Wild: 31779 samples, spoof_frac=0.37
    04_ReplayDF: manifest=52320 matched=52320 (100.0%) cache_extras_dropped=480
  04_ReplayDF: 52320 samples, spoof_frac=0.50
    05_AI4T: manifest=3148 matched=3134 (99.6%) cache_extras_dropped=277
  05_AI4T: 3134 samples, spoof_frac=0.29

63 mixes x 3 OOD targets
  [fit backend: sklearn(CPU) [cuml unavailable: ModuleNotFoundError]]
```

Of the 63 candidate mixes, 2 single-corpus entries (`VCTK` alone, `MLAAD` alone) were skipped —
both are single-class corpora (`spoof_frac` 0.00 and 1.00 respectively) and cannot train a binary
`LogisticRegression`. **61 mixes are valid** and used for every table below.

## 3. Full ranked table (all 61 valid mixes, by mean OOD EER)

| # | Mix | Mean OOD EER | ITW | ReplayDF | AI4T | n_train |
|---|-----|--------------|-----|----------|------|---------|
| 1 | ASVspoof5+DiffSSD+FakeOrReal | 0.1785 | 0.0833 | 0.2668 | 0.1853 | 300,000 |
| 2 | ASVspoof5+DiffSSD+VCTK+ODSS_CLEAN | 0.1787 | 0.1001 | 0.2480 | 0.1880 | 300,000 |
| 3 | ASVspoof5+DiffSSD+FakeOrReal+ODSS_CLEAN | 0.1790 | 0.0806 | 0.2644 | 0.1920 | 300,000 |
| 4 | ASVspoof5+DiffSSD | 0.1803 | 0.0991 | 0.2620 | 0.1799 | 300,000 |
| 5 | ASVspoof5+DiffSSD+FakeOrReal+VCTK+ODSS_CLEAN | 0.1804 | 0.0816 | 0.2604 | 0.1992 | 300,000 |
| 6 | ASVspoof5+DiffSSD+FakeOrReal+VCTK | 0.1808 | 0.0869 | 0.2659 | 0.1898 | 300,000 |
| 7 | ASVspoof5+DiffSSD+VCTK | 0.1826 | 0.1000 | 0.2665 | 0.1812 | 300,000 |
| 8 | ASVspoof5+DiffSSD+ODSS_CLEAN | 0.1851 | 0.0947 | 0.2692 | 0.1916 | 300,000 |
| 9 | DiffSSD+FakeOrReal+VCTK+MLAAD+ODSS_CLEAN | 0.1868 | 0.0753 | 0.3042 | 0.1808 | 300,000 |
| 10 | ASVspoof5+DiffSSD+FakeOrReal+VCTK+MLAAD+ODSS_CLEAN | 0.1888 | 0.0724 | 0.2993 | 0.1947 | 300,000 |
| 11 | FakeOrReal+MLAAD+ODSS_CLEAN | 0.1894 | 0.0656 | 0.3069 | 0.1956 | 300,000 |
| 12 | ASVspoof5+FakeOrReal+MLAAD+ODSS_CLEAN | 0.1919 | 0.0632 | 0.3112 | 0.2014 | 300,000 |
| 13 | FakeOrReal+VCTK+MLAAD+ODSS_CLEAN | 0.1923 | 0.0653 | 0.3143 | 0.1974 | 300,000 |
| 14 | DiffSSD+FakeOrReal+MLAAD+ODSS_CLEAN | 0.1940 | 0.0732 | 0.3128 | 0.1961 | 300,000 |
| 15 | ASVspoof5+DiffSSD+FakeOrReal+MLAAD+ODSS_CLEAN | 0.1942 | 0.0728 | 0.3043 | 0.2055 | 300,000 |
| 16 | ASVspoof5+FakeOrReal+VCTK+MLAAD+ODSS_CLEAN | 0.1944 | 0.0672 | 0.3122 | 0.2037 | 300,000 |
| 17 | ASVspoof5+DiffSSD+FakeOrReal+VCTK+MLAAD | 0.1954 | 0.0764 | 0.3067 | 0.2032 | 300,000 |
| 18 | ASVspoof5+FakeOrReal+VCTK+MLAAD | 0.2013 | 0.0672 | 0.3130 | 0.2239 | 300,000 |
| 19 | ASVspoof5+DiffSSD+VCTK+MLAAD+ODSS_CLEAN | 0.2072 | 0.1115 | 0.2924 | 0.2176 | 300,000 |
| 20 | ASVspoof5+DiffSSD+MLAAD+ODSS_CLEAN | 0.2078 | 0.1110 | 0.3113 | 0.2010 | 300,000 |
| 21 | DiffSSD+FakeOrReal+MLAAD | 0.2088 | 0.0808 | 0.3248 | 0.2207 | 300,000 |
| 22 | ASVspoof5+DiffSSD+FakeOrReal+MLAAD | 0.2099 | 0.0842 | 0.3186 | 0.2270 | 300,000 |
| 23 | DiffSSD+FakeOrReal+VCTK+MLAAD | 0.2125 | 0.0979 | 0.3271 | 0.2127 | 300,000 |
| 24 | ASVspoof5+FakeOrReal+MLAAD | 0.2142 | 0.0801 | 0.3157 | 0.2467 | 300,000 |
| 25 | FakeOrReal+VCTK+MLAAD | 0.2144 | 0.0767 | 0.3425 | 0.2239 | 300,000 |
| 26 | DiffSSD+VCTK+MLAAD+ODSS_CLEAN | 0.2166 | 0.1186 | 0.3257 | 0.2055 | 300,000 |
| 27 | DiffSSD+FakeOrReal+VCTK+ODSS_CLEAN | 0.2187 | 0.1135 | 0.2923 | 0.2503 | 234,722 |
| 28 | DiffSSD+ODSS_CLEAN | 0.2197 | 0.1593 | 0.2750 | 0.2248 | 121,180 |
| 29 | ASVspoof5+DiffSSD+VCTK+MLAAD | 0.2198 | 0.1050 | 0.3153 | 0.2391 | 300,000 |
| 30 | ASVspoof5+FakeOrReal+ODSS_CLEAN | 0.2203 | 0.0649 | 0.3159 | 0.2799 | 300,000 |
| 31 | ASVspoof5+FakeOrReal+VCTK+ODSS_CLEAN | 0.2206 | 0.0662 | 0.3229 | 0.2728 | 300,000 |
| 32 | ASVspoof5+DiffSSD+MLAAD | 0.2214 | 0.1130 | 0.3127 | 0.2387 | 300,000 |
| 33 | DiffSSD+FakeOrReal+ODSS_CLEAN | 0.2221 | 0.1210 | 0.2965 | 0.2490 | 190,480 |
| 34 | DiffSSD+FakeOrReal+VCTK | 0.2236 | 0.1379 | 0.3287 | 0.2041 | 207,768 |
| 35 | DiffSSD+FakeOrReal | 0.2244 | 0.1396 | 0.3241 | 0.2095 | 163,526 |
| 36 | DiffSSD+MLAAD+ODSS_CLEAN | 0.2246 | 0.1251 | 0.3250 | 0.2239 | 300,000 |
| 37 | ASVspoof5+VCTK+MLAAD+ODSS_CLEAN | 0.2271 | 0.1191 | 0.3092 | 0.2530 | 300,000 |
| 38 | ASVspoof5+VCTK+MLAAD | 0.2280 | 0.1035 | 0.3193 | 0.2611 | 300,000 |
| 39 | ASVspoof5+VCTK+ODSS_CLEAN | 0.2280 | 0.0638 | 0.3175 | 0.3028 | 300,000 |
| 40 | FakeOrReal+MLAAD | 0.2286 | 0.0785 | 0.3408 | 0.2665 | 300,000 |
| 41 | DiffSSD+VCTK+MLAAD | 0.2287 | 0.1291 | 0.3322 | 0.2248 | 300,000 |
| 42 | DiffSSD+VCTK+ODSS_CLEAN | 0.2312 | 0.1425 | 0.2940 | 0.2571 | 165,422 |
| 43 | ASVspoof5+FakeOrReal+VCTK | 0.2322 | 0.0762 | 0.3440 | 0.2764 | 300,000 |
| 44 | ASVspoof5+ODSS_CLEAN | 0.2326 | 0.0698 | 0.3340 | 0.2939 | 300,000 |
| 45 | ASVspoof5+VCTK | 0.2326 | 0.0695 | 0.3305 | 0.2979 | 300,000 |
| 46 | ASVspoof5+FakeOrReal | 0.2333 | 0.0744 | 0.3437 | 0.2817 | 300,000 |
| 47 | ASVspoof5 | 0.2360 | 0.0771 | 0.3357 | 0.2952 | 300,000 |
| 48 | DiffSSD+VCTK | 0.2370 | 0.1633 | 0.3500 | 0.1978 | 138,468 |
| 49 | DiffSSD | 0.2418 | 0.1834 | 0.3443 | 0.1978 | 94,226 |
| 50 | ASVspoof5+MLAAD+ODSS_CLEAN | 0.2421 | 0.1391 | 0.3372 | 0.2499 | 300,000 |
| 51 | DiffSSD+MLAAD | 0.2473 | 0.1507 | 0.3571 | 0.2342 | 300,000 |
| 52 | ASVspoof5+MLAAD | 0.2574 | 0.1580 | 0.3294 | 0.2849 | 300,000 |
| 53 | FakeOrReal+VCTK+ODSS_CLEAN | 0.2661 | 0.0919 | 0.3263 | 0.3800 | 140,496 |
| 54 | FakeOrReal+ODSS_CLEAN | 0.2710 | 0.0996 | 0.3305 | 0.3827 | 96,254 |
| 55 | VCTK+ODSS_CLEAN | 0.2759 | 0.1060 | 0.3355 | 0.3863 | 71,196 |
| 56 | ODSS_CLEAN | 0.2862 | 0.1229 | 0.3398 | 0.3957 | 26,954 |
| 57 | FakeOrReal+VCTK | 0.3169 | 0.1444 | 0.4603 | 0.3459 | 113,542 |
| 58 | VCTK+MLAAD+ODSS_CLEAN | 0.3214 | 0.1544 | 0.5013 | 0.3087 | 300,000 |
| 59 | FakeOrReal | 0.3244 | 0.1449 | 0.4618 | 0.3665 | 69,300 |
| 60 | MLAAD+ODSS_CLEAN | 0.4589 | 0.2469 | 0.7257 | 0.4042 | 300,000 |
| 61 | VCTK+MLAAD | 0.8135 | 0.7315 | 0.9785 | 0.7304 | 300,000 |

Skipped (single-class, cannot train LR): `VCTK` alone, `MLAAD` alone.

## 4. Single-corpus baselines

| Corpus | Mean OOD EER | ITW | ReplayDF | AI4T | n_train |
|--------|--------------|-----|----------|------|---------|
| ASVspoof5 | 0.2360 | 0.0771 | 0.3357 | 0.2952 | 300,000 |
| DiffSSD | 0.2418 | 0.1834 | 0.3443 | 0.1978 | 94,226 |
| ODSS_CLEAN | 0.2862 | 0.1229 | 0.3398 | 0.3957 | 26,954 |
| FakeOrReal | 0.3244 | 0.1449 | 0.4618 | 0.3665 | 69,300 |
| VCTK | — (skipped, single-class, spoof_frac=0.00) | | | | |
| MLAAD | — (skipped, single-class, spoof_frac=1.00) | | | | |

## 5. Per-corpus inclusion effect (mean OOD EER, with vs. without, over all 61 valid mixes)

| Corpus | n (with) | Mean EER (with) | n (without) | Mean EER (without) | Δ (with − without) |
|--------|----------|------------------|--------------|----------------------|----------------------|
| DiffSSD | 32 | 0.2071 | 29 | 0.2673 | **−0.0601** |
| ASVspoof5 | 32 | 0.2088 | 29 | 0.2654 | **−0.0566** |
| FakeOrReal | 32 | 0.2159 | 29 | 0.2576 | **−0.0417** |
| ODSS_CLEAN | 32 | 0.2267 | 29 | 0.2457 | −0.0191 |
| VCTK | 31 | 0.2404 | 30 | 0.2308 | +0.0096 |
| MLAAD | 31 | 0.2432 | 30 | 0.2280 | +0.0152 |

Lower (negative) delta = inclusion helps. ASVspoof5 and DiffSSD are the two strongest positive
contributors; VCTK and MLAAD are mildly negative on average across the full mix population.

## 6. MLAAD per-target effect (with vs. without MLAAD, over all 61 valid mixes)

| Target | n (with) | Mean EER (with) | n (without) | Mean EER (without) | Δ (with − without) |
|--------|----------|------------------|--------------|----------------------|----------------------|
| AI4T | 31 | 0.2476 | 30 | 0.2618 | **−0.0142 (helps)** |
| ITW | 31 | 0.1230 | 30 | 0.1053 | +0.0177 (slightly hurts) |
| ReplayDF | 31 | 0.3589 | 30 | 0.3169 | +0.0420 (hurts) |

This is the headline nuance for MLAAD: the average per-corpus effect (§5, +0.0152 mean OOD) hides
a divergent, target-specific pattern — MLAAD inclusion **helps** AI4T, **hurts** ReplayDF
noticeably, and **slightly hurts** ITW.

## 7. Best mix per target

| Target | Best mix | Target EER | Mean OOD EER (that mix) | n_train |
|--------|----------|------------|---------------------------|---------|
| ITW | ASVspoof5+FakeOrReal+MLAAD+ODSS_CLEAN | 0.0632 | 0.1919 | 300,000 |
| ReplayDF | ASVspoof5+DiffSSD+VCTK+ODSS_CLEAN | 0.2480 | 0.1787 | 300,000 |
| AI4T | ASVspoof5+DiffSSD | 0.1799 | 0.1803 | 300,000 |

No single mix wins on all three targets — the factor-specific divergence (§6) shows up again here:
the ITW-best mix contains MLAAD, the AI4T-best mix does not.

## 8. Decisions carried forward

**(a) Working hypothesis for Step 5/6:** small ASVspoof5+DiffSSD-anchored mixes. Every mix in the
top 8 by mean OOD EER contains both ASVspoof5 and DiffSSD, and the plain 2-corpus
`ASVspoof5+DiffSSD` mix (rank 4, mean=0.1803, n_train=300,000) outperforms the full 6-corpus mix
(rank 10, mean=0.1888, n_train=300,000). More corpora is not monotonically better; the two
strongest single-corpus contributors (§5) already carry most of the OOD generalization signal.

**(b) VCTK slight negative inclusion effect** (§5, Δ=+0.0096) is consistent with the project's
§4.4 bona-only confound policy — VCTK contributes only bona-fide samples (spoof_frac=0.00), so its
presence in a mix shifts the bona/spoof balance without adding spoof diversity, matching the known
confound this policy was written to guard against.

**(c) MLAAD is not included in the base mix at the frozen-probe level.** Per-target results (§6)
show it helps AI4T but hurts ReplayDF and (slightly) ITW; a flat "MLAAD hurts" reading of the
mean-effect table (§5) is **not** the finding — it is reserved for the multilingual evaluation axis
planned for Step 5, where its value is expected to show up on non-English-heavy targets that this
sweep's three dev-tier targets cannot measure (see §9 (c) below).

## 9. Pre-registered interpretation limits

- Frozen-LR screening ≠ fine-tuned verdict. These numbers rank candidate mixes for a frozen
  XLS-R-300M layer-9 probe with a linear classifier; they are not a prediction of fine-tuned model
  performance and should not be read as final corpus-selection ground truth.
- MLAAD's spoof-only mass (spoof_frac=1.00, 456,000 samples) crowds out bona-fide mass under the
  300k stratified-train cap whenever MLAAD is included, which can distort the effective class
  balance seen by the classifier independent of MLAAD's intrinsic sample quality.
- The three OOD targets (ITW, ReplayDF, AI4T) are English-heavy and therefore cannot measure
  MLAAD's multilingual value — this sweep is structurally blind to the axis MLAAD is expected to
  contribute on.
- The per-target divergence (§6, §7) is a factor-specific data effect consistent with the
  project's reliance hypothesis, but it is observational here, not causal — no controlled
  ablation isolates MLAAD's contribution from correlated changes in mix composition or
  effective training size.

## 10. Provenance note

Sweep v1 (`_mix_sweep_results.json`, in-pool split) is **invalid for selection** — its evaluation
split is in-distribution/in-pool and the resulting metric is saturated, so it cannot discriminate
between candidate mixes on OOD generalization. Sweep v2 (`_mix_sweep_v2_results.json`, this
document) uses genuinely held-out OOD dev-tier targets (ITW, ReplayDF, AI4T) and **supersedes v1**
for all corpus-mix decisions from this point forward.
