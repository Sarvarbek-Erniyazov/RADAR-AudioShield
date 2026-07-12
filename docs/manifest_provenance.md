# Factor-metadata provenance & known gaps (manifests/v2)
Backfilled by scripts/extend_manifests.py from utt_id/path structure. Audit §4.6/§4.8; Roadmap v3 Step 2a Commit 3.

## Disclosed limitations (carry into paper data appendix)
- attack column is a constant placeholder (single value `openvoicev2`) in inthewild & replaydf; NOT used for generator_id.
- replaydf generator_id derives from path (authoritative). inthewild/fakeorreal/ai4t spoofs have no per-file generator → NA.
- asvspoof5 generator = collapsed 'asvspoof5spoof' (84% = spoof fraction); per-attack codes need protocol files (future hook).
- ai4t/inthewild/asvspoof5 speaker_id, ai4t language = NA pending corpus metadata. diffssd source/speaker/lang only for LibriSpeech reals (26%).

## Usable factor axes today
generator: replaydf, diffssd, asvspoof5(collapsed), mlaad(when embedded)
language: asvspoof5, fakeorreal, inthewild, replaydf, vctk, mlaad
channel: replaydf | speaker/source: vctk(100%), replaydf, ai4t(source), diffssd(partial)

## Sampler provenance (audit §4.4, Commit 4)
- e002–e007 used WeightedRandomSampler with weights (1/n_corpora)(1/n_classes_in_corpus)(1/count).
  The 1/n_classes_in_corpus term gave bona-only VCTK double per-cell mass → measured 62.5/37.5 class
  skew AND MI(corpus;class) > 0 (corpus predicted class). Fixed in Commit 4 (joint weighting + MI guard).
- BMIQuotaSampler in samplers.py was NEVER used by the training scripts — its per-batch bona-domain
  guarantees did not run in any e002–e007 experiment. (Second inactive-module instance; cf. audit §1.)
