"""BMI-aware batch sampler.

Each batch is built to a quota so BMI's pairwise Kwok term is always defined:
  - >= n_bona_domains distinct bona-fide domains, each with >= min_per_domain samples
  - the rest of the batch filled with spoof drawn from >= 2 corpora
Draws ONLY from rows passed in (caller passes v1-training rows only -> held-out
corpora can never enter a training batch).
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterator, Sequence

from torch.utils.data import Sampler

from .manifest import ManifestRow


class BMIQuotaSampler(Sampler[list[int]]):
    def __init__(
        self,
        rows: Sequence[ManifestRow],
        batch_size: int = 8,
        n_bona_domains: int = 3,
        min_per_domain: int = 2,
        seed: int = 13,
        drop_last: bool = True,
    ) -> None:
        self.rows = list(rows)
        self.batch_size = batch_size
        self.n_bona_domains = n_bona_domains
        self.min_per_domain = min_per_domain
        self.rng = random.Random(seed)
        self.drop_last = drop_last

        # index pools
        self.bona_by_domain: dict[str, list[int]] = defaultdict(list)
        self.spoof_by_corpus: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(self.rows):
            if r.target == 0:
                self.bona_by_domain[r.bona_fide_source].append(i)
            else:
                self.spoof_by_corpus[r.corpus].append(i)

        self.bona_domains = list(self.bona_by_domain)
        self.spoof_corpora = list(self.spoof_by_corpus)
        if len(self.bona_domains) < 2:
            raise ValueError(f"BMI needs >=2 bona-fide domains, got {self.bona_domains}")

        bona_quota = self.n_bona_domains * self.min_per_domain
        if bona_quota >= batch_size:
            raise ValueError(
                f"bona quota {bona_quota} >= batch_size {batch_size}; "
                "increase batch_size or reduce n_bona_domains/min_per_domain")
        self.bona_quota = bona_quota
        self.spoof_quota = batch_size - bona_quota
        self._len = max(1, len(self.rows) // batch_size)

    def __len__(self) -> int:
        return self._len

    def _draw(self, pool: list[int], k: int) -> list[int]:
        if len(pool) >= k:
            return self.rng.sample(pool, k)
        return [self.rng.choice(pool) for _ in range(k)]  # with replacement if small

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self._len):
            batch: list[int] = []
            # bona: pick n distinct domains, min_per_domain each
            n_dom = min(self.n_bona_domains, len(self.bona_domains))
            chosen = self.rng.sample(self.bona_domains, n_dom)
            for d in chosen:
                batch += self._draw(self.bona_by_domain[d], self.min_per_domain)
            # any remaining bona slots (if n_dom < n_bona_domains) -> extra from chosen
            while len(batch) < self.bona_quota:
                d = self.rng.choice(chosen)
                batch += self._draw(self.bona_by_domain[d], 1)
            # spoof: spread across >=2 corpora
            n_sc = min(2, len(self.spoof_corpora)) if len(self.spoof_corpora) >= 2 else 1
            sc = self.rng.sample(self.spoof_corpora, n_sc)
            per = max(1, self.spoof_quota // len(sc))
            for c in sc:
                batch += self._draw(self.spoof_by_corpus[c], per)
            while len(batch) < self.batch_size:
                c = self.rng.choice(self.spoof_corpora)
                batch += self._draw(self.spoof_by_corpus[c], 1)
            batch = batch[: self.batch_size]
            self.rng.shuffle(batch)
            yield batch
