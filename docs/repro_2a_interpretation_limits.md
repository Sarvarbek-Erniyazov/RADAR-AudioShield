\# Pre-registered interpretation limits — 2a reproduction run (expected 2026-07-16)



Committed before any result from the collaborator run exists.



\## What PASS licenses

\- PASS (exit 0, all hash checks green, all EERs within ±0.002 of committed

&#x20; references) licenses exactly one claim: the repaired stack reproduces the

&#x20; e005-A / e007-A/B numbers as originally computed. It verifies the repairs

&#x20; did not change the measured outcomes.

\- PASS does NOT validate the numbers as scientifically meaningful, does not

&#x20; retroactively bless the pre-audit "breakthrough" framing, and does not

&#x20; promote ITW/ReplayDF/AI4T beyond development-tier status.

\- PASS unlocks: merge to main, tag v0.8-2a-correctness, Step 3 execution

&#x20; against real checkpoints.



\## What DIVERGENT triggers

\- Any EER outside ±0.002, any hash mismatch, or nonzero exit = DIVERGENT.

\- DIVERGENT triggers investigation, never adjustment: no tolerance widening,

&#x20; no selective re-runs, no "close enough." The log is committed either way.

\- Investigation order: (1) environment delta (his lockfile fallback path per

&#x20; A2, if taken), (2) data copy integrity (corpus hashes), (3) genuine

&#x20; behavioral change introduced by a 2a repair — in which case the repair

&#x20; chain is audited commit-by-commit before anything merges.

\- A DIVERGENT caused by a correctness repair (e.g., the e005-C manifest

&#x20; repoint) is a finding about the original numbers, not about the repair.

&#x20; The original number does not win by seniority.



\## What ±0.002 does and does not mean

\- The tolerance absorbs nondeterminism the stack cannot pin (GPU kernel

&#x20; ordering, BLAS differences across hardware). It is not a license to accept

&#x20; drift from code-path changes.

\- Two runs both "within tolerance" of the reference may still differ from

&#x20; each other by up to 0.004; cross-machine agreement claims are not licensed.



\## Environment caveat (pre-declared)

\- If A2's fallback path (requirements.txt + fresh freeze) is taken, the run

&#x20; is still admissible, but any DIVERGENT result is first attributed to

&#x20; environment delta until the lockfile diff rules it out.

