# Phase 08 - Report: specification

Status: draft
Approved by the author: -
Master plan: `../0_plan_maestro.md`

## Objective

One committed write-up, `report.md`, that answers the tournament's question with the figures Phases 00-07 already measured.
The question, from the master plan: does a cheap, fast strategy sit on the quality-cost frontier against the literature's ghosts on the terrain, and keep its place on a locked exam corpus?
The milestone "After 07" asks for this write-up and no phase owns it; the game (Phase 09) needs a measured-results export that does not exist yet (charter section 5: "The page reads a JSON exported from the measured artifacts").
Both are built here, from one source, so the report and the game cannot disagree.

## Scope

In:
- `tournament.json`, written by code from the phase run files (`data/phase02/results.json` to `data/phase07/results.json` and `data/phase07/annex.json`), with the SHA-256 of every input.
  It holds, per set and system, the figures of record (FS@2,048 and the other metrics each phase reports), the verdict states and the paired-test counts as the outcome code wrote them, the cost cells with their labels (`not recorded` kept as such), and the money per phase from the master plan's decisions table.
- `report.md`: hand-written prose around tables generated from `tournament.json` by the same code, in this order:
  question and answer first; setup (terrain, exam, metric, reading budget, cost classes, ghosts); each candidate phase (03-05) with its verdict and cost; the literature bar (06); the exam (07); money spent; limits and biases; what the result does and does not show; what would come next.
- A number checker: every figure in the hand-written prose matches a value in `tournament.json` or carries a pointer to the committed page it is read from; untraced figures fail the check.
- README: the Status line and a short results section linking to `report.md`.

Out:
- Any new measurement, rerun, refit or spend; filling a `not recorded` cost cell after the fact.
- Editing any closed phase's frozen documents or run files.
- The voice-to-order side studies (`data/side-v2o/`): not part of the tournament, publication not decided.
- The game itself (Phase 09) and any change to how a verdict is decided.

## Acceptance criteria

Frozen at approval. Changing them needs a deviation approved by the author.

| ID | Observable criterion | How it is checked |
|---|---|---|
| C1 | `tournament.json` is written by one command from the phase run files, records the SHA-256 of each input, and a second run writes it byte-equal | run the command twice; compare digests; a test on a small fixture |
| C2 | Every verdict state and comparison label in `tournament.json` and `report.md` is the one the phase outcome code wrote; none is retyped by hand | test: states read from each `results.json` equal those in the export; the report's tables come from the export |
| C3 | Every figure in the hand-written prose of `report.md` is traced: it matches an export value or names the committed page it is read from; 0 untraced figures | the number checker exits 0 on `report.md` and fails on a planted wrong figure |
| C4 | Every reported value carries a label (measured, derived, exploratory, interpretation, projection); cost cells keep their phase's label and `not recorded` stays visible | checker plus a reading pass over the tables |
| C5 | The answer states the measured outcome without promotion: class L loses to G-L on the exam, classes R and A tie G-R; the terrain bar wins 8 of 9 set-class cells and loses MuSiQue class A; the HotpotQA wins are an upper bound; G-A2 not run; QASPER has few multi-evidence questions | adversarial review compares the answer section with the phase 06 and 07 pages |
| C6 | Money per phase and in total is reported with its three figures where they exist (time x rate, balance delta, invoice), read from the master plan and phase plans | table in `report.md` with a source per row |
| C7 | No closed phase's documents or run files change; no money is spent | `git diff main -- docs/plans/fase-0[0-7]*` empty; RunPod balance unchanged |
| C8 | README links to `report.md`; `npm run check` exits 0; the sdd-review adversarial pass leaves no open blocker | command output and the review record |

## Constraints and risks

- The run files live only in the git-ignored `data/`; the report is reproducible on this machine, and the export plus its input digests are what a reader outside it can check.
- The metric set differs by phase (Phase 02 lacks Share@5 and R@2/10/20; Phase 07 reports FS@2,048 only); the report shows what each phase measured and no common column is invented.
- Some costs were never recorded (inherited old-pod Dense embeddings in Phases 03-05, several Phase 07 first-stage timings); the report says so instead of estimating them.
- Laptop USD is 0 by assumption (energy not counted); stated once in the setup.
- No control is needed: this phase measures nothing and declares no new winner.
- Risk: prose drifting from the figures; C3 and C2 are the guard.

## Open decisions

The agent's recommendation comes first in each.

- D1. Commit `tournament.json` next to `report.md` (recommended: it is aggregate figures, no corpus text, like the committed Phase 00 `ledger.json`, and it is what Phase 09 reads) or keep it under `data/` only.
- D2. Where the export code lives: a new `report` command in `src/edge_rag/` reusing `phase_results.py` readers (recommended: one source with the phase pages) or a script under the phase's `tools/`.
- D3. Include the old project's systems (p10-a/b/c, p14, j-p10b, j-union) in the report's tables as context (recommended: yes, labelled context, as Phase 02 did) or only the tournament's own systems and ghosts.
