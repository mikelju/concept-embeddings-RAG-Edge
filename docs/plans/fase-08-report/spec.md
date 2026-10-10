# Phase 08 - Report: specification

Status: approved (frozen)
Approved by the author: 2026-10-10
Master plan: `../0_plan_maestro.md`

## Objective

One committed write-up, `report.md`, that answers the tournament's question with the figures Phases 00-07 already measured.
The question, from the master plan: does a cheap, fast strategy sit on the quality-cost frontier against the literature's ghosts on the terrain, and keep its place on a locked exam corpus?
The milestone "After 07" asks for this write-up and no phase owns it.
The project closes after this phase (author, 2026-10-10; the game is dropped), so the report is the project's last word and its figures are also left machine-readable in `tournament.json`, built from the same source as the report's tables.

## Scope

In:
- `tournament.json`, written by code from the phase run files (`data/phase02/results.json` to `data/phase07/results.json` and `data/phase07/annex.json`), with the SHA-256 of every input.
  It holds, per set and system, the figures of record (FS@2,048 and the other metrics each phase reports), the verdict states and the paired-test counts as the outcome code wrote them, the cost cells with their labels (`not recorded` kept as such), and the money per phase from the master plan's decisions table.
- `report.md`: hand-written prose around tables generated from `tournament.json` by the same code, in this order:
  question and answer first; setup (terrain, exam, metric, reading budget, cost classes, ghosts); each candidate phase (03-05) with its verdict and cost; the literature bar (06); the exam (07); money spent; limits and biases; what the result does and does not show; what a successor could try.
- A number checker: every figure in the hand-written prose matches a value in `tournament.json` or carries a pointer to the committed page it is read from; untraced figures fail the check.
- README: the Status line and a short results section linking to `report.md`.

Out:
- Any new measurement, rerun, refit or spend; filling a `not recorded` cost cell after the fact.
- Editing any closed phase's frozen documents or run files.
- The voice-to-order side studies (`data/side-v2o/`): not part of the tournament, publication not decided.
- The game (dropped by the author) and any change to how a verdict is decided.

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

None. Resolved by the author on 2026-10-10, each on the agent's recommendation:
- D1. `tournament.json` is committed next to `report.md`: aggregate figures, no corpus text, like the committed Phase 00 `ledger.json`.
- D2. The export and the report tables are written by a new `report` command in `src/edge_rag/`, reusing the `phase_results.py` readers, with tests.
- D3. The old project's systems (p10-a/b/c, p14, j-p10b, j-union) appear in the tables labelled context, as in Phase 02.
