# Phase 08 - Report: results by criterion

Spec: `spec.md` (frozen); plan: `plan.md`.
Measured on 2026-10-10 on branch `fase-08-report`, on this machine (the run files live in the git-ignored `data/`).

## C1 - one command, input digests, byte-equal rerun

- Command: `uv run edge-rag report`, run twice in a row; both runs printed the same digest.
- `tournament.json` SHA-256 after each run: `7a480d93b9c90c770da3ac24abe16966ee95df28109288b6a747263d86b3da02` (identical).
- The export's `inputs` section records the SHA-256 of each phase run file (Phases 02-07 `results.json` and the Phase 07 `annex.json`).
- Fixture test: `tests/test_report.py::test_rerun_is_byte_equal_and_records_every_input_digest` passes.
- Status: met.

## C2 - states and labels as the outcome code wrote them

- Fixture test: `tests/test_report.py::test_states_counts_and_cost_labels_are_copied_from_the_run_files` passes.
- Every table in `report.md` sits in a generated block written from the export; `--check` fails if any block differs from what the export renders.
- Status: met.

## C3 - every prose figure traced

- `uv run edge-rag report --check`: exit 0, "tables equal to the export, every prose figure traced".
- Planted figure: replacing `231` by `98,765` in the exam reading made `--check` exit 1 with `untraced figure 98,765`; the reverted page passes again (exit 0).
- A first plant, `232`, passed the check because that integer is also a value elsewhere in the export: the checker traces a figure to any export value at its precision, not to the value the sentence means; this is a limit of the C3 guard, and the reading pass of C5 covers it.
- Checker defect found and fixed in this phase: the `048` of `FS@2,048` was read as a figure of its own; the number pattern now refuses a comma before a figure, with a regression assertion in `test_checker_passes_traced_figures_and_fails_a_planted_one` that fails on the old pattern.
- Status: met.

## C4 - every value labelled; `not recorded` visible

- The checker refuses a prose line with figures and no label; `--check` exit 0.
- Reading pass over the generated tables: each row carries its phase's label column; the `unrecorded` block lists the cost cells never recorded, and the exam cost column keeps `not recorded`.
- Status: met.

## C5 - the answer without promotion

- The answer section of `report.md` states: no Phase 03-05 entrant; terrain bar 8 of 9 set-class cells won, MuSiQue class A lost to G-A1; HotpotQA wins an upper bound; G-A2 not run (cost gate); exam class L `rrf4` loses to G-L, classes R and A `j-rrf4` tie G-R (not losing, not a win); QASPER has few multi-evidence questions.
- Adversarial review against the Phase 06 and 07 pages: pending (review by the coordinator).
- Status: pending review.

## C6 - money per phase and in total

- The `money` block gives per phase the time x rate (derived), balance delta (measured) and invoice (measured) where they exist, with the source file of each row, and the derived totals.
- Gaps kept visible, not estimated: Phases 00 and 01 rented no pod; Phase 03 has no clean balance delta (deviation 03.1); the Phase 06 invoice was pending at its close, so no invoice total.
- Status: met.

## C7 - closed phases untouched, no spend

- `git diff origin/main --stat -- docs/plans/fase-00* docs/plans/fase-01* ... docs/plans/fase-07*`: empty (base `origin/main` 8154bb1; the local `main` ref is stale and was not used).
- `git status --short data`: empty; nothing under `data/` changed.
- No money spent: no pod, no paid API; the phase reads files only.
- Status: met.

## C8 - README, checks, review

- README: Status line says the project closed after Phase 08; a results section links to `report.md` and `tournament.json`.
- `npm run check`: exit 0 (hooks test, pytest with one skip, ruff, mypy).
- sdd-review adversarial pass: pending (review by the coordinator).
- Status: pending review.

## Limits

- The report is reproducible only where the git-ignored `data/` run files exist; outside this machine a reader checks the export against its input digests.
- The C3 guard matches a figure to any export value at its precision, so a wrong figure that happens to equal another export value passes it.
