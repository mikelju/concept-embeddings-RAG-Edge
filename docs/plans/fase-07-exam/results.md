# Phase 07 - Exam on QASPER: results

In scope: 1,337 questions (1,106 single-evidence, 231 multi-evidence); FS and paired tests measured; states and verdict written by code.

## Verdict

- **Class L** (pooled QASPER test, in-scope questions): rrf4 928 against G-L 1058 (the bar): 55 wins, 185 losses, exact p 1.24e-17, **loss**.
- **Class R** (pooled QASPER test, in-scope questions): j-rrf4 1107 against G-R 1102 (the bar): 8 wins, 3 losses, exact p 0.227, **tie**. Not losing, not a win. Against the literature control j-rrf3 (report only): tie, 2 wins, 0 losses, exact p 0.5.
- **Class A** (pooled QASPER test, in-scope questions): j-rrf4 1107 against G-R 1102 (the bar): 8 wins, 3 losses, exact p 0.227, **tie**. Not losing, not a win. Against the literature control j-rrf3 (report only): tie, 2 wins, 0 losses, exact p 0.5.

## Systems

| System | Role | FS@2,048 pooled | Single | Multi | FS@2,048 within-paper | Upper reference | Cost |
|---|---|---:|---:|---:|---:|---|---|
| rrf4 | entrant | 928 | 832 | 96 | - | - | not recorded |
| j-rrf4 | entrant | 1107 | 983 | 124 | - | - | not recorded |
| j-rrf3 | control | 1105 | 983 | 122 | - | - | not recorded |
| p10-b | context | 838 | 755 | 83 | - | - | not recorded |
| p14 | context | 811 | 734 | 77 | - | - | not recorded |
| G-L | ghost | 1058 | 948 | 110 | - | - | not recorded |
| G-R | ghost | 1102 | 982 | 120 | - | - | not recorded |
| G-R2 | ghost | 1056 | 928 | 128 | - | - | not recorded |
| G-A1 | ghost | 591 | 568 | 23 | - | - | not recorded |
| bm25 | component | 770 | 683 | 87 | 969 | - | not recorded |
| dense | component | 775 | 716 | 59 | 1038 | - | not recorded |
| hop | component | 68 | 64 | 4 | - | - | not recorded |
| within-j-strong | within-paper control only | - | - | - | 1129 | - | not recorded |
