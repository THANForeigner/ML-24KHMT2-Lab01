# Verification Results

Dataset SHA-256: `4d911088c4b12d60a450a9acae6b606f4119ebbb48679518e427a4fc00778472`

| Criterion | Status | Evidence |
|---|---|---|
| AC01 | PASS | nbformat validation, 12 sections, centralized config |
| AC02 | PASS | Actual archive CSV read; T01-T04 and metadata |
| AC03 | PASS | T05-T09; coercion/category/domain action logs |
| AC04 | PASS | Stable assignments + T41; raw SHA-256 unchanged |
| AC05 | PASS | T10-T13; F06-F11; per-feature interpretations |
| AC06 | PASS | T14-T18; paired GPA/Change; unverified floor explicitly N/A |
| AC07 | PASS | T19-T22; valid pair/group n; predictor-only matrices |
| AC08 | PASS | T23-T26; F32-F39; exact train bin edges and cell counts |
| AC09 | PASS | T27-T32; design rank/VIF/OLS + residual/influence diagnostics |
| AC10 | PASS | T33-T38; formulas, constraints, centering, fixed quadratic comparison |
| AC11 | PASS | T39-T42; explicit source, timing, role and bias restrictions |
| AC12 | PASS | All 105 planned IDs accounted; each saved artifact exists and decodes |
| AC13 | PASS | 12 generated FND records map RQ/evidence/implication; checked against current outputs |
| AC14 | PASS | 3 reports + manifest + split/run metadata exist |
| AC15 | PASS | Fresh-kernel nbclient execution on actual CSV; zero cell errors |
| AC16 | PASS | Second fresh kernel: split and key statistics compared |
| AC17 | PASS | Data access review: no test frame/target extraction, counts/integrity only; T41; predictors and preprocessing scopes checked |
| AC18 | PASS | T47; M01-M05 scenarios, evidence, eligibility, same metric protocol |

PASS means the implemented check passed on the attached CSV; it does not resolve the explicit provenance/timing/course-rubric limitations.
Formal optional tests remain disabled; the documented unknown GPA floor and hours additivity are intentionally not fabricated.