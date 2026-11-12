# V18 completed experiment

**Status: development and export complete. Retain V14: no V18 candidate qualified.**

Both admitted fine-tuning recipes completed ten development folds each, with ten matched unadapted control folds. All ten fixed blends were compared. The best was 50% FT512 plus 50% V14: mean relative local gain **0.039920%**, below the frozen **0.20%** requirement, and negative gain on assignment 42. FT1024 also failed qualification. EXA8 was excluded before accuracy testing because its forced Flash Attention kernel was unsupported on the owned T4.

Confirmation and new submission artifacts were skipped under the fixed protocol. The scored V14 CSV and matching executed notebook are preserved byte-for-byte under KEEP_V14 names. V18 has no portal result.

- [Final result summary](delivery/V18_Result_Summary.md)
- [Full measured report](delivery/V18_Report.md)
- [Independent reconstruction](delivery/independent_review.json)
- [Verification and reproduction](REPRODUCTION.md)

Read [PLAN_PROPOSED.md](PLAN_PROPOSED.md) for the complete experiment and resource envelope. It proposes eighteen elapsed hours on free T4/Mac compute, shared research time, and a protected three-hour delivery reserve. The user instructed proceeding without waiting for additional deadline information; the eighteen-hour cap and three-hour reserve set this run’s cutoffs.

Three fixed components: FT512, EXA8 and FT1024. At most twenty-five blend comparisons and one locked finalist. Keep V14 unless every qualification and reproduction gate passes.

- [Research and source decisions](RESEARCH_SOURCES.md)
- [Independent V17 reconstruction](review_evidence.json)
- [Proposed machine-readable protocol](protocol_proposed.json)
- [Exact reviewed public sources](source_review_registry.json)
- [Published release and checkpoint metadata](release_review_registry.json)

Reproduce the read-only audit with the existing compatible runtime from the project root:

```sh
PYTHONPATH=.runtime_deps improvement_v9/colab_cli_env/bin/python improvement_v18/review_v17.py
```

The model implementation and all original labels, folds, controls, source hashes and the original execution clock are retained. A local completion handoff required manually advancing the already-declared no-candidate decision/export steps on the same runtime; its receipt is included. No extra training recipe or allocation was introduced. Prior V17 files and ledgers remain unchanged.
