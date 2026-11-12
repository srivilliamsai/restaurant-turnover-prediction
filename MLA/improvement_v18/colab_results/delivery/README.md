# V18 planning package

**Status: authorized and implementing. Local tests, audits and original-Mac control reproduction passed. The owned free T4 is running control and candidate engineering pilots.**

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

The execution coordinator and fixed model adapters are implemented. No V18 submission is produced before qualification and fresh reproduction. Prior V17 files and ledgers remain unchanged.
