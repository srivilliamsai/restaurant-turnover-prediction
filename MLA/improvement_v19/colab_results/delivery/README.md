# V19 research and plan

**V19 started. New Colab account verified; owned free T4 and isolated environments prepared. Engineering admission precedes development scoring.**

Start with [PLAN.md](PLAN.md). The proposed core models are Causilo, Kumo Tabular Small and TabDPT v1.3, with Kumo Medium admitted only if engineering and the complete workload fit. Keep the scored V14 until a qualified candidate actually improves its portal score.

- [Official-source review and screenshot corrections](RESEARCH_SOURCES.md)
- [Machine-readable proposed protocol](protocol_proposed.json)
- [Independent V14–V18 audit: 110 fold outputs, 40 comparisons](history_audit.json)
- [Published checkpoint metadata](checkpoint_review.json)
- [New account verification](account_check.json)
- [Preserved V14 CSV](reference/scored_v14.csv) and [notebook](reference/scored_v14.ipynb)

Reproduce the historical audit from the project root without training:

```sh
PYTHONPATH=.runtime_deps improvement_v9/colab_cli_env/bin/python improvement_v19/audit_history.py
```

The sign-in helper is user-operated and keeps credentials in the existing Colab CLI private store. It does not allocate a runtime. Public research snapshots are evidence, not an installed model environment or completed feasibility test.
