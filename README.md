# Agentic SOCaaS Validation

Simulation and validation suite for the experimental validation section of:

> *Agentic AI-Driven SOC-as-a-Service for Optimizing Incident Response in
> Cloud Environments: A Conceptual Framework.* Submitted to *Sensors* (MDPI).
>
> Abdulaziz Y. Alhumaidi, Faisal A. Al-Qadda, Albandri Alsumayt, and Majed Alshammari.

This repository contains the code, generated data, figures, and tables for
four seeded, fully reproducible simulation experiments that validate the
proposed Agentic AI-driven SOC-as-a-Service (SOCaaS) framework:

| Experiment | Framework layer | What it validates |
|---|---|---|
| E1 — Automated triage and enrichment | Layer 2 | Random Forest alert classification; alert-volume reduction vs. threat leakage under two suppression policies |
| E2 — Retrieval grounding | Layer 3 | Playbook / MITRE ATT&CK mapping accuracy, grounded (TF-IDF corpus) vs. ungrounded (titles only) |
| E3 — Governance-gated orchestration | Layers 4–5 | Discrete-event simulation of manual, narrow-AI, and agentic SOC models; load scaling; risk-threshold sensitivity sweep |
| E4 — Closed-loop scenario and audit | Layer 6 | Monte Carlo privilege-escalation workflow; evidence-chain completeness; audit-logging overhead |

## Reproducing the results

**Google Colab (recommended):** upload `socaas_validation_colab.py` (or paste
it into a cell) and run. All figures and tables are regenerated and a zip of
outputs is downloaded automatically.

**Locally:**

```bash
pip install -r requirements.txt
python socaas_validation_colab.py
```

All random processes are seeded (`SEED = 42`); every number reported in the
manuscript is exactly reproducible from this code. Outputs are written to
`./outputs/` (PNG figures at 300 dpi, tables as CSV) and packaged as
`socaas_validation_outputs.zip`.

## Repository contents

```
socaas_validation_colab.py   # complete simulation suite (E1–E4)
requirements.txt             # pinned Python dependencies
outputs/                     # pre-generated reference outputs
  figure1_e1_triage_classification.png
  figure2_e2_retrieval_grounding.png
  figure3_e3_orchestration.png
  figure4_e3_threshold_sensitivity.png
  figure5_e4_audit_traceability.png
  table1_e1_classification_report.csv
  table2_e1_operational_metrics.csv
  table3_e2_retrieval_accuracy.csv
  table4_e3_orchestration_comparison.csv
  table5_e3_load_scaling.csv
  table6_e3_threshold_sweep.csv
  table7_e4_audit_metrics.csv
  table8_e4_per_step_coverage.csv
CITATION.cff                 # citation metadata
LICENSE                      # MIT
```

Mapping to the manuscript: repository tables 1–8 correspond to manuscript
Tables 3–8 (tables 1–2 here provide the per-class report and operational
metrics that jointly populate manuscript Tables 3–4); repository figures 1–5
correspond to manuscript Figures 5–9.

## Data

All data are synthetic and generated at runtime by the script itself under
the calibration described in Section 6.1 of the manuscript (overlapping
class-conditional distributions, 3% label noise, ambiguous retrieval queries,
per-step evidence-logging failure probabilities). No proprietary, operational,
or human-subject data are used.

## Citation

Archived at Zenodo: version DOI to be added upon the v1.0.0 release. Until then, cite via `CITATION.cff`.

## License

MIT — see `LICENSE`.
