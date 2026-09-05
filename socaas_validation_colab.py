"""
==============================================================================
Experimental Validation Package
Agentic AI-Driven SOC-as-a-Service for Optimizing Incident Response in
Cloud Environments: A Conceptual Framework
------------------------------------------------------------------------------
Simulation and validation suite (Google Colab compatible).

Experiments:
  E1 - Automated triage and enrichment (Layer 2): Random Forest alert
       classification on synthetic multi-source cloud telemetry.
  E2 - Retrieval grounding (Layer 3): playbook / MITRE ATT&CK retrieval
       accuracy, grounded (TF-IDF) vs. ungrounded (keyword) baseline.
  E3 - Governance-gated orchestration (Layers 4-5): discrete-event
       simulation of manual, narrow-AI, and agentic-governed SOC models,
       with a risk-threshold sensitivity sweep.
  E4 - Closed-loop scenario and audit completeness (Layer 6): Monte Carlo
       privilege-escalation workflow with evidence-chain verification.

All figures (PNG, 300 dpi) and tables (CSV) are written to ./outputs and,
when run in Google Colab, packaged into a single zip and auto-downloaded.

Reproducibility: all random processes are seeded (SEED = 42).
==============================================================================
"""

import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
rng = np.random.default_rng(SEED)
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 9, "figure.dpi": 300,
                     "axes.grid": True, "grid.alpha": 0.3})

CLASSES = ["Benign / noise", "Credential abuse", "Privilege escalation",
           "Lateral movement", "Data exfiltration"]

# =============================================================================
# EXPERIMENT 1 - Automated triage and enrichment (Layer 2)
# =============================================================================
print("=" * 70)
print("E1: Automated triage and enrichment (Random Forest)")
print("=" * 70)

N_EVENTS = 20000
# Class mix reflects realistic cloud SOC telemetry: dominated by benign noise.
class_probs = [0.70, 0.09, 0.07, 0.08, 0.06]
y = rng.choice(len(CLASSES), size=N_EVENTS, p=class_probs)

# 12 features derived from multi-source telemetry (IAM, API, network,
# endpoint, SIEM enrichment). Class-conditional means overlap deliberately
# so the task is non-trivial and results remain credible.
feat_names = [
    "api_call_rate", "failed_auth_ratio", "new_geo_flag", "token_reuse_count",
    "iam_policy_changes", "priv_role_assumptions", "east_west_conn_rate",
    "unique_targets_contacted", "storage_read_volume_mb", "egress_volume_mb",
    "off_hours_flag", "asset_criticality"]

# mean matrix: rows = classes, cols = features
M = np.array([
    # benign
    [1.0, 0.05, 0.05, 0.2, 0.1, 0.1, 1.0, 2.0, 40,  20,  0.15, 1.5],
    # credential abuse
    [2.2, 0.55, 0.45, 2.6, 0.3, 0.5, 1.3, 2.5, 55,  30,  0.45, 2.0],
    # privilege escalation
    [2.0, 0.25, 0.20, 1.2, 1.9, 2.4, 1.5, 3.0, 70,  35,  0.40, 2.8],
    # lateral movement
    [2.4, 0.20, 0.15, 0.9, 0.5, 0.8, 3.4, 6.5, 60,  40,  0.35, 2.3],
    # data exfiltration
    [1.8, 0.15, 0.25, 0.8, 0.4, 0.6, 1.6, 3.2, 210, 260, 0.50, 2.6],
])
S = np.array([0.8, 0.18, 0.25, 1.0, 0.7, 0.9, 1.0, 1.8, 45, 45, 0.30, 0.9])

X = M[y] + rng.normal(0, 1, (N_EVENTS, len(feat_names))) * S
X = np.clip(X, 0, None)

# 3% label noise (mislabeled ground truth), typical of SOC-derived datasets
noise_idx = rng.choice(N_EVENTS, size=int(0.03 * N_EVENTS), replace=False)
y_noisy = y.copy()
y_noisy[noise_idx] = rng.integers(0, len(CLASSES), size=len(noise_idx))

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_noisy, test_size=0.30, stratify=y_noisy, random_state=SEED)

rf = RandomForestClassifier(n_estimators=200, max_depth=14,
                            min_samples_leaf=3, random_state=SEED, n_jobs=-1)
rf.fit(X_tr, y_tr)
y_pred = rf.predict(X_te)
y_proba = rf.predict_proba(X_te)

rep = classification_report(y_te, y_pred, target_names=CLASSES,
                            output_dict=True, zero_division=0)
t1 = pd.DataFrame(rep).T.round(3)
t1.to_csv(f"{OUT}/table1_e1_classification_report.csv")
print(t1)

# Operational metrics under two suppression policies:
#   argmax policy      - suppress whenever predicted class is benign
#   conservative gate  - suppress only when P(benign) >= 0.90 (risk-averse)
benign = 0
mask_benign = (y_te == benign)
macro_f1 = f1_score(y_te, y_pred, average="macro")

def suppression_metrics(suppress_mask):
    vol_red = suppress_mask.mean()                       # not escalated
    fp_sup = suppress_mask[mask_benign].mean()           # benign recall
    leak = suppress_mask[~mask_benign].mean()            # threats suppressed
    return vol_red, fp_sup, leak

sup_argmax = (y_pred == benign)
sup_conserv = (y_proba[:, benign] >= 0.90)
va, fa, la = suppression_metrics(sup_argmax)
vc, fc, lc = suppression_metrics(sup_conserv)

op = pd.DataFrame({
    "Metric": ["Test events", "Macro F1", "Weighted F1",
               "Alert volume reduction - argmax policy",
               "Benign suppression rate - argmax policy",
               "Threat leakage - argmax policy",
               "Alert volume reduction - conservative gate (P>=0.90)",
               "Benign suppression rate - conservative gate",
               "Threat leakage - conservative gate"],
    "Value": [len(y_te), round(macro_f1, 3),
              round(f1_score(y_te, y_pred, average="weighted"), 3),
              f"{va:.1%}", f"{fa:.1%}", f"{la:.2%}",
              f"{vc:.1%}", f"{fc:.1%}", f"{lc:.2%}"]})
op.to_csv(f"{OUT}/table2_e1_operational_metrics.csv", index=False)
print(op.to_string(index=False))

# Figure 1: confusion matrix + per-class F1
cm = confusion_matrix(y_te, y_pred, normalize="true")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
im = ax[0].imshow(cm, cmap="Blues", vmin=0, vmax=1)
ax[0].set_xticks(range(5)); ax[0].set_yticks(range(5))
short = ["Benign", "Cred. abuse", "Priv. esc.", "Lateral", "Exfil."]
ax[0].set_xticklabels(short, rotation=35, ha="right")
ax[0].set_yticklabels(short)
ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("True")
ax[0].set_title("(a) Normalized confusion matrix")
ax[0].grid(False)
for i in range(5):
    for j in range(5):
        ax[0].text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                   color="white" if cm[i, j] > 0.5 else "black", fontsize=7)
fig.colorbar(im, ax=ax[0], fraction=0.046)

f1s = [rep[c]["f1-score"] for c in CLASSES]
ax[1].bar(short, f1s, color="#33658A")
ax[1].axhline(macro_f1, ls="--", color="crimson", lw=1,
              label=f"Macro F1 = {macro_f1:.3f}")
ax[1].set_ylim(0, 1); ax[1].set_ylabel("F1-score")
ax[1].set_title("(b) Per-class F1")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/figure1_e1_triage_classification.png", bbox_inches="tight")
plt.close()

# =============================================================================
# EXPERIMENT 2 - Retrieval grounding (Layer 3)
# =============================================================================
print("\n" + "=" * 70)
print("E2: Retrieval grounding (playbook / ATT&CK mapping)")
print("=" * 70)

PLAYBOOKS = {
    "PB01 Credential compromise (T1078 Valid Accounts)":
        "valid account abuse stolen credentials login anomalous authentication "
        "password spray token theft mfa fatigue identity provider",
    "PB02 Privilege escalation (T1548 Abuse Elevation)":
        "privilege escalation role assumption iam policy modification admin "
        "rights elevation permission grant escalate root sts assume-role",
    "PB03 Lateral movement (T1021 Remote Services)":
        "lateral movement east west internal remote service ssh rdp smb "
        "pivot host to host propagation internal reconnaissance",
    "PB04 Data exfiltration (T1567 Exfil to Cloud Storage)":
        "exfiltration data transfer egress bucket copy large download "
        "storage sync external destination staging compression",
    "PB05 Ransomware in cloud workloads (T1486)":
        "ransomware encryption of data mass file modification extension "
        "rename ransom note backup deletion snapshot removal",
    "PB06 Cryptomining (T1496 Resource Hijacking)":
        "cryptomining resource hijacking cpu spike gpu instance spawn "
        "coin miner pool connection billing anomaly compute abuse",
    "PB07 API abuse and enumeration (T1580 Cloud Discovery)":
        "api enumeration discovery describe list calls reconnaissance "
        "cloud infrastructure mapping unusual read-only bursts",
    "PB08 Persistence via new IAM user (T1136 Create Account)":
        "persistence create account new iam user access key creation "
        "backdoor user long-term credentials rogue identity",
    "PB09 Security control tampering (T1562 Impair Defenses)":
        "disable logging cloudtrail stop guardduty suspend agent removal "
        "impair defenses tamper monitoring delete log group",
    "PB10 Compromised CI/CD pipeline (T1195 Supply Chain)":
        "supply chain pipeline build artifact poisoning dependency "
        "malicious package deploy stage secret leak repository",
    "PB11 Denial of service / availability (T1498)":
        "denial of service flood availability degradation request surge "
        "resource exhaustion throttling autoscaling abuse",
    "PB12 Insider data access misuse (T1530 Cloud Storage Object)":
        "insider misuse excessive access sensitive object read unusual "
        "working hours data browsing entitlement creep",
}
pb_names = list(PLAYBOOKS.keys())

# 60 incident descriptions (5 per playbook), written with paraphrase and
# distractor terms; 20% receive cross-domain noise to emulate ambiguity.
TEMPLATES = {
    0: ["multiple failed logins followed by success from new country for service identity",
        "authentication burst using reused token against identity provider endpoints",
        "suspicious sign-in pattern password spray across federated accounts",
        "stolen access key used for anomalous console login",
        "mfa push fatigue then unusual session issued"],
    1: ["sts assume role chain ending in admin policy attachment",
        "iam policy modified to grant wildcard permissions to low-tier role",
        "unexpected elevation of service account to administrator rights",
        "role assumption spike with permission grants outside change window",
        "user escalated to root-equivalent via policy version rollback"],
    2: ["east-west ssh connections fanning out across private subnets",
        "internal host contacting many peers over smb after initial foothold",
        "pivot behavior remote service logins between workloads",
        "propagation of sessions host to host inside the vpc",
        "internal reconnaissance followed by remote desktop hops"],
    3: ["large egress transfer to unknown external bucket during off hours",
        "mass object download then sync to external storage endpoint",
        "compressed archives staged and pushed to outside destination",
        "sustained high-volume data copy from sensitive prefix",
        "abnormal outbound bandwidth to file-sharing service"],
    4: ["mass file modification with extension rename and note dropped",
        "snapshots deleted then volumes encrypted across fleet",
        "backup deletion followed by encryption burst on shares",
        "ransom note artifacts and mass write amplification",
        "workload files encrypted after backup removal"],
    5: ["cpu pegged on fleet with connections to mining pool",
        "unexpected gpu instances spawned billing anomaly detected",
        "coin miner process beaconing to pool from containers",
        "compute abuse spike instances created outside pipeline",
        "resource hijacking sustained cpu with pool traffic"],
    6: ["burst of describe and list api calls across services",
        "read-only enumeration mapping infrastructure inventory",
        "reconnaissance via unusual discovery api sequences",
        "systematic listing of buckets roles and instances",
        "api enumeration from newly seen principal"],
    7: ["new iam user created with long-term access keys off hours",
        "rogue identity added and access key generated for persistence",
        "backdoor user account provisioned outside identity workflow",
        "creation of secondary credentials on dormant account",
        "unsanctioned account created then attached to group"],
    8: ["cloudtrail logging stopped in two regions by api call",
        "guardduty detector suspended and log group deleted",
        "monitoring agent removed from production instances",
        "audit trail disabled prior to sensitive operations",
        "defense tampering log delivery channel deleted"],
    9: ["build pipeline pulled unverified dependency then deployed",
        "artifact poisoning suspected malicious package in build",
        "ci secret leaked and used to push rogue deploy stage",
        "repository dependency swap introduced malicious code",
        "compromised build step injected artifact into release"],
    10: ["request surge exhausting service quota availability degraded",
         "flood traffic triggering throttling and autoscaling abuse",
         "resource exhaustion attack degrading api availability",
         "sustained high request rate causing service brownout",
         "denial of service pattern against public endpoints"],
    11: ["employee browsing sensitive objects far beyond role needs",
         "excessive reads of confidential storage during odd hours",
         "entitlement creep exploited for broad data browsing",
         "insider accessing customer records without ticket linkage",
         "unusual sensitive object access from internal principal"],
}
DISTRACT = ["cloud", "alert raised by siem", "analyst note attached",
            "seen in us-east-1", "ticket opened", "correlated with edr"]

queries, q_labels = [], []
for cls, sents in TEMPLATES.items():
    for s in sents:
        extra = " ".join(rng.choice(DISTRACT, size=2, replace=False))
        # 20% ambiguity: inject one term from a *different* playbook corpus
        if rng.random() < 0.20:
            other = int(rng.integers(0, 12))
            if other != cls:
                tok = rng.choice(PLAYBOOKS[pb_names[other]].split(), size=1)[0]
                extra += " " + tok
        queries.append(s + " " + extra)
        q_labels.append(cls)
q_labels = np.array(q_labels)

# Grounded: TF-IDF vectorization of playbook corpus, cosine retrieval
vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
PB_M = vec.fit_transform(PLAYBOOKS.values())
Q_M = vec.transform(queries)
sims = cosine_similarity(Q_M, PB_M)
rank = np.argsort(-sims, axis=1)
top1 = (rank[:, 0] == q_labels).mean()
top3 = np.mean([q_labels[i] in rank[i, :3] for i in range(len(queries))])

# Ungrounded baseline: the reasoning component has access only to playbook
# identifiers/titles (no retrieval corpus), emulating an LLM prompt that
# lists response-playbook names without RAG grounding.
titles_only = [n for n in pb_names]
vec_u = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
T_M = vec_u.fit_transform(titles_only)
QU_M = vec_u.transform(queries)
usims = cosine_similarity(QU_M, T_M)
usims = usims + rng.random(usims.shape) * 1e-4  # seeded tie-breaking
krank = np.argsort(-usims, axis=1)
ktop1 = (krank[:, 0] == q_labels).mean()
ktop3 = np.mean([q_labels[i] in krank[i, :3] for i in range(len(queries))])

t3 = pd.DataFrame({
    "Configuration": ["Ungrounded (titles only, no corpus)",
                      "Grounded retrieval (TF-IDF corpus)"],
    "Top-1 accuracy": [round(ktop1, 3), round(top1, 3)],
    "Top-3 accuracy": [round(ktop3, 3), round(top3, 3)],
    "Queries": [len(queries)] * 2, "Playbooks": [12] * 2})
t3.to_csv(f"{OUT}/table3_e2_retrieval_accuracy.csv", index=False)
print(t3.to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 3.4))
xpos = np.arange(2); w = 0.32
ax.bar(xpos - w/2, [ktop1, top1], w, label="Top-1", color="#F26419")
ax.bar(xpos + w/2, [ktop3, top3], w, label="Top-3", color="#33658A")
ax.set_xticks(xpos)
ax.set_xticklabels(["Ungrounded\n(titles only)", "Grounded retrieval\n(TF-IDF corpus, Layer 3)"])
ax.set_ylim(0, 1.05); ax.set_ylabel("Playbook mapping accuracy")
for i, v in enumerate([ktop1, top1]):
    ax.text(i - w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
for i, v in enumerate([ktop3, top3]):
    ax.text(i + w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/figure2_e2_retrieval_grounding.png", bbox_inches="tight")
plt.close()

# =============================================================================
# EXPERIMENT 3 - Governance-gated orchestration (Layers 4-5)
# =============================================================================
print("\n" + "=" * 70)
print("E3: Governance-gated orchestration (discrete-event simulation)")
print("=" * 70)

N_INC = 2000
N_ANALYSTS = 6
BASE_RATE = 12          # incidents per hour (baseline load)
DEFAULT_TH = 0.50       # governance risk threshold

def simulate_soc(model, n_inc=N_INC, arrival_rate_per_hr=BASE_RATE,
                 threshold=DEFAULT_TH, seed=SEED):
    """Discrete-event queue simulation of one SOC operating model.

    model in {"manual", "narrow_ai", "agentic"}. Six analysts serve a
    single queue; handling time depends on incident impact class.
    Returns per-incident dataframe.
    """
    r = np.random.default_rng(seed)
    inter = r.exponential(60.0 / arrival_rate_per_hr, n_inc)
    arrivals = np.cumsum(inter)
    # ground-truth impact: is the correct response high-impact? ~30% of cases
    high_impact = r.random(n_inc) < 0.30
    # agent risk score correlates with impact but is imperfect (AUC ~ 0.95)
    risk = np.clip(np.where(high_impact,
                            r.beta(8, 2, n_inc), r.beta(2, 8, n_inc)), 0, 1)

    analysts_free = np.zeros(N_ANALYSTS)  # time each analyst becomes free
    rows = []
    for i in range(n_inc):
        t0 = arrivals[i]
        hi = high_impact[i]
        if model == "manual":
            triage = r.lognormal(np.log(18), 0.45)      # ~18 min median
            auto = False
        elif model == "narrow_ai":
            triage = r.lognormal(np.log(6.5), 0.40)     # ML pre-triage
            auto = False
        else:  # agentic: automated L2+L3+L4 pipeline
            triage = r.lognormal(np.log(2.0), 0.35)
            auto = risk[i] < threshold                  # governance gate

        if model == "agentic" and auto:
            # fully automated low-risk path: execution + audit logging
            resp = r.lognormal(np.log(3.5), 0.35)
            touches = 0
            done = t0 + triage + resp
            queue_wait = 0.0
        else:
            # requires analyst: queue for next free analyst
            k = int(np.argmin(analysts_free))
            start = max(t0 + triage, analysts_free[k])
            queue_wait = max(0.0, analysts_free[k] - (t0 + triage))
            if model == "manual":
                med = 45 if hi else 12
                work = r.lognormal(np.log(med), 0.40)
                touches = int(r.integers(3, 5))
            elif model == "narrow_ai":
                med = 30 if hi else 7
                work = r.lognormal(np.log(med), 0.40)
                touches = int(r.integers(2, 4))
            else:  # agentic escalation: analyst reviews evidence pack
                work = r.lognormal(np.log(12), 0.35)
                touches = 1
            done = start + work
            analysts_free[k] = done

        rows.append({
            "model": model, "arrival": t0, "high_impact": hi,
            "risk": risk[i], "auto": auto if model == "agentic" else False,
            "triage_min": triage, "queue_wait_min": queue_wait,
            "mttr_min": done - t0, "touches": touches})
    return pd.DataFrame(rows)

runs = {m: simulate_soc(m) for m in ["manual", "narrow_ai", "agentic"]}
summary = []
for m, df in runs.items():
    summary.append({
        "Model": {"manual": "Manual SOC", "narrow_ai": "Narrow-AI assisted",
                  "agentic": "Agentic + governance"}[m],
        "MTTT (min)": round(df["triage_min"].mean(), 1),
        "MTTR (min)": round(df["mttr_min"].mean(), 1),
        "P90 TTR (min)": round(df["mttr_min"].quantile(0.9), 1),
        "Analyst touches / incident": round(df["touches"].mean(), 2),
        "Auto-executed (%)": round(100 * df["auto"].mean(), 1),
        "Mean queue wait (min)": round(df["queue_wait_min"].mean(), 1)})
t4 = pd.DataFrame(summary)
t4.to_csv(f"{OUT}/table4_e3_orchestration_comparison.csv", index=False)
print(t4.to_string(index=False))

# Load scaling: MTTR vs arrival rate
rates = [6, 10, 14, 18, 22, 26]
scal = {m: [] for m in runs}
for m in runs:
    for rt in rates:
        d = simulate_soc(m, arrival_rate_per_hr=rt, seed=SEED + rt)
        scal[m].append(d["mttr_min"].mean())
pd.DataFrame({"arrival_rate_per_hr": rates, **{m: scal[m] for m in scal}}
             ).to_csv(f"{OUT}/table5_e3_load_scaling.csv", index=False)

# Threshold sensitivity sweep (agentic model)
ths = np.round(np.arange(0.30, 0.91, 0.05), 2)
sweep = []
for th in ths:
    d = simulate_soc("agentic", threshold=th, seed=SEED)
    mis = d.loc[d["auto"], "high_impact"].mean() if d["auto"].any() else 0.0
    hi = d[d["high_impact"]]
    sweep.append({"threshold": th,
                  "auto_executed_pct": 100 * d["auto"].mean(),
                  "escalated_pct": 100 * (1 - d["auto"].mean()),
                  "mis_automation_pct": 100 * mis,
                  "high_impact_auto_pct": 100 * hi["auto"].mean(),
                  "mttr_min": d["mttr_min"].mean()})
t6 = pd.DataFrame(sweep).round(2)
t6.to_csv(f"{OUT}/table6_e3_threshold_sweep.csv", index=False)
print(t6.to_string(index=False))

# Figure 3: (a) MTTR comparison bars  (b) load scaling
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
labels = ["Manual", "Narrow-AI", "Agentic +\ngovernance"]
mttr = [runs[m]["mttr_min"].mean() for m in ["manual", "narrow_ai", "agentic"]]
mttt = [runs[m]["triage_min"].mean() for m in ["manual", "narrow_ai", "agentic"]]
xpos = np.arange(3); w = 0.34
ax[0].bar(xpos - w/2, mttt, w, label="MTTT", color="#F6AE2D")
ax[0].bar(xpos + w/2, mttr, w, label="MTTR", color="#2F4858")
for i in range(3):
    ax[0].text(i - w/2, mttt[i] + 1, f"{mttt[i]:.1f}", ha="center", fontsize=8)
    ax[0].text(i + w/2, mttr[i] + 1, f"{mttr[i]:.1f}", ha="center", fontsize=8)
ax[0].set_xticks(xpos); ax[0].set_xticklabels(labels)
ax[0].set_ylabel("Minutes"); ax[0].set_title("(a) Mean time to triage / respond")
ax[0].legend()

colors = {"manual": "#B23A48", "narrow_ai": "#F26419", "agentic": "#33658A"}
for m, lab in zip(["manual", "narrow_ai", "agentic"], labels):
    ax[1].plot(rates, scal[m], "o-", label=lab.replace("\n", " "),
               color=colors[m], lw=1.5, ms=4)
ax[1].set_xlabel("Incident arrival rate (per hour)")
ax[1].set_ylabel("MTTR (min, log scale)")
ax[1].set_yscale("log")
ax[1].set_title("(b) MTTR under increasing load (6 analysts)")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/figure3_e3_orchestration.png", bbox_inches="tight")
plt.close()

# Figure 4: threshold sensitivity
fig, ax1 = plt.subplots(figsize=(6.4, 3.6))
ax1.plot(t6["threshold"], t6["auto_executed_pct"], "o-", color="#33658A",
         label="Auto-executed (%)", ms=4)
ax1.plot(t6["threshold"], t6["escalated_pct"], "s-", color="#F6AE2D",
         label="Escalated for approval (%)", ms=4)
ax1.set_xlabel("Governance risk threshold")
ax1.set_ylabel("Share of incidents (%)")
ax2 = ax1.twinx()
ax2.plot(t6["threshold"], t6["high_impact_auto_pct"], "^--", color="#B23A48",
         label="High-impact auto-executed (%)", ms=4)
ax2.set_ylabel("High-impact actions auto-executed (%)", color="#B23A48")
ax2.tick_params(axis="y", labelcolor="#B23A48")
ax2.grid(False)
lines, labs = ax1.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(lines + l2, labs + lb2, fontsize=8, loc="center left")
plt.tight_layout()
plt.savefig(f"{OUT}/figure4_e3_threshold_sensitivity.png", bbox_inches="tight")
plt.close()

# =============================================================================
# EXPERIMENT 4 - Closed-loop scenario and audit completeness (Layer 6)
# =============================================================================
print("\n" + "=" * 70)
print("E4: Closed-loop privilege-escalation scenario (Monte Carlo)")
print("=" * 70)

N_RUNS = 1000
STEPS = ["Telemetry ingestion", "Triage & enrichment", "Retrieval grounding",
         "Agentic reasoning", "Governance decision", "Response execution"]
# per-step evidence-logging failure probabilities (transient I/O, races)
p_fail = np.array([0.004, 0.006, 0.008, 0.010, 0.005, 0.007])
# per-step processing time (sec), lognormal
step_mu = np.array([4.0, 9.0, 6.0, 14.0, 8.0, 11.0])
audit_frac = 0.032  # audit write time as fraction of step time, mean

r4 = np.random.default_rng(SEED)
rows = []
for k in range(N_RUNS):
    logged = r4.random(len(STEPS)) >= p_fail
    times = r4.lognormal(np.log(step_mu), 0.30)
    audit_t = times * (audit_frac + r4.normal(0, 0.004, len(STEPS)).clip(-0.01, 0.01))
    audit_t = np.clip(audit_t, 0, None)
    high_risk = r4.random() < 0.55  # priv-esc scenario often crosses gate
    approval = r4.lognormal(np.log(240), 0.5) if high_risk else 0.0  # sec
    rows.append({
        "run": k, "chain_complete": bool(logged.all()),
        "steps_logged": int(logged.sum()),
        "workflow_time_s": times.sum() + approval,
        "audit_time_s": audit_t.sum(),
        "audit_overhead_pct": 100 * audit_t.sum() / (times.sum() + approval),
        "high_risk_path": high_risk})
e4 = pd.DataFrame(rows)
per_step_cov = [(1 - p) for p in p_fail]  # analytic; report empirical below
emp_step_cov = []
r4b = np.random.default_rng(SEED)  # re-simulate to get per-step empirical
logged_mat = np.array([r4b.random(len(STEPS)) >= p_fail for _ in range(N_RUNS)])
emp_step_cov = logged_mat.mean(axis=0)

t7 = pd.DataFrame({
    "Metric": ["Monte Carlo runs", "Complete evidence chains (%)",
               "Mean steps logged (of 6)", "Mean audit overhead (%)",
               "P95 audit overhead (%)", "Mean end-to-end time, low-risk path (s)",
               "Mean end-to-end time, high-risk path incl. approval (s)"],
    "Value": [N_RUNS, round(100 * e4["chain_complete"].mean(), 1),
              round(e4["steps_logged"].mean(), 2),
              round(e4["audit_overhead_pct"].mean(), 2),
              round(e4["audit_overhead_pct"].quantile(0.95), 2),
              round(e4.loc[~e4["high_risk_path"], "workflow_time_s"].mean(), 1),
              round(e4.loc[e4["high_risk_path"], "workflow_time_s"].mean(), 1)]})
t7.to_csv(f"{OUT}/table7_e4_audit_metrics.csv", index=False)
print(t7.to_string(index=False))

pd.DataFrame({"step": STEPS, "empirical_logging_coverage": emp_step_cov.round(4)}
             ).to_csv(f"{OUT}/table8_e4_per_step_coverage.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].bar(range(6), emp_step_cov * 100, color="#33658A")
ax[0].set_xticks(range(6))
ax[0].set_xticklabels(["Ingest", "Triage", "Retrieve", "Reason",
                       "Govern", "Execute"], rotation=25, ha="right")
ax[0].set_ylim(97, 100.2); ax[0].set_ylabel("Evidence logged (%)")
ax[0].axhline(100 * e4["chain_complete"].mean(), ls="--", color="crimson",
              lw=1, label=f"Complete chains: {100*e4['chain_complete'].mean():.1f}%")
ax[0].set_title("(a) Per-step evidence-logging coverage")
ax[0].legend(fontsize=8)

ax[1].hist(e4["audit_overhead_pct"], bins=30, color="#F6AE2D",
           edgecolor="white")
ax[1].axvline(e4["audit_overhead_pct"].mean(), ls="--", color="#2F4858",
              label=f"Mean = {e4['audit_overhead_pct'].mean():.2f}%")
ax[1].set_xlabel("Audit-logging overhead (% of workflow time)")
ax[1].set_ylabel("Runs")
ax[1].set_title("(b) Distribution of audit overhead")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/figure5_e4_audit_traceability.png", bbox_inches="tight")
plt.close()

# =============================================================================
# Package & (Colab) auto-download
# =============================================================================
zip_path = "socaas_validation_outputs.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(OUT)):
        z.write(os.path.join(OUT, f), arcname=f)
print(f"\nAll outputs written to ./{OUT} and packaged as {zip_path}")

try:  # Colab auto-download; silently skipped elsewhere
    from google.colab import files  # type: ignore
    files.download(zip_path)
except Exception:
    pass
