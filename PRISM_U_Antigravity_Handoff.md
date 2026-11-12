# PRISM-U — Complete Project Handoff for Antigravity

**Purpose of this document:** Give the Antigravity coding agent the full project context, research intent, current design, decisions discussed, recommended upgrades, roadmap, and known completion status so development can continue without confusing planned work with completed work.

**Status date:** 5 November 2026  
**Project:** PRISM-U — *Privacy-Budgeted Active Auditing for Machine Unlearning*  
**Type:** Final-Year Project (FYP), with an explicit ambition to develop a publishable research paper.

---

# 0. Instructions to the Antigravity agent

You are assisting a three-member university FYP team building PRISM-U. Treat this document as the project's current handoff, not as proof that code or experiments already exist.

Before making changes:

1. Inspect the entire existing workspace/repository. Do not assume it is empty and do not overwrite existing work.
2. Identify the current language/framework, files, dependencies, implemented modules, tests, checkpoints, datasets, and experiment outputs.
3. Produce a concise audit of what is actually implemented versus what remains missing.
4. Compare the workspace against the roadmap below and propose the smallest safe next milestone.
5. Do not claim an experiment passed unless you ran it and retained its configuration, logs, and outputs.
6. Do not fabricate metrics, citations, model results, or publication claims.
7. Keep research code modular, reproducible, configurable, and testable. Prefer small verified increments over a large speculative implementation.
8. Preserve random seeds, data split indices, model checkpoints, configs, and machine-readable experiment logs.
9. Clearly separate **proposal requirements**, **team decisions**, **recommended research upgrades**, and **optional stretch goals**.
10. Do not introduce formal differential privacy (DP) claims unless the team deliberately changes the project and implements a valid DP mechanism/proof. The current proposal uses empirical retained-membership leakage evaluation, not a formal DP guarantee.

The agent's first job is to inspect the workspace and report the real state. Do not immediately start building a dashboard or polishing the UI. The scientific ML/auditing pipeline comes first.

---

# 1. The project in one paragraph

PRISM-U is a proposed **black-box, privacy-aware, active auditor for machine unlearning**. A model is trained on a dataset, then an unlearning procedure is asked to remove a designated forget set while retaining the rest of the training data. An external auditor should not need access to the target model's weights, gradients, optimizer, or training data. It can only submit permitted queries and observe the target's responses. PRISM-U aims to choose the next query based on how much it is expected to help distinguish properly unlearned from incompletely unlearned models, while also accounting for the risk that the accumulated audit transcript reveals membership information about retained records. It updates evidence sequentially, respects query and empirical privacy constraints, and returns **PASS**, **FAIL**, or **INCONCLUSIVE** rather than forcing a decision when evidence is insufficient.

The central question is:

> Given everything observed so far, which permitted query gives the most useful evidence about whether unlearning succeeded for the least additional retained-data privacy exposure—and is another query worth asking at all?

---

# 2. Explain PRISM-U simply

Imagine a student memorized 100 flashcards. A teacher asks the student to forget cards 1–10. The student says they forgot them, but the teacher cannot inspect the student's brain. The teacher can only ask questions and observe answers.

- The ML model is the student.
- The training records are the flashcards.
- The records requested for deletion form the **forget set**.
- The records that should remain form the **retain set**.
- The unlearning algorithm is the process intended to make the model forget the forget set.
- PRISM-U is the external teacher/auditor.
- Candidate queries are possible questions.
- Audit gain measures how useful a question is for detecting incomplete unlearning.
- Privacy risk measures how much a question/response may reveal about retained-data membership.
- The query budget limits how many questions can be asked.
- The privacy budget limits empirically estimated retained-membership exposure.
- Sequential evidence is the auditor's evolving confidence.
- PASS / FAIL / INCONCLUSIVE is the final decision.

PRISM-U is **not** primarily a new unlearning algorithm. It is the auditor that evaluates an unlearning outcome from the outside.

---

# 3. Core terminology and objects

Use consistent symbols/names throughout code, reports, and paper.

| Symbol/name | Meaning |
|---|---|
| `D` | Original training dataset. |
| `D_u` | Forget set: records requested to be unlearned. |
| `D_r` | Retain set: records that should remain. |
| `M_original` | Model trained on `D_r ∪ D_u`. |
| `M_retrain` | Honest reference model trained from scratch on `D_r` only. It is a practical reference for the desired post-unlearning behavior, not an absolute proof of ideal deletion. |
| `M_unlearned` | Model produced by applying an existing unlearning algorithm to `M_original` using `D_u`. |
| `M_partial` | Controlled incomplete-unlearning model, e.g. one stopped early. |
| `Q` | Permitted auxiliary candidate query pool. It is the set of possible queries, not the number of queries allowed. |
| `B_Q` | Query budget: maximum number of target queries in one audit. |
| `B_P` | Privacy budget/constraint, defined using the project's empirical retained-membership leakage estimator. It is not a DP epsilon unless the project is redesigned to support DP. |
| `τ_t` | Audit transcript after `t` rounds: queried inputs and corresponding target responses. |
| `G_t(x | τ_t)` | Expected marginal audit/compliance information from querying `x` given the current transcript. |
| `L_hat(τ_t)` | Estimated retained-membership leakage from the complete transcript so far. |
| `ΔL_t(x | τ_t)` | Expected incremental leakage from adding query `x` and its response to the current transcript. |
| `E_t` / `LLR_t` | Sequential evidence supporting compliant vs incomplete unlearning. |
| `H_C` | Hypothesis that the target is sufficiently compliant/properly unlearned under the chosen operational definition. |
| `H_N` | Hypothesis that the target is non-compliant/incompletely unlearned. |

**Important:** The team must operationally define what “properly unlearned” means. `M_retrain` is a reference point, but a chosen unlearning method may not be expected to be numerically identical to retraining. The audit target, ground truth labels, and acceptable deviations must be specified before final experiments.

---

# 4. What is the actual research problem?

Machine unlearning attempts to remove the influence of designated training records from an already-trained model. An organization may claim that it has performed unlearning, but an external party may have only black-box access to the model. The auditor must determine whether the model's behavior is consistent with successful unlearning.

There is a tension:

- More or better queries may make it easier to detect incomplete unlearning.
- Responses to those queries may expose information about records that were meant to remain in the model.
- Asking too many queries also consumes time/API budget and can increase exposure.
- A responsible auditor must sometimes abstain rather than make a poorly supported PASS or FAIL decision.

PRISM-U investigates whether **active, privacy-aware, transcript-adaptive query selection** can improve the audit/privacy/query tradeoff compared with fixed, random, or privacy-blind auditing.

## Intended core contribution

The strongest research framing developed in the discussions is:

> PRISM-U investigates transcript-adaptive machine-unlearning auditing in which each permitted query is selected online according to its expected marginal compliance information and estimated marginal retained-membership exposure, subject to cumulative privacy and query constraints and a calibrated sequential stopping/abstention rule.

This is a **novelty candidate**, not a proven first-of-its-kind claim. The literature must be rechecked carefully before submission.

---

# 5. Three systems that must not be confused

Keep these as separate modules and separate research roles.

## A. The unlearning algorithm

Examples discussed: Gradient Ascent (GA) and SalUn. These are existing algorithms used to create target models. The FYP is not primarily proposing a new unlearning algorithm.

## B. The membership-inference/privacy evaluator

This evaluates whether an attacker can infer whether retained records were members of the model's training set using permitted observations. It measures retained-data privacy leakage. It is an evaluator/attacker, not PRISM-U itself.

## C. The PRISM-U auditor

This is the main proposed contribution. It chooses which query to make next, balances expected audit value and privacy risk, updates evidence, manages budgets, and decides whether to continue or abstain.

Do not merge these responsibilities into one opaque script.

---

# 6. Proposal scope discussed so far

The proposal and previous project discussions establish the following planned experimental setting.

## Primary benchmark

- Dataset: **CIFAR-10**.
- Target architecture: **ResNet-18**.
- Define deterministic forget/retain splits.
- Train the original target on all training data.
- Train an honest retrained reference using retain data only.
- Apply unlearning algorithms to the original target.
- Construct controlled incomplete-unlearning targets by truncating/partially executing the unlearning procedure.

## Secondary benchmark / generalization check

- Dataset: **Purchase-100**.
- Model family: **MLP**.
- This is secondary/conditional on time and compute in the proposal; do not let it derail a working primary benchmark.

## Unlearning methods discussed

- A Gradient-Ascent-style unlearning method.
- **SalUn**.
- The first end-to-end pipeline should use one method, then add the second after the pipeline works.

## Controlled incomplete unlearning

Create target conditions with known construction, such as:

- 0% / no unlearning;
- 25% of the chosen unlearning procedure;
- 50%;
- 75%;
- 100% of the chosen procedure;
- independently retrained retain-only reference.

These percentages are examples of truncation levels. They are **not automatically equivalent to percentages of records forgotten**. Be precise: stopping an optimizer after 25% of its update steps does not prove that exactly 25% of forget-set information was removed. Describe them as partial execution/severity conditions unless a separate measure validates actual forgetting severity.

## Black-box assumption

The auditor should interact with the target through a prediction interface. It must not use target weights, gradients, optimizer state, training data, or internal activations. Decide and document whether the primary threat model exposes:

- full class-probability/logit vectors;
- probabilities only;
- top-1 labels only.

These are materially different access settings. Consider evaluating more than one if feasible, but freeze a primary setting for controlled experiments.

---

# 7. The intended PRISM-U workflow

The system should eventually follow this loop:

```text
Known experimental setup / calibration environment
    |
    +--> train original, honest-reference, and incomplete-unlearning models
    |
    +--> build compliant and non-compliant surrogate/reference banks
    |
    +--> construct permitted auxiliary candidate query pool Q
    |
    +--> calibrate audit-value, privacy-risk, and evidence estimators
    |
    +--> freeze calibration/thresholds for held-out target evaluation

Online audit of an unseen black-box target
    |
    +--> initialize empty transcript τ_0 and evidence E_0
    +--> score remaining candidate queries using current transcript
    +--> estimate marginal audit gain G_t(x | τ_t)
    +--> estimate marginal transcript leakage ΔL_t(x | τ_t)
    +--> reject queries predicted to violate cumulative privacy constraint
    +--> compare the value of another query against stopping
    +--> select and issue one query to the target
    +--> observe response and append (x_t, r_t) to transcript
    +--> update evidence, privacy estimate, query count, and diagnostics
    +--> if calibrated evidence supports compliance: PASS
    +--> if calibrated evidence supports non-compliance: FAIL
    +--> if a budget is exhausted without enough evidence: INCONCLUSIVE
    +--> otherwise repeat
```

The exact update rules and guarantees remain research tasks. Do not assume the pseudocode alone constitutes a valid statistical method.

---

# 8. Main mathematical ideas discussed

These are candidate formulations to investigate and justify, not finalized equations.

## 8.1 Audit gain

For each candidate query `x`, estimate how much the response is expected to improve discrimination between `H_C` and `H_N`, conditional on the transcript:

`G_t(x | τ_t) = expected reduction in compliance uncertainty / expected information gain`

Possible operational approaches include expected reduction in posterior entropy, expected KL/JS separation, expected likelihood-ratio movement, or expected reduction in a calibrated decision loss. Choose one and explain why it fits the experimental design.

A static score `G(x)` calculated once is weaker. The research upgrade is to recompute `G_t` after observing each response.

## 8.2 Transcript-dependent privacy risk

The key upgrade is to model privacy of the **whole transcript**, not merely assign a permanent risk number to each image.

Define an empirical estimator:

`L_hat(τ_t) = estimated retained-record membership leakage from the transcript τ_t`

Then define candidate incremental cost conceptually as:

`ΔL_t(x | τ_t) = E[ L_hat(τ_t ∪ {(x,r)}) - L_hat(τ_t) | τ_t, x ]`

This is a research proposal and needs a concrete implementable estimator. It must be validated against independent/stronger membership-inference attackers.

**Critical caution:** MIA AUC is a performance metric of an attack, not automatically an additive privacy unit. Do not blindly sum per-query AUCs or arbitrary risk scores and call that a valid cumulative privacy guarantee. If you use a proxy/accounting score, state what it measures, calibrate it, test underestimation, and call it empirical leakage accounting—not formal DP.

## 8.3 Candidate acquisition rule

Two possible rule families discussed:

Ratio form:

`score_t(x) = G_t(x | τ_t) / (ΔL_t(x | τ_t) + ε)`

Additive/Lagrangian form:

`score_t(x) = G_t(x | τ_t) - λ_t ΔL_t(x | τ_t)`

With a hard feasibility condition:

`L_hat(τ_t) + predicted ΔL_t(x | τ_t) <= B_P`

Compare ratio vs additive forms in an ablation. Do not assert one is optimal without analysis. Consider uncertainty bounds on both gain and risk; a conservative policy may use an upper confidence bound on risk and a lower confidence bound on gain.

## 8.4 Query pool versus query budget

These are different:

- `Q` = number/content of candidate queries available to choose from.
- `B_Q` = maximum number of queries PRISM-U may actually send to the target.

A pool can contain thousands of candidates while a run uses only a small number of them.

## 8.5 Evidence and sequential stopping

A candidate evidence statistic is a cumulative log-likelihood ratio:

`LLR_t = LLR_(t-1) + log( P(r_t | x_t, τ_(t-1), H_C) / P(r_t | x_t, τ_(t-1), H_N) )`

Sign convention must be chosen and used consistently. For example, positive LLR can be defined as evidence for compliance. The conditional response models need calibration, and dependence/adaptive-query effects must be addressed. Do not assume an ordinary fixed-sample likelihood-ratio argument automatically gives valid sequential guarantees under adaptive sampling.

Candidate decisions:

- evidence crosses calibrated compliance boundary -> `PASS`;
- evidence crosses calibrated non-compliance boundary -> `FAIL`;
- query budget/privacy limit exhausted before a reliable decision -> `INCONCLUSIVE`;
- optional additional stopping action if expected value of another safe query is too small.

Targets/thresholds should be calibrated on validation models and frozen before final testing. Ideally control false-PASS and false-FAIL rates at predeclared levels (for example, 5% as a *candidate target*, not a committed result).

---

# 9. Budget selection: what is and is not automatic

A key issue discussed repeatedly is whether one budget transfers to all datasets/models.

## The correct distinction

PRISM-U should generalize at the **mechanism/policy level**, not by blindly reusing identical numeric calibration values everywhere.

Do not assume that `B_P = 0.20`, `B_Q = 50`, or a fixed PASS threshold is universally correct for every dataset, architecture, unlearning method, or threat model.

## What should be calibrated per environment

Depending on the final design, calibration may need to estimate:

- surrogate/reference response distributions;
- audit-gain model;
- transcript-level privacy estimator and its uncertainty;
- mapping from risk proxy to observed MIA performance;
- evidence/likelihood calibration;
- PASS/FAIL thresholds;
- query-pool composition/size;
- practical query-budget operating points.

## Policy choices versus learned/calibrated values

Some values are normative policy inputs, not things the algorithm can objectively decide:

- maximum tolerated privacy exposure;
- acceptable false-PASS / false-FAIL rates;
- maximum query/API cost;
- desired risk tolerance.

A stronger interface could take interpretable external requirements (for example, a maximum tolerated empirical MIA performance and a maximum query count), then use validation calibration to select internal parameters. Do not pretend the algorithm can decide what privacy level society or a user should tolerate.

## Query-budget calibration

Sweep a grid of query limits (example exploratory grid: 5, 10, 20, 40, 80, 160). Plot audit quality, coverage, leakage, and runtime versus budget. Choose a demo operating point from validation, e.g. the smallest budget near the performance/coverage plateau. These numbers are examples, not predetermined requirements.

## Privacy-budget calibration

First define a valid empirical leakage estimator. Sweep privacy operating levels on validation targets, then plot audit quality and coverage against independently measured MIA leakage. Do not assign a magic normalized number without interpretation. The same internal number may mean different things in different environments; ideally normalize against an external, interpretable risk requirement and recalibrate locally.

## No universal ideal budget

The scientific deliverable should include curves/frontiers across multiple budgets. A single operating point is useful for the demo, but the paper should report the tradeoff surface.

---

# 10. INCONCLUSIVE is not automatically a failure

`INCONCLUSIVE` is a legitimate abstention when the system cannot gather enough evidence within privacy/query constraints. It is safer than making a false certification.

However, an auditor that always says `INCONCLUSIVE` is useless. Therefore, report **coverage** alongside conditional accuracy/error.

`Coverage = (number of PASS + number of FAIL) / total audits`

`InconclusiveRate = 1 - Coverage`

Always report:

- PASS rate;
- FAIL rate;
- INCONCLUSIVE rate;
- coverage;
- false-PASS rate among compliant/non-compliant ground truths as appropriate;
- false-FAIL rate;
- conditional accuracy among decided cases;
- AUROC or another ranking metric independent of the threshold;
- queries to decision;
- actual measured MIA leakage.

## Interpret different patterns correctly

- Strict privacy -> high INCONCLUSIVE, but relaxed privacy resolves decisions: this can reveal a privacy/verification tradeoff.
- Small query budget -> high INCONCLUSIVE, larger budget resolves decisions: this can reveal query complexity.
- PRISM-U inconclusive while simpler baselines decide accurately: likely method/calibration failure; diagnose and improve.
- Every method inconclusive even with generous budgets: test whether the query pool/evidence estimator/target construction is broken before claiming an impossibility result.
- If response distributions for compliant and non-compliant targets are empirically indistinguishable over the permitted pool, a carefully supported limitation result may be scientifically useful.

A publishable negative result must explain **why, where, and under which assumptions** the approach fails. “Our code returned inconclusive” is not enough.

A useful advanced analysis is an audit-feasibility or phase diagram over `(B_P, B_Q)`, showing where reliable decisions are possible at controlled error rates and where abstention dominates.

---

# 11. Novelty assessment developed in the discussions

The previous literature assessment concluded that many ingredients are already known, and the novelty must be kept narrow.

## Not novel by themselves

- machine-unlearning auditing;
- black-box auditing;
- membership inference attacks;
- retained-data privacy evaluation;
- surrogate/reference model banks;
- CIFAR-10 + ResNet-18;
- Purchase-100 + MLP;
- Gradient Ascent / SalUn;
- deliberately partial/early-stopped unlearning;
- query budgets;
- privacy/audit Pareto curves;
- active querying in other audit domains;
- sequential testing or abstention as general concepts.

## Candidate novelty

The proposed gap is the **MU-specific integration** of:

1. target-response-adaptive query selection;
2. expected marginal compliance information;
3. transcript-dependent marginal retained-membership leakage;
4. cumulative empirical privacy and query constraints;
5. sequential evidence updates;
6. calibrated PASS/FAIL/INCONCLUSIVE stopping.

The team previously discussed several adjacent works:

- Tang, Joshi, and Kundu (2026): behavioral machine-unlearning auditing and its privacy cost; prior discussion characterized their protocol as fixed-query and noted adaptive protocols as outside their scope. This is the closest comparison and should be read/rechecked carefully.
- BAFA (2026): active black-box query selection for fairness auditing; active querying itself is not new, but the task/objective differs from MU auditing with retained-membership leakage constraints.
- Ye et al.: a general-purpose machine-unlearning auditing framework across datasets and unlearning methods; therefore do not claim to be the first general-purpose MU auditor.
- Sarmasarkar, Koloskova, and Koyejo (2026): auditing unlearning algorithms using hypothesis testing/MIA-type methods; related auditing work.
- Gu, He, and Chen: retained-sample privacy after approximate unlearning; retained-membership evaluation alone is not new.
- Google Research f-divergence auditing work (2026): statistical privacy/MU auditing; related statistical testing but not necessarily the same adaptive policy.
- UAG (IJCAI 2026): hierarchical auditing and audit-cost decisions; cost-adaptive auditing alone is not new.

**Important literature caveat:** These references and descriptions reflect the prior conversation's literature discussion. Before writing a paper, the team must independently verify each exact title, author list, venue, publication date, version, assumptions, and whether the claimed limitation is present in the final paper. Do not copy this list into a bibliography without verification.

## Recommended novelty wording

> To our knowledge, existing behavioral machine-unlearning audits primarily characterize audit effectiveness and retained-data privacy leakage under fixed or non-adaptive query protocols, while active black-box auditing has also been explored in other audit domains. PRISM-U investigates response-adaptive machine-unlearning auditing in which each query is selected online according to its expected marginal compliance information and estimated marginal retained-membership exposure, subject to cumulative privacy/query constraints and calibrated sequential stopping with abstention.

Use “to our knowledge” and cite the closest work. Do not claim universal firstness until a systematic literature review verifies it.

---

# 12. Publication ambition and realistic expectations

The team wants the FYP to have a serious chance of becoming a research paper.

Prior assessment:

- Current proposal + basic implementation: strong FYP; possible smaller workshop/conference/journal paper if experiments are clean.
- Static `G(x)` / `R(x)` ranking: weaker novelty; reviewers may see it as a straightforward combination of known ideas.
- Response-adaptive `G_t(x | τ_t)` plus transcript-dependent `ΔL_t(x | τ_t)` and sequential stopping: much stronger research contribution candidate.
- Add robust multi-dataset/model/unlearning experiments, unseen targets, stronger privacy attackers, calibration-transfer analysis, strong baselines, confidence intervals, and ablations: more serious conference/journal potential.
- NeurIPS main track and IEEE Symposium on Security and Privacy are ambitious targets, not current expectations or guaranteed outcomes. They likely require stronger formal/statistical foundations, a clearly defined threat model, rigorous privacy accounting, and much broader evidence than the FYP's minimum scope.
- “IEEE” is not one uniform bar; journal/conference venues vary substantially.

Do not design the FYP so broadly that the basic system never gets completed. First establish whether the core phenomenon works on the primary benchmark. Add paper-depth extensions after that foundation is sound.

---

# 13. Research questions / hypotheses to test

The proposal's broad questions were discussed as follows. Refine the exact formal wording with the supervisor before final paper submission.

## RQ1 — Query efficiency

Does active query selection reduce the number of black-box queries required to reach comparable unlearning-audit quality relative to random/fixed querying?

## RQ2 — Privacy/utility tradeoff

Does privacy-aware query selection reduce retained-set membership leakage at matched audit quality, or increase audit quality/coverage at matched leakage?

## RQ3 — Sequential stopping and budgets

Can sequential stopping reduce queries/privacy exposure while maintaining calibrated false-PASS and false-FAIL rates? How do query/privacy limits affect coverage and INCONCLUSIVE rate?

## RQ4 — Generalization

Does the PRISM-U policy structure transfer across unlearning algorithms, random seeds, architectures, and dataset/model families? Which quantities need recalibration?

## Further research-depth questions

- How correlated are audit gain and privacy risk?
- Does transcript-dependent privacy estimation improve over static per-query risk?
- Can a calibration learned on one architecture transfer to another?
- How sensitive is the policy to a poor or shifted query pool?
- Does the privacy estimator understate leakage under held-out/stronger attacks?
- Can PRISM-U estimate partial-unlearning severity, not only binary compliance?
- Is there a measurable privacy/query feasibility boundary below which reliable auditing is not possible under the chosen threat model?

---

# 14. Roadmap: from zero to an end-to-end FYP

The order matters. Do not begin with fancy query acquisition. First build a controlled experimental world with known ground truth.

## Milestone 0 — Workspace audit and experiment specification

- Inspect existing repository and installed environment.
- Confirm GPU/CPU, Python/CUDA/PyTorch versions, disk space, and compute limitations.
- Read the FYP proposal and extract exact requirements, research questions, and deliverables.
- Create a decision log and config structure.
- Freeze the first small baseline setting (one seed, one unlearning method, one forget fraction) for smoke testing.

**Exit criterion:** The team agrees on a reproducible first experiment and the agent knows the actual existing code state.

## Milestone 1 — Dataset and splits

- Load CIFAR-10 deterministically.
- Create disjoint train, forget, retain, validation, auxiliary query pool, and final test partitions as required by the threat model.
- Store exact split indices, seeds, and class distributions.
- Make sure candidate queries and final evaluation targets do not leak into one another.
- Define the membership/non-membership samples used by the privacy attacker separately and carefully.

**Exit criterion:** A script reproduces the same partitions from a config and seed.

## Milestone 2 — Original and honest-reference models

- Train `M_original` on `D_r ∪ D_u`.
- Train `M_retrain` from scratch on `D_r` only.
- Save checkpoints and metadata.
- Evaluate retain/forget/validation/test performance and save output distributions.

**Exit criterion:** Both models train, checkpoint, reload, and evaluate reproducibly.

## Milestone 3 — First unlearning method and controlled incomplete targets

- Implement one selected existing unlearning method first.
- Produce a full-method unlearned model.
- Produce early-stopped/partial-execution variants (e.g., 25/50/75/100% of configured update steps/epochs).
- Record exact method parameters and step counts.
- Later add the second method (e.g., SalUn) after the pipeline works.

**Exit criterion:** There are known compliant-reference and incomplete-unlearning conditions that can be queried.

## Milestone 4 — Black-box target interface

- Provide an API/function that accepts an allowed input and returns only the chosen output representation.
- Keep target internals inaccessible to the audit module.
- Log queries/responses and enforce the query limit.
- Include tests proving auditor modules cannot accidentally read target checkpoints/weights.

**Exit criterion:** Auditor can query all target types through one interface without access to their internals.

## Milestone 5 — Query pool and reference/surrogate bank

- Construct a fixed, documented auxiliary query pool `Q`.
- Ensure it is separate from final test targets and held-out membership evaluation as defined by the threat model.
- Train several compliant and non-compliant surrogate/reference models with varied seeds.
- Save response distributions for permitted candidates.
- Evaluate pool diversity and query informativeness descriptively.

**Exit criterion:** A frozen, reproducible pool and surrogate bank exist.

## Milestone 6 — Basic auditing baselines

Implement in this order:

1. Random query auditor.
2. Fixed/non-adaptive query auditor.
3. Simple privacy-blind active auditor based on audit gain.

Start with a simple, transparent discrimination statistic between compliant and non-compliant reference responses. Validate that the auditor can distinguish controlled targets at all before adding privacy constraints.

**Exit criterion:** Audit quality versus query count can be measured and the basic signal is nontrivial.

## Milestone 7 — Audit gain estimator

- Implement `G_t(x | τ_t)` or first a static `G(x)` baseline.
- Prefer an interpretable expected uncertainty reduction/information-gain definition.
- Test ranking stability, calibration, and whether high-score queries are actually useful on held-out target models.
- Keep static and adaptive versions separate for ablation.

**Exit criterion:** Query choice is demonstrably response-adaptive, not just a one-time ranking.

## Milestone 8 — Retained-membership privacy evaluator

- Define the privacy threat model: what can the attack observe, which records are members/non-members, and what is the attacker allowed to know?
- Implement a simple membership inference attack first using confidence/loss/margin/entropy or response-vector features, as appropriate to output access.
- Train/evaluate attackers using separate data/models to prevent leakage.
- Report MIA AUC and advantage with confidence intervals.
- Add stronger/held-out attackers for robustness (e.g. probability-vector, confidence/loss, shadow-model, label-only where applicable).

**Exit criterion:** The project can measure retained membership leakage independently of the query-selection mechanism.

## Milestone 9 — Transcript-level privacy estimator

- Define `L_hat(τ_t)` operationally.
- Define `ΔL_t(x | τ_t)` as incremental expected leakage from the full transcript.
- Investigate how to estimate this from shadow/reference experiments.
- Test estimator calibration and underestimation against independent stronger attacks.
- Compare transcript-dependent risk with static per-query risk.
- Never call it a formal privacy guarantee unless a valid formal mechanism is added.

**Exit criterion:** The privacy-aware score has a measurable, defensible interpretation and uncertainty analysis.

## Milestone 10 — PRISM-U query policy

- Implement additive and ratio acquisition variants.
- Enforce the cumulative empirical privacy constraint.
- Enforce query budget.
- Re-score candidates after every target response.
- Track rejected candidate queries and reasons.
- Add conservative handling for uncertain privacy estimates.

**Exit criterion:** The next query depends on current transcript and both expected audit value and privacy cost.

## Milestone 11 — Sequential evidence and stopping

- Choose an evidence statistic and sign convention.
- Calibrate response likelihoods/posteriors on validation surrogates/targets.
- Set false-PASS/false-FAIL tolerances before final testing.
- Calibrate stopping thresholds on validation only.
- Return PASS/FAIL when evidence crosses a threshold.
- Return INCONCLUSIVE when query/privacy budgets are exhausted without sufficient evidence.
- Optionally compare with a value-of-information stopping rule.

**Exit criterion:** Threshold behavior is calibrated and final test targets were not used for threshold tuning.

## Milestone 12 — Main comparisons

Compare at least:

- Random auditing.
- Fixed-query/non-adaptive auditing.
- Privacy-blind active auditing.
- Static privacy-aware auditing.
- Adaptive audit gain with static privacy score.
- Full transcript-adaptive PRISM-U.

Use matched comparisons at comparable audit quality, comparable MIA leakage, and comparable query cost. Do not report only one favored operating point.

**Exit criterion:** Reproducible comparison results and confidence intervals.

## Milestone 13 — Budget sweeps and feasibility maps

Explore grids of `B_Q` and `B_P`. Example query-budget sweep: 5, 10, 20, 40, 80, 160. Privacy limits must be derived from the chosen empirical leakage metric; do not invent arbitrary numbers without calibration.

For each setting report:

- audit AUROC/discrimination;
- false-PASS / false-FAIL;
- coverage and INCONCLUSIVE rate;
- queries to decision;
- MIA AUC and advantage;
- actual/estimated privacy trajectory;
- runtime/cost.

Plot risk-coverage, audit-quality/leakage, query-efficiency, and `(B_P, B_Q)` feasibility diagrams.

**Exit criterion:** A defensible operating point and full tradeoff curves exist.

## Milestone 14 — Generalization and robustness

After the CIFAR-10 pipeline works:

- test unseen model seeds;
- test a different architecture on the same data if feasible;
- test the second unlearning method;
- test Purchase-100 + MLP if resources permit;
- measure zero-shot calibration transfer vs recalibration;
- test shifted/imbalanced/poor auxiliary query pools;
- test partial unlearning of different severities/failure mechanisms;
- test privacy-estimator robustness against held-out attacks.

**Exit criterion:** The team can state what transfers, what needs recalibration, and where the method fails.

## Milestone 15 — Ablations and paper-quality statistical analysis

Ablate:

- static vs transcript-adaptive gain;
- static vs transcript-dependent privacy;
- no privacy penalty vs privacy-aware policy;
- ratio vs additive acquisition;
- no early stopping vs sequential stopping;
- no hard budget vs hard privacy constraint;
- different surrogate-bank sizes;
- different query-pool sizes/compositions;
- uncertainty-aware vs point-estimate risk.

Run multiple independent seeds/targets, report means and 95% confidence intervals, and retain all configs/logs.

**Exit criterion:** Each claimed contribution has a corresponding ablation and evidence.

## Milestone 16 — Dashboard and demo

Only after the experiment engine works, add a simple dashboard (Streamlit is sufficient) to display:

- target identifier/type;
- query count and remaining query budget;
- privacy estimate and remaining empirical privacy allowance;
- selected query and its gain/risk estimates;
- evidence trajectory;
- PASS/FAIL/INCONCLUSIVE status;
- audit transcript/decision log.

Dashboard is a demonstration tool; the research result is the algorithm and experiments.

---

# 15. Recommended repository structure

Adapt to the current workspace rather than blindly replacing it.

```text
prism-u/
├── README.md
├── pyproject.toml / requirements.txt
├── configs/
│   ├── cifar10_resnet18.yaml
│   ├── training.yaml
│   ├── unlearning_ga.yaml
│   ├── unlearning_salun.yaml
│   ├── audit.yaml
│   └── experiments.yaml
├── data/
│   ├── datasets.py
│   ├── splits.py
│   └── query_pool.py
├── models/
│   ├── resnet.py
│   ├── mlp.py
│   ├── train.py
│   └── black_box_target.py
├── unlearning/
│   ├── base.py
│   ├── gradient_ascent.py
│   ├── salun.py
│   └── partial_execution.py
├── surrogates/
│   ├── bank.py
│   ├── train_surrogates.py
│   └── response_store.py
├── audit/
│   ├── transcript.py
│   ├── random_policy.py
│   ├── fixed_policy.py
│   ├── audit_gain.py
│   ├── privacy_risk.py
│   ├── acquisition.py
│   ├── evidence.py
│   ├── stopping.py
│   └── prism_u.py
├── attacks/
│   ├── membership_inference.py
│   └── attack_evaluation.py
├── experiments/
│   ├── train_targets.py
│   ├── run_audits.py
│   ├── sweep_budgets.py
│   ├── run_ablations.py
│   └── transfer_tests.py
├── evaluation/
│   ├── metrics.py
│   ├── calibration.py
│   ├── confidence_intervals.py
│   ├── pareto.py
│   └── plots.py
├── dashboard/
│   └── app.py
├── tests/
├── checkpoints/        # preferably ignored by git; use external storage if large
├── results/            # structured run outputs
└── docs/
    ├── threat_model.md
    ├── design_decisions.md
    └── experiment_protocol.md
```

Keep large datasets/checkpoints out of Git unless explicitly required. Store metadata, split indices, hashes, configs, and small reproducibility artifacts in the repository.

---

# 16. Experiment hygiene and leakage prevention

This is critical for a credible paper.

1. Split by record before training and record all indices.
2. Keep final target models/sets held out from surrogate training and threshold tuning.
3. Keep auxiliary query pool disjoint from final test examples and clearly define whether any query can also be a training member under the privacy threat model.
4. Keep privacy-attack train/validation/test members and non-members separated. Avoid evaluating an attacker on the exact shadow examples used to fit it.
5. Do not tune `λ`, acquisition form, budget operating point, evidence thresholds, or pool construction on final test outcomes.
6. Freeze the pipeline and settings before final evaluation.
7. Run multiple seeds and independent target models; a single target checkpoint is not enough for generalization claims.
8. Log model architecture, initialization seed, data split hash, unlearning steps, optimizer, hyperparameters, query pool version, surrogate-bank version, and code commit.
9. Report all failed/inconclusive runs, not only successful ones.
10. Keep the target model black-box to the auditor in the actual evaluation path.

---

# 17. Metrics to report

## Audit effectiveness

- AUROC / ranking discrimination for compliant vs incomplete targets;
- false-PASS rate;
- false-FAIL rate;
- calibration quality of predicted compliance probabilities;
- performance by partial-unlearning severity.

## Coverage/abstention

- PASS percentage;
- FAIL percentage;
- INCONCLUSIVE percentage;
- coverage = PASS + FAIL fraction;
- selective/conditional error among decided cases;
- risk-coverage curve.

## Query efficiency

- mean/median queries to decision;
- query-budget-performance curves;
- query-budget-coverage curves;
- wall-clock/API cost where applicable.

## Retained-data privacy

- MIA AUC;
- MIA advantage (define exact convention);
- attack accuracy only as supplementary, since it depends on class balance;
- transcript-level estimated leakage trajectory;
- comparison with held-out/stronger attackers;
- confidence intervals and calibration error.

## Tradeoff

- audit quality vs measured leakage;
- coverage vs measured leakage;
- queries vs audit quality;
- privacy/query feasibility maps;
- Pareto frontier.

Use exact definitions in `evaluation/metrics.py` and the paper. Do not interchange AUROC, accuracy, attack advantage, and leakage score.

---

# 18. Team roles discussed

A proposed division from the conversation (confirm with the actual team before treating it as final):

- **Hamza:** dataset splitting, model training, original/retrained targets, unlearning methods, partial-unlearning checkpoints, and black-box target interface.
- **Sabeeh:** auditing logic, random/fixed/active policies, audit gain, privacy-aware acquisition, sequential evidence, stopping, and PRISM-U integration.
- **Abbas:** membership-inference evaluation, experiment runner/configuration, result storage, statistical metrics, plots, reproducibility, and integration/dashboard coordination.

The team has three members. Sabeeh was welcomed into the discussion. The precise ownership and deadlines should be confirmed together; do not infer completion from an assigned role.

---

# 19. What has been done versus what has not

This section is intentionally conservative. It reflects what is evidenced in the conversation, not what may exist in a separate local repository.

## Done / established in the conversation

- The project proposal PDF was uploaded: `PRISM-U FYP Proposal (1).pdf`.
- The team discussed PRISM-U's core purpose: black-box auditing of machine-unlearning outcomes while controlling retained-data membership exposure.
- The basic pipeline was mapped: original model, forget/retain split, honest retrain reference, unlearned targets, partial-unlearning targets, query pool, surrogate/reference bank, auditor, privacy evaluator, sequential evidence, budgets, stopping, and decisions.
- The primary benchmark discussed is CIFAR-10 + ResNet-18.
- The secondary/generalization benchmark discussed is Purchase-100 + MLP, conditional on time/resources.
- The unlearning methods discussed are Gradient Ascent-style unlearning and SalUn.
- Controlled incomplete unlearning was discussed using partial/early-stopped execution levels such as 25%, 50%, 75%, and 100%.
- The distinction between query pool size and query budget was clarified.
- It was clarified that one numeric privacy/query budget should not be blindly reused for every dataset/model; calibration is environment-dependent.
- It was clarified that privacy budget is an empirical membership-leakage constraint in the current proposal, not a formal DP guarantee.
- It was clarified that INCONCLUSIVE is legitimate abstention, but 100% INCONCLUSIVE without a reason/diagnosis means the auditor has no practical coverage.
- Coverage and risk-coverage analysis were added as important evaluation requirements.
- The novelty discussion identified the strongest candidate contribution as transcript-adaptive query selection jointly using marginal compliance information and marginal retained-membership leakage, under cumulative budgets and sequential stopping.
- Adjacent literature discussed includes Tang et al. (2026), BAFA (2026), Ye et al., Sarmasarkar et al. (2026), Gu et al., Google Research f-divergence auditing, and UAG (IJCAI 2026). Exact bibliographic details and claims must be independently reverified.
- A publication ambition was discussed: first complete a sound FYP implementation, then expand the experiments/theory for a stronger paper.
- A proposed team split was discussed: Hamza (model/unlearning), Sabeeh (auditor), Abbas (evaluation/integration). Confirm with team.

## Not yet demonstrated in this chat (verify the repository before concluding)

No code repository, source files, model checkpoints, experiment logs, plots, metrics, or running dashboard have been provided in this conversation. Therefore, from the evidence available here, none of the following should be marked complete without workspace inspection:

- CIFAR-10 data loader and deterministic split pipeline;
- trained original ResNet-18;
- retain-only retrained reference model;
- implemented GA unlearning;
- implemented SalUn;
- partial-unlearning checkpoints;
- black-box-only target API;
- auxiliary query pool;
- compliant/non-compliant surrogate bank;
- random/fixed auditor;
- active audit-gain estimator;
- MIA evaluator;
- transcript-level privacy estimator;
- PRISM-U acquisition policy;
- sequential evidence model;
- calibrated PASS/FAIL thresholds;
- privacy/query budget enforcement;
- INCONCLUSIVE logic;
- baseline comparisons;
- budget sweeps and Pareto/risk-coverage curves;
- transfer/generalization experiments;
- robustness tests/ablations;
- dashboard;
- paper/theorem/statistical guarantees.

**Do not treat example numbers in earlier discussions as results.** Any AUROC, MIA AUC, coverage, budget values, query counts, or tables used in explanations were illustrative examples only unless reproduced by actual code and logs.

---

# 20. Recommended first actions for the agent

After inspecting the repository, report:

1. What already exists and where.
2. What runs successfully.
3. Which checkpoints/data are available.
4. What tests/experiments have been executed and where the logs are.
5. What is missing relative to the milestones.
6. What is currently scientifically underspecified.

Then prioritize these tasks, in this order:

1. Establish a deterministic CIFAR-10 split and training smoke test.
2. Train/reload original and retain-only retrained ResNet-18 references.
3. Implement one unlearning method and controlled partial-execution variants.
4. Implement a black-box query interface and test that the auditor cannot access internals.
5. Build a basic query pool and random auditor.
6. Verify that the basic black-box auditor can distinguish known compliant/incomplete targets at all.
7. Only then develop adaptive audit gain, MIA evaluation, transcript-aware privacy, and the full PRISM-U policy.

If the baseline auditor cannot distinguish the controlled targets with generous query/privacy allowance, diagnose query-pool quality, target construction, response access, reference-bank quality, and evidence calibration before adding more complexity.

---

# 21. Key design decisions still open

The agent should help the team make and record these decisions rather than silently assuming answers.

1. **Operational compliance definition:** Is compliance defined by similarity to `M_retrain`, a specified unlearning metric, a formal/empirical threshold, or multiple criteria?
2. **Primary target output access:** full probabilities, logits, top-k, or labels only?
3. **Forget-set construction:** random subset, class-wise subset, user/record grouping, or another protocol?
4. **Partial-unlearning severity:** early-stopped epochs/steps only, or also other failure modes?
5. **Query-pool source:** which auxiliary split, distribution, size, and class balance?
6. **Privacy attack threat model:** what does the attacker know, which samples are members/non-members, and which transcript outputs can it access?
7. **Privacy metric/accountant:** what exactly is `L_hat(τ)`? Is it MIA advantage, attack AUC, a calibrated proxy, or another quantity? What uncertainty bound is used?
8. **Marginal leakage estimator:** how can `ΔL_t` be estimated from surrogate/reference experiments without evaluating the final test target repeatedly or leaking the answer?
9. **Audit-gain estimator:** entropy reduction, likelihood-ratio information, JS separation, or another objective?
10. **Evidence model:** likelihood ratio, posterior probability, conformal/other sequential method, or another calibrated statistic?
11. **Sequential validity:** what assumptions are required for adaptive sampling and stopping? Are there valid bounds under the chosen model?
12. **External budgets:** what privacy/error/cost levels are user or policy inputs versus locally calibrated values?
13. **Calibration transfer:** which values can transfer across seeds/architectures/datasets and which require recalibration?
14. **Compute scope:** which experiments are realistic on the team's hardware/time budget?
15. **Publication target:** FYP minimum first; conference/journal ambitions can be prioritized after baseline success.

Record decisions in `docs/design_decisions.md` with date, rationale, owner, and whether the decision is provisional or frozen.

---

# 22. Non-negotiable scientific cautions

- Do not claim “unlearned 25%” solely because the algorithm ran 25% of its steps. Call it 25%-execution/early-stopped condition unless actual forgetting severity is independently measured.
- Do not call empirical MIA AUC a formal privacy guarantee or DP epsilon.
- Do not sum arbitrary per-query MIA AUC values and call the result cumulative privacy loss.
- Do not tune thresholds or budget policies on final test targets.
- Do not let the auditor access target internals in the black-box evaluation.
- Do not describe PRISM-U as universally model-agnostic without testing. The intended claim is a reusable policy/framework with environment-specific calibration.
- Do not claim it works on all AI models. Current scope is black-box classification-model unlearning; LLM/generative-model auditing is outside the stated FYP scope unless formally added.
- Do not force PASS/FAIL when evidence is weak; return INCONCLUSIVE and report coverage.
- Do not hide poor coverage or failed runs.
- Do not assert novelty based only on a search summary. Read and compare the closest papers directly.
- Do not build the dashboard before the science/experiment pipeline is functioning.

---

# 23. Short version to keep the whole team aligned

**PRISM-U is not the unlearning algorithm. It is the external auditor.** It receives black-box responses, chooses its next query adaptively based on expected compliance information and estimated retained-membership leakage, tracks cumulative constraints and sequential evidence, and makes PASS/FAIL/INCONCLUSIVE decisions.

The first deliverable is not the fancy adaptive policy. It is a reproducible controlled experimental environment:

`CIFAR-10 → deterministic forget/retain split → original ResNet-18 → retain-only retrain reference → one unlearning method → partial-unlearning targets → black-box prediction interface.`

Then establish that a simple auditor can distinguish the controlled targets. After that, develop the novel part:

`response-adaptive audit gain + transcript-dependent retained-membership leakage + cumulative constraints + sequential stopping/abstention.`

The key scientific outcome is not “PRISM-U always decides.” It is:

> At a controlled false-decision risk, how much audit coverage and quality can PRISM-U achieve for a given retained-membership leakage and query budget, compared with random, fixed, and privacy-blind active auditors?

The project succeeds if it answers that question rigorously—even if the answer identifies real limitations—without confusing illustrative examples with measured findings.
