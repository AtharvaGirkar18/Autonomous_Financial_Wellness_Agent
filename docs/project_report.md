# Autonomous Financial Wellness Agent — Project Report Draft

## A. Dataset

### Dataset creation methodology

The benchmark is generated programmatically with a fixed random seed of **42**. It contains 18 synthetic Indian user profiles over six months and represents salary deposits, bank transactions, UPI payments, credit-card subscription charges, rent, EMI, utilities, essential spending, discretionary spending, recurring expenses, anomalies and duplicate charges.

No real personal banking information is used. The generator is intentionally reproducible so the same dataset, ground truth and evaluation metrics can be recreated before a project demonstration or report submission.

### Dataset summary

| Property | Current result |
|---|---:|
| Synthetic users | 18 |
| Transactions | 3,294 |
| Time range | 6 months |
| Ground-truth anomalies | 90 |
| Ground-truth duplicate rows | 72 |
| Subscription transaction rows | 324 |
| Detected recurring subscription merchant records | 54 |
| Categories | 15 |

The subscription count is intentionally distinguished from transaction-row count: the dataset has 324 subscription charge rows, while the detector produces recurring merchant records for each synthetic user/merchant combination.

### Ground-truth scenarios

`USER001` is the judge-facing demo profile. Its food spending is deliberately set to **₹5,000, ₹6,500 and ₹8,000** in January–March 2026, followed by continued pressure in later months. It also contains five unusual transactions, duplicate Swiggy charges, recurring subscriptions, a ₹32,000 starting cash buffer and an at-risk ₹12 lakh car goal.

### Schema

Core transaction fields are:

`transaction_id`, `user_id`, `date`, `description`, `merchant`, `amount`, `transaction_type`, `account_type`, `payment_mode`, `category`, `subcategory`, `is_recurring`, `is_anomaly`, `is_duplicate`, `is_subscription`.

Where appropriate, ground-truth labels are retained for classification, anomaly and duplicate evaluation. Synthetic statements are also exported as CSVs and generated as PDF fixtures using ReportLab.

## B. Problem Definition and Scope of Project

### Problem

Traditional personal finance applications often present transaction records, charts and static budgets. These tools are useful for awareness but do not necessarily determine what should happen next, verify recommendations, remember past interventions or proactively identify emerging risks.

### Motivation

The goal is to build a financial wellness system that behaves as a continuing decision-support workflow: ingest → understand → analyze → detect → predict → plan → recommend → review → alert → remember → reevaluate.

### Objectives

1. Normalize heterogeneous financial transaction data.
2. Categorize transactions using a hybrid rule/ML strategy.
3. Analyze spending trends, savings and budget utilization.
4. Detect unusual and duplicate transactions.
5. Identify recurring subscription charges conservatively.
6. Forecast short-horizon cash-flow risk.
7. Compute a deterministic Financial Health Score.
8. Support goals and life-event simulations through a financial twin.
9. Coordinate specialized agents with a planner and reviewer.
10. Produce evidence-grounded recommendations and proactive alerts.
11. Persist state in SQLite and expose an observability trace.
12. Quantitatively evaluate both analytical models and agent behavior.

### Functional scope

The Streamlit application supports synthetic demo mode, CSV uploads, PDF statement ingestion, budget editing, goal creation, what-if simulation, advisor queries, alerts and agent-trace inspection.

### Non-functional scope

The implementation emphasizes reproducibility, modularity, deterministic financial calculations, no hard-coded secrets, graceful failure, unit testing, persistent state, and deployment readiness.

### Assumptions and limitations

The project uses synthetic financial data, simplified forecasting, heuristic financial-health thresholds and limited PDF parsing patterns. The result is an academic demonstration, not a banking-grade financial planning or fraud platform. No individualized securities recommendation is produced.

## C. Literature Review and Project Report

A 14-paper literature review is maintained in `docs/literature_review.md`. It covers personal financial management, transaction classification, anomaly/fraud detection, cash-flow forecasting, financial LLMs, recommendation-oriented finance applications, multi-agent systems, autonomous-agent evaluation and explainable AI.

### Research gap

The literature contains strong component-level work, but the proposed contribution is an integration architecture for a personal-finance use case: a planner selects a request-specific route; specialized agents perform analytical work; deterministic tools handle financial arithmetic; a shared state connects the agents; a reviewer validates output; proactive alerts are generated; and SQLite retains state for subsequent feedback.

This is intentionally stated as a project-level systems gap rather than a claim that no prior publication has combined any of these components.

### Proposed contribution

The **Autonomous Financial Wellness Agent** provides a reproducible college-scale reference implementation for integrating deterministic financial analytics, ML models and LLM-based orchestration into a persistent planner–specialist–reviewer–alert feedback loop.

## D. Conceptual System Design

### Architecture

See `docs/architecture.md` and the Mermaid diagram in `README.md`.

```text
Streamlit
   ↓
Master Orchestrator / LangGraph
   ↓
Planner
   ↓
Request-specific Specialist Agents
   ↓
Deterministic Tools + Shared Financial State
   ↓
Financial Advisor
   ↓
Reviewer
   ↓
Alerts
   ↓
SQLite Memory / Observability
```

### Planner / reviewer workflow

For a question such as **“Can I afford a ₹12 lakh car?”**, the planner selects Health → Forecast → Financial Twin → Advisor → Reviewer. The system retrieves the financial state, computes the EMI deterministically, simulates the impact on savings/cash buffer, produces a recommendation and then passes it to the reviewer before alerts and persistence occur.

For a complete data-analysis event, the workflow expands to ingestion, categorization, spending, budget, anomaly, subscription, forecast, health, financial twin, advisor and reviewer nodes.

### Agent communication and tools

Agents communicate through the typed shared state. Explicit deterministic functions exist for transaction retrieval, category analysis, budgets, savings rate, debt ratio, emergency-fund coverage, anomaly detection, subscriptions, forecasting, EMI, scenarios, goal status and memory persistence.

### Memory and database

SQLite + SQLAlchemy stores users, transactions, budgets, goals, subscriptions, alerts, agent runs, recommendations, simulations and financial-health history. Streamlit `session_state` is not the sole state store.

### Explainability

Every recommendation contains evidence metadata, while alerts contain severity, explanation and a recommended action. The observability tab exposes the selected plan, agent order, tool labels, execution time and reviewer outcome.

## E. Technologies Used

| Technology | Reason for selection |
|---|---|
| Python | Core data, ML and orchestration implementation |
| Streamlit | Fast interactive academic demonstration and deployment target |
| LangGraph | Explicit graph/state orchestration and request-specific agent routing |
| Configurable LLM API | Planning, language reasoning and controlled explanation layer |
| pandas / NumPy | Deterministic financial analytics and transformations |
| scikit-learn | Categorization and Isolation Forest anomaly detection |
| SQLite + SQLAlchemy | Lightweight persistent state with migration path to PostgreSQL |
| PyMuPDF | Fast PDF text extraction |
| ReportLab | Reproducible synthetic PDF fixture generation |
| Plotly | Interactive Streamlit charts |
| Pydantic / TypedDict | Structured validation and shared state typing |
| python-dateutil | Goal timeline calculation |
| Docker | Portable deployment path |

## F. Implementation

### Major modules

**Ingestion Agent:** validates and normalizes uploaded transaction tables and PDF extraction output.

**Categorization Agent:** combines merchant rules with TF-IDF + Logistic Regression when the rules are not confident enough.

**Spending Analysis Agent:** computes monthly income/expense totals, category totals, trend direction, volatility, essential/discretionary split and savings.

**Budget Agent:** evaluates utilization in the latest observed month and allows user overrides.

**Financial Health Agent:** calculates an auditable 0–100 score from deterministic components.

**Anomaly Agent:** combines Isolation Forest, an interpretable local-spend rule and duplicate matching.

**Subscription Agent:** detects recurring subscription patterns only when supported by merchant/category/label evidence and avoids unsupported “unused” claims.

**Forecast Agent:** generates a 30-day balance forecast with a volatility-based uncertainty interval and supports monthly experimental baselines.

**Financial Twin Agent:** stores income, expenses, savings capacity, debt ratio, emergency coverage, goals and health score as a scenario-ready financial state.

**Advisor Agent:** converts deterministic analytical outputs into evidence-based natural-language recommendations. When an LLM is configured, it can rephrase the already-grounded result without being allowed to supply new financial facts.

**Reviewer:** checks required outputs and evidence metadata and can reject/repair unsupported recommendations.

**Alert & Notification Agent:** raises budget, anomaly, duplicate, subscription-review, savings-rate, cash-flow and goal-risk alerts.

### UI screenshots

During final report preparation, capture screenshots of:

- Overview
- Transactions
- Spending Analysis
- Anomalies / Duplicate Candidates
- Subscriptions
- Budget
- Cash Flow
- Goals
- What-If
- AI Advisor
- Alerts
- Agent Trace

The application already provides dedicated views for these surfaces.

## G. Results and Conclusions

### Model and analytical results

The following values are from the reproducible benchmark generated by `python -m ml.evaluation`.

| Component | Metric | Result |
|---|---|---:|
| Categorization — rule baseline | Accuracy | 79.61% |
| Categorization — rule baseline | F1 | 77.42% |
| Categorization — ML | Accuracy | 100.00% |
| Categorization — ML | F1 | 100.00% |
| Categorization — hybrid | Accuracy | 100.00% |
| Categorization — hybrid | F1 | 100.00% |
| Anomaly Detection | Precision | 100.00% |
| Anomaly Detection | Recall | 100.00% |
| Anomaly Detection | F1 | 100.00% |
| Duplicate Detection | Precision | 100.00% |
| Duplicate Detection | Recall | 100.00% |
| Duplicate Detection | F1 | 100.00% |
| Forecast — naive | MAE | ₹4,454.06 |
| Forecast — naive | RMSE | ₹5,550.36 |
| Forecast — naive | MAPE | 6.37% |
| Forecast — moving average | MAE | ₹5,161.82 |
| Forecast — moving average | RMSE | ₹6,413.92 |
| Forecast — moving average | MAPE | 7.33% |

### Agentic-system evaluation

| Measure | Result |
|---|---:|
| Planning success rate | 100.00% |
| Tool-selection success rate | 100.00% |
| Grounded recommendation rate | 100.00% |
| Context-relevant response rate | 100.00% |
| Correct response rate | 100.00% |
| Deterministic hallucination proxy | 0.00% |
| Reviewer rejection on seeded invalid cases | 100.00% (2/2) |

**Important:** the hallucination figure is explicitly a deterministic-mode proxy: it counts recommendation output lacking evidence metadata. A semantic hallucination study for an LLM-backed provider requires a separate labeled human/LLM evaluation set.

### Demo profile snapshot

For `USER001`, the current reproducible run reports:

| Measure | Result |
|---|---:|
| Current balance | ₹32,000 |
| Monthly income | ₹75,000 |
| Monthly expenses | ₹71,731.36 |
| Monthly savings | ₹3,268.64 |
| Savings rate | 4.36% |
| Financial Health Score | 30.37 / 100 |
| Debt ratio | 13.33% |
| Emergency-fund coverage | 0.71 months |
| Detected anomalies | 5 |
| Duplicate candidates | 4 |
| Detected recurring subscriptions | 3 |
| Lowest 30-day projected balance | ₹829.92 |
| Projected low-balance date | 2026-07-06 |
| Car goal status | At Risk |
| Car-goal contribution gap | ₹50,731.36/month |
| Generated alerts | 16 |
| Trace steps | 13 |
| Reviewer | PASSED |

The exact machine-readable snapshot is stored in `artifacts/demo_snapshot.json`.

### Sample agent outputs

**Car affordability:** the deterministic model calculates an estimated EMI of approximately **₹19,928** for the ₹12 lakh scenario under the demo assumptions of a 20% down payment, 9% annual rate and 60-month tenure. Because the down payment itself exceeds the ₹32,000 cash buffer, the system marks the scenario as challenging.

**Salary increase:** increasing monthly salary by **₹15,000** changes modeled savings capacity from approximately **₹3,269** to **₹18,269**, assuming spending remains unchanged.

**Savings goal:** the advisor identifies observed discretionary categories that could contribute toward an additional **₹5,000/month** reduction without directly cutting essential categories in the deterministic heuristic.

### Interpretation

The benchmark demonstrates the intended architecture and measurable behavior. The 100% ML/anomaly scores should not be presented as production banking performance: the synthetic benchmark intentionally contains strong merchant/category patterns and clearly injected anomaly cases. The forecasting experiment also shows that the naive last-value baseline is better than the simple three-month moving average for this particular synthetic dataset, illustrating why method choice should be empirical instead of assumed.

### Limitations

1. The dataset is synthetic and cannot establish real-world fraud or categorization performance.
2. PDF extraction currently targets the repository's line-oriented synthetic statement format, not arbitrary bank layouts.
3. The cash-flow model is intentionally lightweight and short-horizon.
4. Subscription usage cannot be proven from transaction data alone.
5. SQLite is appropriate for the academic deployment target but not a substitute for banking-grade data infrastructure.
6. LLM semantic hallucination requires a separate provider-backed benchmark.

### Future enhancements

Production-oriented extensions include institution-specific statement parsers, secure financial-data APIs, learned user embeddings, richer forecasting models, subscription-usage telemetry, human approval for high-impact actions, PostgreSQL, stronger agent benchmarks, encrypted sensitive data and longitudinal outcome evaluation.

## Final conclusion

The completed system is best characterized as an **AI-powered financial-wellness operating system prototype** rather than an expense tracker with a chatbot attached. Its main evidence is the request-specific planner, specialized agents, deterministic tools, shared persistent state, reviewer repair path, proactive monitoring, scenario simulation, explainable recommendations and quantitative evaluation. The architecture is designed so that each major claim can be traced to deterministic calculations, model outputs, measured experiments or verified literature.
