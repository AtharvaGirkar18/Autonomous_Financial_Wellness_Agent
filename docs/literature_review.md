# Literature Review

The review targets the ten research areas required by the project specification: personal financial management, transaction classification, anomaly/fraud detection, spending analysis, cash-flow forecasting, recommendation systems, financial LLMs, multi-agent systems, agentic orchestration, and explainable AI. Bibliographic details below were checked against publisher, IEEE, Springer, Elsevier, arXiv, or bibliographic records during project preparation.

## 1. Literature survey

| Paper | Year | Method | Dataset | Key Contribution | Limitation | Relevance |
|---|---:|---|---|---|---|---|
| Jadhav & Patil, **AI-ML based Expense Categorization and Budgeting System for Personalized Financial Management** | 2025 | AI/ML, NLP, hybrid ML, predictive analytics | Financial transaction data; prototype-oriented | Integrates automated expense categorization, anomaly detection and personalized budgeting | Prototype framing; limited evidence of long-term autonomous feedback | Directly motivates hybrid categorization, anomaly analysis and personalized budgeting |
| Yang, Biibosunov, Biibosunova, Kalmanbetova & Wang, **A deep learning framework for automated transaction classification and budget prediction in personal financial management** | 2026 | Transformer attention + BiLSTM + TCN, multi-task learning | 52,847 real Chinese transactions | Jointly models transaction classification and budget prediction and reports cross-dataset backbone evaluation | Chinese-language focus; requires substantially larger real data and a deep model | Supports the classification/forecasting combination in this project and motivates a lightweight hybrid alternative |
| Hilal, Gadsden & Yawney, **Financial Fraud: A Review of Anomaly Detection Techniques and Recent Advances** | 2022 | Review of statistical, supervised, semi-supervised and unsupervised anomaly detection | Literature survey | Organizes effective financial anomaly/fraud techniques and recent advances | Review rather than a single deployable personal-finance system | Supports Isolation Forest + interpretable rules for anomaly detection |
| Al-Hashedi & Magalingam, **Financial fraud detection applying data mining techniques: A comprehensive review from 2009 to 2019** | 2021 | Systematic review of data-mining approaches | 75 reviewed studies | Surveys 34 data-mining techniques across financial fraud categories | Covers institutional fraud broadly rather than personal-finance workflows | Supports comparative fraud/anomaly evaluation and explains why multiple methods are useful |
| Pang, Shi, Shi & Zhao, **A nonlinear dynamic approach to cash flow forecasting** | 2022 | Nonlinear dynamic grey-box model | Financial cash-flow data | Models nonlinear, dynamic cash-flow parameters | More complex than needed for a compact personal-finance demo; not designed for daily individual statements | Motivates explicit cash-flow forecasting rather than LLM-only estimation |
| Guo, Hu & Xiao, **Cash flow forecasting: dealing with serial correlation and idiosyncratic heterogeneity** | 2026 | Local-learning predictive model combining cross-sectional and time-series information | Operating cash-flow observations | Addresses serial correlation and heterogeneity to improve forecasting | Corporate CFO setting and longer-history assumptions | Supports the project’s emphasis on measured forecast error and limits of simple baselines |
| Wu et al., **BloombergGPT: A Large Language Model for Finance** | 2023 | 50B-parameter finance-domain LLM | 363B financial tokens + 345B general tokens | Demonstrates domain-specialized LLM capabilities for finance tasks | Proprietary-scale compute/data; unsuitable as a direct student deployment | Motivates using an LLM for finance-oriented language reasoning while keeping calculations deterministic |
| Yang, Liu & Wang, **FinGPT: Open-Source Financial Large Language Models** | 2023 | Data-centric financial LLM + parameter-efficient adaptation | Open financial data/resources | Emphasizes accessible, transparent finance LLM development and applications including robo-advising | Financial LLM quality depends heavily on curated data and safe use | Supports configurable LLM integration and finance-specific reasoning |
| Wu et al., **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation** | 2024 | Multi-agent LLM conversations with tools/humans/code | Multiple pilot application environments | Shows how specialized agents can coordinate through flexible workflows | Generic framework; not a personal-finance evaluation | Supports the project’s multi-agent coordination design |
| Liu et al., **AgentBench: Evaluating LLMs as Agents** | 2024 | Multi-dimensional agent benchmark | 8 interactive environments | Establishes quantitative evaluation of reasoning and decision-making in agent settings | General benchmark rather than financial-domain evaluation | Motivates explicit planning/tool-use/reviewer metrics in this project |
| Wang et al., **A survey on large language model based autonomous agents** | 2024 | Survey / unified agent framework | Literature survey | Organizes agent construction, applications and evaluation strategies | Survey-level abstraction; no single application architecture | Provides conceptual support for planning, memory, tools and evaluation |
| Černevičienė & Kabašinskas, **Explainable artificial intelligence (XAI) in finance: a systematic literature review** | 2024 | Systematic review of XAI in finance | 138 relevant articles from 2005–2022 | Reviews finance tasks, XAI methods and unresolved challenges | Focuses on explainability broadly rather than personal-finance agents | Supports evidence-linked recommendations and transparent health scoring |
| Weber, Carl & Hinz, **Applications of Explainable Artificial Intelligence in Finance—a systematic review of Finance, Information Systems, and Computer Science literature** | 2024 | Systematic literature review | 2,022 screened articles; 60 relevant articles | Maps XAI methods and goals in finance, highlighting transparency needs in regulated settings | Broad finance scope; anti-money-laundering evidence remains comparatively sparse | Reinforces the need for traceable, explainable financial recommendations |
| **Money Map: The Personal Finance Tracker** | 2025 | Data-driven ML, forecasting, visualization and recommendations | Personal financial data | Combines spending analysis, budget forecasting, personalized insights and visualization for individual finance | Primarily application-oriented; limited evidence of autonomous planner/reviewer feedback loops | Closest comparator for the integrated dashboard/forecast/recommendation experience |

## 2. Findings by research area

### Personal financial management and recommendations
Recent personal-finance systems demonstrate the value of automated categorization, analytics, budgeting and recommendation generation. Jadhav & Patil explicitly combine expense classification, anomaly detection and personalized budgeting, while *Money Map* integrates forecasting and personalized insights. The 2026 DFTS paper further shows that classification and budget prediction can be linked in one learning architecture. These systems motivate the project's analytics layer but do not, by themselves, establish a reusable planner-reviewer-memory loop for continuing intervention.

### Transaction classification
Rule matching remains attractive for known merchants because it is fast and explainable. Modern work increasingly applies TF-IDF/SVM, ensemble models, transformers, and multi-task/deep architectures. The proposed system intentionally uses a hybrid architecture: merchant rules first, then TF-IDF + Logistic Regression for ambiguous cases, with an LLM available only as a controlled fallback/rephrasing layer.

### Anomaly and fraud detection
The survey literature shows strong roles for supervised, semi-supervised and unsupervised methods. The project therefore combines Isolation Forest with an interpretable local spending rule and a deterministic duplicate detector. Because the benchmark is synthetic and injected anomalies are intentionally separable, the resulting anomaly metrics must be presented as benchmark measurements rather than as production fraud-detection claims.

### Cash-flow forecasting
Cash-flow research confirms that time-series structure matters and that more advanced models can represent nonlinear or heterogeneous behavior. The project intentionally keeps the production demo lightweight: recurring-schedule modeling plus short-horizon daily forecasting, while the experiment runner compares naive and moving-average monthly baselines with MAE/RMSE/MAPE.

### Financial LLMs and agentic systems
BloombergGPT and FinGPT show the growing role of domain-aware LLMs in finance, while AutoGen and the autonomous-agent survey establish a broader pattern of planning, tool use, coordination and iterative workflows. AgentBench reinforces the need to evaluate agent behavior quantitatively rather than treating an LLM call as proof of agency. The proposed architecture therefore separates deterministic tools from language reasoning and records planner, tool, reviewer and memory events in an execution trace.

### Explainability
The two XAI reviews show that explainability and traceability are particularly important in finance. This directly motivates deterministic financial-health scoring, evidence metadata for recommendations, alert explanations, and a visible agent trace rather than opaque LLM-only outputs.

## 3. Research gap

The literature is strong on individual building blocks—transaction classification, fraud/anomaly detection, forecasting, financial language models, recommendation systems, multi-agent frameworks and XAI—but comparatively fragmented when applied to a single personal-finance workflow. The project-level gap addressed here is the **integration of these components into an autonomous financial-wellness loop** that:

1. plans a request-specific workflow;
2. coordinates specialized analytical agents;
3. invokes deterministic tools for financial arithmetic;
4. carries a shared persistent financial state;
5. reviews and repairs recommendations before delivery;
6. produces proactive alerts without an explicit user question;
7. simulates goals/life events through a financial twin; and
8. measures both ML performance and agent behavior.

This is deliberately framed as an integration/design contribution, not as a claim that no prior system contains any of these components individually.

## 4. Proposed contribution

The proposed **Autonomous Financial Wellness Agent** contributes a reproducible, college-scale reference architecture for financial wellness rather than a production banking product. Its main technical contribution is the combination of deterministic financial tools, ML analytics and LLM-based orchestration within one planner → specialist agents → reviewer → alert → memory loop. A secondary contribution is a judge-friendly synthetic benchmark with explicit anomaly, duplicate, category and subscription ground truth, enabling reproducible evaluation without using real personal banking data.

## 5. References / verification metadata

1. Rajshree Jadhav & G. A. Patil, *AI-ML based Expense Categorization and Budgeting System for Personalized Financial Management*, Communications on Applied Nonlinear Analysis, 2025. DOI: `10.52783/cana.v32.5266`.
2. Zhengxiang Yang, Bbolotbek Biibosunov, Saltanat Biibosunova, Gulzat Kalmanbetova & Xuzhang Wang, *A deep learning framework for automated transaction classification and budget prediction in personal financial management*, Discover Artificial Intelligence, 2026. DOI: `10.1007/s44163-026-02025-1`.
3. Waleed Hilal, S. Andrew Gadsden & John Yawney, *Financial Fraud: A Review of Anomaly Detection Techniques and Recent Advances*, Expert Systems with Applications, 2022. DOI: `10.1016/j.eswa.2021.116429`.
4. Khaled Gubran Al-Hashedi & Pritheega Magalingam, *Financial fraud detection applying data mining techniques: A comprehensive review from 2009 to 2019*, Computer Science Review, 2021. DOI: `10.1016/j.cosrev.2021.100402`.
5. Yang Pang, Shimeng Shi, Yukun Shi & Yang Zhao, *A nonlinear dynamic approach to cash flow forecasting*, Review of Quantitative Finance and Accounting, 2022. DOI: `10.1007/s11156-022-01066-8`.
6. Hongtao Guo, Yushan Hu & Zhijie Xiao, *Cash flow forecasting: dealing with serial correlation and idiosyncratic heterogeneity*, Review of Quantitative Finance and Accounting, 2026. DOI: `10.1007/s11156-025-01442-0`.
7. Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg & Gideon Mann, *BloombergGPT: A Large Language Model for Finance*, arXiv:2303.17564, 2023. DOI: `10.48550/arXiv.2303.17564`.
8. Hongyang Yang, Xiao-Yang Liu & Christina Dan Wang, *FinGPT: Open-Source Financial Large Language Models*, arXiv:2306.06031, 2023. DOI: `10.48550/arXiv.2306.06031`.
9. Qingyun Wu et al., *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*, COLM 2024 / arXiv:2308.08155. DOI: `10.48550/arXiv.2308.08155`.
10. Xiao Liu et al., *AgentBench: Evaluating LLMs as Agents*, ICLR 2024 / arXiv:2308.03688. DOI: `10.48550/arXiv.2308.03688`.
11. Lei Wang et al., *A survey on large language model based autonomous agents*, Frontiers of Computer Science, 2024. DOI: `10.1007/s11704-024-40231-1`.
12. Jurgita Černevičienė & Audrius Kabašinskas, *Explainable artificial intelligence (XAI) in finance: a systematic literature review*, Artificial Intelligence Review, 2024. DOI: `10.1007/s10462-024-10854-8`.
13. Patrick Weber, K. Valerie Carl & Oliver Hinz, *Applications of Explainable Artificial Intelligence in Finance—a systematic review of Finance, Information Systems, and Computer Science literature*, Management Review Quarterly, 2024. DOI: `10.1007/s11301-023-00320-0`.
14. *Money Map: The Personal Finance Tracker*, 2025 International Conference on Knowledge Engineering and Communication Systems (ICKECS), IEEE. DOI: `10.1109/ICKECS65700.2025.11035699`.
