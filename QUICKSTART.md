# 🚀 Quickstart & Demonstration Guide

This guide contains everything you need to run, test, and demonstrate the **Autonomous Financial Wellness Agent** on any laptop.

---

## 💻 1. Quick Setup (Under 2 Minutes)

### Prerequisites
- **Python 3.10, 3.11, or 3.12** installed on your system.
- *(Windows)* Make sure **"Add Python to PATH"** was checked during installation.

### Option A: Windows 1-Click Launch (Easiest)
Simply **double-click** the file:
```
run.bat
```
*It will automatically create the virtual environment, install requirements if needed, and open the app in your browser at `http://localhost:8501`.*

---

### Option B: Manual Command Line Launch (Windows / Mac / Linux)

1. **Open a terminal** in this project folder:
   ```bash
   # Windows PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   streamlit run app.py
   ```

   ```bash
   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   streamlit run app.py
   ```

2. The application will open in your browser at `http://localhost:8501`.

---

## 🔑 2. API Key & AI Status

The project comes with a pre-configured `.env` file:
- In the sidebar, you will see **`✅ AI connected`**.
- If internet is unavailable or Google API rate limits hit (503/429), the system's **deterministic financial calculation engine and synthesis fallback** will smoothly handle the responses with 0 errors.
- To switch or update API keys, edit `.env`:
  - `GOOGLE_API_KEY=...` (free at https://aistudio.google.com/apikey)
  - Or switch to Grok: `LLM_PROVIDER=grok` and `XAI_API_KEY=...`

---

## 🎯 3. Live Demonstration Walkthrough (Viva / Presentation)

### Step 1: Load Data
- **Option 1 (Instant):** In the sidebar, click **"Load Demo Financial Profile"**.
- **Option 2 (Statement Upload - Recommended for Viva):**
  - Click **"Browse files"** under **"Upload CSV or PDF statement"**.
  - Choose either:
    - `data/sample_statements/healthy_bank_statement.pdf`
    - `data/sample_statements/healthy_bank_statement.csv`
  - *Result:* Ingests 39 real transactions across 3 months and generates a **healthy, balanced score of 66–69 / 100** with positive metrics.

### Step 2: Showcase Multi-Agent Tabs
- **Overview:** Shows Health Score, Current Balance, Savings Rate, Debt-to-Income, Emergency Runway, and Feasibility of Goals.
- **Spending Analysis:** Displays category distributions, transaction frequency, and merchant breakdowns.
- **Budget:** Shows category utilization and overspend risk badges.
- **Cash Flow:** 30-day deterministic cash runway forecast.
- **Goals:** Add custom financial goals; calculates monthly required savings dynamically.
- **Agent Trace:** Full observability log showing timestamps, tool calls, and execution latencies across all 11 specialist agents.

### Step 3: AI Advisor Demonstrations
Switch to the **AI Advisor** tab and test these questions:

1. **Car Affordability Simulation:**
   > `Can I afford a ₹15 lakh car?`
   - *Agent behavior:* Simulates auto loan EMI, checks debt-to-income limits, and advises on down payment buffer.

2. **Savings Optimization:**
   > `How can I save ₹10,000 more per month?`
   - *Agent behavior:* Audits discretionary categories (dining, subscriptions) and suggests practical reallocation.

3. **Salary Scenario:**
   > `What if my salary increases by ₹20,000?`
   - *Agent behavior:* Recalculates new disposable income and shows how fast goals can be reached.

4. **Investment Readiness Check:**
   > `Am I investment ready?`
   - *Agent behavior:* Evaluates emergency fund runway (>3 months) and low debt ratio before advising index fund / SIP allocation.

5. **Domain Guardrail (Crucial Demo Point):**
   > `how are u` OR `do you like flowers`
   - *Agent behavior:* Catches non-financial queries via input guardrails and replies with:
     `⚠️ Query not related to financial advice. Please ask questions concerning your personal budget, savings, expenses, debts, or investments.`

---

## 🧪 4. Running Unit Tests
To demonstrate software quality and test coverage to examiners, run:
```bash
pytest -q
```
*Expected: All 12/12 unit tests pass (`100%`).*
