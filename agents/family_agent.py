from agents.base import run_agent

def family_finance_node(state: dict) -> dict:
    def _run(s):
        # Lightweight architecture support; aggregation can be extended with linked profile IDs.
        return {"family_finance":{"enabled":False,"profiles":[],"note":"Architecture supports multiple linked profiles."}}
    return run_agent(state,"Family Finance Agent",_run)
