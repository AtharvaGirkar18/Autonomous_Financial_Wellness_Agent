from agents.base import run_agent
from tools.goal_tools import goal_status
from datetime import date

def goal_node(state: dict) -> dict:
    def _run(s):
        goals=s.get("goals",[])
        cap=s.get("health",{}).get("monthly_savings",s.get("health",{}).get("monthly_savings",0))
        out=[]
        for g in goals:
            target_date=date.fromisoformat(g["target_date"])
            out.append({**g,**goal_status(g["target_amount"],g.get("current_amount",0),target_date,cap)})
        return {"goals":out}
    return run_agent(state,"Goal Planning Agent",_run)
