from aura_data_analyst.decision_orchestrator import FinalDecisionMatrix

def main():
    print("Starting Data-Analyst Agent (Running with Mock Data Engine)...\n")
    agent = FinalDecisionMatrix()
    result = agent.analyze()
    print("\n[Result Object]", result)

if __name__ == "__main__":
    main()
