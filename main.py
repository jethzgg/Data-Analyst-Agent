from src.decision_orchestrator import FinalDecisionMatrix

def main():
    print("Starting Data-Analyst Agent (Running with Mock Data Engine)...\n")
    agent = FinalDecisionMatrix()
    agent.process_pipeline()

if __name__ == "__main__":
    main()
