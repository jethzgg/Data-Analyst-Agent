from src.decision_orchestrator import FinalDecisionMatrix

def main():
    print("Khởi động Data-Analyst Agent (Chạy với Mock Data Engine)...\n")
    agent = FinalDecisionMatrix()
    agent.process_pipeline()

if __name__ == "__main__":
    main()
