from src.agent_workflow import AgentOrchestrator

def main():
    print("Khởi động Data-Analyst Agent (Chạy với MOCK Test Data)...\n")
    agent = AgentOrchestrator()
    agent.process_pipeline()

if __name__ == "__main__":
    main()