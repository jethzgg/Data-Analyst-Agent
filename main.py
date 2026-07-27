from data_analyst.decision_orchestrator import FinalDecisionMatrix

def main():
    agent = FinalDecisionMatrix()
    result = agent.analyze()
    print(result)

if __name__ == "__main__":
    main()

