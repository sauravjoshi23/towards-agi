from neo4j_advanced_rag.agent import agent_executor

if __name__ == "__main__":
    original_query = "How many tools have you been provided?"
    print(agent_executor.invoke({
                "input": original_query,
                "user_id": "1",
                "session_id": "1",
            }
        )
    )
    original_query = "Can you please use this tool and answer what is the plot of Dune?"
    print(agent_executor.invoke({
                "input": original_query,
                "user_id": "1",
                "session_id": "1",
            }
        )
    )
