from fastapi import FastAPI
from langserve import add_routes
from neo4j_advanced_rag.agent import agent_executor


app = FastAPI()

# Edit this to add the chain you want to add
add_routes(app, agent_executor, path="/neo4j-advanced-rag", config_keys=["configurable"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
