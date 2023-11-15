from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.pydantic_v1 import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools import tool
from langchain.agents import AgentExecutor
from neo4j_advanced_rag.neo4j_vector import VectorTool
from neo4j_advanced_rag.neo4j_cypher import GraphTool
from langchain.schema.runnable import ConfigurableField
from neo4j_advanced_rag.retrievers import (
    hypothetic_question_vectorstore,
    parent_vectorstore,
    summary_vectorstore,
    typical_rag,
)

class AgentInput(BaseModel):
    input: str
    user_id: str
    session_id: str

llm = ChatOpenAI()
vector_tool = VectorTool()
graph_tool = GraphTool()
tools = [vector_tool, graph_tool]
llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

assistant_system_message = """You are a helpful assistant. \
Use one of the tools provided to you if necessary."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", assistant_system_message),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = (
    {
        "input": lambda x: x["input"],
        "user_id": lambda x: x["user_id"],
        "session_id": lambda x: x["session_id"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        )
    }
    | prompt 
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True).with_types(input_type=AgentInput)
agent_executor = agent_executor | (lambda x: x["output"])
