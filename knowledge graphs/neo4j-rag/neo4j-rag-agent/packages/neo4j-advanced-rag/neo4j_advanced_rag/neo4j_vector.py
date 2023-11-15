from operator import itemgetter
from typing import Optional, Type
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import ConfigurableField, RunnablePassthrough
from neo4j_advanced_rag.history import get_vector_history, save_vector_history
from langchain.tools import BaseTool
from pydantic import BaseModel, BaseSettings
from typing import Type, Any
from neo4j_advanced_rag.retrievers import (
    hypothetic_question_vectorstore,
    parent_vectorstore,
    summary_vectorstore,
    typical_rag,
)

class Question(BaseModel):
    question: str
    user_id: str
    session_id: str

class VectorTool(BaseTool, BaseSettings):
    name = "vector_tool"
    description = "Useful Tool for retrieving general open ended information about Dune"
    args_schema: Type[Question] = Question

    def _run(self, question, user_id, session_id):
        retriever = typical_rag.as_retriever().configurable_alternatives(
            ConfigurableField(id="strategy"),
            default_key="typical_rag",
            parent_strategy=parent_vectorstore.as_retriever(),
            hypothetical_questions=hypothetic_question_vectorstore.as_retriever(),
            summary_strategy=summary_vectorstore.as_retriever(),
        )

        # Define LLM
        llm = ChatOpenAI()

        # Condense a chat history and follow-up question into a standalone question
        condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
        Make sure to include all the relevant information.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""  # noqa: E501
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

        # RAG answer synthesis prompt
        answer_template = """Answer the question based only on the following context:
        <context>
        {context}
        </context>"""

        ANSWER_PROMPT = ChatPromptTemplate.from_messages(
            [
                ("system", answer_template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{question}"),
            ]
        )

        chain = (
            RunnablePassthrough.assign(chat_history=get_vector_history)
            | RunnablePassthrough.assign(rephrased_question=CONDENSE_QUESTION_PROMPT | llm | StrOutputParser())
            | RunnablePassthrough.assign(context=itemgetter("rephrased_question") | retriever)
            | RunnablePassthrough.assign(output=ANSWER_PROMPT | llm | StrOutputParser())
            | save_vector_history
        ).with_types(input_type=Question)

        return chain.invoke(
            {
                "question": question,
                "user_id": user_id,
                "session_id": session_id,
            },
            {"configurable": {"strategy": "typical_rag"}} #todo
        )