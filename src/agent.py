"""Multi-step ReAct agent using LangGraph with RAG + tool-calling."""
from typing import TypedDict, Annotated, List
import operator

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    question: str
    final_answer: str


def build_react_agent(vectorstore: FAISS, model: str = "gpt-4o-mini"):
    """
    ReAct agent with retrieval + calculator tools.
    Cuts report turnaround from days to hours by automating
    multi-step reasoning over domain documents.
    """

    @tool
    def retrieve_documents(query: str) -> str:
        """Search the knowledge base for relevant information."""
        docs = vectorstore.similarity_search(query, k=4)
        return "\n\n".join(doc.page_content for doc in docs)

    @tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression safely."""
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Invalid expression"
        try:
            return str(eval(expression))  # noqa: S307 — restricted to numeric chars above
        except Exception as e:
            return f"Error: {e}"

    tools = [retrieve_documents, calculate]
    llm = ChatOpenAI(model=model, temperature=0).bind_tools(tools)

    def call_model(state: AgentState):
        from langchain_core.messages import HumanMessage
        messages = state["messages"]
        if not messages:
            messages = [HumanMessage(content=state["question"])]
        response = llm.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


def run_agent(agent, question: str) -> str:
    from langchain_core.messages import HumanMessage
    result = agent.invoke({"messages": [HumanMessage(content=question)], "question": question, "final_answer": ""})
    return result["messages"][-1].content
