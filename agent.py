"""Travel booking agent using LangChain v1 and LangGraph."""
import os
from typing import TypedDict, Annotated, Sequence
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain_community.vectorstores import Chroma
from langchain.tools import tool
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import operator

from pydantic import BaseModel, Field

from tools import (
    search_flights,
    search_hotels,
    create_booking,
    lookup_booking,
    get_weather_forecast
)


class TravelAgentState(TypedDict):
    """State for the travel booking agent."""
    messages: Annotated[Sequence[BaseMessage], operator.add]


def setup_rag():
    embeddings = OllamaEmbeddings()

    vectorstore = load_vector_store(embeddings)

    # return 3 most similar docs
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

def load_vector_store(embeddings: Embeddings) -> VectorStore:
    if os.path.exists("chroma_db"):
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )

        return vectorstore
    else:
        raise ValueError("Knowledge base not initialized. Run 'python setup_kb.py' first.")

class KnowledgeBaseInput(BaseModel):
    query: str = Field(...)

@tool("search_knowledge_base", args_schema=KnowledgeBaseInput)
def search_knowledge_base(query: str) -> str:
    """
    Search the travel knowledge base for information about destinations, policies, FAQs, etc.

    Args:
        query: The search query

    Returns:
        Relevant information from the knowledge base
    """
    retriever = setup_rag()
    docs = retriever.invoke(query)

    results = []
    for doc in docs:
        results.append(f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}\n")

    return "\n---\n".join(results)


def create_travel_agent():
    """Create the travel booking agent with LangChain."""
    model_name = os.getenv("MODEL", "llama3.2")
    temperature = float(os.getenv("MODEL_TEMPERATURE", "0"))

    model = ChatOllama(
        model=model_name,
        temperature=temperature,
        verbose=False,
    )

    tools = [
        search_flights,
        search_hotels,
        create_booking,
        lookup_booking,
        get_weather_forecast,
        search_knowledge_base,
    ]

    system_prompt = """You are a helpful travel booking assistant. Your role is to:
1. Help customers find and book flights and hotels
2. Answer questions about destinations, policies, and travel requirements
3. Provide personalized travel recommendations
4. Create bookings when customers are ready

Guidelines:
- Always search the knowledge base when asked about destinations, policies, or FAQs
- Be friendly, professional, and helpful
- Ask clarifying questions if travel details are missing (dates, destinations, passengers)
- Present options clearly with prices and key details
- Confirm all details before creating a booking
- Use the weather forecast tool when relevant
- Redact sensitive information when displaying booking details

When a customer wants to book:
1. Search for flights/hotels based on their requirements
2. Present the options clearly
3. Once they confirm, create the booking
4. Provide the booking confirmation"""

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[
            PIIMiddleware(
                "email",
                strategy="redact",
                apply_to_input=True,
                apply_to_output=True
            ),
            PIIMiddleware(
                "phone_number",
                detector=r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4}",
                strategy="redact",
                apply_to_input=True,
                apply_to_output=True
            ),
        ]
    )

    return agent


def create_travel_graph():
    """
    Create LangGraph workflow for travel booking.

    Graph structure:
    [Entry] → [Agent] → [End]
    """

    agent = create_travel_agent()

    workflow = StateGraph(TravelAgentState)

    def agent_node(state: TravelAgentState):
        """
        Node that runs the agent.
        """
        messages = state["messages"]
        initial_count = len(messages)

        response = agent.invoke({"messages": messages})

        new_messages = response["messages"][initial_count:]
        return {"messages": new_messages}

    workflow.add_node("agent", agent_node)

    workflow.set_entry_point("agent")

    workflow.add_edge("agent", END)

    return workflow.compile()


def run_agent_streaming(graph, query: str, state: TravelAgentState):
    """
    Run the agent with streaming output and update state.
    
    Returns a generator of events and updates the state in-place.
    """
    state["messages"].append(HumanMessage(content=query))

    for event in graph.stream(state, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "agent":
                if "messages" in node_output:
                    messages = node_output["messages"]
                    for msg in messages:
                        if isinstance(msg, AIMessage):
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    yield {
                                        "type": "tool_call",
                                        "tool": tool_call.get("name", "unknown"),
                                        "args": tool_call.get("args", {})
                                    }
                            elif msg.content:
                                yield {
                                    "type": "response",
                                    "content": msg.content
                                }

    final_state = graph.invoke(state)
    state["messages"] = final_state["messages"]

