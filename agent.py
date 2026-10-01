"""
MILESTONE 3 - Multimodal RAG + LangGraph Agent
-----------------------------------------------
Combines:
    1. Vision findings from YOLO
    2. Safety documents from ChromaDB
    3. Gemini for grounded answers

Flow:
START -> optional vision context -> retrieve -> answer -> END
"""

from typing import TypedDict
from langgraph.graph import END, START, StateGraph
from knowledge_base import search
from llm import ask_llm


PROMPT = """You are a workplace safety assistant.

You have two different sources of information:

1. CAMERA FINDINGS:
These are observations produced by the YOLO vision model.
Treat them as facts about what the model detected.

2. SAFETY DOCUMENTS:
These contain the official safety rules and requirements.
Use them as the authoritative source for safety rules.

IMPORTANT RULES:
- For questions about what the camera detected, use CAMERA FINDINGS.
- For questions about safety requirements, use SAFETY DOCUMENTS.
- For questions involving both, clearly separate camera observations
  from documented safety rules.
- Do NOT invent detections or safety requirements.
- Do NOT subtract PPE counts from person counts to determine who is
  not wearing PPE.
  Example: 3 people and 1 vest does NOT prove that 2 people are
  without vests.
- Only report a specific PPE violation when the camera explicitly
  detects a violation such as no_helmet, no_gloves, no_goggle, etc.
- If the camera information is insufficient to answer a question,
  clearly say that the exact value cannot be determined from the
  available camera findings.
- Do not conclude that workers are fully compliant merely because
  no explicit PPE violation was detected.
- If compliance depends on context not provided by the camera, such as
  the work area or activity, say that compliance cannot be fully determined.
- Cite document-derived facts using [source #chunk].
- Do not use document citations for camera observations.
- If a requested safety rule is not present in the documents, say:
  "Not found in documents."

CAMERA FINDINGS:
{vision}

RETRIEVED SAFETY DOCUMENTS:
{docs}

QUESTION:
{question}

Answer clearly and concisely:
"""


class State(TypedDict, total=False):
    question: str
    vision: str
    violations: list
    search_query: str
    docs: list
    answer: str
    steps: list


def use_image(state):
    """Use vision findings to improve document retrieval."""
    question = state["question"]
    violations = state.get("violations", [])
    extra = " ".join(sorted(set(violations)))
    query = f"{question} {extra}".strip()

    return {
        "search_query": query,
        "steps": state.get("steps", []) + ["use_image"],
    }


def retrieve(state):
    """Retrieve relevant safety-document chunks."""
    query = state.get("search_query") or state["question"]
    docs = search(query, top_k=3)

    return {
        "docs": docs,
        "steps": state.get("steps", []) + ["retrieve"],
    }


def answer(state):
    """Generate a grounded answer using Gemini."""
    docs = state.get("docs", [])

    docs_text = (
        "\n\n".join(
            f"[{d['source']} #{d['chunk']}] {d['text']}"
            for d in docs
        )
        if docs else "(No relevant documents found.)"
    )

    prompt = PROMPT.format(
        vision=state.get("vision") or "No camera findings.",
        docs=docs_text,
        question=state["question"],
    )

    reply = ask_llm(prompt)

    if reply is None:
        if docs:
            reply = (
                "Gemini is unavailable. The most relevant "
                "safety-document passages are:\n\n"
                + "\n\n".join(
                    f"- [{d['source']} #{d['chunk']}] {d['text']}"
                    for d in docs
                )
            )
        else:
            reply = "No relevant information was found in the documents."

    return {
        "answer": reply,
        "steps": state.get("steps", []) + ["answer"],
    }


def route(state):
    """Choose whether camera context is available."""
    return "use_image" if state.get("vision") else "retrieve"


graph = StateGraph(State)

graph.add_node("use_image", use_image)
graph.add_node("retrieve", retrieve)
graph.add_node("answer", answer)

graph.add_conditional_edges(
    START,
    route,
    {
        "use_image": "use_image",
        "retrieve": "retrieve",
    },
)

graph.add_edge("use_image", "retrieve")
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", END)

agent = graph.compile()


def ask_agent(question, vision="", violations=None):
    """Run the complete multimodal RAG pipeline."""
    return agent.invoke({
        "question": question,
        "vision": vision,
        "violations": violations or [],
        "steps": [],
    })


if __name__ == "__main__":
    result = ask_agent("What PPE is required on site?")

    print("Steps:", result["steps"])
    print("\nAnswer:\n", result["answer"])