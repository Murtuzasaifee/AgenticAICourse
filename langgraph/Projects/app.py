from dotenv import load_dotenv
load_dotenv()
import os
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langgraph.graph import add_messages
from typing import Annotated, List
from langchain_core.messages import AIMessage
from IPython.display import Image, display
from langgraph.graph import START, StateGraph, END


os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
llm=ChatGroq(model="qwen-2.5-32b")


class BlogState(TypedDict):
    topic: str
    title: str
    blog_content: Annotated[List, add_messages]
    reviewed_content: Annotated[List, add_messages]
    is_blog_ready: str
    


def generate_title(state: BlogState):
    prompt = f"""Generate compelling blog title option about {state["topic"]} that is:
    - SEO-friendly
    - Attention-grabbing
    - Topic-agnostic (works for multiple niches)
    - Between 6-12 words
    """
    
    response = llm.invoke(prompt)
    return {"title": response.content}

def generate_content(state: BlogState):
    prompt = f"""Write a comprehensive blog post titled "{state["title"]}" with:
    1. Engaging introduction with hook
    2. 3-5 subheadings with detailed content
    3. Practical examples/statistics
    4. Clear transitions between sections
    5. Actionable conclusion
    Style: Professional yet conversational (Flesch-Kincaid 60-70). Use markdown formatting
    """
    
    response = llm.invoke(prompt)
    return {"blog_content": response.content}

def review_content(state: BlogState):
    prompt = f"""Critically review this blog content (scale 1-10):
    - Clarity & Structure
    - Grammar & Style
    - SEO optimization
    - Reader engagement
    Provide specific improvement suggestions. Content:\n{state["blog_content"]}"""
    
    feedback = llm.invoke(prompt)
    
    return {"reviewed_content": feedback.content}

def evaluate_content(state: BlogState):

    # Create evaluation prompt
    prompt = f"""Evaluate blog content against editorial feedback (Pass/Fail):
    
    Original Blog Content:
    {state["blog_content"]}

    Editor Feedback:
    {state["reviewed_content"]}

    Evaluation Criteria:
    1. Coherent structure (logical flow between sections)
    2. Grammar/mechanics (no errors in spelling/punctuation)
    3. Brief compliance (matches topic: {state["topic"]})
    4. Engagement factors (storytelling, examples, hooks)
    5. Formatting (proper headers, lists, whitespace)
    6. Feedback addressed (revisions based on editor comments)

    Final Verdict (Pass = meets all criteria, Fail = any major issues):
    Answer only Pass or Fail:"""

    # Get LLM response
    response = llm.invoke(prompt)
    verdict = response.content.strip().upper()
    
    # Update state
    state["is_blog_ready"] = "Pass" if "PASS" in verdict else "Fail"
    state["reviewed_content"].append(AIMessage(
        content=f"Final Verdict: {state['is_blog_ready']}\n{response.content}"
    ))
    
    return state

def route_based_on_verdict(state: BlogState):
    """Conditional routing function"""
    return "Pass" if state["is_blog_ready"] == "Pass" else "Fail"




builder = StateGraph(BlogState)


builder.add_node("title_generator", generate_title)
builder.add_node("content_generator", generate_content)
builder.add_node("content_reviewer", review_content)
builder.add_node("quality_check", evaluate_content)  # New evaluation node

builder.add_edge(START, "title_generator")
builder.add_edge("title_generator", "content_generator")
builder.add_edge("content_generator", "content_reviewer")
builder.add_edge("content_reviewer", "quality_check")

# Add conditional edge after quality check
builder.add_conditional_edges(
    "quality_check",
    route_based_on_verdict,
    {"Pass": END, "Fail": "content_generator"}
)

graph = builder.compile()

graph.invoke({"topic": "Langgrah Orchestrator"})

display(Image(graph.get_graph().draw_mermaid_png()))