import os
from typing import Literal, TypedDict, Annotated
from operator import add

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from pprint import pprint
import re
import json
from .scraper_tool import web_table_summarizer_tool
from .executor_tool import python_repl_tool
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=GOOGLE_API_KEY)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")

# # Initialize the DeepSeek model via OpenRouter
# llm = ChatOpenAI(
#     model="qwen/qwen3-coder:free",
#     base_url="https://openrouter.ai/api/v1",
#     api_key=OPENROUTER_API_KEY,
#     temperature=0.1,
#     max_tokens=4000,
# )

# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     base_url="https://aiproxy.sanand.workers.dev/openai/v1/",
#     api_key="eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjMwMDEzNThAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.iZ1JwaudkarInqmBYQBhkxHJjGgsslp8ZBu7_ppvXS4",

# )

def extract_json_from_string(input_string):
    """
    Extracts JSON from a string that may contain markdown code blocks or other formatting.
    
    Args:
        input_string (str): Input string containing JSON data
        
    Returns:
        dict/list: Parsed JSON object, or None if no valid JSON found
        
    Raises:
        ValueError: If JSON is found but cannot be parsed
    """
    
    # Handle list input (take the relevant part)
    if isinstance(input_string, list):
        # Look for the element that contains JSON
        for item in input_string:
            if isinstance(item, str) and ('```json' in item or '{' in item or '[' in item):
                input_string = item
                break
        else:
            # If no JSON-like string found in list, join all elements
            input_string = ' '.join(str(item) for item in input_string)
    
    # Pattern 1: Extract from markdown code blocks (```json ... ```)
    json_block_pattern = r'```json\s*\n(.*?)\n```'
    match = re.search(json_block_pattern, input_string, re.DOTALL)
    
    if match:
        json_str = match.group(1).strip()
    else:
        # Pattern 2: Look for JSON-like structures (starting with [ or {)
        # Find the first occurrence of [ or { and try to extract valid JSON
        json_start = -1
        for i, char in enumerate(input_string):
            if char in '[{':
                json_start = i
                break
        
        if json_start == -1:
            return None
            
        # Try to find the matching closing bracket/brace
        json_str = input_string[json_start:]
        
        # Try to parse progressively smaller substrings until we find valid JSON
        for end_pos in range(len(json_str), 0, -1):
            candidate = json_str[:end_pos].strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        
        return None
    
    # Try to parse the extracted JSON string
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Found JSON block but failed to parse: {e}")


class AnalysisState(TypedDict):
    query: str
    context: str
    final_output: str
    messages: Annotated[list[BaseMessage], add]


def create_system_prompt(content: str) -> SystemMessage:
    """Helper to create system message"""
    return SystemMessage(content=content)


# Context agent
context_agent = create_react_agent(
    llm,
    tools=[web_table_summarizer_tool, python_repl_tool],
    prompt=
        """You are a data context agent. Analyze the user's query and provide a brief overview of the data.

If the query mentions a website URL:
1. Use web_table_summarizer_tool to get the overview of the tables on the page
2. pick the relevant table and provide it's name, coulmn names and sample rows.

If the query mentions local data or files:
1. Use python_tool to explore the data structure
2. Show sample data and key columns

Keep your response under 150 words and end with "CONTEXT_COMPLETE"."""
    
)

# Analysis agent  
analysis_agent = create_react_agent(
    llm,
    tools=[python_repl_tool],
    prompt=
        """You are a data analysis agent. You have the user's query and context about the data.

Your task:
1. Write Python code to fetch and analyze the data. Do not directly use sample rows from the CONTEXT.
2. Use BeautifulSoup for web scraping (don't use pd.read_html)
3. Convert data to pandas DataFrame for analysis
4. Answer the user's specific question
5. Use print() statements to show results
6. Rewrite the correct code and re-execute it if errors occur


Execute your code using python_repl_tool and respond with final json result of the execution only which will comply with json.loads function. """
    
)


def context_node(state: AnalysisState) -> Command[Literal["analysis", END]]:
    """Get data context"""
    if state.get("context"):
        return Command(goto="analysis")
    
    try:
        messages = [HumanMessage(content=state["query"])]
        result = context_agent.invoke({"messages": messages})
        
        # Extract context from all messages, focusing on AI responses
        context_parts = []
        for msg in result["messages"]:
            if hasattr(msg, 'content') and msg.content:
                content = str(msg.content)
                # Skip if it's just a tool call or empty
                if content and len(content.strip()) > 10 and not content.startswith('{"'):
                    context_parts.append(content)
        
        # Join all context parts
        context = "\n\n".join(context_parts)
        
        # More lenient check - just need some meaningful content
        if not context or len(context.strip()) < 20:
            # Debug: print what we got
            print("DEBUG - Result messages:")
            for i, msg in enumerate(result["messages"]):
                print(f"Message {i}: {type(msg).__name__} - {str(msg.content)[:200]}")
            
            return Command(
                update={"final_output": f"Failed to get meaningful context. Got: {context[:200]}"},
                goto=END
            )
        
        print(f"DEBUG - Context extracted successfully: {len(context)} characters")
        
        return Command(
            update={"context": context},
            goto="analysis"
        )
        
    except Exception as e:
        return Command(
            update={"final_output": f"Context error: {str(e)}"},
            goto=END
        )


def analysis_node(state: AnalysisState) -> Command[Literal[END]]:
    """Perform data analysis"""
    if not state.get("context"):
        return Command(
            update={"final_output": "No context available for analysis"},
            goto=END
        )
    
    try:
        analysis_prompt = f"""
USER QUERY: {state['query']}

CONTEXT: {state['context']}

Please analyze the data to answer the user's query. Write complete Python code to:
1. Scrape the web page and extract the relevant table if data is available in webpage else go on with user's guidance.
2. Process the data to answer all the questions
3. Generate any requested visualizations
4. Format the final answer as requested
"""
        
        messages = [HumanMessage(content=analysis_prompt)]
        result = analysis_agent.invoke({"messages": messages})
        
        # Extract final output from all AI messages
        # output_parts = []
        # for msg in result["messages"]:
        #     if hasattr(msg, 'content') and msg.content:
        #         content = str(msg.content)
        #         # Skip tool calls and very short responses
        #         if content and len(content.strip()) > 10 and not content.startswith('{"'):
        #             output_parts.append(content)
        
        # output = "\n\n".join(output_parts) if output_parts else "Analysis completed but no output generated"
        output = result["messages"][-1].content if result["messages"] else "Analysis completed but no output generated"
        output = extract_json_from_string(output)

        
        # Debug output
        print(f"DEBUG - Analysis agent returned {len(result['messages'])} messages")
        for i, msg in enumerate(result["messages"]):
            print(f"Message {i}: {type(msg).__name__} - {str(msg.content)[:100]}...")
        
        return Command(
            update={"final_output": output},
            goto=END
        )
        
    except Exception as e:
        return Command(
            update={"final_output": f"Analysis error: {str(e)}"},
            goto=END
        )


# Build the graph
def create_analysis_graph():
    """Create the analysis workflow graph"""
    workflow = StateGraph(AnalysisState)
    
    # Add nodes
    workflow.add_node("context", context_node)
    workflow.add_node("analysis", analysis_node)
    
    # Add edges
    workflow.add_edge(START, "context")
    
    return workflow.compile()


# Main execution function
def run_analysis(query: str) -> str:
    """
    Run the data analysis workflow
    
    Args:
        query: User's data analysis query
        
    Returns:
        Final analysis result
    """
    graph = create_analysis_graph()
    
    initial_state = AnalysisState(
        query=query,
        context="",
        final_output="",
        messages=[]
    )
    
    try:
        final_state = graph.invoke(initial_state)
        return final_state
    except Exception as e:
        return f"Workflow error: {str(e)}"