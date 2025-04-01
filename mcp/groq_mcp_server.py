from mcp.server.fastmcp import FastMCP, Context
import os
import httpx
import json
from datetime import datetime

# Create an MCP server with a descriptive name
mcp = FastMCP("GroqChatbot")

# Add a tool to generate responses using Groq's API
@mcp.tool()
async def generate_with_groq(prompt: str, model: str = "llama3-70b-8192") -> str:
    """Generate a response using Groq's LLM API"""
    # Get API key from environment variable
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY environment variable not set."
    
    # Call Groq API
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30.0  # 30-second timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"Error calling Groq API: {response.status_code} - {response.text}"

# Add a more complex assistant tool that can access resources for context
@mcp.tool()
async def chat_with_context(user_message: str, ctx: Context) -> str:
    """Generate a response that incorporates available context"""
    # Get time information as context
    current_time, _ = await ctx.read_resource("info://time")
    
    # Create a prompt that includes context
    prompt = f"""
Current time: {current_time}
User: {user_message}

As an AI assistant, please respond to the user's message. Be helpful, concise, and friendly.
"""
    
    # Call the Groq generation tool
    return await generate_with_groq(prompt)

# Add a helper tool for answering questions
@mcp.tool()
def answer_question(question: str) -> str:
    """Answer a specific question"""
    knowledge_base = {
        "who are you": "I'm a simple chatbot built with MCP and powered by Groq's LLMs.",
        "what is mcp": "MCP (Model Context Protocol) is a standardized protocol that allows applications to provide context to language models in a structured way.",
        "what is groq": "Groq is a company offering high-performance LLM inference with extremely low latency.",
        "how does this work": "I use the Model Context Protocol (MCP) to structure my capabilities and Groq's API to generate responses."
    }
    
    # Check if we have a direct answer in our knowledge base
    for key, value in knowledge_base.items():
        if key in question.lower():
            return value
    
    return "I don't have a specific answer to that question in my knowledge base."

# Add some resources that provide context
@mcp.resource("info://time")
def get_current_time() -> str:
    """Get the current time information"""
    current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    return f"Current time (UTC): {current_time}"

@mcp.resource("info://user")
def get_user_info() -> str:
    """Get information about the current user"""
    return f"User: {os.environ.get('USER', 'Unknown')}"

# Run the server when executed directly
if __name__ == "__main__":
    mcp.run()
