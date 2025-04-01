import asyncio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
import os

async def main():
    # Make sure we pass the GROQ_API_KEY to the server
    if "GROQ_API_KEY" not in os.environ:
        api_key = input("Please enter your Groq API key: ")
        os.environ["GROQ_API_KEY"] = api_key
    
    # Configure MCP server connection
    server_params = StdioServerParameters(
        command="python",
        args=["groq_mcp_server.py"],
        env={"GROQ_API_KEY": os.environ["GROQ_API_KEY"]}
    )
    
    print("Starting chatbot with MCP and Groq...\n")
    
    # Connect to the MCP server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            
            # List available tools and resources
            tools = await session.list_tools()
            resources = await session.list_resources()
            
            print("=== Available Tools ===")
            for tool in tools:
                print(f"- {tool.name}: {tool.description}")
            
            print("\n=== Available Resources ===")
            for resource in resources:
                print(f"- {resource.pattern}: {resource.description}")
            
            print("\n=== Chat with the Bot (type 'exit' to quit) ===")
            print("Type '/help' to see available commands\n")
            
            # Simple chat loop
            while True:
                user_input = input("\nYou: ")
                if user_input.lower() == "exit":
                    break
                
                # Process commands and chat messages
                if user_input.startswith("/"):
                    await process_command(user_input, session)
                else:
                    # Use the chat_with_context tool for regular messages
                    print("Bot is thinking...")
                    result = await session.call_tool("chat_with_context", {"user_message": user_input})
                    print(f"\nBot: {result}")

async def process_command(command: str, session: ClientSession):
    """Process special commands"""
    if command == "/help":
        print("\nAvailable commands:")
        print("  /help - Show this help message")
        print("  /time - Get current time")
        print("  /user - Get user information")
        print("  /direct [question] - Get a direct answer from the knowledge base")
        print("  /generate [prompt] - Generate text with Groq LLM")
        print("  /exit - Exit the chat")
    
    elif command == "/time":
        content, _ = await session.read_resource("info://time")
        print(f"\nBot: {content}")
    
    elif command == "/user":
        content, _ = await session.read_resource("info://user")
        print(f"\nBot: {content}")
    
    elif command.startswith("/direct "):
        _, question = command.split(" ", 1)
        result = await session.call_tool("answer_question", {"question": question})
        print(f"\nBot: {result}")
    
    elif command.startswith("/generate "):
        _, prompt = command.split(" ", 1)
        print("Generating with Groq...")
        result = await session.call_tool("generate_with_groq", {"prompt": prompt})
        print(f"\nBot: {result}")
    
    else:
        print("\nBot: Unknown command. Type /help for available commands.")

if __name__ == "__main__":
    asyncio.run(main())
