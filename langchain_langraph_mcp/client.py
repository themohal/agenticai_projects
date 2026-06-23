from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

import asyncio

async def main():
    client=MultiServerMCPClient(
        {
            "math":{
                "command":"python",
                "args":["mathserver.py"], ## Ensure correct absolute path
                "transport":"stdio",
            
            },
            "weather": {
                "url": "http://localhost:8000/mcp",  # Ensure server is running here
                "transport": "streamable_http",
            }

        }
    )

    import os
    os.environ["GEMINI_API_KEY"]=os.getenv("GEMINI_API_KEY")

    tools=await client.get_tools()
    model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    agent=create_agent(model, tools)

    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    )

    print("Math response:", math_response['messages'][-1].content)

    # Print the last few messages to see the tool interaction
    for msg in math_response['messages'][-3:]:
        print(f"[{type(msg).__name__}]: {msg.content}\n")

    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is the weather in California?"}]}
    )
    print("Weather response:", weather_response['messages'][-1].content)
    # Print the last few messages to see the tool interaction
    for msg in weather_response['messages'][-3:]:
        print(f"[{type(msg).__name__}]: {msg.content}\n")

    

asyncio.run(main())