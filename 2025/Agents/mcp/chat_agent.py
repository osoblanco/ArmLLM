# app.py
import asyncio, os, streamlit as st
import os

from mcp_use import MCPClient, MCPAgent
from langchain_openai import ChatOpenAI


@st.cache_resource  # one client/LLM per session
def build_agent():
    # ----- 1. connect to your servers -------------
    client = MCPClient.from_config_file("mcp_servers.json")
    # ----- 2. initialize the LLM (agent policy) -----
    llm = ChatOpenAI(
        # Uncomment the following lines to use your own vLLM server
        # model_name="Qwen/Qwen3-32B-FP8",
        # base_url="http://YOUR_IP:8000/v1",
        model_name="gpt-4.1",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    # ----- 3. initialize an agent that can use the MCP client ----------
    return MCPAgent(
        llm=llm,
        client=client,
        verbose=True,
        max_steps=30,
        system_prompt="""
You are a highly personalized assistant. You have access to a memory. Use it as much as possible. 
When you assist the user, you must log information about all entities that you encouter.
For example, if the user mentions they have a relative, log their name and info. Same if they mention a pet, 
a hobby, an interest, a preference, etc. It is crucial that you don't just assume you will remember and that
you actually use the memory tools. If you need information, check your memory before disturbing the user.
There is a good chance you already have the info. Disturbing the user with clarifying questions is a completely
last resort. If some memory queries return empty results, retry using another query at least once.
It's very important that you use the memory as much as possible.
Use a lot of emojis in your responses to make it more fun and engaging.
""",
    )


async def _run(prompt):
    return await agent.run(prompt)


agent = build_agent()

st.title("🤖 ArmLLM Chatbot")
st.markdown("Ask the agent to do things for you!")

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.history.append(
        (
            "assistant",
            asyncio.run(
                _run(
                    """
Welcome the user with a friendly message.
Search your memory to find the user's name (usually the "user" entity). If you find it, greet them by name.
It's also fine to ask them a friendly question about something you know about them.
If you don't find it, also ask them for their name.
Don't mention your memory because that's not very natural.
"""
                )
            ),
        )
    )

# render chat so far
for role, msg in st.session_state.history:
    st.chat_message(role).markdown(msg)

# user input
if prompt := st.chat_input("Ask anything…"):
    st.session_state.history.append(("user", prompt))
    with st.spinner("Running agent..."):
        answer = asyncio.run(_run(prompt))
    st.session_state.history.append(("assistant", answer))
    st.rerun()
