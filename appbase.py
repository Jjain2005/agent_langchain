import streamlit as st

from langchain_groq import ChatGroq

from langchain_community.tools import DuckDuckGoSearchRun

from langchain.agents import initialize_agent
from langchain.agents import AgentType

# ---------------- PAGE ---------------- #

st.title("AI Search Chatbot")

# ---------------- API KEY ---------------- #

api_key = st.sidebar.text_input(
    "Enter Groq API Key",
    type="password"
)

# ---------------- MEMORY ---------------- #

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Ask me anything."
        }
    ]

# ---------------- DISPLAY OLD MESSAGES ---------------- #

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------- USER INPUT ---------------- #

if prompt := st.chat_input("Ask something..."):

    # show user message
    with st.chat_message("user"):
        st.write(prompt)

    # save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # ---------------- LLM ---------------- #

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    # ---------------- TOOLS ---------------- #

    search = DuckDuckGoSearchRun()

    tools = [search]

    # ---------------- AGENT ---------------- #

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    # ---------------- RESPONSE ---------------- #

    with st.chat_message("assistant"):

        response = agent.run(prompt)

        st.write(response)

    # save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })