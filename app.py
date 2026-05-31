import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import Tool
import wikipediaapi
import arxiv as arxiv_client
from duckduckgo_search import DDGS
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="LangChain Search Agent", page_icon="🔍")
st.title("LangChain - Chat with Search")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")

# ──────────────────────────────────────────
# ONLY THIS SECTION CHANGED — tool definitions
# ──────────────────────────────────────────

def duckduckgo_search(query):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    if not results:
        return "No results found."
    return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])

def arxiv_search(query):
    client = arxiv_client.Client()
    results = list(client.results(arxiv_client.Search(query=query, max_results=2)))
    if not results:
        return "No papers found."
    return "\n\n".join([f"Title: {r.title}\nSummary: {r.summary[:300]}" for r in results])

def wikipedia_search(query):
    wiki = wikipediaapi.Wikipedia(language="en", user_agent="LangchainBot/1.0")
    page = wiki.page(query)
    return page.summary[:500] if page.exists() else f"No page found for '{query}'."

search = Tool(name="Search",    func=duckduckgo_search, description="Search the web for current information.")
arxiv  = Tool(name="Arxiv",     func=arxiv_search,      description="Search arxiv for scientific papers.")
wiki   = Tool(name="Wikipedia", func=wikipedia_search,  description="Search Wikipedia for general knowledge.")

# ──────────────────────────────────────────
# EVERYTHING BELOW IS EXACTLY YOUR OLD CODE
# ──────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm a chatbot that can search the web, Arxiv, and Wikipedia."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar.")
        st.stop()

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        streaming=True
    ) 

    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        max_iterations=5,
        verbose=True,
        early_stopping_method="generate",
        agent_kwargs={
            "prefix": """You are a helpful assistant. You have access to the following tools.
Always use a tool to find information before answering.
After getting the tool result, provide a clean final answer to the user.
Do NOT repeat the Thought/Action/Observation steps in your final answer."""
        }
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            response = search_agent.run(prompt, callbacks=[st_cb])
        except Exception as e:
            response = f"Sorry, ran into an error: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)