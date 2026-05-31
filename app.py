import streamlit as st
from groq import Groq
from duckduckgo_search import DDGS
import wikipediaapi
import arxiv as arxiv_client

st.set_page_config(page_title="AI Search Agent", page_icon="🔍")
st.title("AI Search Agent")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")

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

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I can search the web, Arxiv, and Wikipedia. What do you want to know?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    if not api_key:
        st.warning(" Please enter your Groq API key in the sidebar.")
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            web_results = duckduckgo_search(user_input)
            wiki_results = wikipedia_search(user_input)

            context = f"Web Search Results:\n{web_results}\n\nWikipedia:\n{wiki_results}"

            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Use the provided search results to answer the question accurately and concisely."},
                    {"role": "user", "content": f"Question: {user_input}\n\nSearch Results:\n{context}"}
                ]
            )
            output = response.choices[0].message.content

        st.write(output)
        st.session_state.messages.append({"role": "assistant", "content": output})
