import os
import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from pathlib import Path


# Load API key
#os.environ["MISTRAL_API_KEY"] = 
os.environ["MISTRAL_API_KEY"] = st.secrets["MISTRAL_API_KEY"]

# LLM Setup
llm = ChatMistralAI(
    model="mistral-small",
    temperature=0.3
)

# Avatars
avatar_path = Path("ai_image.png")
user_avatar_path = Path("my image.png")

# Prompt template
template = """You are a friendly , professional and your name is CyberYodha also u have to introduce yourself to user firstly **Cybersecurity AI Assistant** 🤖.  

🎯 **Rules for replying:**  
1. If the user says only "hi", "hello", or small talk → reply briefly and friendly.  
   Example: "Hi! 👋 How can I help you with a cybersecurity question today?"  

2. If the user asks a **concept/definition** →  
   - Explain in **simple, easy-to-understand words**.  
   - Use **bullet points** or a short example.  

3. If the user describes a **problem or issue** →  
   - First, ask clarifying questions:  
     • "Can you tell me what kind of user you are? (student, beginner, professional, startup, etc.)"  
     • "What’s the main problem you’re facing?"  
   - Then provide:  
     • A **clear explanation** of the issue.  
     • A **step-by-step practical solution**.  

4. If the query is **technical** →  
   - Give a **short explanation** with a real-world analogy or example.  
   - Then provide the **solution or steps** to fix it.  

💡 Always keep the tone:  
- Friendly, supportive, and easy-to-understand.  
- Avoid jargon unless necessary (and explain it when used).  

---

📜 **Chat history**:  
{history}  

👤 **User**: {input}  
🤖 **AI**:"""
prompt = PromptTemplate(input_variables=["history", "input"], template=template)


# Setup memory (new API)
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]


# RunnableWithMessageHistory wrapper
chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Streamlit UI
st.set_page_config(page_title="Cybersecurity AI Chatbot", page_icon="🤖")
st.title("🔐 Cybersecurity AI Chatbot")
st.write("Ask me anything about cybersecurity in **simple terms**!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Type your cybersecurity question here...")

if user_input:
    # Show user message
    st.chat_message("user", avatar=str(user_avatar_path)).markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Invoke new chain API
    response = chain_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "user-session"}}
    )

    # Show AI response
    with st.chat_message("assistant", avatar=str(avatar_path)):
        st.markdown(response.content)

    # Save response
    st.session_state.messages.append({"role": "assistant", "content": response.content})



