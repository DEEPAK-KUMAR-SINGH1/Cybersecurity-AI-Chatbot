import os
import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from pathlib import Path


# Set API Key (from environment variable)
load_dotenv()

# LLM Setup
llm = ChatMistralAI(
    model="mistral-small",
    temperature=0.3
)

avatar_path = Path("ai_image.png")
user_avatar_path = Path("my image.png")

#  Memory Setup
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)


# set tamplate
template ="""You are a friendly , professional and your name is CyberYodha also u have to introduce yourself to user firstly **Cybersecurity AI Assistant** 🤖.  

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
🤖 **AI**:  
"""
# set prompt

prompt = PromptTemplate(input_variables=["history", "input"], template=template)


# 4. Conversation Chain

conversation = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    prompt=prompt,
    verbose=False
)

# 5. Streamlit UI

st.set_page_config(page_title="Cybersecurity AI Chatbot", page_icon="🤖")

st.title("🔐 Cybersecurity AI Chatbot")
st.write("Ask me anything about cybersecurity in **simple terms**!")

# Chat input
user_input = st.chat_input("Type your cybersecurity question here...")


# Display chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# When user enters a message
if user_input:
    # Show user message
    st.chat_message("user",avatar="my image.png").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get AI response
    response = conversation.run(user_input)

    # Show AI response
    with st.chat_message("assistant",avatar=avatar_path):
        st.markdown(response)


    # Save response
    st.session_state.messages.append({"role": "assistant", "content": response})
