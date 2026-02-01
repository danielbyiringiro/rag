import streamlit as st
from decouple import config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="ARAP",
    page_icon="🤖",
    layout="wide"
)

st.title("How can I help you today ?")

# Initialize chat messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there, I am your Ashesi Assistant"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_prompt = st.chat_input()

def build_history_text(messages, turns: int = 4) -> str:
    """
    Build a text chat history of the last `turns` (user+assistant pairs).
    turns=4 => keep ~8 messages at most (excluding any system-like first message).
    """
    # Keep only user/assistant messages
    convo = [m for m in messages if m["role"] in ("user", "assistant")]

    # Keep last 2*turns messages
    window = convo[-2 * turns :]

    lines = []
    for m in window:
        role = "Human" if m["role"] == "user" else "AI"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)

prompt = ChatPromptTemplate.from_template(
    """You are a very kind and friendly AI assistant.
You are currently having a conversation with a human.
Answer in a kind and friendly tone with a sense of professionalism.

Chat history:
{chat_history}

Human: {question}
AI:"""
)

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=config("OPENAI_API_KEY"),
)

# LCEL chain (modern replacement for LLMChain)
chain = prompt | llm

if user_prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    # Build last-k history
    chat_history = build_history_text(st.session_state.messages, turns=4)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            ai_msg = chain.invoke({"chat_history": chat_history, "question": user_prompt})
            ai_response = ai_msg.content
            st.write(ai_response)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
