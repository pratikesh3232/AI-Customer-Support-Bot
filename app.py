# app.py
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

st.set_page_config(page_title="Support Chat", page_icon="💬")
st.title("💬 Customer Support")
st.caption("Ask me anything about our products, orders, or policies.")

# ── Load FAISS KB (cached so it loads once) ──
@st.cache_resource
def load_kb():
    embeddings = OpenAIEmbeddings()
    vs = FAISS.load_local("faiss_kb", embeddings, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_kwargs={"k": 4})

retriever = load_kb()

# ── Session state: chat history + display messages ──
def get_chat_history():
    return st.session_state.setdefault("chat_history", [])


def save_chat_history(question, answer):
    history = get_chat_history()
    history.extend([
        HumanMessage(content=question),
        AIMessage(content=answer),
    ])


st.session_state.setdefault("messages", [])

# ── LCEL chain with memory ──
def build_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful customer support assistant.
Answer ONLY using the context below.
If the answer isn't in the context, say "I don't have that information — please contact our support team."
Be polite and concise. Mention the source document when possible.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(
            f"[{d.metadata.get('source', 'doc')}]\n{d.page_content}"
            for d in docs
        )

    chain = (
        {
            "context": RunnableLambda(lambda x: retriever.invoke(x["question"])) | RunnableLambda(format_docs),
            "question": RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnableLambda(lambda _: get_chat_history()),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

chain = build_chain()

# ── Render previous messages ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Handle new input ──
if question := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = chain.invoke({"question": question})
        st.markdown(answer)

    # save to chat history
    save_chat_history(question, answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})