# eduai_app.py
import streamlit as st
import pandas as pd
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import smtplib
from email.mime.text import MIMEText
import re
import chromadb

# LangChain / embeddings / llm
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ----------------------------
# CONFIG / SECRETS
# ----------------------------
st.set_page_config(page_title="EduAI", page_icon="🎓", layout="wide")

# MongoDB (from secrets.toml)
MONGO_URI = st.secrets["mongo"]["connection_string"]
DB_NAME = st.secrets["mongo"]["database"]
USERS_COLL = st.secrets["mongo"]["collection"]
ISSUES_DB = st.secrets["mongo"]["issues_db"]
ISSUES_COLL = st.secrets["mongo"]["issues_collection"]

# Email / SMTP
MAIL_USERNAME = st.secrets["email"]["username"]
MAIL_PASSWORD = st.secrets["email"]["password"]
MAIL_SENDER = st.secrets["email"]["sender"]

# Chroma Cloud
CHROMA_API_KEY = st.secrets["chroma"]["api_key"]
CHROMA_TENANT = st.secrets["chroma"]["tenant_id"]
CHROMA_DATABASE = st.secrets["chroma"]["database_id"]
COLLECTION_NAME = st.secrets["chroma"]["collection_id"]

# OpenRouter (LLM)
OPENROUTER_API_KEY = st.secrets["openai"]["api_key"]
OPENROUTER_BASE_URL = st.secrets["openai"]["base_url"]

# Streamlit App secret
SECRET_KEY = st.secrets["app"]["secret_key"]

# Embeddings model (local)
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"

# System prompt for RAG
SYSTEM_PROMPT = """You are a helpful RAG assistant using data from the LearnPal database.
Use retrieved context to answer questions clearly and concisely.
If information is missing, say so and suggest what to ask next.
Cite sources as (source: <filename>:<page>) when possible.
"""

# ----------------------------
# DB & Auth Helpers
# ----------------------------
@st.cache_resource
def get_client():
    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=50000)

def get_db():
    return get_client()[DB_NAME]

def users_collection():
    return get_db()[USERS_COLL]

def issues_collection():
    return get_client()[ISSUES_DB][ISSUES_COLL]

def register_user(username, password):
    if users_collection().find_one({"username": username}):
        return "Username already exists"
    hashed = generate_password_hash(password, method="pbkdf2:sha256")
    users_collection().insert_one({"username": username, "password": hashed})
    return None

def login_user(username, password):
    user = users_collection().find_one({"username": username})
    return user and check_password_hash(user["password"], password)

# ----------------------------
# Mail Helper
# ----------------------------
def send_mail(to_email, subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = MAIL_SENDER
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(MAIL_USERNAME, MAIL_PASSWORD)
        server.sendmail(MAIL_SENDER, [to_email], msg.as_string())

# ----------------------------
# RAG (Chroma Cloud + LangChain) Helpers
# ----------------------------
@st.cache_resource
def get_chroma_client():
    return chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
    )

@st.cache_resource
def get_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

@st.cache_resource
def get_vectorstore_and_retriever():
    client = get_chroma_client()
    vectorstore = Chroma(client=client, collection_name=COLLECTION_NAME, embedding_function=get_embeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return vectorstore, retriever

@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("messages"),
    ("system", "Retrieved context:\n{context}"),
])

def build_context_from_chroma(query: str) -> str:
    collection = get_collection()
    try:
        result = collection.query(
            query_texts=[query],
            n_results=5,
            include=["metadatas", "documents"],
        )
    except Exception as e:
        st.warning(f"[RAG] chroma query failed: {e}")
        return ""

    parts = []
    docs = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    for i, doc in enumerate(docs):
        meta = metadatas[i] if i < len(metadatas) else {}
        src = meta.get("source", "unknown")
        page = meta.get("page", "NA")
        snippet = str(doc).replace("\n", " ").strip()[:1000]
        parts.append(f"[source: {src}:{page}] {snippet}")

    ctx = "\n\n".join(parts)
    return ctx[:3500] + " ...[truncated]" if len(ctx) > 3500 else ctx

def generate_answer_with_rag(messages):
    llm = get_llm()
    last_user = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
    context = build_context_from_chroma(last_user or "")
    filled_prompt = prompt.invoke({"messages": messages, "context": context})
    try:
        resp = llm.invoke(filled_prompt)
    except Exception as e:
        return f"Error calling LLM: {e}"
    return getattr(resp, "content", str(resp))

# ----------------------------
# App State
# ----------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "lc_messages" not in st.session_state:
    st.session_state.lc_messages = [SystemMessage(content="Conversation started.")]

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("📌 Navigation")
if st.session_state.authenticated:
    page = st.sidebar.radio("Go to", ["Home", "Chatbot", "FAQs", "Submit Issue", "Logout"])
else:
    page = "Login"

# ----------------------------
# Pages
# ----------------------------
if page == "Login":
    st.title("🎓 EduAI")
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab_register:
        new_user = st.text_input("New Username", key="reg_user")
        new_pass = st.text_input("New Password", type="password", key="reg_pass")
        if st.button("Register"):
            err = register_user(new_user, new_pass)
            if err:
                st.error(err)
            else:
                st.success("Registration successful")

elif page == "Home":
    st.title("🏠 Home - EduAI")
    st.write(f"Welcome, **{st.session_state.username}**! 👋")
    st.write("Use the Chatbot tab for RAG-powered answers from your LearnPal collection.")

elif page == "Chatbot":
    st.title("🤖 EduAI — RAG Chat")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask your question...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.lc_messages.append(HumanMessage(content=user_input))

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = generate_answer_with_rag(st.session_state.lc_messages)
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.lc_messages.append(AIMessage(content=answer))

elif page == "FAQs":
    st.title("❓ FAQs - EduAI")
    st.markdown("Common queries and REAP-related FAQs here...")

elif page == "Submit Issue":
    st.title("🛠️ Submit an Issue")
    name = st.text_input("Name")
    email = st.text_input("Email")
    issue = st.text_area("Describe your issue")

    if st.button("Submit"):
        if not (name and email and issue):
            st.error("Please fill all fields.")
        else:
            issues_collection().insert_one({"name": name, "email": email, "issue": issue})
            try:
                send_mail(email, "Issue Submitted", f"Hello {name}, we received your issue:\n\n{issue}")
                st.success("Issue submitted successfully.")
            except Exception as e:
                st.warning(f"Issue saved, but email failed: {e}")

elif page == "Logout":
    st.title("🚪 Logout")
    if st.button("Confirm Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.success("You have been logged out")
        st.rerun()
