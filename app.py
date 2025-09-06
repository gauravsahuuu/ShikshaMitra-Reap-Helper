import streamlit as st
import pandas as pd
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import smtplib
from email.mime.text import MIMEText
import google.generativeai as genai
import re

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="ShikshaMitra", page_icon="🎓", layout="wide")

# Load secrets from .streamlit/secrets.toml
MONGO_URI = st.secrets["MONGO_URI"]
DB_NAME = st.secrets.get("DB_NAME", "SIH")
USERS_COLL = st.secrets.get("USERS_COLL", "users")
ISSUES_DB = st.secrets.get("ISSUES_DB", "issue_tracker")
ISSUES_COLL = st.secrets.get("ISSUES_COLL", "issues")
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

MAIL_USERNAME = st.secrets["MAIL_USERNAME"]
MAIL_PASSWORD = st.secrets["MAIL_PASSWORD"]
MAIL_SENDER = st.secrets.get("MAIL_SENDER", MAIL_USERNAME)

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
# Gemini Helper
# ----------------------------
PROMPT_PREFIX = (
    "If the user says 'hey' or 'hello', respond that you are ShikshaMitra "
    "(REAP admission counselling helper). "
    "I am applying for REAP (Rajasthan's engineering colleges' admission) counselling in Rajasthan. "
    "Give a brief result. Generate data only on the basis of REAP counselling which you have or is available on the internet. "
    "I don't want real-time data, I just want a rough idea. Even if you have no idea or it may vary, "
    "if you have zero idea, you can say that you don't know about this and ask the user to raise a ticket."
)

@st.cache_resource
def get_model():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-1.5-flash")

def format_text(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", lambda m: f"<b>{m.group(1)}</b>", text)
    text = text.replace("*", "•")
    text = text.replace("\n", "<br>")
    return text

def ask_gemini(user_message: str) -> str:
    model = get_model()
    resp = model.generate_content(PROMPT_PREFIX + user_message)
    txt = str(getattr(resp, "text", "") or "")
    return format_text(txt)

# ----------------------------
# App State
# ----------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
if "chat_history" not in st.session_state:
    # Use Streamlit chat schema: role in {"user","assistant"}, content is HTML-ready
    st.session_state.chat_history = []

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("📌 Navigation")
if st.session_state.authenticated:
    page = st.sidebar.radio(
        "Go to",
        ["Home", "College Predictor", "Chatbot", "FAQs", "Submit Issue", "Logout"],
    )
else:
    page = "Login"

# ----------------------------
# Pages
# ----------------------------
if page == "Login":
    st.title("🎓 ShikshaMitra")
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        st.subheader("Login")
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
        st.subheader("Register (Admin only)")
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        if st.button("Register"):
            err = register_user(new_user, new_pass)
            if err:
                st.error(err)
            else:
                st.success("Registration successful")

elif page == "Home":
    st.title("🏠 Home - ShikshaMitra")
    st.write(f"Welcome, **{st.session_state.username}**! 👋")
    st.write("Choose an option from the sidebar.")

elif page == "College Predictor":
    st.title("🎯 College Predictor (REAP)")

    @st.cache_data
    def load_data():
        return pd.read_csv("data/cutoffs_modified.csv")

    df = load_data()

    gender = st.selectbox("Gender", ["male", "female"])
    sfs_gas = st.selectbox(
        "SFS or GAS category", sorted(df["category"].dropna().astype(str).unique())
    )
    category = st.selectbox("Reservation Category", ["Gen", "EWS", "OBC", "SC", "ST"])
    input_rank = st.number_input("Your Rank", min_value=1, step=1)

    if st.button("Predict"):
        filtered_df = df[df["category"].astype(str).str.strip() == sfs_gas.strip()]
        category_column_map = {
            "Gen": "gen",
            "EWS": "mews" if gender == "male" else "fews",
            "OBC": "mobc" if gender == "male" else "fobc",
            "SC": "msc" if gender == "male" else "fsc",
            "ST": "mst" if gender == "male" else "fst",
        }
        category_column = category_column_map.get(category)

        if category_column not in filtered_df.columns:
            st.error("Category column not found")
        else:
            filtered_df = filtered_df.copy()
            filtered_df[category_column] = pd.to_numeric(
                filtered_df[category_column], errors="coerce"
            )
            result_df = filtered_df[
                (filtered_df[category_column] >= int(input_rank))
                | (filtered_df[category_column].isna())
            ][["Institute", "Branch", category_column]].rename(
                columns={category_column: "Cutoff"}
            )
            st.dataframe(result_df, use_container_width=True)

elif page == "Chatbot":
    st.title("🤖 ShikshaMitra Chatbot")

    # Display existing conversation with default Streamlit chat styling
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            # Assistant messages may include <b>, bullets, and <br>
            st.markdown(msg["content"], unsafe_allow_html=True)

    # Chat input stays docked at the bottom center, submits on Enter by default
    prompt = st.chat_input("Type your question about REAP...")
    if prompt:
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant reply
        reply_html = ask_gemini(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": reply_html})
        with st.chat_message("assistant"):
            st.markdown(reply_html, unsafe_allow_html=True)

elif page == "FAQs":
    st.title("❓ FAQs - ShikshaMitra")

    st.subheader("General")
    st.markdown("""
**Is there any helpline for REAP -2024?**  
Yes. The Helpline No.- 0141-2702344, 9462015808, 9462015080 (Call between 9 AM to 6 PM Only)

**I passed my 10th and 12th from Rajasthan still do I need domicile certificate?**  
Yes, you require domicile certificate of Rajasthan otherwise you will be considered as out of Rajasthan candidate.

**What is the validity of the Category Certificate (SC/ST/OBC/EWS-Gen.) and PwD certificate?**  
- OBC certificate should not be issued before **01/09/2023**. A grace period of **two years** is admissible with an undertaking.  
- Undertaking by OBC/MBC (non-creamy layer) if within grace period.  
- Certificates for PwD and Ex-Servicemen if applicable.
    
**What are the requirements for EWS candidates?**  
Income/eligibility & document requirements as per official guidelines.

**When will the Online Application form for REAP 2024-25 be available?**  
Registration starts **27/06/2024 onward**. Websites: `www.reapbtech24.com`, `www.reapbarch24.com`.

**Whether the application fee excludes the Common Service charge?**  
Fee is **₹590 (₹500 + 18% GST)** per counseling mode. Non-refundable and non-transferable.
    
**What is reservation criteria and seat matrix of REAP 2024-25?**  
See page **8** of the REAP booklet (web portal).

**What documents to carry at reporting?**  
See page **14** of the REAP booklet (web portal).
""")

    st.subheader("Data Correction")
    st.markdown("""
**How can I change the subject group I selected incorrectly?**  
Raise a ticket on the **candidate panel** and keep the ticket number.

**How can I change my options/choice if I filled them incorrectly?**  
Raise a ticket on the **candidate panel** and keep the ticket number.

**Procedure for Filling Online Application & College Choice Form?**  
1) Pay application/registration fee  
2) Fill online application & choice form  
3) Receive confirmation email/SMS  
4) Print hardcopies after final submission

**Where can I check my subject group?**  
See page **28** of the REAP booklet.

**How can I correct personal details/income if I can’t go back?**  
Raise a ticket on the **candidate panel** with all credentials.
""")

    st.subheader("Transaction")
    st.markdown("""
**My registration fee is deducted but registration not done.**  
Use **Check Transaction Status** with your temporary transaction number. You’ll get the permanent number by email (check spam too).

**Problem due to a failed transaction?**  
If the gateway doesn’t confirm in real time, payment isn’t complete. Pay again online; the failed transaction amount will be refunded.
""")

    st.subheader("Technical Issue")
    st.markdown("""
**What should I do if I face technical issues on the website?**  
Clear browser cache or contact technical support.

**How do I reset my account password?**  
Use the **Forgot Password** option on the login page.
""")

elif page == "Submit Issue":
    st.title("🛠️ Submit an Issue")
    name = st.text_input("Name")
    email = st.text_input("Email")
    mobile = st.text_input("Mobile")
    issue = st.text_area("Describe your issue")

    if st.button("Submit"):
        if not (name and email and issue):
            st.error("Name, Email, and Issue are required.")
        else:
            issues_collection().insert_one(
                {"name": name, "email": email, "mobile": mobile, "issue": issue}
            )
            try:
                body = f"Hello {name},\n\nThank you for submitting your issue.\n\nYour Issue: {issue}\n\nBest,\nShikshaMitra Support Team"
                send_mail(email, "Issue Submission Confirmation", body)
                st.success("Issue submitted. Confirmation email sent.")
            except Exception as e:
                st.warning(f"Issue saved, but email failed: {e}")

elif page == "Logout":
    st.title("🚪 Logout")
    if st.button("Confirm Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.success("You have been logged out")
        st.rerun()
