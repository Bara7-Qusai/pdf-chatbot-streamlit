# -*- coding: utf-8 -*-
"""
PDF Chatbot - واجهة احترافية جدًا باستخدام Streamlit + LangChain + Google GenAI
"""

import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import streamlit as st

# ==========================
# تحميل متغيرات البيئة
# ==========================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("⚠️ الرجاء وضع GOOGLE_API_KEY في ملف .env")
    st.stop()

# ==========================
# إعداد واجهة Streamlit
# ==========================
st.set_page_config(page_title="📄 PDF Chatbot", layout="wide")


st.markdown("""
<style>
body {
    background-color: #f7f7f8;
}
.chat-container {
    max-width: 900px;
    margin: auto;
    padding: 10px;
}
.user-msg {
    background-color: #DCF8C6;
    padding: 12px 18px;
    border-radius: 20px;
    margin: 5px 0;
    text-align: right;
    font-size: 16px;
    color: #000000;
}
.bot-msg {
    background-color: #FFFFFF;
    padding: 12px 18px;
    border-radius: 20px;
    margin: 5px 0;
    text-align: left;
    font-size: 16px;
    box-shadow: 0px 1px 3px rgba(0,0,0,0.1);
    color: #000000; 
}
.bot-icon {
    width: 30px;
    vertical-align: middle;
    margin-right: 8px;
}
.user-icon {
    width: 30px;
    vertical-align: middle;
    margin-left: 8px;
}
</style>
""", unsafe_allow_html=True)


st.title("📄 PDF Chatbot")
st.markdown("ارفع ملف PDF وابدأ محادثة تفاعلية  !")

# ==========================
# رفع PDF
# ==========================
uploaded_file = st.file_uploader("اختر ملف PDF", type="pdf")

if uploaded_file is not None:

    # ==========================
    # استخراج النص
    # ==========================
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    if not text.strip():
        st.error("❌ الملف لا يحتوي على نص قابل للقراءة.")
        st.stop()

    # ==========================
    # تقسيم النص بالـ separators الهرمي
    # ==========================
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    st.success(f"✅ تم معالجة النص إلى {len(chunks)} chunks")

    # ==========================
    # إنشاء قاعدة المعرفة
    # ==========================
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # ==========================
    # إعداد LLM وذاكرة المحادثة
    # ==========================
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=knowledge_base.as_retriever(),
        memory=memory
    )

    # ==========================
    # محادثة مع المستخدم
    # ==========================
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("💬 اكتب سؤالك هنا:")

    if user_input:
        response = qa_chain({"question": user_input})
        # حفظ الرسائل
        st.session_state.chat_history.append({"user": user_input, "bot": response["answer"]})

    # عرض الرسائل بشكل فقاعات مثل ChatGPT
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        st.markdown(f'<div class="user-msg">🧑 {chat["user"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bot-msg">🤖 {chat["bot"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

