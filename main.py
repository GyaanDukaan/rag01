import streamlit as st
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile
import os

# Load Gemini API key
#GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"] if "GOOGLE_API_KEY" in st.secrets else os.getenv("GOOGLE_API_KEY")
#if not GOOGLE_API_KEY:
#    st.warning("Please set GOOGLE_API_KEY.")
#    st.stop()

os.environ["GOOGLE_API_KEY"] = "AIzaSyDfpL0ndH73Hfz1HRbksaKNDI_tC9wyjJg"


st.title("🔍 Gemini RAG App")

uploaded_file = st.file_uploader("Upload a text file (.txt)", type="txt")
query = st.text_input("Ask something about the uploaded document:")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Load and chunk the document
    loader = TextLoader(file_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Use Gemini Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)


    # Setup Gemini Chat Model
    retriever = vectorstore.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

    # Setup Retrieval QA
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    if query:
        response = rag_chain.run(query)
        st.success(response)
