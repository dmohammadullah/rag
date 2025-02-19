import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from PyPDF2 import PdfReader

# Set API configuration (Deepseek only)
DEEPSEEK_API_KEY = ""  # Your Deepseek key
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"

# Configure page
st.set_page_config(page_title="Ask your PDF")
st.header("Ask questions about your PDF 📚")

# File upload
pdf = st.file_uploader("Upload your PDF", type="pdf")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Process PDF and create vector store
if pdf is not None:
    # Read PDF
    pdf_reader = PdfReader(pdf)
    text = "".join([page.extract_text() for page in pdf_reader.pages])

    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Create LOCAL embeddings (no API key needed)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)

    st.success("PDF processed successfully!")

# Question handling
if st.session_state.vector_store is not None:
    question = st.text_input("Ask a question about your PDF:")

    if question:
        # Search for similar chunks
        docs = st.session_state.vector_store.similarity_search(question)
        # Initialize Deepseek LLM
        llm = ChatOpenAI(
            model_name="deepseek-chat",
            temperature=0,
            api_key=DEEPSEEK_API_KEY,
            openai_api_base=DEEPSEEK_API_BASE,  # Changed from base_url to openai_api_base
            streaming=False
        )
        
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.invoke({"input_documents": docs, "question": question})["output_text"]
  
        st.write("Answer:")
        st.write(response)
