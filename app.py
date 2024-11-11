from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import streamlit as st

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Check for API key before proceeding
if not api_key:
    st.error("OpenAI API key not found! Please set OPENAI_API_KEY in your .env file")
    st.write("1. Create a .env file in your project directory")
    st.write("2. Add your OpenAI API key like this: OPENAI_API_KEY=sk-your_api_key_here")
    st.write("3. Make sure the .env file is in the same directory as your Python script")
    st.stop()

@st.cache_resource
def initialize_qa_chain():
    try:
        pdfpath = "./CTP Project Design Doc.pdf"
        
        # Check if PDF exists
        if not os.path.exists(pdfpath):
            st.error(f"PDF file not found at: {pdfpath}")
            st.stop()
            
        st.write(f"Loading PDF from: {pdfpath}")
        
        # Load PDF
        loader = PyPDFLoader(pdfpath)
        pages = loader.load()
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
        
        # Create vector store
        vectorstore = FAISS.from_documents(pages, embeddings)
        
        # Initialize ChatOpenAI with explicit API key
        llm = ChatOpenAI(
            temperature=0.7,
            api_key=api_key,  # Explicitly pass the API key
            model="gpt-3.5-turbo",
            max_tokens=100,
        )
        
        # Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
        )
        
        return qa_chain
        
    except Exception as e:
        st.error(f"Error in initialize_qa_chain: {str(e)}")
        raise e

# Page title
st.title("Ask about our Wordle Final Project")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize QA chain
try:
    with st.spinner("Loading PDF and initializing QA system..."):
        qa_chain = initialize_qa_chain()
    st.success("QA system initialized successfully!")
except Exception as e:
    st.error(f"Error initializing QA chain: {str(e)}")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask a question about the Wordle Final Project")

# Process the input
if prompt:
    try:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get response from QA chain with a spinner
        with st.spinner("Searching document for answer..."):
            response = qa_chain.run(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")