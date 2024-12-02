import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Page config
st.set_page_config(page_title="Document Q&A", layout="wide")

# Load environment variables
load_dotenv()

# Function to get API key from either secrets or environment
def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        return os.getenv("OPENAI_API_KEY")

api_key = get_api_key()

# Set environment variable to handle tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def check_pdf_file():
    """Check if PDF file exists and return its path"""
    # Allow user to upload PDF
    uploaded_file = st.file_uploader("Upload your PDF document", type=['pdf'])
    
    if uploaded_file:
        # Save uploaded file temporarily
        with open("temp_doc.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        return "temp_doc.pdf"
    
    # If no upload, check for local file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_pdf = os.path.join(script_dir, "CTP Project Design Doc.pdf")
    
    if os.path.exists(local_pdf):
        return local_pdf
    
    return None

@st.cache_resource
def initialize_qa_chain(pdf_path):
    """Initialize the QA chain with the given PDF"""
    try:
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        if not pages:
            st.error("No content found in PDF!")
            return None
            
        st.success(f"Successfully loaded {len(pages)} pages from PDF")
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create vector store
        vectorstore = FAISS.from_documents(pages, embeddings)
        
        # Initialize ChatOpenAI
        llm = ChatOpenAI(
            temperature=0.2,
            api_key=api_key,
            model="gpt-3.5-turbo",
            max_tokens=500,
        )
        
        # Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True,
        )
        
        return qa_chain
        
    except Exception as e:
        st.error(f"Error in initialize_qa_chain: {str(e)}")
        return None

def main():
    st.title("💬 Document Q&A Bot")
    
    # Check for API key
    if not api_key:
        st.error("OpenAI API key not found! Please set it in your .env file or Streamlit secrets")
        st.write("Add your OpenAI API key as OPENAI_API_KEY=sk-your_api_key_here")
        st.stop()
    
    # Get PDF path
    pdf_path = check_pdf_file()
    
    if not pdf_path:
        st.error("No PDF document found! Please upload a PDF file.")
        st.stop()
    
    # Initialize QA chain
    with st.spinner("Loading document and initializing QA system..."):
        qa_chain = initialize_qa_chain(pdf_path)
    
    if not qa_chain:
        st.error("Failed to initialize QA system!")
        st.stop()
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    prompt = st.chat_input("Ask a question about the document")
    
    # Process the input
    if prompt:
        try:
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Get response from QA chain
            with st.spinner("Searching document for answer..."):
                response = qa_chain.invoke({"query": prompt})
            
            result = response["result"]
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(result)
                
                # If you want to show source documents
                if "source_documents" in response:
                    with st.expander("View Sources"):
                        for doc in response["source_documents"]:
                            st.write(f"Page {doc.metadata.get('page', 'unknown')}")
                            st.write(doc.page_content)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": result})
            
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
