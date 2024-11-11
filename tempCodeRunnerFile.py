import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")

# Test if API key is loaded
if not api_key:
    st.error("No API key found. Make sure you have API_KEY in your .env file")
    st.stop()

# Initialize ChatOpenAI with error handling
try:
    llm = ChatOpenAI(
        temperature=0.7,
        openai_api_key=api_key,
        model="gpt-3.5-turbo",  # explicitly specify model
        max_tokens=100,
    )
    # Test the LLM with a simple query
    test_response = llm.invoke("Say 'API is working!'")
    st.success("LLM initialized successfully!")
except Exception as e:
    st.error(f"Error initializing LLM: {str(e)}")
    st.stop()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat interface
st.title("Simple Chat Test")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Type something to test the chat..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    try:
        # Get assistant response with a spinner
        with st.spinner("Thinking..."):
            response = llm.invoke(prompt)
            response_content = response.content
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response_content)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_content})
    
    except Exception as e:
        st.error(f"Error getting response: {str(e)}")