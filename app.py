import os
import sys
import streamlit as st
from streamlit_msal import Msal
from langchain_community.callbacks.manager import get_openai_callback
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import uuid
import msal
import pyodbc
import re
import logging
import time
import queue
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from streamlit_modal import Modal
from langchain.schema.runnable import RunnablePassthrough

#from login_ui import login_ui
import config
from config import SQL_DATABASE as sql_database
from login_ui import login_ui
import functions

# SETUP
### DATABASE CONNECTION
# Initialize connection variable 
connection = None
cursor = None

### LOGGING
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

### PROGRAM PATHS
base_directory = os.getcwd()
folder_name = "faiss_index"
vectorstore_path = os.path.join(base_directory, folder_name)

### SET THE APPLICATION RUN TYPE
#app_type = 'dev'
app_type = 'prod'

# SET THE PAGE CONFIGURATION (PAGE TITLE, PAGE ICON)
functions.render_page_config(base_directory)

# Global cache for LLM models and vectorstores
MODEL_CACHE = {}
VECTORSTORE_CACHE = {}
CHAIN_CACHE = {}

# Cache the common resources with longer TTL
@st.cache_resource(ttl=24*3600)  # Cache for 24 hours
def get_embeddings():
    """Get cached embeddings model"""
    return config.instantiate_embeddings()

# Cache the LLM model with minimal parameters
def get_llm(temperature, bot_type):
    """Get cached LLM with specific temperature"""
    cache_key = f"{bot_type}_{temperature}"
    if cache_key not in MODEL_CACHE:
        MODEL_CACHE[cache_key] = config.instantiate_llm(temperature)
    return MODEL_CACHE[cache_key]

# Cache the FAISS vectorstore with long TTL
def get_vectorstore(bot_type_selected):
    """Get cached vectorstore for specific bot type"""
    if bot_type_selected in VECTORSTORE_CACHE:
        return VECTORSTORE_CACHE[bot_type_selected]
    
    embeddings = get_embeddings()
    sub_folder = bot_type_selected.lower().replace(' ', '_')  
    load_path = os.path.join(vectorstore_path, sub_folder)
                                        
    # Load the FAISS index
    if os.path.exists(load_path):
        vectorstore = FAISS.load_local(
            load_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        VECTORSTORE_CACHE[bot_type_selected] = vectorstore
        return vectorstore
    else:
        st.error(f"No vectorstore found at {load_path}. Please ensure the FAISS index has been created and saved.")
        return None

# Get retrieval chain with caching - updated for safe but accurate source focus
def get_retrieval_chain(bot_type_selected):
    """Get cached retrieval chain for specific bot type with accurate source representation and reranking"""
    if bot_type_selected in CHAIN_CACHE:
        return CHAIN_CACHE[bot_type_selected]
    
    # Get cached vectorstore
    vectorstore = get_vectorstore(bot_type_selected)
    if vectorstore is None:
        return None
    
    # Get cached LLM with appropriate temperature
    llm = get_llm(0.2, bot_type_selected)  # Slightly higher but still low for consistency
    
    # Create a balanced source-focused prompt that won't trigger safety alerts
    balanced_prompt = """You are a helpful knowledge assistant that provides accurate information from reference documents.

GUIDANCE FOR ACCURATE RESPONSES:
- Focus on sharing information that comes directly from the provided sources
- When appropriate, include relevant quotes and excerpts from the source material
- Please clearly attribute information by mentioning source documents
- Aim to preserve the original context and meaning of the information
- Include complete information from sources when answering questions
- If sources contain contradictory information, acknowledge both perspectives
- When the sources don't address a specific question, politely explain this limitation
- For technical details or specific procedures, refer closely to the source wording

Your goal is to be accurate, helpful, and transparent about where information comes from.
"""
    
    # Get the base prompt from config and combine with our source focus guidance
    base_prompt = config.llm_prompt_dictonary.get(bot_type_selected, "")
    combined_prompt = base_prompt + "\n\n" + balanced_prompt
    
    # Create prompt template with balanced instructions
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", combined_prompt), 
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", "Here's relevant context from our knowledge base: {context}"),
        ]
    )
    
    # Create document chain
    document_chain = create_stuff_documents_chain(
        llm, 
        prompt_template,
    )
    
    # Set up retriever with increased number of chunks for initial retrieval
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 10,       # Retrieve more chunks for reranking
            "fetch_k": 20  # Consider more candidates
        }
    )

    # Create a custom retrieval function with reranking
    def retrieve_and_rerank(query):
        # First get documents using the standard FAISS retriever
        initial_docs = retriever.get_relevant_documents(query)
        
        # Apply reranking to these documents
        reranked_docs = config.rerank_documents(initial_docs, query, top_k=5)
        
        return reranked_docs

    # Create a custom retrieval chain that uses our reranking function
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    retrieval_chain = {
        "context": lambda x: format_docs(retrieve_and_rerank(x["input"])),
        "input": lambda x: x["input"],
        "chat_history": lambda x: x.get("chat_history", [])
    } | RunnablePassthrough() | document_chain
    
    # Cache the chain
    CHAIN_CACHE[bot_type_selected] = retrieval_chain
    return retrieval_chain

# Function to get a database connection
def get_db_connection():
    """Get a database connection"""
    global connection, cursor
    if connection is None:
        connection = config.sql_connect()
        cursor = connection.cursor()
    return connection, cursor

# Function to display sources with expanders
def display_sources(source_documents, answer_text=""):
    """
    Display sources as Streamlit expanders with highlighted text.
    """
    if not source_documents:
        return
    
    # Use the function to prepare source documents with highlighting
    prepared_sources = functions.prepare_source_documents(source_documents, answer_text)
    
    if prepared_sources:
        st.markdown("### Source Documents")
        st.markdown("*Information was drawn from these reference documents:*")
        
        # Create an expander for each source
        for source in prepared_sources:
            with st.expander(source["title"], expanded=False):
                # Get the highlighted text
                highlighted_text = functions.render_highlighted_text(source["content"])
                
                # Display the highlighted content
                st.markdown(f'<div class="source-content">{highlighted_text}</div>', unsafe_allow_html=True)

# Streamlined application function
def application():
    # IMPORT THE CUSTOM CSS FROM THE CONFIG FILE
    functions.add_custom_css()
    
    # Initialize feedback state
    functions.init_feedback_state()
            
    # Set up the conversation aliases:
    user_name = st.session_state.display_name
    assistant_name = "ABC Knowledge Assistant"

    try:
        # RENDER THE HEADER OF THE MAIN SCREEN 
        functions.render_app_header(base_directory)
        
        # RENDER POP UP MENU
        functions.render_popup_menu()
                
        st.write('Hi', st.session_state.display_name,)
        st.write(f"I am your helpful ABC Knowledge Assistant. Ask me any questions about XYZ products or policies.")
        
        # RENDER THE SIDEBAR MENU
        bot_type_selected = functions.render_sidebar_menu(base_directory, config.app_version, config.bot_options)               

        # Get retrieval chain for the selected bot type
        retrieval_chain = get_retrieval_chain(bot_type_selected)
        if retrieval_chain is None:
            return
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = str(uuid.uuid4())
            
        # Display existing messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=(os.path.join(base_directory, 'assets', 'user.png') if message["role"] == "user" else os.path.join(base_directory, 'assets', 'chatbot.png'))):
                if message["role"] == "user":
                    st.markdown(f"<div class='user-name' style='color: lightblue;'>{user_name}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='user-name' style='color: orange;'>{assistant_name}</div>", unsafe_allow_html=True)
                st.write(' ')
                
                # For assistant messages, show the answer and sources separately
                if message["role"] == "assistant":
                    # Check if the message has sources
                    if "sources" in message:
                        # Display just the answer part
                        st.markdown(message["answer"])
                        
                        # Display the sources using expanders
                        display_sources(message["sources"], message["answer"])
                    else:
                        # If no sources, just display the content as before
                        st.markdown(message["content"])
                else:
                    st.markdown(message["content"])
                
                # Only show feedback options for assistant messages
                if message["role"] == "assistant" and message.get("id"):
                    response_id = message["id"]
                    
                    # Get prompt from message or use empty string
                    prompt = message.get("prompt", "")
                    
                    # Check if feedback has already been submitted for this message
                    feedback_submitted_key = f"feedback_submitted_{response_id}"
                    
                    if feedback_submitted_key not in st.session_state or not st.session_state[feedback_submitted_key]:
                        # Display feedback buttons in a single container
                        feedback_container = st.container()
                        with feedback_container:
                            cols = st.columns([3, 1, 1, 1, 3])
                            
                            # Thumbs up button
                            with cols[1]:
                                if st.button("👍", key=f"thumbs_up_{response_id}"):
                                    # Store the feedback type in session state
                                    st.session_state[f"feedback_type_{response_id}"] = "thumbs_up"
                                    # Show the feedback form
                                    st.session_state[f"show_form_{response_id}"] = True
                                    st.rerun()
                            
                            # Thumbs down button
                            with cols[2]:
                                if st.button("👎", key=f"thumbs_down_{response_id}"):
                                    # Store the feedback type in session state
                                    st.session_state[f"feedback_type_{response_id}"] = "thumbs_down"
                                    # Show the feedback form
                                    st.session_state[f"show_form_{response_id}"] = True
                                    st.rerun()
                            
                            # Flag button
                            with cols[3]:
                                if st.button("🚩", key=f"flag_{response_id}"):
                                    # Store the feedback type in session state
                                    st.session_state[f"feedback_type_{response_id}"] = "flag_message"
                                    # Show the feedback form
                                    st.session_state[f"show_form_{response_id}"] = True
                                    st.rerun()
                        
                        # Check if the feedback form should be shown
                        if f"show_form_{response_id}" in st.session_state and st.session_state[f"show_form_{response_id}"]:
                            # Get the feedback type from session state
                            feedback_type = st.session_state[f"feedback_type_{response_id}"]
                            
                            # Map feedback types to display labels
                            feedback_labels = {
                                "thumbs_up": "👍 Positive feedback",
                                "thumbs_down": "👎 Negative feedback",
                                "flag_message": "🚩 Report this response"
                            }
                            
                            # Display feedback form with popover
                            with st.popover(f"{feedback_labels.get(feedback_type, 'Feedback')}", use_container_width=True):
                                # Comment text area
                                comment = st.text_area("Additional comments (optional):", 
                                                     key=f"comment_{response_id}", 
                                                     height=100)
                                
                                # Submit and cancel buttons
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    if st.button("Submit", key=f"submit_{response_id}"):
                                        # Get database connection
                                        conn, cur = get_db_connection()
                                        
                                        # Directly update the database with the feedback
                                        success = functions.direct_feedback_update(
                                            sql_database, 
                                            cur, 
                                            logger, 
                                            st.session_state.conversation_id, 
                                            prompt, 
                                            feedback_type, 
                                            comment
                                        )
                                        
                                        if success:
                                            # Mark as submitted and show success message
                                            st.session_state[feedback_submitted_key] = True
                                            st.session_state[f"show_form_{response_id}"] = False
                                            st.success("Feedback submitted successfully!")
                                            st.rerun()
                                        else:
                                            st.error("Failed to submit feedback. Please try again.")
                                
                                with col2:
                                    if st.button("Cancel", key=f"cancel_{response_id}"):
                                        # Reset form state and rerun
                                        st.session_state[f"show_form_{response_id}"] = False
                                        st.rerun()
            
        # Handle new messages
        if prompt := st.chat_input("Ask me something..."):
            # Start timing for performance measurement
            start_time = datetime.now()
            
            # Get database connection early
            conn, cur = get_db_connection()
            
            # Preprocess the prompt
            processed_prompt = functions.preprocess_query(prompt)
            
            # Add user message to session and display it
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar=os.path.join(base_directory, 'assets', 'user.png')):
                st.markdown(f"<div class='user-name' style='color: lightblue;'>{user_name}</div>", unsafe_allow_html=True)
                st.write(' ')
                st.markdown(prompt)
            
            # Prepare chat history - optimize by using only last few messages for context
            recent_messages = st.session_state.messages[-7:-1] if len(st.session_state.messages) > 1 else []
            chat_history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in recent_messages
            ]
            
            # Create a container for the assistant's response
            with st.chat_message("assistant", avatar=os.path.join(base_directory, 'assets', "chatbot.png")):
                message_placeholder = st.empty()
                
                # Show thinking animation while generating response
                with message_placeholder:
                    functions.thinking_animation()
                    
                    try:
                        # Generate response
                        with get_openai_callback() as cb:
                            # Generate response
                            result = retrieval_chain.invoke({
                                "input": processed_prompt,
                                "chat_history": chat_history
                            })
                            
                            # Get just the answer part 
                            answer_only = result["answer"]
                            
                            # Get source documents
                            source_docs = result.get("context", [])
                            
                            # Replace thinking animation with the answer
                            message_placeholder.markdown(f"<div class='user-name' style='color: orange;'>{assistant_name}</div>", unsafe_allow_html=True)
                            message_placeholder.write(' ')
                            message_placeholder.markdown(answer_only)
                            
                            # Display sources below the answer
                            if source_docs:
                                display_sources(source_docs, answer_only)
                        
                    except Exception as e:
                        logger.error(f"Error generating response: {str(e)}")
                        st.error("I had trouble generating a response. Please try again or rephrase your question.")
                        return
            
            # Generate a unique ID for this response
            response_id = str(uuid.uuid4())
            
            # For logging purposes, combine the answer and sources
            if source_docs:
                full_response = answer_only + "\n\nSources:\n" + "\n".join([f"- {doc.metadata.get('source', 'Unknown')}" for doc in source_docs])
            else:
                full_response = answer_only
            
            # Add the response to session state with the answer, sources, and ID
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,  # For backward compatibility
                "answer": answer_only,     # Just the answer part
                "sources": source_docs,    # Source documents
                "id": response_id,
                "prompt": prompt
            })
            
            # Log conversation IMMEDIATELY instead of in background
            try:
                logger.info(f"Logging conversation to database")
                # Use direct call to log_conversation function
                success = functions.log_conversation(
                    sql_database, cur, logger, 
                    st.session_state.conversation_id, 
                    prompt, full_response, None, 
                    st.session_state['display_name'], 
                    st.session_state['user_email'], 
                    bot_type_selected
                )
                if success:
                    logger.info(f"Successfully logged conversation to database")
                else:
                    logger.error(f"Failed to log conversation to database")
            except Exception as log_error:
                logger.error(f"Error logging conversation: {str(log_error)}")
            
            # Calculate and log response time for monitoring
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            logger.info(f"Response generated in {response_time:.2f} seconds")
            
            # Force refresh to show the new message with feedback buttons
            st.rerun()
        
        else:
            st.write(" ")
    except Exception as e:
        print(e)
        logger.error(f"An error occurred: {str(e)}")
        st.error("An error occurred while processing your request. Please try again.")


def main():
    
    # CONTROL THE APPLICATION FLOW BASED ON APP_TYPE (DEV OR PROD)
    
    if app_type == 'prod':
        if st.session_state.get("authenticated", False): 
            application()
        else:
            login_ui()
    
    else: # NON-PROD
        st.session_state.display_name = "user"
        st.session_state.user_email = "user@gmail.com"
        application()

    
if __name__ == '__main__':
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    main()