import streamlit as st
import os
from dotenv import load_dotenv
import json
import datetime
import base64
from langchain.chains import ConversationalRetrievalChain
from OllamaChatBot import OllamaChatBot

from langchain_community.document_loaders import (
    DirectoryLoader, 
    PyPDFLoader, 
    TextLoader, 
    JSONLoader, 
    CSVLoader,
    NotebookLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from LLMReranker import LLMReranker
import shutil
from chromadb import Settings
import chromadb
import pandas as pd
import time
from langchain.schema import Document

# Set page config
st.set_page_config(
    page_title="RAG Chatbot", 
    page_icon="🤖",
    layout="wide"  # Use wide layout for more space
)

# Load environment variables from .env file
load_dotenv(os.path.join(os.getcwd(), ".env"))

def split_csv(csv_string):
    """
    Split a comma-separated string into a list of strings.
    Works for single values as well.
    
    Args:
        csv_string (str): A comma-separated string
        
    Returns:
        list: List of individual string values
    """
    if not csv_string:
        return []
    
    # Split the string by commas and strip whitespace
    result = [item.strip() for item in csv_string.split(',')]
    return result
    
def create_embeddings(embedding_type, embedding_model, ollama_base_url=None):
    """Create the appropriate embedding model based on type"""
    if embedding_type == "Ollama":
        return OllamaEmbeddings(
            base_url=ollama_base_url,
            model=embedding_model
        )
    else:  # OpenAI embeddings
        return OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

def create_vector_store(docs_dir, db_path, embedding_type, embedding_model, ollama_base_url=None):
    """Create a vector store from documents in the specified directory"""
    try:
        # Remove existing database if it exists
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        
        # Get document loaders for different file types
        loaders = get_document_loaders(docs_dir)
        
        # Load all documents
        all_documents = []
        for loader_name, loader in loaders.items():
            try:
                documents = loader.load()
                all_documents.extend(documents)
                st.sidebar.write(f"Loaded {len(documents)} {loader_name} documents")
            except Exception as e:
                st.sidebar.warning(f"Error loading {loader_name} documents: {str(e)}")
        
        if not all_documents:
            st.error("No documents were loaded. Please check your directory path.")
            return None
        
        # Split documents into chunks using the configured chunk size and overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        chunks = text_splitter.split_documents(all_documents)
        
        # Create embeddings
        embeddings = create_embeddings(embedding_type, embedding_model, ollama_base_url)
        
        # Initialize Chroma with specific settings
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_path,
            client=client,
            collection_name="knowledge_docs"
        )
        return vector_store
    
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def load_vector_store(db_path, embedding_type, embedding_model, ollama_base_url=None):
    """Load an existing vector store"""
    try:
        # Create embeddings
        embeddings = create_embeddings(embedding_type, embedding_model, ollama_base_url)
        
        # Initialize Chroma client
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings,
            client=client,
            collection_name="knowledge_docs"
        )
        return vector_store
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def create_llm(model_type, model_name, ollama_base_url=None):
    """Create the appropriate LLM based on model type and name"""
    if model_type == "Ollama":
        return ChatOllama(
            base_url=ollama_base_url,
            model=model_name,
            temperature=0.5
        )
    elif model_type == "OpenAI":
        return ChatOpenAI(
            model=model_name,
            temperature=0.5,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    elif model_type == "Anthropic":
        return ChatAnthropic(
            model=model_name,
            temperature=0.5,
            anthropic_api_key=os.getenv("CLAUDE_API_KEY")
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
def extract_markdown_content(text: str, type: str = "json") -> str:
    """Extract content from markdown code blocks"""
    start = f"""```{type}"""
    end = """```"""

    start_idx = text.find(start)
    end_idx = text.rfind(end)

    if start_idx >= 0 and end_idx >= 0:
        start_idx += len(type) + 3
        end_idx -= 1
        return (text[start_idx:end_idx]).strip()

    return text.strip()

# Function to rerank documents using LLMReranker
def rerank_documents(reranker, documents, query):
    """
    Rerank documents using LLMReranker
    
    Args:
        reranker: LLMReranker instance 
        documents: List of retrieved documents
        query: User query
        
    Returns:
        List of reranked documents
    """
    # Convert LangChain documents to format expected by LLMReranker
    reranker_docs = []
    for doc in documents:
        reranker_docs.append({
            "text": doc.page_content,
            "metadata": doc.metadata
        })
    
    # Rerank documents
    ranked_chunks = reranker.rerank(query, reranker_docs)
    
    # Convert back to LangChain document format
    reranked_docs = []
    
    for chunk in ranked_chunks:
        doc = Document(
            page_content=chunk["passage"],
            metadata=chunk["metadata"]
        )
        reranked_docs.append(doc)
    
    return reranked_docs

# Function to process a query and generate a response
def process_query(llm, retriever, reranker, memory, prompt):
    """Process a query using retrieval and reranking"""
    try:
        # Get initial documents from retriever
        initial_docs = retriever.get_relevant_documents(prompt)
        
        # Rerank documents
        reranked_docs = rerank_documents(reranker, initial_docs, prompt)
        
        # Format context from reranked documents
        context = "\n\n".join([doc.page_content for doc in reranked_docs])
        
        # Get chat history as formatted string
        chat_history = ""
        if memory.chat_memory.messages:
            for message in memory.chat_memory.messages:
                role = "User" if message.type == "human" else "Assistant"
                chat_history += f"{role}: {message.content}\n"
        
        # Generate prompt with context and chat history
        final_prompt = f"""
        Answer the following question based on the provided context and chat history:
        
        Chat History:
        {chat_history}
        
        Context:
        {context}
        
        Question: {prompt}
        
        Please provide a comprehensive and accurate answer based on the information in the context.
        """
        
        # Get response
        response = llm.invoke(final_prompt)
        answer = response.content
        
        # Update memory with the new exchange
        memory.chat_memory.add_user_message(prompt)
        memory.chat_memory.add_ai_message(answer)
        
        return answer, reranked_docs
    
    except Exception as e:
        error_message = f"Error processing query: {str(e)}"
        st.error(error_message)
        return error_message, []

# Function to execute chain of thought reasoning with preserved context
def execute_chain_of_thought(llm, retriever, reranker, prompt, max_steps=5):
    """Execute chain of thought reasoning with reranking"""
    # Initial prompt to get chain of thought planning
    cot_system_prompt = """
    You are a helpful assistant that thinks through problems step by step.
    Given the user's question, create a plan to answer it with clear reasoning steps.
    If you output equations, use LaTex.
    Output your reasoning plan in the following JSON format:
    {
        "reasoning_steps": [
            "Step 1: ...",
            "Step 2: ...",
            "Step 3: ...",
            "Step 4: ..."
        ],
        "total_steps": 4
    }
    Use as many steps as needed for the problem.
    """
    
    # Get the initial documents
    initial_docs = retriever.get_relevant_documents(prompt)
    
    # Perform reranking
    reranked_docs = rerank_documents(reranker, initial_docs, prompt)
    
    context = "\n\n".join([doc.page_content for doc in reranked_docs])
    
    # Initial prompt with context
    initial_prompt = f"""
    System: {cot_system_prompt}
    
    Context information for the question:
    {context}
    
    User question: {prompt}
    
    Think step by step about how to solve this. Return your plan in the required JSON format.
    """
    
    # Get the plan
    plan_response = llm.invoke(initial_prompt)
    plan_text = plan_response.content

    # Extract JSON plan
    try:
        json_str = extract_markdown_content(plan_text)
        plan_json = json.loads(json_str)
    except:
        # If JSON parsing fails, create a default plan
        plan_json = {
            "reasoning_steps": ["Step 1: Analyze the question", 
                                "Step 2: Review contextual information", 
                                "Step 3: Formulate answer that generates the expected output from the original prompt", 
                                "Step 4: Output results exactly as specified in the original prompt"],
            "total_steps": 4
        }
    
    steps = plan_json.get("reasoning_steps", [])
    total_steps = min(len(steps), max_steps)  # Limit to max_steps
    
    # Create progress bar
    progress_bar = st.progress(0)
    step_output = []
    
    # Initialize the growing context that will accumulate through steps
    growing_context = context
    
    # Execute each step
    for i in range(total_steps):
        # Update progress
        progress = (i) / total_steps
        progress_bar.progress(progress)
        
        current_step = steps[i]
        
        # Create prompt for this step with the growing context
        if i < total_steps:
            step_prompt = f"""
            System: You are solving a problem step by step. 
            
            Context information:
            {growing_context}
            
            User question: {prompt}
            
            Previous steps completed:
            {' '.join(step_output)}
            
            Current step to execute: {current_step}
            
            Provide your detailed reasoning for this step.
            """
        else:
            step_prompt = f"""
            Context information:
            {growing_context}
            
            Previous steps completed:
            {' '.join(step_output)}
            
            Now answer this prompt exactly using the reasoning and context given: {prompt}
            """

        # Get response for this step
        step_response = llm.invoke(step_prompt)
        step_result = step_response.content
        
        # Add to outputs
        step_output.append(f"Step {i+1}: {step_result}")
        
        # Add this step's reasoning to the growing context
        growing_context += f"\n\nStep {i+1} reasoning: {step_result}"
        
        # Display intermediate step
        with st.expander(f"Step {i+1}: {current_step}", expanded=False):
            st.write(step_result)
        
        # Small delay to show progress visually
        time.sleep(0.5)
    
    # Final step - synthesize all reasoning into an answer
    progress_bar.progress(1.0)
    
    # Create prompt for final answer using the complete growing context
    final_prompt = f"""
    System: Now that you have thought through all the steps to solve this problem, 
    provide a complete, well-reasoned final answer.
    
    Original context information:
    {context}
    
    User question: {prompt}
    
    Your step-by-step reasoning process:
    {' '.join(step_output)}
    
    Complete accumulated context and reasoning:
    {growing_context}
    
    Based on this comprehensive reasoning and context, provide your final comprehensive answer to: {prompt}
    """
    
    # Get final answer
    final_response = llm.invoke(final_prompt)
    final_answer = final_response.content
    
    return final_answer, step_output, reranked_docs

def convert_chat_to_qmd(chat_history):
    """
    Convert chat history to Quarto markdown (.qmd) format
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create the Quarto YAML header
    qmd_content = f"""---
title: "RAG Chatbot Conversation"
author: "Generated by RAG Chatbot"
date: "{now}"
format: 
  html:
    theme: cosmo
    toc: true
    code-fold: true
---

# RAG Chatbot Conversation

This document contains a conversation with the RAG Chatbot exported on {now}.

"""
    
    # Add each message to the QMD content
    for i, message in enumerate(chat_history):
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            qmd_content += f"\n## User Question {i//2 + 1}\n\n"
            qmd_content += f"{content}\n\n"
        else:  # assistant
            qmd_content += f"\n## Assistant Response {i//2 + 1}\n\n"
            qmd_content += f"{content}\n\n"
    
    return qmd_content

def get_download_link(qmd_content, filename="chat_export.qmd"):
    """
    Generate a download link for the QMD content
    """
    b64 = base64.b64encode(qmd_content.encode()).decode()
    href = f'<a href="data:text/markdown;base64,{b64}" download="{filename}">Download Conversation as Quarto Document</a>'
    return href

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'reranker' not in st.session_state:
    st.session_state.reranker = None
if 'memory' not in st.session_state:
    st.session_state.memory = None
if 'initial_chunks' not in st.session_state:
    st.session_state.initial_chunks = 50
if 'top_n' not in st.session_state:
    st.session_state.top_n = 10
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = 1000
if 'chunk_overlap' not in st.session_state:
    st.session_state.chunk_overlap = 200
if 'reranker_chunk_size' not in st.session_state:
    st.session_state.reranker_chunk_size = 1000
if 'reranker_chunk_overlap' not in st.session_state:
    st.session_state.reranker_chunk_overlap = 200
if 'batch_size' not in st.session_state:
    st.session_state.batch_size = 5

# Add this to your session state initialization section after the memory initialization
if 'conversation' not in st.session_state:
    st.session_state.conversation = None

# Then modify the initialize_rag_components function to create and return the conversation chain
def initialize_rag_components(vector_store, model_type, model_name, ollama_base_url=None):
    """Initialize the RAG components: LLM, retriever, reranker, memory, and conversation chain"""
    try:
        # Create the LLM
        llm = create_llm(model_type, model_name, ollama_base_url)
        
        # Create memory for conversation history
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create retriever - no k parameter as reranker handles this
        retriever = vector_store.as_retriever()
        
        # Create LLMReranker (always uses OllamaChatBot)
        reranker = LLMReranker(
            llm_bot=OllamaChatBot(model_name, ollama_base_url, 0.0),
            chunk_size=st.session_state.reranker_chunk_size,
            chunk_overlap=st.session_state.reranker_chunk_overlap,
            initial_chunks=st.session_state.initial_chunks,
            top_n=st.session_state.top_n,
            batch_size=st.session_state.batch_size
        )
        
        # Create a ConversationalRetrievalChain
        conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True
        )
        
        return llm, retriever, reranker, memory, conversation
    
    except Exception as e:
        st.error(f"Error initializing RAG components: {str(e)}")
        return None, None, None, None, None

# Title and description
st.title("🤖 RAG Chatbot")
st.markdown("Ask questions about your documents with LLM reranking.")

# Sidebar for configuration
with st.sidebar:
    st.subheader("LLM Settings")

    # Model selection
    model_type = st.selectbox(
        "Select Model Provider",
        ["Ollama", "OpenAI", "Anthropic"],
        index=0
    )
    
    # Model selection based on provider
    if model_type == "Ollama":
        model_name = st.selectbox(
            "Select Ollama Model",
            split_csv(os.getenv("OLLAMA_MODEL")),
            index=0
        )
        ollama_base_url = st.text_input("Ollama Base URL", os.getenv("OLLAMA_END_POINT"))
    elif model_type == "OpenAI":
        model_name = st.selectbox(
            "Select OpenAI Model",
            split_csv(os.getenv("OPENAI_MODEL")),
            index=0
        )
        ollama_base_url = None
    elif model_type == "Anthropic":
        model_name = st.selectbox(
            "Select Anthropic Model",
            split_csv(os.getenv("ANTHROPIC_MODEL")),
            index=0
        )
        ollama_base_url = None
    
    # Add Chain of Thought option
    use_cot = st.checkbox("Use Chain of Thought", value=False)

    reset_chat = st.button("🔄 Reset Chat")

    # Document retrieval settings
    st.subheader("Docs & Chunking")

    # Input for documents directory
    docs_dir = st.text_input("Documents Directory", os.getenv("SOURCE_DOC_DIR"))
    
    # Chunk size slider
    chunk_size = st.slider(
        "Chunk Size",
        min_value=500,
        max_value=8000,
        value=1000,
        step=100,
        help="Size of text chunks when processing documents (in characters)"
    )
    st.session_state.chunk_size = chunk_size
    
    # Chunk overlap slider
    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=1000,
        value=200,
        step=50,
        help="Amount of overlap between consecutive chunks (in characters)"
    )
    st.session_state.chunk_overlap = chunk_overlap

    # Embedding model selection
    embedding_type = st.selectbox(
        "Select Embedding Model",
        ["Ollama", "OpenAI"],
        index=0
    )
    
    if embedding_type == "Ollama":
        embedding_model = st.text_input("Ollama Embedding Model", "nomic-embed-text:latest")
    else:
        embedding_model = st.text_input("OpenAI Embedding Model", "text-embedding-3-small")

    build_db = st.button("🔨 Build/Rebuild Vector Database")

    # LLM Reranking settings
    st.subheader("LLM Reranking Settings")

    # Initial chunks slider
    initial_chunks = st.slider(
        "Initial chunks to retrieve",
        min_value=10,
        max_value=100,
        value=50,
        step=5,
        help="Controls how many initial chunks to retrieve before reranking"
    )
    st.session_state.initial_chunks = initial_chunks
    
    # Top N slider
    top_n = st.slider(
        "Top N chunks to retain",
        min_value=1,
        max_value=20,
        value=10,
        step=1,
        help="Number of top chunks to keep after reranking"
    )
    st.session_state.top_n = top_n
    
    # Reranker chunk size slider
    reranker_chunk_size = st.slider(
        "Reranker Chunk Size",
        min_value=500,
        max_value=4000,
        value=1000,
        step=100,
        help="Size of text chunks for reranking"
    )
    st.session_state.reranker_chunk_size = reranker_chunk_size
    
    # Reranker chunk overlap slider
    reranker_chunk_overlap = st.slider(
        "Reranker Chunk Overlap",
        min_value=0,
        max_value=500,
        value=200,
        step=50,
        help="Amount of overlap between consecutive chunks for reranking"
    )
    st.session_state.reranker_chunk_overlap = reranker_chunk_overlap
    
    # Batch size slider
    batch_size = st.slider(
        "Batch Size",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Number of chunks to process in a sub-ranking batch"
    )
    st.session_state.batch_size = batch_size
    
    # Database operations
    db_path = os.path.join(docs_dir, "vectorstore")
     
    # Handle chat reset
    if reset_chat:
        st.session_state.chat_history = []
        if st.session_state.vector_store:
            llm, retriever, reranker, memory, conversation = initialize_rag_components(
                st.session_state.vector_store, 
                model_type, 
                model_name,
                ollama_base_url if model_type == "Ollama" else None
            )
            st.session_state.llm = llm
            st.session_state.retriever = retriever
            st.session_state.reranker = reranker
            st.session_state.memory = memory
            st.session_state.conversation = conversation
        st.rerun()

def get_document_loaders(docs_dir):
    """Get document loaders for different file types"""
    loaders = {
        "pdf": DirectoryLoader(
            docs_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        ),
        "txt": DirectoryLoader(
            docs_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        ),
        "json": DirectoryLoader(
            docs_dir,
            glob="**/*.json",
            loader_cls=lambda path: JSONLoader(
                file_path=path,
                jq_schema=".",
                text_content=False
            ),
            show_progress=True
        ),
        "jsonl": DirectoryLoader(
            docs_dir,
            glob="**/*.jsonl",
            loader_cls=lambda path: JSONLoader(
                file_path=path,
                jq_schema=".",
                text_content=False
            ),
            show_progress=True
        ),
        "csv": DirectoryLoader(
            docs_dir,
            glob="**/*.csv",
            loader_cls=CSVLoader,
            show_progress=True
        ),
        "ipynb": DirectoryLoader(
            docs_dir,
            glob="**/*.ipynb",
            loader_cls=NotebookLoader,
            show_progress=True
        )
    }
    return loaders

# Handle database building
if build_db:
    with st.spinner("Building vector database from documents..."):
        vector_store = create_vector_store(
            docs_dir, 
            db_path, 
            embedding_type, 
            embedding_model,
            ollama_base_url if model_type == "Ollama" else None
        )
        if vector_store:
            st.session_state.vector_store = vector_store
            st.success("Vector database built successfully!")
            llm, retriever, reranker, memory, conversation = initialize_rag_components(
                vector_store, 
                model_type, 
                model_name,
                ollama_base_url if model_type == "Ollama" else None
            )
            st.session_state.llm = llm
            st.session_state.retriever = retriever
            st.session_state.reranker = reranker
            st.session_state.memory = memory
            st.session_state.conversation = conversation

# Load existing database if it exists and we haven't loaded it yet
if not st.session_state.vector_store and os.path.exists(db_path):
    with st.spinner("Loading existing vector database..."):
        vector_store = load_vector_store(
            db_path, 
            embedding_type, 
            embedding_model,
            ollama_base_url if model_type == "Ollama" else None
        )
        if vector_store:
            st.session_state.vector_store = vector_store
            llm, retriever, reranker, memory, conversation = initialize_rag_components(
                vector_store, 
                model_type, 
                model_name,
                ollama_base_url if model_type == "Ollama" else None
            )
            st.session_state.llm = llm
            st.session_state.retriever = retriever
            st.session_state.reranker = reranker
            st.session_state.memory = memory
            st.session_state.conversation = conversation
            st.success("Existing vector database loaded successfully!")

# Display API key status
with st.sidebar:
    st.markdown("---")
    st.subheader("API Key Status")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    claude_key = os.getenv("CLAUDE_API_KEY")
    
    if openai_key:
        st.success("OpenAI API Key: Loaded ✅")
    else:
        st.error("OpenAI API Key: Missing ❌")
    
    if claude_key:
        st.success("Claude API Key: Loaded ✅")
    else:
        st.error("Claude API Key: Missing ❌")
    
    # Add Quarto export button
    st.markdown("---")
    st.subheader("Export Conversation")
    
    if st.button("📥 Download Chat as Quarto (.qmd)"):
        if st.session_state.chat_history:
            qmd_content = convert_chat_to_qmd(st.session_state.chat_history)
            st.markdown(get_download_link(qmd_content), unsafe_allow_html=True)
            st.success("Quarto document ready for download!")
        else:
            st.warning("No chat history to export yet.")


# Add helpful information in the sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    ### Instructions
    1. Enter the path to your documents directory
    2. Adjust the document processing settings (chunk size, overlap, and initial chunks to retrieve)
    3. Configure LLM Reranking settings
    4. Select your preferred model provider and model
    5. Select your embedding model
    6. Click 'Build Vector Database' if this is your first time or you want to rebuild
    7. Toggle 'Use Chain of Thought' if you want to see step-by-step reasoning
    8. Start chatting! 🚀
    
    Note: The system will process PDF, TXT, JSON, JSONL, CSV, and Jupyter Notebook files in the specified directory.
    """)

# Main layout for chat
if st.session_state.vector_store:
    # Display chat messages
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user", avatar="🧑").write(message["content"])
        else:
            st.chat_message("assistant", avatar="🤖").write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.chat_message("user", avatar="🧑").write(prompt)
        
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Process based on Chain of Thought setting
        if use_cot and st.session_state.llm and st.session_state.retriever:
            with st.spinner("Thinking step by step..."):
                # Execute chain of thought reasoning
                final_answer, step_outputs, reranked_docs = execute_chain_of_thought(
                    st.session_state.llm,
                    st.session_state.retriever,
                    st.session_state.reranker,
                    prompt
                )
                
                # Display assistant response
                st.chat_message("assistant", avatar="🤖").write(final_answer)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": final_answer})
        
        # Regular conversation flow (without CoT)
        else:  # Not using Chain of Thought
            with st.spinner("Thinking..."):
                # Process the query using our process_query function
                answer, reranked_docs = process_query(
                    st.session_state.llm,
                    st.session_state.retriever,
                    st.session_state.reranker,
                    st.session_state.memory,
                    prompt
                )
                
                # Display assistant response
                st.chat_message("assistant", avatar="🤖").write(answer)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.warning("Please build or load a vector database to start chatting!")