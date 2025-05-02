import streamlit as st
import os
from dotenv import load_dotenv
import json
import datetime
import base64

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
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter

import shutil
from chromadb import Settings
import chromadb
import pandas as pd
import time

# Set page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"  # Use wide layout for more space
)

# Load environment variables from .env file
load_dotenv(os.path.join(os.getcwd(), ".env"))

def extract_document_metadata(document_path, llm):
    """
    Extract metadata from an academic paper using the LLM.
    Returns author, title, year, journal, full APA reference, and abstract.
    """
    try:
        # Read the document content
        if document_path.endswith('.pdf'):
            loader = PyPDFLoader(document_path)
            doc = loader.load()[0]  # Get the first page for metadata extraction
            content = doc.page_content
        else:
            loader = TextLoader(document_path)
            doc = loader.load()[0]
            content = doc.page_content
        
        # Truncate content to a reasonable size for the LLM (first 10000 chars)
        content_preview = content[:2000]
        
        # Create prompt for metadata extraction
        prompt = f"""
        Extract the following metadata from this academic paper:
        - Author(s)
        - Title
        - Year
        - Journal/Conference
        - Full APA reference
        - Abstract

        Return the extracted metadata in a strict JSON format without any additional text:
        {{
            "author": "Author name(s)",
            "title": "Paper title",
            "year": "Publication year",
            "journal": "Journal or conference name",
            "apa_reference": "Full APA style reference",
            "abstract": "Paper abstract"
        }}

        If any field cannot be extracted, use "Unknown" as the value.

        Here's the beginning of the paper:
        {content_preview}
        """
        
        # print("prompt: ", prompt, "\n\n")

        # Query the LLM for metadata extraction
        response = llm.invoke(prompt)

        # print("raw response for metadata: ", response, "\n\n")
        
        # Extract JSON from the response
        metadata_str = extract_markdown_content(response.content, "json")

        # print("extracted JSON: ", metadata_str, "\n\n")

        if not metadata_str:
            metadata_str = response.content
            
        metadata = json.loads(metadata_str)
        return metadata
    
    except Exception as e:
        print(f"Error extracting metadata from {document_path}: {str(e)}")
        # Return default metadata if extraction fails
        return {
            "author": "Unknown",
            "title": os.path.basename(document_path),
            "year": "Unknown",
            "journal": "Unknown",
            "apa_reference": f"Unknown. ({datetime.datetime.now().year}). {os.path.basename(document_path)}.",
            "abstract": "No abstract available"
        }

def save_metadata_json(metadata, source_path, metadata_dir):
    """
    Save metadata to a JSON file in the metadata directory with the same basename as the source.
    """
    # Create metadata directory if it doesn't exist
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Create JSON filename based on source document name
    base_name = os.path.splitext(os.path.basename(source_path))[0]
    json_path = os.path.join(metadata_dir, f"{base_name}.json")
    
    # Save metadata to JSON file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return json_path

def prepend_reference_to_chunks(chunks, metadata):
    """
    Prepend the full APA reference to each chunk as context.
    """
    for chunk in chunks:
        # Add the reference as context at the beginning of each chunk
        chunk.page_content = f"Reference: {metadata['apa_reference']}\n\n{chunk.page_content}"
    
    return chunks

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
    elif embedding_type == "OpenAI":
        return OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    elif embedding_type == "Google":
        return GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")


def create_vector_store(docs_dir, db_path, embedding_type, embedding_model, ollama_base_url=None):
    """Create a vector store from documents in the specified directory"""
    try:
        # Remove existing database if it exists
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            
            # Ensure the directory exists after removal
            os.makedirs(db_path, exist_ok=True)

        # Create metadata directory
        metadata_dir = os.path.join(docs_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)

        # Get document loaders for different file types
        loaders = get_document_loaders(docs_dir)
        
        # Create the selected LLM for metadata extraction
        metadata_llm = create_llm(model_type, model_name, ollama_base_url if model_type == "Ollama" else None)
        
        st.sidebar.write(f"Using {model_type} model: {model_name} for metadata extraction...")

        # Load all documents and extract metadata
        all_documents = []
        for loader_name, loader in loaders.items():
            try:
                documents = loader.load()

                # Process each document for metadata extraction
                processed_docs = []
                for doc in documents:
                    # Get the source path
                    source_path = doc.metadata.get('source')
                    if source_path:
                        # Extract metadata
                        st.sidebar.write(f"Extracting metadata from {os.path.basename(source_path)}...")

                        # Check if the metadata file already exists
                        base_name = os.path.splitext(os.path.basename(source_path))[0]
                        json_path = os.path.join(metadata_dir, f"{base_name}.json")

                        if os.path.exists(json_path):
                            # Load the existing metadata
                            with open(json_path, 'r', encoding='utf-8') as f:
                                try:
                                    metadata = json.load(f)
                                    st.sidebar.write(f"Metadata loaded from {os.path.basename(json_path)}")
                                except json.JSONDecodeError as e:
                                    st.sidebar.warning(f"Error decoding JSON from {json_path}: {e}.  Attempting to re-extract metadata.")
                                    metadata = extract_document_metadata(source_path, metadata_llm)
                                    json_path = save_metadata_json(metadata, source_path, metadata_dir)
                        else:
                            # Extract metadata if it doesn't exist
                            metadata = extract_document_metadata(source_path, metadata_llm)

                            # Save metadata to JSON
                            json_path = save_metadata_json(metadata, source_path, metadata_dir)
                            st.sidebar.write(f"Saved metadata to {os.path.basename(json_path)}")

                        # Add metadata to document's metadata
                        doc.metadata.update(metadata)

                        processed_docs.append(doc)
                    else:
                        processed_docs.append(doc)

                all_documents.extend(processed_docs)
                st.sidebar.write(f"Loaded and processed {len(documents)} {loader_name} documents")
            except Exception as e:
                st.sidebar.warning(f"Error loading {loader_name} documents: {str(e)}")
                st.error(f"Failed to load docs {str(e)}")  # Add specific logging

        if not all_documents:
            st.error("No documents were loaded. Please check your directory path.")
            return None

        # Split documents into chunks using the configured chunk size and overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", " ", ""]  # Prioritize splitting by paragraph, then newline, then space
        )

        # Process documents one by one to apply the reference prepending
        all_chunks = []
        for doc in all_documents:
            # Split the individual document
            doc_chunks = text_splitter.split_documents([doc])

            # Get the metadata for this document
            doc_metadata = doc.metadata

            # Prepend reference to each chunk
            if 'apa_reference' in doc_metadata and doc_metadata['apa_reference'] != "Unknown":
                doc_chunks = prepend_reference_to_chunks(doc_chunks, doc_metadata)

            all_chunks.extend(doc_chunks)

        # Create embeddings
        embeddings = create_embeddings(embedding_type, embedding_model, ollama_base_url)

        # Process in batches to avoid token limits
        batch_size = 100
        total_chunks = len(all_chunks)
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        # Initialize an empty vector store
        vector_store = Chroma.from_documents(
            documents=all_chunks[:1],  # Start with just one document
            embedding=embeddings,
            persist_directory=db_path,
            collection_name="knowledge_docs"
        )
        
        # If there are more documents, add them in batches
        if len(all_chunks) > 1:
            for i in range(1, total_chunks, batch_size):
                batch_num = (i // batch_size) + 1
                end_idx = min(i + batch_size, total_chunks)
                batch = all_chunks[i:end_idx]
                
                st.sidebar.write(f"Processing batch {batch_num}/{total_batches}...")
                
                # Add batch to vector store
                vector_store.add_documents(documents=batch)
                
                st.sidebar.write(f"Batch {batch_num} completed")
        
        st.sidebar.write("Vector store created successfully!")
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
            temperature=0.8
        )
    elif model_type == "OpenAI":
        return ChatOpenAI(
            model=model_name,
            temperature=0.8,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    elif model_type == "Anthropic":
        return ChatAnthropic(
            model=model_name,
            temperature=0.8,
            anthropic_api_key=os.getenv("CLAUDE_API_KEY")
        )
    elif model_type == "Google":
         return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.8,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def initialize_conversation(vector_store, model_type, model_name, k_value=10, ollama_base_url=None, use_reranking=False, num_chunks_kept=4):
    """Initialize the conversation chain"""
    try:
        # Create the LLM
        llm = create_llm(model_type, model_name, ollama_base_url)

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": k_value})

        # Log retrieved documents (for debugging)
        sample_query = "What is the main topic of this document?"  # Or a relevant query
        retrieved_docs = retriever.get_relevant_documents(sample_query)
        print("Retrieved Documents for Debugging:")
        for doc in retrieved_docs:
            print(f"Source: {doc.metadata.get('source')}")  # Assuming 'source' is the file path
            print(f"Content: {doc.page_content[:200]}...")  # Print first 200 chars

        # Reranking with ContextualCompressionRetriever
        if use_reranking:
            compressor = LLMChainExtractor.from_llm(llm)
            # Ensure num_chunks_kept is not greater than k_value
            num_chunks_kept = min(num_chunks_kept, k_value)
            retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever, k=num_chunks_kept)


        # Make the chain verbose so we can see prompts in stdout
        conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=True
        )

        return conversation, llm, retriever

    except Exception as e:
        st.error(f"Error initializing conversation: {str(e)}")
        return None, None, None

def extract_markdown_content(text: str, type: str = "json") -> str:
        start = f"""```{type}"""
        end = """```"""

        start_idx = text.find(start)
        end_idx = text.rfind(end)

        if start_idx >= 0 and end_idx >= 0:
            start_idx += len(type) + 3
            end_idx -= 1
            return (text[start_idx:end_idx]).strip()

        return text.strip()

# Function to execute chain of thought reasoning with preserved context
def execute_chain_of_thought(llm, retriever, prompt, max_steps=5):
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

    # Get the relevant documents
    retrieved_docs = retriever.get_relevant_documents(prompt)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

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
        print("plan:", json_str)
        plan_json = json.loads(json_str)
    except:
        # If JSON parsing fails, create a default plan
        plan_json = {
            "reasoning_steps": ["Step 1: Analyze the question",
                                "Step 2: Review contextual information",
                                "Step 3: Formulate answer that generates the expected output from the original prompt",
                                "Step 4: Output results exactly as specified in the original prompt",
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
        # This is the key change - we're updating the context with each step's output
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

    return final_answer, step_output

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
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'k_value' not in st.session_state:
    st.session_state.k_value = 10
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = 1000
if 'chunk_overlap' not in st.session_state:
    st.session_state.chunk_overlap = 200
if 'use_reranking' not in st.session_state:
    st.session_state.use_reranking = False
if 'num_chunks_kept' not in st.session_state:
    st.session_state.num_chunks_kept = 4

# Title and description
st.title("🤖 RAG Chatbot")
st.markdown("Ask questions about your documents.")

# Sidebar for configuration
with st.sidebar:
    # Document retrieval settings
    st.subheader("Vector DB Settings")

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
        split_csv(os.getenv("EMBEDDING_PROVIDERS")),
        index=0
    )

    if embedding_type == "Ollama":
        embedding_model = st.text_input("Ollama Embedding Model", "nomic-embed-text:latest")
    elif embedding_type == "OpenAI":
        embedding_model = st.text_input("OpenAI Embedding Model", "text-embedding-3-small")
    elif embedding_type == "Google":
        embedding_model = st.text_input("Google Embedding Model", "models/embedding-001")

    build_db = st.button("🔨 Build Vector Database")

    st.subheader("LLM Settings")

    # Number of retrieved documents
    k_value = st.slider(
        "Number of retrieved documents (K)",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        help="Controls how many relevant documents to retrieve for each query"
    )
    st.session_state.k_value = k_value

     # Number of chunks kept for reranking
    num_chunks_kept = st.slider(
        "Number of Chunks Kept After Reranking",
        min_value=1,
        max_value=st.session_state.k_value,  # Limit to the number of retrieved documents
        value=4,
        step=1,
        help="Number of chunks to keep after LLM reranking.  Must be less than or equal to the number of retrieved documents."
    )
    st.session_state.num_chunks_kept = num_chunks_kept

    # Model selection
    model_type = st.selectbox(
        "Select Model Provider",
        split_csv(os.getenv("LLM_PROVIDERS")),
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
    elif model_type == "Anthropic":
        model_name = st.selectbox(
            "Select Anthropic Model",
            split_csv(os.getenv("ANTHROPIC_MODEL")),
            index=0
        )
    elif model_type == "Google":
        model_name = st.selectbox(
            "Select Google Model",
            split_csv(os.getenv("GOOGLE_MODEL")),
            index=0
        )

    # Add Chain of Thought option
    use_cot = st.checkbox("Use Chain of Thought", value=False)

    # Add LLM Reranking option
    use_reranking = st.checkbox("Use LLM Reranking", value=True)
    st.session_state.use_reranking = use_reranking

    # Database operations
    db_path = os.path.join(docs_dir, "vectorstore")
    print("Using: ", db_path, " for vector database...")

    col1, col2 = st.columns(2)
    reset_chat = st.button("🔄 Reset Chat")

    # Handle chat reset
    if reset_chat:
        st.session_state.chat_history = []
        if st.session_state.vector_store:
            conversation, llm, retriever = initialize_conversation(
                st.session_state.vector_store,
                model_type,
                model_name,
                st.session_state.k_value,
                ollama_base_url if model_type == "Ollama" else None,
                st.session_state.use_reranking,
                st.session_state.num_chunks_kept
            )
            st.session_state.conversation = conversation
            st.session_state.llm = llm
            st.session_state.retriever = retriever
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
                jq_schema=".", # CHECK THIS - IS IT RIGHT?
                text_content=False
            ),
            show_progress=True
        ),
        "jsonl": DirectoryLoader(
            docs_dir,
            glob="**/*.jsonl",
            loader_cls=lambda path: JSONLoader(
                file_path=path,
                jq_schema=".", # CHECK THIS - IS IT RIGHT?
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
        ),
        "md": DirectoryLoader(
            docs_dir,
            glob="**/*.md",
            loader_cls=TextLoader,
            show_progress=True
        ),
        "qmd": DirectoryLoader(
            docs_dir,
            glob="**/*.qmd",
            loader_cls=TextLoader,
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
            # Initialize conversation with new vector store
            conversation, llm, retriever = initialize_conversation(
                vector_store,
                model_type,
                model_name,
                st.session_state.k_value,
                ollama_base_url if model_type == "Ollama" else None,
                st.session_state.use_reranking,
                st.session_state.num_chunks_kept
            )
            st.session_state.conversation = conversation
            st.session_state.llm = llm
            st.session_state.retriever = retriever

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
            # Re-initialize conversation with current k_value
            conversation, llm, retriever = initialize_conversation(
                vector_store,
                model_type,
                model_name,
                st.session_state.k_value,
                ollama_base_url if model_type == "Ollama" else None,
                st.session_state.use_reranking,
                st.session_state.num_chunks_kept
            )
            st.session_state.conversation = conversation
            st.session_state.llm = llm
            st.session_state.retriever = retriever
            st.success("Existing vector database loaded successfully!")

# Display API key status
with st.sidebar:
    st.markdown("---")
    st.subheader("API Key Status")

    openai_key = os.getenv("OPENAI_API_KEY")
    claude_key = os.getenv("CLAUDE_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if openai_key:
        st.success("OpenAI API Key: Loaded ✅")
    else:
        st.error("OpenAI API Key: Missing ❌")

    if claude_key:
        st.success("Claude API Key: Loaded ✅")
    else:
        st.error("Claude API Key: Missing ❌")

    if google_key:
        st.success("Google API Key: Loaded ✅")
    else:
        st.error("Google API Key: Missing ❌")

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
    2. Adjust the document processing settings (chunk size, overlap, and K value)
    3. Select your preferred model provider and model
    4. Select your embedding model
    5. Click 'Build Vector Database' if this is your first time or you want to rebuild
    6. Toggle 'Use Chain of Thought' if you want to see step-by-step reasoning
    7. Toggle 'Use LLM Reranking' if you want the LLM to pick the best chunks from retrieved documents
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
                final_answer, step_outputs = execute_chain_of_thought(
                    st.session_state.llm,
                    st.session_state.retriever,
                    prompt
                )

                # Display assistant response
                st.chat_message("assistant", avatar="🤖").write(final_answer)

                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": final_answer})

        # Regular conversation flow (without CoT)
        elif st.session_state.conversation:
            with st.spinner("Thinking..."):
                # Execute query
                response = st.session_state.conversation({"question": prompt})
                answer = response["answer"]

                # Display assistant response
                st.chat_message("assistant", avatar="🤖").write(answer)

                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.warning("Please build or load a vector database to start chatting!")