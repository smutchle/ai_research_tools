import streamlit as st
import os
from dotenv import load_dotenv
import json
import datetime
import base64
import glob # Import glob for file listing
import time
import shutil
import chromadb
from chromadb.config import Settings
from chromadb.errors import InvalidCollectionException # Import specific exception
import pandas as pd

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
    Extract metadata from a document using the LLM.
    Attempts to identify type (academic paper, code, data, etc.) and extract relevant fields.
    Returns a dictionary with metadata.
    """
    try:
        # Read the document content - limit to first few pages/lines for efficiency
        content_preview = ""
        if document_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(document_path)
            # Load only the first few pages for efficiency in metadata extraction
            doc_pages = loader.load()
            if not doc_pages:
                 print(f"Error loading content from {document_path}")
                 content_preview = "Could not load document content."
            else:
                content_preview = " ".join([p.page_content for p in doc_pages[:5]]) # Use first 5 pages
        elif document_path.lower().endswith(('.txt', '.md', '.qmd')):
            loader = TextLoader(document_path)
            doc = loader.load()[0]
            content_preview = doc.page_content
        elif document_path.lower().endswith('.ipynb'):
             loader = NotebookLoader(document_path)
             doc = loader.load()[0]
             content_preview = doc.page_content
        elif document_path.lower().endswith('.csv'):
             loader = CSVLoader(document_path)
             # Load first row or a small sample
             doc = loader.load()[0]
             content_preview = doc.page_content
        elif document_path.lower().endswith(('.json', '.jsonl')):
             # Assuming JSONLoader provides usable text content from a record
             loader = JSONLoader(file_path=document_path, jq_schema=".", text_content=False) # Load as dict/list
             try:
                 doc = loader.load()[0]
                 content_preview = str(doc.page_content) # Convert dict/list to string for preview
             except Exception as e:
                 print(f"Error loading/converting JSON content for preview {document_path}: {e}")
                 content_preview = "Could not load or process JSON content."
        else:
             print(f"Unsupported file type for metadata extraction preview: {document_path}")
             content_preview = "Unsupported file type."

        # Truncate content to a reasonable size for the LLM
        content_preview = content_preview[:10000]

        # Create prompt for metadata extraction
        prompt = f"""
        Analyze the following document content preview to determine its type (e.g., Academic Paper, Report, Code, Data, General Text, Image Description, etc.).
        Then, extract relevant metadata based on the detected type.

        If it's an academic paper, report, or similar document, extract:
        - Author(s)
        - Title
        - Year
        - Journal/Conference/Publication (if applicable)
        - Full APA reference (if applicable)
        - Abstract (if applicable)

        If it's Code, Data, or another type, provide a brief description.

        Return the extracted metadata in a strict JSON format without any additional text. Structure it like this:
        {{
            "file_type_detected": "e.g., Academic Paper",
            "author": "Author name(s) or N/A",
            "title": "Document title or N/A",
            "year": "Publication year or N/A",
            "journal": "Journal or conference name or N/A",
            "apa_reference": "Full APA style reference or N/A",
            "abstract": "Document abstract or N/A",
            "description": "Brief description if not an academic paper or N/A"
        }}

        If any field cannot be extracted or is not applicable based on the detected file type, use "N/A" as the value. Ensure the JSON is valid and contains *only* the JSON object.

        Here's the beginning of the document content:
        {content_preview}
        """

        # print("metadata prompt: ", prompt, "\n\n") # Debug print

        # Query the LLM for metadata extraction
        response = llm.invoke(prompt)

        # print("raw response for metadata: ", response, "\n\n") # Debug print

        # Extract JSON from the response
        metadata_str = extract_markdown_content(response.content, "json")

        # print("extracted JSON: ", metadata_str, "\n\n") # Debug print

        # Fallback if markdown extraction fails
        if not metadata_str:
            metadata_str = response.content
            # Attempt to find the first and last curly brace as a last resort
            json_start = metadata_str.find('{')
            json_end = metadata_str.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                metadata_str = metadata_str[json_start : json_end + 1]
            else:
                 st.warning(f"Could not find valid JSON format for metadata from {os.path.basename(document_path)}. Attempting manual parsing or using defaults.")
                 metadata_str = "{}" # Provide empty JSON to trigger robust default handling


        try:
            metadata = json.loads(metadata_str)
            if not isinstance(metadata, dict):
                 st.warning(f"LLM response for {os.path.basename(document_path)} was not a JSON object. Using defaults.")
                 metadata = {} # Use empty dict if not a dict

            # Ensure required keys exist with default "N/A" if missing
            required_keys = ["file_type_detected", "author", "title", "year", "journal", "apa_reference", "abstract", "description"]
            for key in required_keys:
                if key not in metadata:
                    metadata[key] = "N/A"

            # Use filename if title is N/A
            if metadata.get("title", "N/A") == "N/A" or not metadata.get("title"): # Check for empty string too
                 metadata["title"] = os.path.basename(document_path)

             # Generate basic APA if N/A and title is available
            if metadata.get("apa_reference", "N/A") == "N/A" and metadata["title"] != "N/A":
                 year = metadata.get("year", "Unknown")
                 if year == "N/A": year = "Unknown"
                 author = metadata.get("author", "Unknown")
                 if author == "N/A": author = "Unknown"
                 journal = metadata.get("journal", "Unknown")
                 if journal == "N/A": journal = "Unknown"

                 # Basic APA attempt
                 if author != "Unknown":
                     metadata["apa_reference"] = f"{author}. ({year}). {metadata['title']}. {journal}."
                 else:
                    metadata["apa_reference"] = f"Unknown. ({year}). {metadata['title']}. {journal}."


            return metadata

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON metadata from LLM response for {document_path}: {e}")
            print(f"Faulty JSON string: {metadata_str}")
             # Fallback to default metadata if JSON parsing fails
            return {
                "file_type_detected": "Unknown",
                "author": "Unknown",
                "title": os.path.basename(document_path),
                "year": "Unknown",
                "journal": "Unknown",
                "apa_reference": f"Unknown. ({datetime.datetime.now().year}). {os.path.basename(document_path)}.",
                "abstract": "No abstract available (JSON parse error)",
                "description": "Metadata extraction failed due to JSON error."
            }

    except Exception as e:
        print(f"Unexpected error extracting metadata from {document_path}: {str(e)}")
        # Return default metadata if any exception occurs
        return {
            "file_type_detected": "Unknown",
            "author": "Unknown",
            "title": os.path.basename(document_path),
            "year": "Unknown",
            "journal": "Unknown",
            "apa_reference": f"Unknown. ({datetime.datetime.now().year}). {os.path.basename(document_path)}.",
            "abstract": "No abstract available (error during processing)",
            "description": "Metadata extraction failed."
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
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return json_path
    except Exception as e:
        st.warning(f"Error saving metadata to {json_path}: {e}")
        return None


def load_metadata_json(source_path, metadata_dir):
     """
     Load metadata from a JSON file in the metadata directory.
     Returns metadata dict or None if file doesn't exist or is invalid.
     """
     base_name = os.path.splitext(os.path.basename(source_path))[0]
     json_path = os.path.join(metadata_dir, f"{base_name}.json")

     if os.path.exists(json_path):
         try:
             with open(json_path, 'r', encoding='utf-8') as f:
                 metadata = json.load(f)
                 # Basic validation
                 if isinstance(metadata, dict):
                     return metadata
                 else:
                     st.warning(f"Metadata file {json_path} is not a valid JSON object.")
                     return None
         except json.JSONDecodeError as e:
             st.warning(f"Error decoding JSON from {json_path}: {e}. Metadata may be corrupt.")
             return None
         except Exception as e:
             st.warning(f"Unexpected error reading metadata from {json_path}: {e}.")
             return None
     return None


def prepend_reference_to_chunks(chunks, metadata):
    """
    Prepend relevant context information (APA reference, title, or description)
    to each chunk from the document's metadata.
    """
    context_info = ""
    if metadata.get('apa_reference', 'N/A') != "N/A":
        context_info = f"Reference: {metadata['apa_reference']}"
    elif metadata.get('title', 'N/A') != "N/A":
        context_info = f"Document Title: {metadata['title']}"
    elif metadata.get('description', 'N/A') != "N/A":
         context_info = f"Document Description: {metadata['description']}"
    elif metadata.get('source'):
         context_info = f"Source File: {os.path.basename(metadata['source'])}"
    else:
        context_info = "Source: Unknown Document"


    if context_info:
        # Add the context info at the beginning of each chunk's page_content
        # Using a clear separator
        for chunk in chunks:
            chunk.page_content = f"{context_info}\n\n---\n\n{chunk.page_content}"

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
    try:
        if embedding_type == "Ollama":
            if not ollama_base_url:
                st.error("Ollama Base URL is not set.")
                return None
            return OllamaEmbeddings(
                base_url=ollama_base_url,
                model=embedding_model
            )
        elif embedding_type == "OpenAI":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("OpenAI API key is not set in environment variables.")
                return None
            return OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=api_key
            )
        elif embedding_type == "Google":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                 st.error("Google API key is not set in environment variables.")
                 return None
            return GoogleGenerativeAIEmbeddings(
                model=embedding_model,
                google_api_key=api_key
            )
        else:
            st.error(f"Unsupported embedding type: {embedding_type}")
            return None
    except Exception as e:
        st.error(f"Error creating embedding model {embedding_model} ({embedding_type}): {e}")
        return None


def create_or_update_vector_store(docs_dir, db_path, embedding_type, embedding_model, model_type, model_name, ollama_base_url=None, llm_temperature=0.2):
    """Create or update a vector store from documents in the specified directory"""
    st.sidebar.write("---")
    st.sidebar.write("🛠️ Create/Update Vector DB button pressed...")
    st.sidebar.write("Scanning documents and checking status...")

    metadata_dir = os.path.join(docs_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True) # Ensure metadata directory exists

    # Create the selected LLM for metadata extraction (using the new temperature)
    metadata_llm = create_llm(model_type, model_name, ollama_base_url if model_type == "Ollama" else None, llm_temperature)
    if metadata_llm is None:
        st.sidebar.error("Failed to initialize LLM for metadata extraction. Aborting DB process.")
        return None

    st.sidebar.write(f"Using {model_type} model: {model_name} (temp={llm_temperature}) for metadata extraction.")

    # Create embeddings
    embeddings = create_embeddings(embedding_type, embedding_model, ollama_base_url)
    if embeddings is None:
         st.sidebar.error("Failed to initialize embeddings model. Aborting DB process.")
         return None

    collection_name = "knowledge_docs"
    vector_store = None
    db_needs_full_recreate = False # Flag if we need to start from scratch

    # --- Attempt to load existing DB ---
    try:
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        # Check if the collection exists. Use get_collection which is the standard way post-v0.6
        # Catch InvalidCollectionException if it doesn't exist.
        try:
             collection = client.get_collection(collection_name)
             vector_store = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings,
                client=client,
                collection_name=collection_name
            )
             st.sidebar.write(f"Existing database found and collection '{collection_name}' loaded.")
             db_exists = True
             try:
                 count = collection.count()
                 st.sidebar.write(f"Collection contains {count} existing items.")
             except Exception as e:
                 st.warning(f"Could not retrieve count from existing collection: {e}")


        except InvalidCollectionException:
            st.sidebar.write(f"Database directory exists at '{db_path}', but collection '{collection_name}' not found. Will create new collection.")
            db_exists = False # Treat as if DB didn't exist
            db_needs_full_recreate = True # Indicate we need to create the collection/DB
            # No need to delete directory here, Chroma client will create the collection.

    except Exception as e:
        st.sidebar.warning(f"Error connecting to or loading database at '{db_path}': {e}. Will attempt to create a new database.")
        db_exists = False
        db_needs_full_recreate = True # Indicate we need to create the DB from scratch
        shutil.rmtree(db_path, ignore_errors=True) # Clean up potential corrupt DB
        os.makedirs(db_path, exist_ok=True)


    # --- Identify documents to process ---
    # Get list of all potential source files (exclude metadata directory itself)
    # Use glob to find all files, then filter out the metadata directory
    all_source_files = [f for f in glob.glob(os.path.join(docs_dir, "**/*"), recursive=True) if os.path.isfile(f)]
    all_source_files = [f for f in all_source_files if not os.path.normpath(f).startswith(os.path.normpath(metadata_dir))] # Exclude files inside metadata_dir

    st.sidebar.write(f"Found {len(all_source_files)} potential source documents in '{docs_dir}'.")

    docs_to_add_paths = [] # Paths of documents that need to be added

    # Determine which documents need adding
    for source_path in all_source_files:
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        json_path = os.path.join(metadata_dir, f"{base_name}.json")

        # If DB needs full recreate, process ALL found documents
        if db_needs_full_recreate:
             st.sidebar.write(f"✅ Marking {os.path.basename(source_path)} for addition (New DB or Collection).")
             docs_to_add_paths.append(source_path)
        else:
            # In update mode (DB/Collection exists), check for existing metadata file
            if os.path.exists(json_path):
                # Document metadata exists, assume it's already in the DB
                st.sidebar.write(f"⏭️ Skipping {os.path.basename(source_path)}: Metadata exists at {os.path.basename(json_path)}.")
            else:
                # Metadata does not exist, this document needs to be added
                st.sidebar.write(f"✅ Marking {os.path.basename(source_path)} for addition (Missing metadata).")
                docs_to_add_paths.append(source_path)

    if not docs_to_add_paths:
        if db_exists:
            st.sidebar.write("No new documents found to add to the existing database.")
            return vector_store # Return the loaded existing DB
        else:
             st.error("No documents found or marked for processing. Cannot create a database.")
             return None

    st.sidebar.write(f"Processing {len(docs_to_add_paths)} documents for addition.")

    # --- Process documents to be added ---
    processed_docs_for_addition = []
    for source_path in docs_to_add_paths:
        st.sidebar.write(f"Processing: {os.path.basename(source_path)}")
        try:
            loader = get_document_loader(source_path)
            if loader is None:
                st.sidebar.warning(f"Unsupported file type, skipping: {os.path.basename(source_path)}")
                continue

            documents = loader.load()
            if not documents:
                 st.sidebar.warning(f"Loader returned no documents for {os.path.basename(source_path)}, skipping.")
                 continue

            doc = documents[0] # Assume one Langchain Document object per file

            # Extract metadata
            metadata = extract_document_metadata(source_path, metadata_llm)

            # Save metadata to JSON
            json_path = save_metadata_json(metadata, source_path, metadata_dir)
            if json_path:
                st.sidebar.write(f"Saved metadata for {os.path.basename(source_path)}")
            else:
                 st.sidebar.warning(f"Failed to save metadata for {os.path.basename(source_path)}. Proceeding without saved metadata.")


            # Add metadata to document's metadata (ensuring 'source' is included)
            if 'source' not in doc.metadata:
                 doc.metadata['source'] = source_path
            doc.metadata.update(metadata)
            processed_docs_for_addition.append(doc)

        except Exception as e:
            st.sidebar.error(f"Error processing document {os.path.basename(source_path)}: {str(e)}. Skipping.")
            # st.error(f"Failed to process {os.path.basename(source_path)}: {str(e)}") # Avoid cluttering main screen with errors during DB build

    if not processed_docs_for_addition:
        if not db_exists:
            st.error("No documents were successfully processed to create a new database.")
            return None
        else:
             st.sidebar.write("No *new* documents were successfully processed to add to the database.")
             return vector_store # Return existing DB if no new docs could be processed

    # --- Split documents into chunks ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=st.session_state.chunk_size,
        chunk_overlap=st.session_state.chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""]  # Prioritize splitting by paragraph, then newline, then space
    )

    # Process documents one by one to apply the context prepending and splitting
    chunks_to_add = []
    st.sidebar.write("Splitting documents into chunks...")
    for doc in processed_docs_for_addition:
        try:
            # Split the individual document
            doc_chunks = text_splitter.split_documents([doc])

            # Get the metadata for this document
            doc_metadata = doc.metadata

            # Prepend reference/context to each chunk
            doc_chunks = prepend_reference_to_chunks(doc_chunks, doc_metadata)

            chunks_to_add.extend(doc_chunks)
            st.sidebar.write(f"- Split '{os.path.basename(doc.metadata.get('source', 'Unknown'))}' into {len(doc_chunks)} chunks.")
        except Exception as e:
            st.sidebar.error(f"Error splitting or prepping chunks for {os.path.basename(doc.metadata.get('source', 'Unknown'))}: {e}. Skipping chunks for this document.")


    if not chunks_to_add:
        if not db_exists:
            st.error("No text chunks were generated to create a new database.")
            return None
        else:
            st.sidebar.write("No text chunks were generated from new documents to add to the database.")
            return vector_store # Return existing DB if no new chunks

    # --- Add the chunks to the vector store ---
    batch_size = 100 # Process in batches to avoid potential limits
    total_chunks = len(chunks_to_add)
    total_batches = (total_chunks + batch_size - 1) // batch_size

    if db_needs_full_recreate:
        # Create a new vector store from all chunks
        st.sidebar.write(f"Creating a new database and collection '{collection_name}' with {total_chunks} chunks...")
        try:
            vector_store = Chroma.from_documents(
                documents=chunks_to_add, # Use all chunks
                embedding=embeddings,
                persist_directory=db_path,
                collection_name=collection_name
            )
            st.sidebar.write("✅ New vector store created successfully!")
            return vector_store
        except Exception as e:
            st.error(f"Error creating new vector store: {str(e)}")
            st.sidebar.error(f"Error creating new vector store: {str(e)}")
            shutil.rmtree(db_path, ignore_errors=True) # Clean up failed creation
            return None

    elif db_exists and vector_store:
        # Add documents to the existing vector store
        st.sidebar.write(f"Adding {total_chunks} new chunks to the existing database...")
        for i in range(0, total_chunks, batch_size):
            batch_num = (i // batch_size) + 1
            end_idx = min(i + batch_size, total_chunks)
            batch = chunks_to_add[i:end_idx]

            st.sidebar.write(f"Adding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

            try:
                vector_store.add_documents(documents=batch)
                st.sidebar.write(f"Batch {batch_num} completed.")
            except Exception as e:
                 st.sidebar.error(f"Error adding batch {batch_num}: {e}. Attempting to continue with next batch.")
                 # Decide how to handle errors here - continue, break, etc.
                 # For now, we'll just log and continue
                 pass

        st.sidebar.write("✅ New documents added to the existing vector store!")
        # After adding, re-instantiate Chroma to ensure it reflects the new state?
        # Langchain's Chroma should handle this, but explicitly reloading might be safer depending on impl.
        # Let's try reloading the Chroma wrapper instance
        try:
             client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
             vector_store = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings,
                client=client,
                collection_name=collection_name
            )
             st.sidebar.write("Re-loaded vector store instance after adding documents.")
        except Exception as e:
             st.sidebar.warning(f"Could not re-load vector store instance after adding documents: {e}")
             # Keep the potentially old vector_store instance if reload fails


        return vector_store

    else:
         # This case should ideally not be reached if the logic is correct
         st.error("Unexpected state in create_or_update_vector_store. Aborting.")
         return None


def load_vector_store(db_path, embedding_type, embedding_model, ollama_base_url=None):
    """Load an existing vector store"""
    st.sidebar.write("Attempting to load existing vector database...")
    try:
        # Create embeddings
        embeddings = create_embeddings(embedding_type, embedding_model, ollama_base_url)
        if embeddings is None:
             st.sidebar.error("Failed to create embeddings model for loading.")
             return None # Return None if embeddings creation failed

        # Initialize Chroma client
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True # Allows clearing the database programmatically if needed
            )
        )

        # Check if the collection exists using get_collection
        collection_name = "knowledge_docs"
        try:
             collection = client.get_collection(collection_name)
             st.sidebar.write(f"Chroma collection '{collection_name}' found.")
        except InvalidCollectionException:
             st.warning(f"Vector database directory exists at `{db_path}`, but the collection '{collection_name}' was not found. Please rebuild the database.")
             st.sidebar.warning(f"Chroma collection '{collection_name}' not found.")
             return None
        except Exception as e:
             st.error(f"Error accessing Chroma collection '{collection_name}': {e}. Database might be corrupt. Please rebuild.")
             st.sidebar.error(f"Error accessing Chroma collection: {e}")
             return None


        vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings,
            client=client,
            collection_name=collection_name
        )
        # Optional: Check if the collection is empty
        try:
            count = collection.count() # Use the collection object retrieved earlier
            if count == 0:
                 st.warning(f"Vector database collection '{collection_name}' is empty. Please rebuild the database.")
                 st.sidebar.warning(f"Collection '{collection_name}' is empty.")
                 return None
            else:
                 st.sidebar.write(f"Loaded database with {count} items.")
        except Exception as e:
            st.warning(f"Could not get item count from database collection: {e}. Database might need rebuilding.")
            st.sidebar.warning(f"Could not get item count: {e}")
            # Continue loading anyway, maybe count failed but DB is usable
            pass # Or return None if count is critical

        st.sidebar.write("✅ Existing vector database loaded successfully!")
        return vector_store
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        st.sidebar.error(f"Error loading vector store: {str(e)}")
        return None

def create_llm(model_type, model_name, ollama_base_url=None, temperature=0.2):
    """Create the appropriate LLM based on model type, name, and temperature"""
    try:
        if model_type == "Ollama":
            if not ollama_base_url:
                # st.error("Ollama Base URL is not set.") # Avoid spamming errors if Ollama not intended provider
                return None
            return ChatOllama(
                base_url=ollama_base_url,
                model=model_name,
                temperature=temperature
            )
        elif model_type == "OpenAI":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                # st.error("OpenAI API key is not set in environment variables.")
                return None
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_key=api_key
            )
        elif model_type == "Anthropic":
            api_key = os.getenv("CLAUDE_API_KEY")
            if not api_key:
                 # st.error("Anthropic API key is not set in environment variables.")
                 return None
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
                anthropic_api_key=api_key
            )
        elif model_type == "Google":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                 # st.error("Google API key is not set in environment variables.")
                 return None
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=api_key
            )
        else:
            st.error(f"Unsupported model type: {model_type}")
            return None
    except Exception as e:
         # st.error(f"Error creating LLM {model_name} ({model_type}): {e}") # Avoid spamming errors
         print(f"Error creating LLM {model_name} ({model_type}): {e}") # Log to console
         return None

def initialize_conversation(vector_store, model_type, model_name, k_value=10, ollama_base_url=None, use_reranking=False, num_chunks_kept=4, llm_temperature=0.2):
    """Initialize the conversation chain"""
    st.sidebar.write("Initializing conversation chain...")
    try:
        if vector_store is None:
            st.warning("Cannot initialize conversation: Vector store is not loaded.")
            st.sidebar.warning("Cannot initialize conversation: Vector store is None.")
            return None, None, None

        # Create the LLM using the provided temperature
        llm = create_llm(model_type, model_name, ollama_base_url, llm_temperature)
        if llm is None:
             st.warning(f"Cannot initialize conversation: LLM creation failed for {model_name} ({model_type}). Check API keys/endpoints.")
             st.sidebar.error(f"Cannot initialize conversation: LLM creation failed for {model_name} ({model_type}).")
             return None, None, None

        st.sidebar.write(f"LLM ({model_name}, Temp={llm_temperature}) created.")

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": k_value})
        st.sidebar.write(f"Retriever created (k={k_value}).")

        # Reranking with ContextualCompressionRetriever
        if use_reranking:
            st.sidebar.write(f"Using LLM Reranking, keeping top {num_chunks_kept} chunks...")
            # Ensure num_chunks_kept is not greater than k_value
            num_chunks_kept = min(num_chunks_kept, k_value)
            try:
                compressor = LLMChainExtractor.from_llm(llm)
                retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever, k=num_chunks_kept)
                st.sidebar.write("Reranking retriever created.")
            except Exception as e:
                st.sidebar.error(f"Failed to create Reranking retriever: {e}. Proceeding without reranking.")
                st.warning(f"Failed to set up LLM Reranking: {e}. Reranking disabled.")
                use_reranking = False # Fallback


        # Make the chain verbose so we can see prompts in stdout
        conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=True
        )

        st.sidebar.write("✅ Conversation chain initialized.")
        return conversation, llm, retriever

    except Exception as e:
        st.error(f"Error initializing conversation: {str(e)}")
        st.sidebar.error(f"Error initializing conversation: {str(e)}")
        return None, None, None

def extract_markdown_content(text: str, type: str = "json") -> str:
        """Extract content from markdown code blocks."""
        start_tag = f"```{type}"
        end_tag = """```"""

        start_idx = text.find(start_tag)
        if start_idx == -1:
             return text.strip() # Return original text if no start tag

        start_idx += len(start_tag)
        # Handle potential newline immediately after start_tag
        if text[start_idx] == '\n':
             start_idx += 1

        end_idx = text.rfind(end_tag, start_idx) # Search for end tag after start

        if end_idx != -1:
            return (text[start_idx:end_idx]).strip()
        else:
             return text[start_idx:].strip() # Return from start tag to end if no end tag


# Function to execute chain of thought reasoning with preserved context
# (Keep this function as is, it uses the llm passed to it)
def execute_chain_of_thought(llm, retriever, prompt, max_steps=5):
    """Executes a chain of thought reasoning process using the LLM and retriever."""
    st.sidebar.write("Starting Chain of Thought process...")
    # Initial prompt to get chain of thought planning
    # Ensure prompt is adapted for the specific LLM capabilities and instruction following
    cot_system_prompt = """
    You are a helpful assistant that thinks through problems step by step.
    Given the user's question, create a plan to answer it with clear reasoning steps.
    If you output equations, use LaTex format.
    Output your reasoning plan in the following strict JSON format, including only the JSON object:
    ```json
    {{
        "reasoning_steps": [
            "Step 1: ...",
            "Step 2: ...",
            "Step 3: ...",
            "Step 4: ..."
        ],
        "total_steps": 4
    }}
    ```
    Use as many steps as needed for the problem, but aim for conciseness. Ensure the JSON is strictly formatted and is the only output.
    """

    # Get the relevant documents
    st.sidebar.write("Retrieving documents for Chain of Thought context...")
    try:
        retrieved_docs = retriever.get_relevant_documents(prompt)
        context = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in retrieved_docs])
        if not retrieved_docs:
             st.sidebar.write("No documents retrieved for context.")
             context = "No relevant documents found."
        else:
             st.sidebar.write(f"Retrieved {len(retrieved_docs)} documents for context.")

    except Exception as e:
        st.sidebar.error(f"Error retrieving documents for CoT: {e}")
        context = "Could not retrieve context documents due to an error."


    # Initial prompt with context
    initial_prompt = f"""
    System: {cot_system_prompt}

    Context information relevant to the question:
    ---
    {context}
    ---

    User question: {prompt}

    Based on the user question and the context, create a step-by-step reasoning plan in the specified JSON format.
    """

    # Get the plan
    st.sidebar.write("Generating Chain of Thought plan...")
    plan_response = llm.invoke(initial_prompt)
    plan_text = plan_response.content

    # Extract JSON plan
    try:
        json_str = extract_markdown_content(plan_text, "json")
        # print("plan:", json_str) # Debug print
        plan_json = json.loads(json_str)
        steps = plan_json.get("reasoning_steps", [])
        if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
             st.warning("LLM returned invalid plan format. Using default steps.")
             raise ValueError("Invalid plan format") # Trigger fallback

    except Exception as e:
        print(f"Error extracting/parsing JSON plan: {e}. LLM Output (first 500 chars): {plan_text[:500]}...") # Debug print
        st.warning("Failed to extract valid plan from LLM. Using default steps.")
        # If JSON parsing fails, create a default plan
        steps = [
            "Step 1: Analyze the user's question and identify the core concepts and information required.",
            "Step 2: Review the provided context information to find relevant facts, data, or descriptions related to the question.",
            "Step 3: Synthesize information from the context and the question, combining relevant pieces and performing any necessary analysis or calculations.",
            "Step 4: Formulate a clear and comprehensive final answer based on the synthesis, directly addressing the user's question.",
        ]


    total_steps = min(len(steps), max_steps)  # Limit to max_steps

    st.sidebar.write(f"Executing {total_steps} reasoning steps.")

    # Create progress bar in the main area
    progress_bar = st.progress(0, text=f"Starting Chain of Thought process...")
    step_output_display = [] # List to store step results for display/final context

    # Initialize the growing context that will accumulate through steps
    # Start with the original context from document retrieval
    growing_context = f"Original Document Context:\n---\n{context}\n---\n\n"

    # Execute each step
    for i in range(total_steps):
        # Update progress
        progress = (i) / total_steps
        progress_bar.progress(progress, text=f"Executing Step {i+1}/{total_steps}: {steps[i][:80]}...") # Show part of step description

        current_step_description = steps[i]

        # Create prompt for this step with the growing context
        # The prompt for each step should ask the LLM to *execute* the step,
        # building upon the previous steps' reasoning and the original context.
        step_prompt_template = """
        You are executing step {current_step_number} of a chain of thought process to answer the user's question.
        Your goal is to perform the task described in "Current step to execute" using the available information.

        Original User Question: {user_question}

        Accumulated Information (Original context + reasoning from previous steps):
        ---
        {accumulated_context}
        ---

        Current step to execute ({current_step_number}/{total_steps}): {step_description}

        Based on the "Accumulated Information" and the "Original User Question", execute this single step.
        Provide your detailed reasoning and results specifically for THIS step. Your output for this step will be added to the "Accumulated Information" for the next step.
        Focus only on executing the current step and providing the reasoning for *that* step. Do not try to answer the full question yet.
        """

        step_prompt = step_prompt_template.format(
            current_step_number=i+1,
            total_steps=total_steps,
            user_question=prompt,
            accumulated_context=growing_context,
            step_description=current_step_description
        )

        # Get response for this step
        try:
            step_response = llm.invoke(step_prompt)
            step_result = step_response.content
        except Exception as e:
            step_result = f"Error executing step: {e}"
            st.warning(f"Error in CoT step {i+1}: {e}")
            st.sidebar.warning(f"Error in CoT step {i+1}: {e}")


        # Add to outputs list (for summary/display later)
        step_output_display.append(f"Step {i+1} ({current_step_description}): {step_result}")

        # Add this step's reasoning to the growing context for the *next* step
        growing_context += f"Reasoning for Step {i+1} ({current_step_description}):\n{step_result}\n\n---\n\n"

        # Display intermediate step in an expander
        with st.expander(f"Step {i+1}: {current_step_description}", expanded=False):
            st.write(step_result)

        # Small delay to show progress visually (optional)
        # time.sleep(0.1) # Reduced delay

    # Final step - synthesize all reasoning into an answer
    progress_bar.progress(1.0, text="Synthesizing final answer...")

    # Create prompt for final answer using the complete growing context
    final_answer_prompt = f"""
    System: You have completed a chain of thought process to answer the user's question, accumulating information and reasoning through several steps.
    Your task now is to synthesize all the information and reasoning from the steps you just executed into a single, coherent, and comprehensive final answer.

    Original User Question: {prompt}

    Complete Accumulated Information and Reasoning:
    ---
    {growing_context}
    ---

    Based on the "Complete Accumulated Information and Reasoning", provide your final, comprehensive answer to the "Original User Question".
    Present the answer clearly, logically, and directly address the user's query. Do not include the step-by-step reasoning in the final answer, only the result.
    """

    # Get final answer
    try:
        final_response = llm.invoke(final_answer_prompt)
        final_answer = final_response.content
        st.sidebar.write("✅ Chain of Thought process completed.")
    except Exception as e:
        final_answer = f"Error generating final answer after steps: {e}"
        st.error(final_answer)
        st.sidebar.error(final_answer)
        # As a fallback, just concatenate step outputs
        # final_answer = "Could not synthesize final answer. Here are the step results:\n\n" + "\n\n".join(step_output_display)


    progress_bar.empty() # Clear the progress bar

    # Note: We return step_output_display which contains descriptions + results for display,
    # but the final_answer is generated from the `growing_context` which includes all reasoning.
    return final_answer, step_output_display


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
            qmd_content += f"**User:**\n\n{content}\n\n" # Added markdown bold and newlines
        else:  # assistant
            qmd_content += f"\n## Assistant Response {i//2 + 1}\n\n"
            qmd_content += f"**Assistant:**\n\n{content}\n\n" # Added markdown bold and newlines

    return qmd_content

def get_download_link(qmd_content, filename="rag_chatbot_conversation.qmd"):
    """
    Generate a download link for the QMD content
    """
    # Encode the content as bytes, then base64
    b64 = base64.b64encode(qmd_content.encode('utf-8')).decode()
    href = f'<a href="data:text/markdown;base64,{b64}" download="{filename}">Download Conversation as Quarto Document</a>'
    return href

# Helper to get document loader based on file extension
def get_document_loader(file_path):
    """Returns the appropriate Langchain loader for a given file path or None if unsupported."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            return PyPDFLoader(file_path)
        elif ext == ".txt" or ext == ".md" or ext == ".qmd":
            return TextLoader(file_path)
        elif ext == ".json" or ext == ".jsonl":
            # Note: JSONLoader requires a jq_schema. "." loads the root object.
            # This might load the whole file as one "document".
            # For line-delimited JSONL, need different approach or jq schema for each line.
            # This loader might need customization based on JSON structure.
             st.sidebar.warning(f"Using generic JSONLoader for {os.path.basename(file_path)}. May need custom schema.")
             return JSONLoader(file_path=file_path, jq_schema=".", text_content=False) # text_content=False loads dict/list
        elif ext == ".csv":
            return CSVLoader(file_path)
        elif ext == ".ipynb":
            return NotebookLoader(file_path)
        else:
            return None # Indicate unsupported file type
    except Exception as e:
         st.sidebar.error(f"Error creating loader for {os.path.basename(file_path)} (ext: {ext}): {e}")
         return None


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
# Settings with defaults and persistence
if 'k_value' not in st.session_state:
    st.session_state.k_value = 10
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = 1000
if 'chunk_overlap' not in st.session_state:
    st.session_state.chunk_overlap = 200
if 'use_reranking' not in st.session_state:
    st.session_state.use_reranking = True
if 'num_chunks_kept' not in st.session_state:
    st.session_state.num_chunks_kept = 4
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.2 # Default temperature as requested

# Title and description
st.title("🤖 RAG Chatbot")
st.markdown("Ask questions about your documents.")

# Sidebar for configuration
with st.sidebar:
    # Document retrieval settings
    st.subheader("Vector DB Settings")

    # Input for documents directory
    docs_dir = st.text_input("Documents Directory", os.getenv("SOURCE_DOC_DIR", "docs"))
    # Ensure docs_dir is created if it doesn't exist
    if docs_dir:
        try:
            os.makedirs(docs_dir, exist_ok=True)
            # st.info(f"Using documents directory: {docs_dir}") # Avoid spamming
        except OSError as e:
            st.error(f"Error creating/accessing documents directory {docs_dir}: {e}")


    # Chunk size slider
    chunk_size = st.slider(
        "Chunk Size",
        min_value=100, # Smaller min for diverse docs
        max_value=8000,
        value=st.session_state.chunk_size, # Use session state default/current
        step=100,
        help="Size of text chunks when processing documents (in characters)"
    )
    st.session_state.chunk_size = chunk_size

    # Chunk overlap slider
    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=1000,
        value=st.session_state.chunk_overlap, # Use session state default/current
        step=50,
        help="Amount of overlap between consecutive chunks (in characters)"
    )
    st.session_state.chunk_overlap = chunk_overlap

    # Embedding model selection
    embedding_providers_str = os.getenv("EMBEDDING_PROVIDERS", "Ollama,OpenAI,Google")
    embedding_type = st.selectbox(
        "Select Embedding Model Provider",
        split_csv(embedding_providers_str),
        index=0
    )

    embedding_model = "nomic-embed-text:latest" # Default placeholder, updated below
    if embedding_type == "Ollama":
        embedding_model = st.text_input("Ollama Embedding Model", os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest"))
    elif embedding_type == "OpenAI":
        embedding_model = st.text_input("OpenAI Embedding Model", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    elif embedding_type == "Google":
        embedding_model = st.text_input("Google Embedding Model", os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001"))

    db_path = os.path.join(docs_dir, "vectorstore")
    # print("Using: ", db_path, " for vector database...") # Keep this for server logs


    st.subheader("LLM Settings")

    # Temperature Slider
    temperature = st.slider(
        "LLM Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature, # Use session state default/current
        step=0.05,
        help="Controls randomness in LLM responses. Lower = more deterministic, Higher = more creative/varied. Default is 0.2."
    )
    st.session_state.temperature = temperature


    # Number of retrieved documents
    k_value = st.slider(
        "Number of retrieved documents (K)",
        min_value=1,
        max_value=100,
        value=st.session_state.k_value, # Use session state default/current
        step=1,
        help="Controls how many relevant documents to retrieve for each query *before* optional reranking."
    )
    st.session_state.k_value = k_value

     # Add LLM Reranking option
    use_reranking = st.checkbox("Use LLM Reranking", value=st.session_state.use_reranking)
    st.session_state.use_reranking = use_reranking

     # Number of chunks kept for reranking
    # Only show if reranking is enabled
    if use_reranking:
        num_chunks_kept = st.slider(
            "Number of Chunks Kept After Reranking",
            min_value=1,
            max_value=st.session_state.k_value,  # Limit to the number of retrieved documents
            value=st.session_state.num_chunks_kept, # Use session state default/current
            step=1,
            help="Number of chunks to keep after LLM reranking. Must be less than or equal to the number of retrieved documents (K)."
        )
        st.session_state.num_chunks_kept = num_chunks_kept
    else:
         # If reranking is off, effectively all K chunks are kept
         # We might set this state variable for clarity or just use k_value directly where needed
         # Let's set it for consistency, though it's only used if use_reranking is True
         st.session_state.num_chunks_kept = st.session_state.k_value


    # Model selection
    llm_providers_str = os.getenv("LLM_PROVIDERS", "Ollama,OpenAI,Anthropic,Google")
    model_type = st.selectbox(
        "Select LLM Provider",
        split_csv(llm_providers_str),
        index=0
    )

    # Model selection based on provider
    model_name = "" # Default placeholder
    ollama_base_url = None # Default to None

    if model_type == "Ollama":
        ollama_model_str = os.getenv("OLLAMA_MODEL", "llama3:latest,mistral:latest")
        model_name = st.selectbox(
            "Select Ollama Model",
            split_csv(ollama_model_str),
            index=0
        )
        ollama_base_url = st.text_input("Ollama Base URL", os.getenv("OLLAMA_END_POINT", "http://localhost:11434"))
    elif model_type == "OpenAI":
        openai_model_str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo,gpt-4o")
        model_name = st.selectbox(
            "Select OpenAI Model",
            split_csv(openai_model_str),
            index=0
        )
    elif model_type == "Anthropic":
        anthropic_model_str = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307,claude-3-sonnet-20240229")
        model_name = st.selectbox(
            "Select Anthropic Model",
            split_csv(anthropic_model_str),
            index=0
        )
    elif model_type == "Google":
        google_model_str = os.getenv("GOOGLE_MODEL", "gemini-pro")
        model_name = st.selectbox(
            "Select Google Model",
            split_csv(google_model_str),
            index=0
        )

    # Add Chain of Thought option
    use_cot = st.checkbox("Use Chain of Thought", value=False)


    st.markdown("---")

    # Database operations button
    build_db_button = st.button("🔨 Create/Update Vector DB")

    # Handle database building/updating
    if build_db_button:
        if not docs_dir:
             st.error("Please specify a documents directory.")
        elif not embedding_model:
             st.error("Please specify an embedding model.")
        elif model_type == "Ollama" and not ollama_base_url:
             st.error("Please specify the Ollama Base URL.")
        elif not model_name:
             st.error("Please select or specify an LLM model name.")
        else:
            with st.spinner("Processing documents and updating vector database..."):
                vector_store = create_or_update_vector_store(
                    docs_dir,
                    db_path,
                    embedding_type,
                    embedding_model,
                    model_type, # Pass model_type for metadata LLM
                    model_name, # Pass model_name for metadata LLM
                    ollama_base_url if model_type == "Ollama" else None,
                    st.session_state.temperature # Pass the current temperature
                )
                if vector_store:
                    st.session_state.vector_store = vector_store
                    # Initialize conversation with new/updated vector store and current settings
                    # This is important to reflect the new DB content and potentially changed settings
                    conversation, llm, retriever = initialize_conversation(
                        vector_store,
                        model_type,
                        model_name,
                        st.session_state.k_value,
                        ollama_base_url if model_type == "Ollama" else None,
                        st.session_state.use_reranking,
                        st.session_state.num_chunks_kept,
                        st.session_state.temperature # Pass the current temperature
                    )
                    st.session_state.conversation = conversation
                    st.session_state.llm = llm
                    st.session_state.retriever = retriever
                    if st.session_state.conversation:
                         st.sidebar.success("Conversation chain initialized with updated DB.")
                         # Clear chat history after DB update? Or keep? Let's keep for now.
                         # st.session_state.chat_history = []
                    else:
                         st.sidebar.error("Failed to initialize conversation chain after DB update.")
                else:
                    st.error("Failed to create or update vector database.")
                    st.session_state.vector_store = None
                    st.session_state.conversation = None
                    st.session_state.llm = None
                    st.session_state.retriever = None

    st.markdown("---") # Separator


    # Handle chat reset
    reset_chat = st.button("🔄 Reset Chat")
    if reset_chat:
        st.session_state.chat_history = []
        st.session_state.conversation = None # Clear conversation to force re-init
        st.session_state.llm = None
        st.session_state.retriever = None
        # If vector store exists, re-initialize the conversation chain
        if st.session_state.vector_store:
            st.sidebar.write("Resetting chat and re-initializing conversation chain...")
            conversation, llm, retriever = initialize_conversation(
                st.session_state.vector_store,
                model_type,
                model_name,
                st.session_state.k_value,
                ollama_base_url if model_type == "Ollama" else None,
                st.session_state.use_reranking,
                st.session_state.num_chunks_kept,
                st.session_state.temperature # Pass the current temperature
            )
            st.session_state.conversation = conversation
            st.session_state.llm = llm
            st.session_state.retriever = retriever
            if st.session_state.conversation:
                 st.sidebar.success("Conversation chain re-initialized.")
            else:
                 st.sidebar.error("Failed to re-initialize conversation chain.")
        st.rerun()


# --- Initial Load Logic ---
# This block runs on initial app load or after a reset if DB exists,
# but ONLY if the vector_store and conversation are NOT already in session state.
if docs_dir and os.path.exists(docs_dir): # Only attempt if docs_dir is set and exists
    db_path = os.path.join(docs_dir, "vectorstore")
    if os.path.exists(db_path) and not st.session_state.vector_store:
        # Attempt to load the DB if the directory exists and it's not already loaded
        with st.spinner("Loading existing vector database..."):
            vector_store = load_vector_store(
                db_path,
                embedding_type, # Use current selection
                embedding_model, # Use current selection
                ollama_base_url if model_type == "Ollama" else None # Use current selection
            )
            if vector_store:
                st.session_state.vector_store = vector_store
                # Initialize conversation with loaded vector store and current settings
                conversation, llm, retriever = initialize_conversation(
                    vector_store,
                    model_type, # Use current selection
                    model_name, # Use current selection
                    st.session_state.k_value, # Use current selection
                    ollama_base_url if model_type == "Ollama" else None, # Use current selection
                    st.session_state.use_reranking, # Use current selection
                    st.session_state.num_chunks_kept, # Use current selection
                    st.session_state.temperature # Pass the current temperature
                )
                st.session_state.conversation = conversation
                st.session_state.llm = llm
                st.session_state.retriever = retriever
                if st.session_state.conversation:
                     st.success("Existing vector database loaded and conversation initialized!")
                else:
                    st.warning("Existing database loaded, but conversation chain could not be initialized. Check LLM settings.")
            else:
                st.warning("Could not load existing vector database. Please check settings or build/rebuild it.")
# --- End Initial Load Logic ---


# Display API key status
with st.sidebar:
    st.markdown("---")
    st.subheader("Provider Status")

    openai_key = os.getenv("OPENAI_API_KEY")
    claude_key = os.getenv("CLAUDE_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    ollama_url = os.getenv("OLLAMA_END_POINT")

    if model_type == "OpenAI":
        if openai_key:
            st.success("OpenAI API Key: Loaded ✅")
        else:
            st.error("OpenAI API Key: Missing ❌ - Set OPENAI_API_KEY env var.")
    elif model_type == "Anthropic":
         if claude_key:
            st.success("Claude API Key: Loaded ✅")
         else:
            st.error("Claude API Key: Missing ❌ - Set CLAUDE_API_KEY env var.")
    elif model_type == "Google":
         if google_key:
            st.success("Google API Key: Loaded ✅")
         else:
            st.error("Google API Key: Missing ❌ - Set GOOGLE_API_KEY env var.")
    elif model_type == "Ollama":
         if ollama_url:
             st.success("Ollama Endpoint: Set ✅")
         else:
             st.warning("Ollama Endpoint: Not set ❌ - Set OLLAMA_END_POINT env var or enter manually.")

    # Embedding provider status (optional, can add similar checks)
    # if embedding_type == "OpenAI" and not openai_key:
    #     st.error("OpenAI Embedding requires API Key ❌")
    # elif embedding_type == "Google" and not google_key:
    #      st.error("Google Embedding requires API Key ❌")
    # elif embedding_type == "Ollama" and not ollama_url:
    #      st.warning("Ollama Embedding requires Endpoint ❌")


    # Add Quarto export button
    st.markdown("---")
    st.subheader("Export Conversation")

    if st.button("📥 Download Chat as Quarto (.qmd)"):
        if st.session_state.chat_history:
            qmd_content = convert_chat_to_qmd(st.session_state.chat_history)
            st.markdown(get_download_link(qmd_content), unsafe_allow_html=True)
            # st.success("Quarto document ready for download!") # Link is directly displayed
        else:
            st.warning("No chat history to export yet.")

# Add helpful information in the sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    ### Instructions
    1. Enter the path to your documents directory (`docs/` by default). Place your `.pdf`, `.txt`, `.md`, `.qmd`, `.json`, `.jsonl`, `.csv`, `.ipynb` files inside.
    2. Adjust the document processing settings (chunk size, overlap) if needed.
    3. Select your preferred Embedding and LLM providers and models.
    4. Adjust LLM Temperature (lower = more focused, higher = more creative). Default is 0.2.
    5. Click **'Create/Update Vector DB'**. This builds the database initially or adds new documents if the DB exists. Progress and skipped/added files are shown in the sidebar.
    6. Toggle 'Use Chain of Thought' for step-by-step reasoning (might be slower).
    7. Toggle 'Use LLM Reranking' to improve retrieved chunk quality.
    8. Click 'Reset Chat' to clear history and re-initialize the conversation with current settings.
    9. Ask questions about your documents in the chat input below! 🚀

    Supported file types: `.pdf`, `.txt`, `.md`, `.qmd`, `.json`, `.jsonl`, `.csv`, `.ipynb`.
    """)

# Main layout for chat
if st.session_state.vector_store and st.session_state.conversation:
    # Display chat messages
    for message in st.session_state.chat_history:
        avatar = "🧑" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"]) # Use markdown for message display

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.chat_message("user", avatar="🧑").markdown(prompt)

        # Add user message to chat history immediately
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
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(final_answer)

                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": final_answer})

        # Regular conversation flow (without CoT)
        elif st.session_state.conversation:
            with st.spinner("Thinking..."):
                try:
                    # Prepare chat history for the chain
                    # Langchain's ConversationalRetrievalChain expects chat_history
                    # as a list of tuples (human_message, ai_message).
                    # We need to convert our session state history.
                    langchain_history = []
                    # Iterate in steps of 2 (user, assistant pairs)
                    for i in range(0, len(st.session_state.chat_history) - 1, 2):
                         user_msg = st.session_state.chat_history[i]
                         ai_msg = st.session_state.chat_history[i+1]
                         if user_msg['role'] == 'user' and ai_msg['role'] == 'assistant':
                              langchain_history.append((user_msg['content'], ai_msg['content']))
                         # Handle potential incomplete pair (last user message) - it's the current prompt

                    # Execute query
                    response = st.session_state.conversation.invoke({"question": prompt, "chat_history": langchain_history})

                    answer = response["answer"]

                    # Display assistant response
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(answer)

                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"An error occurred during the conversation: {e}")
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown("Sorry, I encountered an error while processing your request.")
                    # Optional: Log the full traceback
                    import traceback
                    print(traceback.format_exc())


        else:
            st.warning("Conversation chain is not initialized. Please build/load the database first.")

else:
    # Message to display if vector store or conversation is not ready
    docs_dir_exists = os.path.exists(docs_dir) if docs_dir else False
    db_path = os.path.join(docs_dir, "vectorstore") if docs_dir else "path/to/vectorstore"
    db_dir_exists = os.path.exists(db_path) if docs_dir else False

    if not docs_dir_exists:
        st.info(f"Documents directory `{docs_dir}` not found. Please create the directory or enter a valid path in the sidebar.")
    elif not db_dir_exists:
         st.info(f"Vector database directory not found at `{db_path}`. Please configure settings and click 'Create/Update Vector DB'.")
    elif not st.session_state.vector_store:
         st.warning(f"Vector database directory found at `{db_path}`, but could not load the database. Please check settings or click 'Create/Update Vector DB' to rebuild.")
    elif not st.session_state.conversation:
        st.warning("Vector database loaded, but conversation chain failed to initialize. Please check LLM settings (Provider, Model, API Keys/Endpoints) and click 'Create/Update Vector DB' to re-initialize.")
    else:
         st.warning("Please build or load a vector database to start chatting!") # Generic fallback