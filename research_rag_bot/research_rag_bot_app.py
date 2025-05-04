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
from langchain_core.documents import Document # Import Document class for Complete Docs mode


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
            # Handle potential errors during loading
            try:
                doc_pages = loader.load()
                if not doc_pages:
                     print(f"Error loading content from {document_path}")
                     content_preview = "Could not load document content."
                else:
                    # Use first 5 pages or first 10000 characters, whichever is smaller
                    preview_text = " ".join([p.page_content for p in doc_pages[:5]])
                    content_preview = preview_text[:10000]
            except Exception as e:
                 print(f"Error loading PDF content from {document_path}: {e}")
                 content_preview = "Could not load PDF document content due to error."

        elif document_path.lower().endswith(('.txt', '.md', '.qmd')):
             loader = TextLoader(document_path)
             try:
                doc = loader.load()[0]
                content_preview = doc.page_content[:10000] # Limit preview size
             except Exception as e:
                 print(f"Error loading text content from {document_path}: {e}")
                 content_preview = "Could not load text document content due to error."

        elif document_path.lower().endswith('.ipynb'):
             loader = NotebookLoader(document_path)
             try:
                 doc = loader.load()[0]
                 content_preview = doc.page_content[:10000] # Limit preview size
             except Exception as e:
                 print(f"Error loading notebook content from {document_path}: {e}")
                 content_preview = "Could not load notebook document content due to error."

        elif document_path.lower().endswith('.csv'):
             loader = CSVLoader(document_path)
             try:
                 # Load first row or a small sample - CSVLoader loads rows as separate docs
                 doc = loader.load()[0] # Load the first row/document
                 content_preview = doc.page_content[:10000] # Limit preview size
             except Exception as e:
                 print(f"Error loading CSV content from {document_path}: {e}")
                 content_preview = "Could not load CSV document content due to error."

        elif document_path.lower().endswith(('.json', '.jsonl')):
             # Assuming JSONLoader provides usable text content from a record
             # Note: This assumes a structure JSONLoader can handle easily,
             # and loading [0] might not be representative for all JSONL
             loader = JSONLoader(file_path=document_path, jq_schema=".", text_content=False) # Load as dict/list
             try:
                 doc = loader.load()[0]
                 # Convert dict/list to string for preview, handle potential errors
                 content_preview = str(doc.page_content)[:10000] # Limit preview size
             except Exception as e:
                 print(f"Error loading/converting JSON content for preview {document_path}: {e}")
                 content_preview = "Could not load or process JSON content for preview."
        else:
             print(f"Unsupported file type for metadata extraction preview: {document_path}")
             content_preview = "Unsupported file type."

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
        # Ensure response is a string before passing to extract_markdown_content
        response_content_str = str(response.content)
        metadata_str = extract_markdown_content(response_content_str, "json")

        # print("extracted JSON: ", metadata_str, "\n\n") # Debug print

        # Fallback if markdown extraction fails
        if not metadata_str:
            metadata_str = response_content_str
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
                if key not in metadata or not metadata[key]: # Also check for empty strings
                    metadata[key] = "N/A"

            # Use filename if title is N/A or empty
            if metadata.get("title", "N/A") == "N/A":
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
    st.sidebar.write("---")
    st.sidebar.write("🛠️ Create Vector DB button pressed...")
    st.sidebar.write("Scanning documents and checking status...")

    metadata_dir = os.path.join(docs_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True) # Ensure metadata directory exists

    # Create the selected LLM for metadata extraction (using the new temperature)
    metadata_llm = create_llm(model_type, model_name, ollama_base_url if model_type == "Ollama" else None, llm_temperature)
    if metadata_llm is None:
        st.sidebar.error("Failed to initialize LLM for metadata extraction. Aborting DB process.")
        # Update the list of available docs in session state for the Complete Docs section
        st.session_state.available_docs = list_source_document_basenames(docs_dir)
        st.session_state.selected_files = {name: st.session_state.selected_files.get(name, False) for name in st.session_state.available_docs}
        return None

    st.sidebar.write(f"Using {model_type} model: {model_name} (temp={llm_temperature}) for metadata extraction.")

    # Create embeddings
    embeddings = create_embeddings(embedding_type, embedding_model, ollama_base_url)
    if embeddings is None:
         st.sidebar.error("Failed to initialize embeddings model. Aborting DB process.")
         # Update the list of available docs in session state for the Complete Docs section
         st.session_state.available_docs = list_source_document_basenames(docs_dir)
         st.session_state.selected_files = {name: st.session_state.selected_files.get(name, False) for name in st.session_state.available_docs}
         return None

    collection_name = "knowledge_docs"
    vector_store = None
    db_needs_full_recreate = False # Flag if we need to start from scratch

    # --- Determine if we need to create a new DB or load existing ---
    if os.path.exists(db_path):
        st.sidebar.write(f"Database directory found at '{db_path}'. Attempting to load existing DB.")
        try:
            # Attempt to load the existing database
            client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
            # Check if the collection exists *before* creating the Chroma instance
            # This helps differentiate between a missing collection vs other errors
            try:
                 collection = client.get_collection(collection_name)
                 # If get_collection succeeds, the DB and collection exist and are loadable
                 vector_store = Chroma(
                    persist_directory=db_path,
                    embedding_function=embeddings, # Ensure we use the current embeddings
                    client=client,
                    collection_name=collection_name
                )
                 st.sidebar.write(f"Existing database collection '{collection_name}' loaded.")
                 db_exists = True
                 try:
                     count = collection.count()
                     st.sidebar.write(f"Collection contains {count} existing items.")
                 except Exception as e:
                     st.warning(f"Could not retrieve count from existing collection: {e}")

            except InvalidCollectionException:
                 # Directory exists, but collection doesn't -> recreate
                 st.sidebar.warning(f"Database directory exists at '{db_path}', but collection '{collection_name}' not found. Will create a new database.")
                 db_exists = False
                 db_needs_full_recreate = True
                 # No need to clean directory here, Chroma.from_documents will create the collection

            except Exception as e:
                # Any other error during client init or get_collection attempt -> recreate
                st.sidebar.warning(f"Error loading existing database at '{db_path}': {e}. Will create a new database.")
                db_exists = False
                db_needs_full_recreate = True
                st.sidebar.warning(f"Cleaning up potentially conflicting directory: {db_path}")
                shutil.rmtree(db_path, ignore_errors=True) # Clean up anything in the directory
                os.makedirs(db_path, exist_ok=True) # Recreate the empty directory


        except Exception as e:
            # Errors even before getting collection (e.g., client init fails) -> recreate
            st.sidebar.warning(f"Error initializing Chroma client for '{db_path}': {e}. Will create a new database.")
            db_exists = False
            db_needs_full_recreate = True
            st.sidebar.warning(f"Cleaning up potentially conflicting directory: {db_path}")
            shutil.rmtree(db_path, ignore_errors=True) # Clean up anything in the directory
            os.makedirs(db_path, exist_ok=True) # Recreate the empty directory

    else:
        # db_path directory does not exist, definitely create new
        st.sidebar.write(f"Database directory not found at '{db_path}'. Will create a new database.")
        db_exists = False
        db_needs_full_recreate = True
        os.makedirs(db_path, exist_ok=True) # Create directory for the first time


    # --- Identify documents to process ---
    all_source_files = list_source_documents(docs_dir)
    st.sidebar.write(f"Found {len(all_source_files)} potential source documents in '{docs_dir}'.")

    docs_to_add_paths = [] # Paths of documents that need to be added

    if db_needs_full_recreate:
        # If recreating, all documents need to be processed and added
        st.sidebar.write("Adding all documents for a full database recreation.")
        docs_to_add_paths = all_source_files
        # Also clean up existing metadata files if doing a full recreate
        st.sidebar.write("Cleaning up existing metadata files for full recreate...")
        # List files directly in metadata_dir
        metadata_files = glob.glob(os.path.join(metadata_dir, "*.json"))
        if metadata_files:
            for metadata_file in metadata_files:
                 try:
                      os.remove(metadata_file)
                      st.sidebar.write(f"- Removed {os.path.basename(metadata_file)}")
                 except Exception as e:
                      st.sidebar.warning(f"Error removing metadata file {os.path.basename(metadata_file)}: {e}")
        else:
             st.sidebar.write("- No existing metadata files found to clean.")


    else: # db_exists is True, attempt incremental update
        st.sidebar.write("Attempting incremental update to existing database.")
        # Determine which documents need adding (those without metadata files)
        for source_path in all_source_files:
            base_name = os.path.splitext(os.path.basename(source_path))[0]
            json_path = os.path.join(metadata_dir, f"{base_name}.json")
            if os.path.exists(json_path):
                # Document metadata exists, assume it's already in the DB from a prior run
                st.sidebar.write(f"⏭️ Skipping {os.path.basename(source_path)}: Metadata exists at {os.path.basename(json_path)}.")
            else:
                # Metadata does not exist, this document needs to be added
                st.sidebar.write(f"✅ Marking {os.path.basename(source_path)} for addition (Missing metadata).")
                docs_to_add_paths.append(source_path)


    if not docs_to_add_paths and not db_exists:
        st.error("No documents found or marked for processing, and no existing database to load. Cannot proceed.")
        # Update the list of available docs in session state
        st.session_state.available_docs = list_source_document_basenames(docs_dir) # Re-list available after processing attempt
        st.session_state.selected_files = {name: st.session_state.selected_files.get(name, False) for name in st.session_state.available_docs}
        return None
    elif not docs_to_add_paths and db_exists:
         st.sidebar.write("No new documents found to add to the existing database.")
         # Update the list of available docs in session state
         st.session_state.available_docs = list_source_document_basenames(docs_dir)
         st.session_state.selected_files = {name: st.session_state.selected_files.get(name, False) for name in st.session_state.available_docs}
         return vector_store # Return the loaded existing DB


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

            metadata = extract_document_metadata(source_path, metadata_llm)

            json_path = save_metadata_json(metadata, source_path, metadata_dir)
            if json_path:
                st.sidebar.write(f"Saved metadata for {os.path.basename(source_path)}")
            else:
                 st.sidebar.warning(f"Failed to save metadata for {os.path.basename(source_path)}. Proceeding without saved metadata.")

            for doc in documents:
                 if 'source' not in doc.metadata:
                      doc.metadata['source'] = source_path
                 doc.metadata.update(metadata)
                 processed_docs_for_addition.append(doc)

        except Exception as e:
            st.sidebar.error(f"Error processing document {os.path.basename(source_path)}: {str(e)}. Skipping.")

    if not processed_docs_for_addition:
        if not db_exists:
            st.error("No documents were successfully processed to create a new database.")
        else:
            st.sidebar.write("No *new* documents were successfully processed to add to the database.")
        # Update the list of available docs in session state
        st.session_state.available_docs = list_source_document_basenames(docs_dir)
        st.session_state.selected_files = {name: st.session_state.selected_files.get(name, False) for name in st.session_state.available_docs}
        # Return existing DB if it was loaded, otherwise None
        return vector_store if db_exists else None


    # --- Split documents into chunks ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=st.session_state.chunk_size,
        chunk_overlap=st.session_state.chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks_to_add = []
    st.sidebar.write("Splitting documents into chunks...")
    docs_by_source = {}
    for doc in processed_docs_for_addition:
         source = doc.metadata.get('source')
         if source:
              if source not in docs_by_source:
                   docs_by_source[source] = []
              docs_by_source[source].append(doc)
         else:
             st.sidebar.warning(f"Document part found without source path. Splitting without specific reference.")
             chunks_to_add.extend(text_splitter.split_documents([doc]))

    for source_path, docs_from_source in docs_by_source.items():
        try:
            doc_chunks = text_splitter.split_documents(docs_from_source)
            metadata = load_metadata_json(source_path, metadata_dir)
            if metadata is None and docs_from_source:
                 metadata = docs_from_source[0].metadata
            if metadata:
                doc_chunks = prepend_reference_to_chunks(doc_chunks, metadata)
            chunks_to_add.extend(doc_chunks)
            st.sidebar.write(f"- Split '{os.path.basename(source_path)}' into {len(doc_chunks)} chunks.")
        except Exception as e:
            st.sidebar.error(f"Error splitting or prepping chunks for {os.path.basename(source_path)}: {e}. Skipping chunks for this document.")

    if not chunks_to_add:
        if not db_exists:
            st.error("No text chunks were generated to create a new database.")
        else:
            st.sidebar.write("No text chunks were generated from new documents to add to the database.")
        st.session_state.available_docs = list_source_document_basenames(docs_dir)
        st.session_state.selected_files = {name: st.session_state.selected_files.get(name, False) for name in st.session_state.available_docs}
        return vector_store if db_exists else None

    # --- Add the chunks to the vector store ---
    total_chunks = len(chunks_to_add)

    if db_needs_full_recreate:
        # Create a new vector store from all chunks
        st.sidebar.write(f"Creating a new database and collection '{collection_name}' with {total_chunks} chunks...")
        try:
            # This call handles client initialization and collection creation internally
            vector_store = Chroma.from_documents(
                documents=chunks_to_add,
                embedding=embeddings,
                persist_directory=db_path,
                collection_name=collection_name
            )
            st.sidebar.write("✅ New vector store created successfully!")
        except Exception as e:
            st.error(f"Error creating new vector store: {str(e)}")
            st.sidebar.error(f"Error creating new vector store: {str(e)}")
            st.sidebar.warning(f"Cleaning up failed DB creation directory: {db_path}")
            shutil.rmtree(db_path, ignore_errors=True)
            vector_store = None # Ensure vector_store is None on failure

    elif db_exists and vector_store:
        # Add documents to the existing vector store instance loaded earlier
        st.sidebar.write(f"Adding {total_chunks} new chunks to the existing database...")
        try:
            added_ids = vector_store.add_documents(documents=chunks_to_add)
            st.sidebar.write(f"✅ {len(added_ids)} new chunks added to the existing vector store!")
            # Re-load the Chroma wrapper to ensure it reflects the new state, using the same client/embeddings
            # While not strictly necessary for add_documents to work, re-initializing
            # the vector_store object might prevent subtle issues with state sync
            # especially if using features beyond simple add/retrieve.
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

        except Exception as e:
             st.sidebar.error(f"Error adding documents to existing DB: {e}")
             st.error(f"Error adding documents to existing DB: {e}")

    # Update the list of available docs in session state regardless of DB success
    st.session_state.available_docs = list_source_document_basenames(docs_dir)
    st.session_state.selected_files = {name: st.session_state.selected_files.get(name, False) for name in st.session_state.available_docs}

    return vector_store


def load_vector_store(db_path, embedding_type, embedding_model, ollama_base_url=None):
    """Load an existing vector store"""
    st.sidebar.write("Attempting to load existing vector database...")
    if not os.path.exists(db_path):
        st.sidebar.write(f"Database directory not found at '{db_path}'. Cannot load.")
        return None

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


        # Check if the embedding function used in the persisted Chroma matches the current one
        # This check is important for compatibility.
        # Chroma's get_collection doesn't directly expose the embedding function.
        # The primary way incompatibility shows is usually during initialization or adding data.
        # If we got here, client and collection are found. Let's proceed with creating the Chroma wrapper.
        # If there's a deep incompatibility, it might manifest later during retrieval or adding.
        # Relying on the 'Error creating new vector store' during build/update is one way this is caught.
        # For loading, if the Chroma wrapper successfully initializes with the provided embedding_function
        # pointing to an existing directory/collection, it often means it's compatible, or the incompatibility
        # is subtle. Explicitly checking embedding function hash/name could be complex and depends on the
        # embedding implementation. Let's trust Langchain/Chroma's internal checks for now.

        vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings, # Use the current embedding function
            client=client,
            collection_name=collection_name
        )
        # Optional: Check if the collection is empty
        try:
            count = collection.count() # Use the collection object retrieved earlier
            if count == 0:
                 st.warning(f"Vector database collection '{collection_name}' is empty. Please rebuild the database.")
                 st.sidebar.warning(f"Collection '{collection_name}' is empty.")
                 # Decide if empty DB should return None or an empty vector store.
                 # Returning None makes chat ineligible. Let's return None if empty.
                 return None
            else:
                 st.sidebar.write(f"Loaded database with {count} items.")
        except Exception as e:
            st.warning(f"Could not get item count from database collection: {e}. Database might need rebuilding.")
            st.sidebar.warning(f"Could not get item count: {e}")
            # If count fails but DB object exists, maybe it's still usable?
            # Let's allow loading but warn. Rebuild is the safest fix.
            pass # Keep the vector_store object even if count fails

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
            # Attempt to ping Ollama? Could add health check here
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
            return None, None, None # Return None for conversation, llm, retriever

        # Create the LLM using the provided temperature
        llm = create_llm(model_type, model_name, ollama_base_url, llm_temperature)
        if llm is None:
             st.warning(f"Cannot initialize conversation: LLM creation failed for {model_name} ({model_type}). Check API keys/endpoints.")
             st.sidebar.error(f"Cannot initialize conversation: LLM creation failed for {model_name} ({model_type}).")
             return None, None, None # Return None for conversation, llm, retriever

        st.sidebar.write(f"LLM ({model_name}, Temp={llm_temperature}) created for conversation chain.")

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
            num_chunks_kept_actual = min(num_chunks_kept, k_value) # Use potentially adjusted value
            try:
                compressor = LLMChainExtractor.from_llm(llm)
                retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever, k=num_chunks_kept_actual)
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
        return conversation, llm, retriever # Return all three instances

    except Exception as e:
        st.error(f"Error initializing conversation: {str(e)}")
        st.sidebar.error(f"Error initializing conversation: {str(e)}")
        return None, None, None

def extract_markdown_content(text: str, type: str = "json") -> str:
        """Extract content from markdown code blocks."""
        # Ensure text is a string
        text = str(text)
        start_tag = f"```{type}"
        end_tag = """```"""

        start_idx = text.find(start_tag)
        if start_idx == -1:
             # If start tag not found, check for just ``` (often used for simple code blocks)
             start_tag_generic = """```"""
             start_idx_generic = text.find(start_tag_generic)
             if start_idx_generic != -1:
                  start_idx = start_idx_generic + len(start_tag_generic)
                  # Check if the line after ``` starts with the requested type (e.g. "json")
                  newline_after_generic = text.find('\n', start_idx_generic)
                  if newline_after_generic != -1 and text[start_idx_generic:newline_after_generic].strip() == start_tag_generic.strip():
                       # If ``` is on a line by itself, then the content starts after the newline
                       start_idx = newline_after_generic + 1
                  else:
                       # If something like ```python, the start is after the type name + newline
                       newline_after_type = text.find('\n', start_idx_generic + len(start_tag_generic))
                       if newline_after_type != -1:
                            start_idx = newline_after_type + 1
                       else:
                            # Fallback if newline isn't found after ```type or ```
                            start_idx = start_idx_generic + len(start_tag_generic)

             else:
                # If neither ```json nor ``` is found, return the whole text after stripping
                return text.strip()

        else: # Found ```json
            start_idx += len(start_tag)
             # Handle potential newline immediately after start_tag
            if start_idx < len(text) and text[start_idx] == '\n':
                 start_idx += 1


        end_idx = text.rfind(end_tag, start_idx) # Search for end tag after start

        if end_idx != -1:
            return (text[start_idx:end_idx]).strip()
        else:
             # If end tag not found, assume the rest of the text is the content
             return text[start_idx:].strip()


# Function to execute chain of thought reasoning with preserved context
# (Keep this function as is, it uses the llm passed to it)
def execute_chain_of_thought(llm, retriever, prompt, max_steps=5):
    """Executes a chain of thought reasoning process using the LLM and retriever."""
    st.sidebar.write("Starting Chain of Thought process...")
    # Initial prompt to get chain of thought planning
    # Ensure prompt is adapted for the specific LLM capabilities and instruction following
    # Updated system prompt for better JSON adherence and clarity
    cot_system_prompt = """
    You are a helpful assistant that plans step-by-step reasoning to answer complex questions based on provided context.
    Given the user's question and relevant context, create a numbered reasoning plan.
    Ensure your output is ONLY a JSON object in the following format:
    {{
        "reasoning_steps": [
            "Step 1: Describe the first task to perform (e.g., Identify key terms)",
            "Step 2: Describe the second task (e.g., Search context for info on key terms)",
            "Step 3: Describe the third task (e.g., Synthesize findings)",
            "Step 4: Describe the final task (e.g., Formulate answer)"
        ]
    }}
    Adapt the number and description of steps based on the complexity of the question.
    Do not include any other text, explanation, or markdown outside the JSON object.
    """

    # Get the relevant documents
    st.sidebar.write("Retrieving documents for Chain of Thought context...")
    retrieved_docs = []
    try:
        retrieved_docs = retriever.get_relevant_documents(prompt)
        if not retrieved_docs:
             st.sidebar.write("No documents retrieved for context.")
             context = "No relevant documents found."
        else:
             st.sidebar.write(f"Retrieved {len(retrieved_docs)} documents for context.")
             context = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in retrieved_docs])


    except Exception as e:
        st.sidebar.error(f"Error retrieving documents for CoT: {e}")
        context = "Could not retrieve context documents due to an error."


    # Initial prompt with context to get the plan
    initial_planning_prompt = f"""
    System: {cot_system_prompt}

    Context information relevant to the question:
    ---
    {context}
    ---

    User question: {prompt}

    Based on the user question and the context, create a step-by-step reasoning plan in the specified JSON format.
    Remember to output *only* the JSON object.
    """

    # Get the plan
    st.sidebar.write("Generating Chain of Thought plan...")
    plan_response = None
    try:
        plan_response = llm.invoke(initial_planning_prompt)
        plan_text = plan_response.content

        # Extract JSON plan
        json_str = extract_markdown_content(plan_text, "json")
        # print("plan:", json_str) # Debug print
        plan_json = json.loads(json_str)
        steps = plan_json.get("reasoning_steps", [])
        if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
             st.warning("LLM returned invalid plan format. Using default steps.")
             raise ValueError("Invalid plan format") # Trigger fallback

    except (json.JSONDecodeError, ValueError, Exception) as e:
        print(f"Error extracting/parsing JSON plan: {e}")
        if plan_response:
             print(f"LLM Output (first 500 chars): {plan_response.content[:500]}...") # Debug print
        st.warning("Failed to extract valid plan from LLM. Using default steps.")
        # If JSON parsing fails or format is invalid, create a default plan
        steps = [
            "Step 1: Analyze the user's question and identify the core concepts and information required.",
            "Step 2: Review the provided context information to find relevant facts, data, or descriptions related to the question.",
            "Step 3: Synthesize information from the context and the question, combining relevant pieces and performing any necessary analysis or calculations.",
            "Step 4: Formulate a clear and comprehensive final answer based on the synthesis, directly addressing the user's question.",
        ]


    total_steps = min(len(steps), max_steps)  # Limit to max_steps

    st.sidebar.write(f"Executing {total_steps} reasoning steps.")

    # Create progress bar in the main area
    progress_bar = st.progress(0.0, text=f"Starting Chain of Thought process...")
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
        You are executing step {current_step_number} of a multi-step reasoning process to answer the user's question.
        Your task is to complete *only* the current step using the available information.

        Original User Question: {user_question}

        Accumulated Information from previous steps and original context:
        ---
        {accumulated_context}
        ---

        Current Step to Execute ({current_step_number}/{total_steps}): {step_description}

        Based on the "Accumulated Information" and the "Original User Question", execute the "Current Step to Execute".
        Provide your detailed reasoning, intermediate results, and findings for *this specific step*.
        Your output for this step should be a concise summary or the results of executing the step's task.
        This output will be used in the next step's "Accumulated Information".
        Do not provide the final answer to the original question yet, only the outcome of this step.
        """

        step_prompt = step_prompt_template.format(
            current_step_number=i+1,
            total_steps=total_steps,
            user_question=prompt,
            accumulated_context=growing_context,
            step_description=current_step_description
        )

        # Get response for this step
        step_result = ""
        try:
            step_response = llm.invoke(step_prompt)
            step_result = step_response.content
        except Exception as e:
            step_result = f"Error executing step: {e}"
            st.warning(f"Error in CoT step {i+1}: {e}")
            st.sidebar.warning(f"Error in CoT step {i+1}: {e}")


        # Add to outputs list (for summary/display later)
        step_output_display.append(f"Step {i+1} ({current_step_description}):\n{step_result}")

        # Add this step's reasoning result to the growing context for the *next* step
        # Use a clear marker to separate steps in the accumulated context
        growing_context += f"\n---\nReasoning Results from Step {i+1} ({current_step_description}):\n{step_result}\n"

        # Display intermediate step in an expander
        with st.expander(f"Step {i+1}: {current_step_description}", expanded=False):
            st.markdown(step_result) # Use markdown for displaying step results

        # Small delay to show progress visually (optional)
        # time.sleep(0.1) # Reduced delay

    # Final step - synthesize all reasoning into an answer
    progress_bar.progress(1.0, text="Synthesizing final answer...")

    # Create prompt for final answer using the complete growing context
    final_answer_prompt = f"""
    System: You have completed a chain of thought process to answer the user's question, accumulating information and reasoning through several steps.
    Your task now is to synthesize all the information and reasoning from the steps you just executed into a single, coherent, and comprehensive final answer.

    Original User Question: {prompt}

    Complete Accumulated Information and Reasoning from Steps:
    ---
    {growing_context}
    ---

    Based on the "Complete Accumulated Information and Reasoning", provide your final, comprehensive answer to the "Original User Question".
    Present the answer clearly, logically, and directly address the user's query. Do not include the step-by-step reasoning process or intermediate thoughts in the final answer, only the final result. Use markdown for formatting if helpful (e.g., lists, bolding).
    """

    # Get final answer
    final_answer = ""
    try:
        final_response = llm.invoke(final_answer_prompt)
        final_answer = final_response.content
        st.sidebar.write("✅ Chain of Thought process completed.")
    except Exception as e:
        final_answer = f"Error generating final answer after steps: {e}"
        st.error(final_answer)
        st.sidebar.error(final_answer)
        # Fallback: if final synthesis fails, maybe just show the step results
        # final_answer = "Could not synthesize final answer. Here are the step results:\n\n" + "\n\n---\n\n".join(step_output_display)


    progress_bar.empty() # Clear the progress bar

    # Note: We return step_output_display which contains descriptions + results for display in expanders,
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
        elif ext in [".txt", ".md", ".qmd"]:
            return TextLoader(file_path)
        elif ext in [".json", ".jsonl"]:
            # Note: JSONLoader requires a jq_schema. "." loads the root object.
            # This might load the whole file as one "document".
            # For line-delimited JSONL, need different approach or jq schema for each line.
            # This loader might need customization based on JSON structure.
             # st.sidebar.warning(f"Using generic JSONLoader for {os.path.basename(file_path)}. May need custom schema.")
             # Let's use text_content=True for simplicity in Complete Docs mode context concatenation
             return JSONLoader(file_path=file_path, jq_schema=".", text_content=True)
        elif ext == ".csv":
            return CSVLoader(file_path)
        elif ext == ".ipynb":
            return NotebookLoader(file_path)
        else:
            return None # Indicate unsupported file type
    except Exception as e:
         st.sidebar.error(f"Error creating loader for {os.path.basename(file_path)} (ext: {ext}): {e}")
         return None

# Helper to list source documents excluding metadata directory
def list_source_documents(docs_dir):
    """Lists all source document files in the docs directory, excluding the metadata directory."""
    if not docs_dir or not os.path.isdir(docs_dir):
        return []
    metadata_dir = os.path.join(docs_dir, "metadata")
    # Use glob to find all files recursively
    all_files = glob.glob(os.path.join(docs_dir, "**/*"), recursive=True)
    # Filter out directories and files within the metadata directory
    source_files = [
        f for f in all_files
        if os.path.isfile(f) and not os.path.normpath(f).startswith(os.path.normpath(metadata_dir))
    ]
    return source_files

# Helper to list basenames of source documents
def list_source_document_basenames(docs_dir):
    """Lists basenames of all source document files in the docs directory, excluding the metadata directory."""
    return [os.path.basename(f) for f in list_source_documents(docs_dir)]


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

# New session state for Complete Documents feature
if 'use_complete_docs' not in st.session_state:
    st.session_state.use_complete_docs = False
if 'available_docs' not in st.session_state:
     st.session_state.available_docs = [] # Store basenames
if 'selected_files' not in st.session_state:
    st.session_state.selected_files = {} # {basename: bool}

# Session state to track current LLM settings for re-initialization check
if 'current_llm_params' not in st.session_state:
     st.session_state.current_llm_params = None


# Title and description
st.title("🤖 RAG Chatbot")
st.markdown("Ask questions about your documents.")

# Sidebar for configuration
with st.sidebar:
    # Input for documents directory (placed higher as it affects other settings)
    docs_dir = st.text_input("Documents Directory", os.getenv("SOURCE_DOC_DIR", "docs"))
    # Ensure docs_dir is created if it doesn't exist
    if docs_dir:
        try:
            os.makedirs(docs_dir, exist_ok=True)
            # st.info(f"Using documents directory: {docs_dir}") # Avoid spamming
        except OSError as e:
            st.error(f"Error creating/accessing documents directory {docs_dir}: {e}")
            docs_dir = None # Set to None if invalid/inaccessible

    # Use expander for Vector DB Settings
    with st.expander("Vector DB Settings", expanded=False): # Collapsed by default
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
            index=0,
            key="embedding_provider_select" # Ensure unique key
        )

        embedding_model = "" # Default placeholder
        if embedding_type == "Ollama":
            embedding_model = st.text_input("Ollama Embedding Model", os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest"), key="ollama_embedding_model_input")
        elif embedding_type == "OpenAI":
            embedding_model = st.text_input("OpenAI Embedding Model", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"), key="openai_embedding_model_input")
        elif embedding_type == "Google":
            embedding_model = st.text_input("Google Embedding Model", os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001"), key="google_embedding_model_input")

        db_path = os.path.join(docs_dir, "vectorstore") if docs_dir else None

        st.markdown("**Before creating your vector DB** make sure to delete any `vectorstore` or `metadata` subdirectories in the docs dir.")

        # Database operations button (moved to the bottom of this section)
        build_db_button = st.button("🔨 Create Vector DB", key="build_db_button")

    # --- New Complete Documents Section ---
    with st.expander("Pass Complete Documents to LLM", expanded=False): # Collapsed by default
        # Re-list available documents whenever docs_dir might have changed or on explicit DB build
        if docs_dir:
            current_available_doc_basenames = list_source_document_basenames(docs_dir)

            # Update session state list of available document *basenames*
            # Check if the set of available basenames has changed
            if set(st.session_state.available_docs) != set(current_available_doc_basenames):
                 st.session_state.available_docs = sorted(current_available_doc_basenames) # Keep sorted
                 # Reset selection state for any removed files, add new ones (default False)
                 new_selected_files = {}
                 for name in st.session_state.available_docs:
                      new_selected_files[name] = st.session_state.selected_files.get(name, False) # Preserve state if existing
                 st.session_state.selected_files = new_selected_files
                 # st.rerun() # Rerun here is often necessary for immediate checkbox updates, but let's see if Streamlit handles it naturally on state update. Adding rerun makes selection snappier.
                 # Note: Rerunning here *might* interfere with other input widgets if not careful. Test thoroughly.
                 # Let's add it back if needed, but try without first.


            st.session_state.use_complete_docs = st.checkbox(
                "Use Complete Documents (Ignores Vector DB)",
                value=st.session_state.use_complete_docs,
                help="If checked, the LLM will receive the full text of selected documents as context instead of using vector database retrieval. Select documents below.",
                key="use_complete_docs_checkbox" # Unique key
            )

            # Show status message based on mode and selection
            if st.session_state.use_complete_docs:
                selected_count = sum(st.session_state.selected_files.values())
                if selected_count > 0:
                    st.info(f"🟢 Using {selected_count} selected document(s) as context.")
                else:
                    st.warning("🔴 Complete Documents mode active, no documents selected.")
            else:
                if st.session_state.vector_store:
                    st.info("🔵 Using Vector Database retrieval.")
                else:
                    st.warning("⚪ Context mode not active.")


            st.write("Select documents to use as context:")

            # Select All/None links (Use buttons for better visual feedback)
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                 if st.button("Select All", key="select_all_docs_button"):
                      st.session_state.selected_files = {name: True for name in st.session_state.available_docs}
                      # st.rerun() # May need rerun here
                 # Use link_button if preferred, but requires handling click state carefully
                 # if st.link_button("Select All", key="select_all_docs_link"):
                 #      st.session_state.selected_files = {name: True for name in st.session_state.available_docs}
                 #      st.rerun()

            with col_sel2:
                 if st.button("Select None", key="select_none_docs_button"):
                      st.session_state.selected_files = {name: False for name in st.session_state.available_docs}
                      # st.rerun() # May need rerun here
                 # if st.link_button("Select None", key="select_none_docs_link"):
                 #      st.session_state.selected_files = {name: False for name in st.session_state.available_docs}
                 #      st.rerun()


            # Display checkboxes for each document
            if st.session_state.available_docs:
                # Sort the available documents alphabetically for consistent display
                sorted_available_docs = sorted(st.session_state.available_docs)
                for doc_basename in sorted_available_docs:
                     # Use the basename as the key and label
                     # Ensure the state is linked to the session state dict
                     checkbox_state = st.session_state.selected_files.get(doc_basename, False)
                     new_checkbox_state = st.checkbox(
                         doc_basename,
                         value=checkbox_state,
                         key=f"select_doc_{doc_basename}" # Unique key for each checkbox
                     )
                     # Update session state if the value changed
                     if new_checkbox_state != checkbox_state:
                          st.session_state.selected_files[doc_basename] = new_checkbox_state
                          # Rerun might be needed if you want immediate feedback in the count/status message
                          # st.experimental_rerun() # Use only if necessary for UI responsiveness

                # Show how many are selected below the checkboxes
                selected_count_display = sum(st.session_state.selected_files.values())
                st.caption(f"{selected_count_display} of {len(st.session_state.available_docs)} selected.")

            else:
                st.info("No documents found in the documents directory.")
                # Ensure selection state is empty if no docs
                st.session_state.selected_files = {}
                st.session_state.available_docs = [] # Ensure the list is empty in state


        else:
            # If docs_dir is None or invalid
            st.warning("Please specify a valid documents directory to list files.")
            st.session_state.available_docs = []
            st.session_state.selected_files = {}

    with st.expander("Search Vector DB for Documents to Send to LLM", expanded=False):
        # Number of retrieved documents
        k_value = st.slider(
            "Number of retrieved documents (K)",
            min_value=1,
            max_value=100,
            value=st.session_state.k_value, # Use session state default/current
            step=1,
            help="Controls how many relevant documents to retrieve for each query *before* optional reranking.",
            key="k_value_slider" # Unique key
        )
        st.session_state.k_value = k_value

         # Add LLM Reranking option
        use_reranking = st.checkbox("Use LLM Reranking", value=st.session_state.use_reranking, key="use_reranking_checkbox", help="If checked, the documents retrieved from the vector database will be ranked by the LLM for effectiveness and then the top N documents will be used in your query.")
        st.session_state.use_reranking = use_reranking

         # Number of chunks kept for reranking
        # Only show if reranking is enabled
        num_chunks_kept = st.session_state.k_value # Default to K if reranking is off or slider not shown
        if use_reranking:
            num_chunks_kept = st.slider(
                "Number of Chunks Kept After Reranking",
                min_value=1,
                max_value=st.session_state.k_value,  # Limit to the number of retrieved documents
                value=st.session_state.num_chunks_kept, # Use session state default/current
                step=1,
                help="Number of chunks to keep after LLM reranking. Must be less than or equal to the number of retrieved documents (K)."
            )
        st.session_state.num_chunks_kept = num_chunks_kept # Update session state with the chosen or default value


    # Use expander for LLM Settings
    with st.expander("LLM Settings", expanded=True): # Expanded by default
        # Temperature Slider
        temperature = st.slider(
            "LLM Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature, # Use session state default/current
            step=0.05,
            help="Controls randomness in LLM responses. Lower = more deterministic, Higher = more creative/varied. Default is 0.2.",
            key="llm_temperature_slider" # Unique key
        )
        st.session_state.temperature = temperature

        # Model selection
        llm_providers_str = os.getenv("LLM_PROVIDERS", "Ollama,OpenAI,Anthropic,Google")
        model_type = st.selectbox(
            "Select LLM Provider",
            split_csv(llm_providers_str),
            index=0,
            key="llm_provider_select" # Unique key
        )

        # Model selection based on provider
        model_name = "" # Default placeholder
        ollama_base_url = None # Default to None

        if model_type == "Ollama":
            ollama_model_str = os.getenv("OLLAMA_MODEL", "llama3:latest,mistral:latest")
            model_name = st.selectbox(
                "Select Ollama Model",
                split_csv(ollama_model_str),
                index=0,
                 key="ollama_model_select" # Unique key
            )
            ollama_base_url = st.text_input("Ollama Base URL", os.getenv("OLLAMA_END_POINT", "http://localhost:11434"), key="ollama_base_url_input")
        elif model_type == "OpenAI":
            openai_model_str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo,gpt-4o")
            model_name = st.selectbox(
                "Select OpenAI Model",
                split_csv(openai_model_str),
                index=0,
                 key="openai_model_select" # Unique key
            )
        elif model_type == "Anthropic":
            anthropic_model_str = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307,claude-3-sonnet-20240229")
            model_name = st.selectbox(
                "Select Anthropic Model",
                split_csv(anthropic_model_str),
                index=0,
                 key="anthropic_model_select" # Unique key
            )
        elif model_type == "Google":
            google_model_str = os.getenv("GOOGLE_MODEL", "gemini-pro")
            model_name = st.selectbox(
                "Select Google Model",
                split_csv(google_model_str),
                index=0,
                 key="google_model_select" # Unique key
            )

        # Add Chain of Thought option
        # Chain of Thought requires a retriever to get initial context, so it's primarily for Vector DB mode.
        # We will disable/ignore it if "Use Complete Documents" is checked in the chat logic.
        use_cot = st.checkbox("Use Chain of Thought (Vector DB Mode Only)", value=False, key="use_cot_checkbox")


# --- LLM Initialization (runs whenever settings change or on initial load) ---
current_llm_params = (model_type, model_name, ollama_base_url, st.session_state.temperature)

# Check if LLM params have changed or if LLM isn't set yet
# Also check if docs_dir is valid, as create_llm for metadata extraction is needed for DB build
if docs_dir and (st.session_state.current_llm_params != current_llm_params or st.session_state.llm is None):
    st.session_state.current_llm_params = current_llm_params
    # st.sidebar.write("Attempting to initialize LLM...") # Moved message inside create_llm
    llm_instance = create_llm(
        model_type,
        model_name,
        ollama_base_url if model_type == "Ollama" else None,
        st.session_state.temperature
    )

    if llm_instance:
        st.session_state.llm = llm_instance
        # st.sidebar.write("✅ LLM initialized.") # Message moved inside create_llm
        # If Vector DB *is* loaded, and LLM was just initialized/changed, re-initialize the conversation chain
        # Check if LLM instance itself is different, or just the params changed
        # Only re-initialize if we are *not* in complete docs mode OR if LLM is truly None
        if st.session_state.vector_store and not st.session_state.use_complete_docs and st.session_state.llm is not None: # Ensure LLM is valid before trying to re-init chain
             # Re-initialize conversation chain with potentially new LLM and settings
             st.sidebar.write("LLM settings changed or initialized, re-initializing conversation chain...")
             conversation, llm, retriever = initialize_conversation(
                 st.session_state.vector_store,
                 model_type,
                 model_name,
                 st.session_state.k_value,
                 ollama_base_url if model_type == "Ollama" else None,
                 st.session_state.use_reranking,
                 st.session_state.num_chunks_kept,
                 st.session_state.temperature
             )
             st.session_state.conversation = conversation
             # Note: initialize_conversation also returns the created LLM, which might be a *new* instance
             # based on the parameters. We should use the one returned by initialize_conversation
             # if it was successful, as that's the LLM actually hooked into the chain.
             # If initialize_conversation returns None for llm, stick with the llm_instance created just above.
             st.session_state.llm = llm if llm is not None else llm_instance
             st.session_state.retriever = retriever
             if st.session_state.conversation:
                  st.sidebar.success("Conversation chain re-initialized.")
             else:
                  st.sidebar.error("Failed to re-initialize conversation chain.")

    else:
        st.session_state.llm = None
        # st.sidebar.warning("⚠️ Could not initialize LLM. Check settings and environment variables.") # Message moved inside create_llm
        # If LLM fails to initialize, the conversation chain (if using LLM) will also fail/be unusable
        st.session_state.conversation = None # Ensure conversation is reset if LLM fails
        st.session_state.retriever = None # Retriever might still exist, but chain is broken.
elif not docs_dir:
     # If docs_dir is not valid, LLM cannot be initialized (as it's needed for metadata extraction during DB build)
     st.session_state.llm = None
     st.session_state.conversation = None
     st.session_state.retriever = None
# --- End LLM Initialization ---


# Handle database building/updating (placed outside sidebar but triggered by sidebar button)
# Use a button key state to track if the button was actually clicked
if 'build_db_button_clicked' not in st.session_state:
    st.session_state.build_db_button_clicked = False

# Check if the button was clicked on this rerun
if build_db_button:
    st.session_state.build_db_button_clicked = True # Mark button as clicked

# Execute DB build/update logic only if button was clicked and it hasn't been processed yet in this session
if st.session_state.build_db_button_clicked:
    st.session_state.build_db_button_clicked = False # Reset flag immediately

    if not docs_dir:
         st.error("Please specify a documents directory.")
    elif not embedding_model:
         st.error("Please specify an embedding model.")
    elif model_type == "Ollama" and not ollama_base_url:
         st.error("Please specify the Ollama Base URL.")
    elif not model_name:
         st.error("Please select or specify an LLM model name.")
    elif st.session_state.llm is None:
        st.error("LLM failed to initialize. Please check LLM settings before building the database (metadata extraction requires LLM).")
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
                # Only initialize the vector DB chain if we are NOT in complete docs mode
                if not st.session_state.use_complete_docs:
                    conversation, llm, retriever = initialize_conversation(
                        vector_store,
                        model_type,
                        model_name,
                        st.session_state.k_value,
                        ollama_base_url if model_type == "Ollama" else None,
                        st.session_state.use_reranking,
                        st.session_state.num_chunks_kept,
                        st.session_state.temperature
                    )
                    st.session_state.conversation = conversation
                    # Update the main session state LLM and retriever from the chain initialization
                    # Use the LLM instance returned by initialize_conversation if successful
                    st.session_state.llm = llm if llm is not None else st.session_state.llm
                    st.session_state.retriever = retriever
                    if st.session_state.conversation:
                         st.sidebar.success("Conversation chain initialized with updated DB.")
                    else:
                         st.sidebar.error("Failed to initialize conversation chain after DB update.")
                else:
                     # If in complete docs mode, clear the vector DB chain variables
                     st.session_state.conversation = None
                     st.session_state.retriever = None
                     st.sidebar.write("Vector DB updated, but conversation chain is not initialized (Complete Documents mode active).")

            else:
                st.error("Failed to create or update vector database.")
                st.session_state.vector_store = None
                st.session_state.conversation = None
                st.session_state.retriever = None
                # st.session_state.llm might still be set from the independent init block, which is fine

        # Rerun the script after DB operation to update the UI status and potentially chat eligibility
        st.rerun()


# --- Initial Load Logic ---
# This block runs on initial app load or after a reset if DB exists,
# but ONLY if the vector_store and conversation are NOT already in session state AND
# we are *not* in Complete Documents mode.
# Also ensure required settings are available and LLM is initialized.
if docs_dir and os.path.exists(docs_dir) and embedding_model and model_name and (model_type != "Ollama" or ollama_base_url) and st.session_state.llm is not None:
    db_path = os.path.join(docs_dir, "vectorstore")
    # Only attempt to load DB if it exists and we are NOT using complete docs AND the DB isn't already loaded
    if os.path.exists(db_path) and not st.session_state.vector_store and not st.session_state.use_complete_docs:
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
                # Update the main session state LLM and retriever from the chain initialization
                st.session_state.llm = llm # Use the LLM instance returned by initialize_conversation
                st.session_state.retriever = retriever
                if st.session_state.conversation:
                     st.success("Existing vector database loaded and conversation initialized!")
                else:
                    st.warning("Existing database loaded, but conversation chain could not be initialized. Check LLM settings.")
            else:
                st.warning("Could not load existing vector database. Please check settings or build/rebuild it.")
                # Ensure related states are None if loading fails
                st.session_state.vector_store = None
                st.session_state.conversation = None
                st.session_state.retriever = None

# Update available docs list on initial load or rerun if docs_dir is set
if docs_dir:
     current_available_doc_basenames = list_source_document_basenames(docs_dir)
     if set(st.session_state.available_docs) != set(current_available_doc_basenames):
          st.session_state.available_docs = sorted(current_available_doc_basenames) # Keep sorted
          # Reset selection state for any removed files, add new ones (default False)
          new_selected_files = {}
          for name in st.session_state.available_docs:
               new_selected_files[name] = st.session_state.selected_files.get(name, False) # Preserve state if existing
          st.session_state.selected_files = new_selected_files
     # Ensure selected_files reflects current available docs even if the set didn't change
     # This handles cases where state might get slightly out of sync without reruns
     current_selected_files = {}
     for name in st.session_state.available_docs:
          current_selected_files[name] = st.session_state.selected_files.get(name, False)
     st.session_state.selected_files = current_selected_files

# --- End Initial Load Logic ---


# --- Chat Eligibility Check ---
# Chat input is available if LLM is initialized AND (Vector DB chain is ready AND Complete Docs is OFF
# OR Complete Docs mode is ON and files are selected)
chat_eligible = (st.session_state.llm is not None) and \
                ((st.session_state.conversation is not None and not st.session_state.use_complete_docs) or \
                 (st.session_state.use_complete_docs and sum(st.session_state.selected_files.values()) > 0))

# --- Main Chat Area ---
if chat_eligible:
    # Display chat messages
    for message in st.session_state.chat_history:
        avatar = "🧑" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"]) # Use markdown for message display

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents...", disabled=False):
        st.chat_message("user", avatar="🧑").markdown(prompt)

        # Add user message to chat history immediately
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Check which mode is active
        if st.session_state.use_complete_docs:
            # --- Complete Documents Mode ---
            selected_basenames = [name for name, selected in st.session_state.selected_files.items() if selected]
            # This case should ideally not happen if chat_eligible check works, but double check
            if not selected_basenames or not docs_dir:
                 response_content = "Complete Documents mode is active, but no documents are selected or documents directory is invalid. Please check settings in the sidebar."
                 st.warning(response_content)
                 st.chat_message("assistant", avatar="🤖").markdown(response_content)
                 st.session_state.chat_history.append({"role": "assistant", "content": response_content})
            else:
                with st.spinner(f"Processing selected documents ({len(selected_basenames)}) and generating response..."):
                    try:
                        # Re-locate full paths from basenames
                        all_source_paths = list_source_documents(docs_dir)
                        selected_paths = [p for p in all_source_paths if os.path.basename(p) in selected_basenames]

                        concatenated_content = ""
                        st.sidebar.write(f"Loading content from {len(selected_paths)} selected documents for context...")
                        loaded_document_contents = []
                        for doc_path in selected_paths:
                             loader = get_document_loader(doc_path)
                             if loader:
                                  try:
                                      # Load ALL content from the document using the loader
                                      documents = loader.load() # Loader might return list of pages/docs
                                      # Concatenate content from all parts of this document
                                      full_text = "\n\n".join([d.page_content for d in documents])
                                      # Add source information and the full text for this document
                                      loaded_document_contents.append(f"--- Start Document: {os.path.basename(doc_path)} ---\n\n{full_text}\n\n--- End Document: {os.path.basename(doc_path)} ---")
                                      st.sidebar.write(f"- Loaded content from {os.path.basename(doc_path)}")
                                  except Exception as e:
                                      st.sidebar.warning(f"Error loading content from {os.path.basename(doc_path)}: {e}. Skipping.")
                             else:
                                 st.sidebar.warning(f"Unsupported file type for loading complete content: {os.path.basename(doc_path)}. Skipping.")

                        # Join content from all successfully loaded documents
                        concatenated_content = "\n\n".join(loaded_document_contents)

                        if not concatenated_content.strip():
                             response_content = "Could not load readable content from selected documents."
                        else:
                            # Construct the manual prompt for Complete Documents mode
                            # NOTE: In this mode, chat history is *not* included in the prompt
                            # to keep the implementation simple as per the request.
                            # Add instructions to rely *only* on the provided documents
                            manual_prompt_template = """
                            You are a helpful assistant. Use the following documents as your ONLY source of information to answer the user's question.
                            If the answer cannot be found *within these documents*, state clearly that you cannot answer based on the provided context.
                            Do not use any external knowledge or make up information.

                            Documents:
                            ---
                            {context}
                            ---

                            User Question: {question}
                            """
                            manual_prompt = manual_prompt_template.format(
                                context=concatenated_content,
                                question=prompt
                            )

                            # Check if LLM is available (should be due to chat_eligible check)
                            if st.session_state.llm:
                                # Use the selected LLM directly
                                # Chain of thought is NOT supported in this mode as it requires the retriever
                                if use_cot:
                                     st.warning("Chain of Thought is not supported in 'Complete Documents' mode. Using direct query.")

                                response = st.session_state.llm.invoke(manual_prompt)
                                response_content = response.content
                            else:
                                response_content = "LLM is not initialized. Cannot answer in Complete Documents mode."
                                st.error(response_content)


                        # Display assistant response
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(response_content)

                        # Add assistant response to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": response_content})

                    except Exception as e:
                        st.error(f"An unexpected error occurred in Complete Documents mode: {e}")
                        with st.chat_message("assistant", avatar="🤖"):
                             st.markdown("Sorry, an error occurred while processing the documents.")
                        import traceback
                        print(traceback.format_exc())


        elif st.session_state.conversation:
            # --- Vector DB Mode ---
            # (Keep existing logic using the conversation chain)
            with st.spinner("Thinking..."):
                try:
                    # Prepare chat history for the chain
                    langchain_history = []
                    # Iterate in steps of 2 (user, assistant pairs)
                    # Exclude the current user message we just added
                    history_for_chain = st.session_state.chat_history[:-1] if len(st.session_state.chat_history) > 0 else []

                    for i in range(0, len(history_for_chain), 2):
                         if i + 1 < len(history_for_chain): # Ensure there's a pair
                            user_msg = history_for_chain[i]
                            ai_msg = history_for_chain[i+1]
                            if user_msg['role'] == 'user' and ai_msg['role'] == 'assistant':
                                langchain_history.append((user_msg['content'], ai_msg['content']))
                            # Handle cases where roles might be mismatched - maybe skip or log warning

                    # Execute query - check if retriever is also initialized (should be if conversation is)
                    if not st.session_state.retriever or not st.session_state.llm:
                         # This case should be caught by chat_eligible, but defensive check
                         response = {"answer": "Vector DB mode is active, but the required components (LLM/Retriever) are not initialized. Please rebuild/load the DB."}
                         st.warning(response["answer"])
                    elif use_cot and st.session_state.llm and st.session_state.retriever:
                         # Use Chain of Thought if selected and LLM/Retriever are available
                         answer, step_outputs = execute_chain_of_thought(
                             st.session_state.llm,
                             st.session_state.retriever,
                             prompt
                         )
                         response = {"answer": answer} # CoT returns the final answer directly


                    else:
                         # Regular conversation flow using the chain
                         response = st.session_state.conversation.invoke({"question": prompt, "chat_history": langchain_history})
                         # If using ConversationalRetrievalChain, the 'answer' key holds the response.


                    answer = response["answer"] # Access the answer from the response dictionary

                    # Display assistant response
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(answer)

                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"An error occurred during the conversation (Vector DB mode): {e}")
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown("Sorry, I encountered an error while processing your request.")
                    import traceback
                    print(traceback.format_exc())


else:
    # Message to display if chat is NOT eligible
    st.info("Chat is not ready.")

    # Provide specific guidance based on the state
    if st.session_state.llm is None:
         st.warning("LLM is not initialized. Please check LLM settings in the sidebar and ensure API keys/endpoints are correct.")
    elif st.session_state.use_complete_docs and sum(st.session_state.selected_files.values()) == 0:
          st.warning("Complete Documents mode is enabled, but no documents are selected. Please select documents in the sidebar.")
    elif not st.session_state.vector_store:
         db_path_display = os.path.join(docs_dir or "your_docs_dir", "vectorstore")
         st.warning(f"Vector database is not loaded or initialized. Please configure settings in the sidebar and click 'Create Vector DB' to build or load the database at `{db_path_display}`.")
         if docs_dir and not os.path.exists(docs_dir):
              st.info(f"Documents directory `{docs_dir}` not found or not specified. Please create the directory or enter a valid path in the sidebar.")
    elif st.session_state.vector_store and st.session_state.conversation is None and not st.session_state.use_complete_docs:
        st.warning("Vector database loaded, but conversation chain failed to initialize. Please check LLM settings (Provider, Model, API Keys/Endpoints) and click 'Create Vector DB' to re-initialize the chain.")
    elif st.session_state.vector_store and st.session_state.use_complete_docs:
         st.info("Vector database is built/loaded, but Complete Documents mode is active.")
    else:
         # Fallback for any other unhandled state - unlikely with the checks above
         st.warning("Please configure settings in the sidebar and load/build the database or select documents to start chatting.")

# --- Chat Bottom Buttons ---
# Move these buttons to the bottom of the main chat area
st.markdown("---") # Separator before buttons

# Only show buttons if LLM is initialized, as reset might re-initialize conversation
# and export needs history which relies on LLM being available to generate.
if st.session_state.llm is not None:
    col1, col2 = st.columns(2) # Use columns for horizontal layout

    with col1:
        # Handle chat reset
        reset_chat = st.button("🔄 Reset Chat History")
        if reset_chat:
            st.session_state.chat_history = []
            # If vector store exists, re-initialize the conversation chain
            # This ensures memory is cleared but the chain is ready if DB is loaded.
            st.sidebar.write("Resetting chat history.")
            # Only re-initialize the vector DB chain if the vector store is actually loaded AND
            # we are NOT in complete docs mode.
            if st.session_state.vector_store and not st.session_state.use_complete_docs:
                st.sidebar.write("Vector DB loaded and mode is active, re-initializing conversation chain...")
                conversation, llm, retriever = initialize_conversation(
                    st.session_state.vector_store,
                    model_type, # Use current selection
                    model_name, # Use current selection
                    st.session_state.k_value, # Use current selection
                    ollama_base_url if model_type == "Ollama" else None, # Use current selection
                    st.session_state.use_reranking, # Use current selection
                    st.session_state.num_chunks_kept, # Use current selection
                    st.session_state.temperature # Pass the current temperature
                )
                st.session_state.conversation = conversation
                 # Update the main session state LLM and retriever from the chain initialization
                st.session_state.llm = llm # Use the LLM instance returned by initialize_conversation
                st.session_state.retriever = retriever
                if st.session_state.conversation:
                     st.sidebar.success("Conversation chain re-initialized after reset.")
                else:
                     st.sidebar.error("Failed to re-initialize conversation chain after reset.")
            else:
                 # If no vector store or in complete docs mode, ensure chain/retriever are None
                 st.session_state.conversation = None
                 st.session_state.retriever = None
                 # st.session_state.llm remains from the independent init block

            st.rerun() # Rerun the script to clear chat display


    with col2:
        # Add Quarto export button
        if st.button("📥 Download Chat as Quarto (.qmd)"):
            if st.session_state.chat_history:
                qmd_content = convert_chat_to_qmd(st.session_state.chat_history)
                # Display the download link directly after the button click
                st.markdown(get_download_link(qmd_content), unsafe_allow_html=True)
            else:
                st.warning("No chat history to export yet.")
# --- End Chat Bottom Buttons ---