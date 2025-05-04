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
from typing import List, Dict, Tuple
import warnings # Import warnings to handle console warnings

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
# Import necessary message classes for history handling in non-chain modes
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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
    print(f"--> Extracting metadata for: {os.path.basename(document_path)}")
    try:
        # Read the document content - limit to first few pages/lines for efficiency
        content_preview = ""
        # Use a simple approach to get initial text based on extension
        try:
            if document_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(document_path)
                # Load only the first few pages
                doc_pages = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0))
                content_preview = " ".join([p.page_content for p in doc_pages[:5]]) # Use first 5 chunks
            elif document_path.lower().endswith(('.txt', '.md', '.qmd')):
                loader = TextLoader(document_path)
                doc = loader.load()[0]
                content_preview = doc.page_content[:5000] # First 5000 chars
            elif document_path.lower().endswith('.ipynb'):
                 loader = NotebookLoader(document_path)
                 doc = loader.load()[0]
                 content_preview = doc.page_content[:5000] # First 5000 chars
            elif document_path.lower().endswith('.csv'):
                 loader = CSVLoader(document_path)
                 # Load first row or a small sample
                 doc = loader.load()[0]
                 content_preview = doc.page_content[:5000] # First 5000 chars
            elif document_path.lower().endswith(('.json', '.jsonl')):
                 # Attempt to read first few lines for preview
                 with open(document_path, 'r', encoding='utf-8', errors='ignore') as f:
                     lines = [next(f) for x in range(20) if x < 1000] # Read up to 20 lines, max 1000 chars total
                     content_preview = "".join(lines)[:5000]
            else:
                 # Fallback for any other type or error during loading
                 content_preview = "Could not load content preview for this file type."
        except Exception as e:
             print(f"Error loading content preview from {document_path}: {e}")
             content_preview = f"Could not load content preview due to error: {e}"


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
                 warnings.warn(f"Could not find valid JSON format for metadata from {os.path.basename(document_path)}. Attempting manual parsing or using defaults.")
                 metadata_str = "{}" # Provide empty JSON to trigger robust default handling


        try:
            metadata = json.loads(metadata_str)
            if not isinstance(metadata, dict):
                 warnings.warn(f"LLM response for {os.path.basename(document_path)} was not a JSON object. Using defaults.")
                 metadata = {} # Use empty dict if not a dict

            # Ensure required keys exist with default "N/A" if missing
            required_keys = ["file_type_detected", "author", "title", "year", "journal", "apa_reference", "abstract", "description"]
            for key in required_keys:
                if key not in metadata:
                    metadata[key] = "N/A"

            # Use filename if title is N/A or empty
            if metadata.get("title", "N/A") == "N/A" or not metadata.get("title"):
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


            # Add the original source path to metadata (absolute path is safer)
            metadata['source'] = os.path.abspath(document_path)
            print(f"Metadata extracted and source added for {os.path.basename(document_path)}")
            # print(json.dumps(metadata, indent=2)) # Avoid printing full metadata to console unless necessary

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
                "description": "Metadata extraction failed due to JSON error.",
                "source": os.path.abspath(document_path)
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
            "description": "Metadata extraction failed.",
            "source": os.path.abspath(document_path)
        }


def save_metadata_json(metadata: Dict, source_path: str, metadata_dir: str) -> str | None:
    """
    Save metadata to a JSON file in the metadata directory with the same basename as the source.
    Returns the path to the saved JSON file or None on failure.
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
        print(f"Metadata saved to {os.path.basename(json_path)}")
        return json_path
    except Exception as e:
        warnings.warn(f"Error saving metadata to {json_path}: {e}")
        return None


def read_metadata_from_json_path(json_file_path: str) -> Dict | None:
    """
    Reads metadata from a specific JSON file path.
    Returns metadata dict or None if file doesn't exist or is invalid.
    """
    if not os.path.exists(json_file_path):
        # This case shouldn't happen if called from get_documents_with_metadata glob
        return None
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            if isinstance(metadata, dict):
                 # Ensure 'source' key exists and is absolute if possible
                 # We trust that the metadata file was created with an absolute 'source' path
                 if 'source' not in metadata or not os.path.isabs(metadata['source']):
                      warnings.warn(f"Metadata file {os.path.basename(json_file_path)} has missing or non-absolute 'source' key.")
                      metadata['source'] = None # Mark as unresolved source

                 return metadata
            else:
                warnings.warn(f"Metadata file {json_file_path} is not a valid JSON object.")
                return None
    except json.JSONDecodeError as e:
        warnings.warn(f"Error decoding JSON from {json_file_path}: {e}. Metadata may be corrupt.")
        return None
    except Exception as e:
        warnings.warn(f"Unexpected error reading metadata from {json_file_path}: {e}.")
        return None


def prepend_reference_to_chunks(chunks, metadata):
    """
    Prepend relevant context information (APA reference, title, or description)
    to each chunk from the document's metadata. Used only for Vector DB mode.
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
            # Avoid adding if chunk content is extremely short or just whitespace
            if chunk.page_content.strip():
                 chunk.page_content = f"{context_info}\n\n---\n\n{chunk.page_content}"

    return chunks

def split_csv(csv_string: str) -> List[str]:
    """
    Split a comma-separated string into a list of strings.
    Works for single values as well.
    """
    if not csv_string:
        return []

    # Split the string by commas and strip whitespace
    result = [item.strip() for item in csv_string.split(',')]
    return result

def create_embeddings(embedding_type: str, embedding_model: str, ollama_base_url: str | None = None):
    """Create the appropriate embedding model based on type"""
    try:
        if embedding_type == "Ollama":
            if not ollama_base_url:
                # warnings.warn("Ollama Base URL is not set for embeddings.") # Avoid redundant warnings
                return None
            return OllamaEmbeddings(
                base_url=ollama_base_url,
                model=embedding_model
            )
        elif embedding_type == "OpenAI":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                # warnings.warn("OpenAI API key is not set in environment variables for embeddings.") # Avoid redundant warnings
                return None
            return OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=api_key
            )
        elif embedding_type == "Google":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                 # warnings.warn("Google API key is not set in environment variables for embeddings.") # Avoid redundant warnings
                 return None
            return GoogleGenerativeAIEmbeddings(
                model=embedding_model,
                google_api_key=api_key
            )
        else:
            warnings.warn(f"Unsupported embedding type: {embedding_type}")
            return None
    except Exception as e:
        print(f"Error creating embedding model {embedding_model} ({embedding_type}): {e}")
        st.error(f"Error creating embedding model: {e}")
        return None


def create_or_update_vector_store(docs_dir: str, db_path: str, embedding_type: str, embedding_model: str, model_type: str, model_name: str, ollama_embedding_base_url: str | None = None, ollama_llm_base_url: str | None = None, llm_temperature: float = 0.2):
    """Create or update a vector store from documents in the specified directory"""
    print("---")
    print("🛠️ Create/Update Vector DB button pressed...")
    print("Scanning documents and checking status...")

    if not docs_dir or not os.path.exists(docs_dir):
         st.sidebar.error(f"Documents directory not found: {docs_dir}") # Keep user error visible
         print(f"Error: Documents directory not found: {docs_dir}")
         return None

    metadata_dir = os.path.join(docs_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True) # Ensure metadata directory exists

    # Create the selected LLM for metadata extraction (using a low temperature)
    # Use the LLM settings provided for metadata extraction
    metadata_llm = create_llm(model_type, model_name, ollama_llm_base_url, temperature=0.1) # Use low temp for metadata
    if metadata_llm is None:
        st.sidebar.error("Failed to initialize LLM for metadata extraction. Aborting DB process.") # Keep user error visible
        print("Error: Failed to initialize LLM for metadata extraction.")
        return None

    print(f"Using {model_type} model: {model_name} (temp=0.1) for metadata extraction.")

    # Create embeddings
    embeddings = create_embeddings(embedding_type, embedding_model, ollama_embedding_base_url)
    if embeddings is None:
         st.sidebar.error("Failed to initialize embeddings model. Aborting DB process.") # Keep user error visible
         print("Error: Failed to initialize embeddings model.")
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
             print(f"Existing database found and collection '{collection_name}' loaded.")
             db_exists = True
             try:
                 count = collection.count()
                 print(f"Collection contains {count} existing items.")
             except Exception as e:
                 warnings.warn(f"Could not retrieve count from existing collection: {e}")


        except InvalidCollectionException:
            print(f"Database directory exists at '{db_path}', but collection '{collection_name}' not found. Will create new collection.")
            db_exists = False # Treat as if DB didn't exist
            db_needs_full_recreate = True # Indicate we need to create the collection/DB
            # No need to delete directory here, Chroma client will create the collection.

    except Exception as e:
        warnings.warn(f"Error connecting to or loading database at '{db_path}': {e}. Will attempt to create a new database.")
        db_exists = False
        db_needs_full_recreate = True # Indicate we need to create the DB from scratch
        shutil.rmtree(db_path, ignore_errors=True) # Clean up potential corrupt DB
        os.makedirs(db_path, exist_ok=True)


    # --- Identify documents to process ---
    # Get list of all potential source files (exclude metadata directory itself)
    # Use glob to find all files, then filter out the metadata directory
    all_source_files = [f for f in glob.glob(os.path.join(docs_dir, "**/*"), recursive=True) if os.path.isfile(f)]
    all_source_files = [f for f in all_source_files if not os.path.normpath(f).startswith(os.path.normpath(metadata_dir))] # Exclude files inside metadata_dir

    print(f"Found {len(all_source_files)} potential source documents in '{docs_dir}'.")

    docs_to_add_paths = [] # Paths of documents that need to be added
    skipped_docs_count = 0

    # Determine which documents need adding
    for source_path in all_source_files:
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        json_path = os.path.join(metadata_dir, f"{base_name}.json")

        # If DB needs full recreate, process ALL found documents
        if db_needs_full_recreate:
             print(f"✅ Marking {os.path.basename(source_path)} for addition (New DB or Collection).")
             docs_to_add_paths.append(source_path)
        else:
            # In update mode (DB/Collection exists), check for existing metadata file
            if os.path.exists(json_path):
                # Document metadata exists, skip extraction and addition
                print(f"⏭️ Skipping {os.path.basename(source_path)}: Metadata exists.")
                skipped_docs_count += 1
                # NO append to docs_to_add_paths if skipping
            else:
                # Metadata does not exist, this document needs to be added
                print(f"✅ Marking {os.path.basename(source_path)} for addition (Missing metadata).")
                docs_to_add_paths.append(source_path)

    print(f"Summary: {len(docs_to_add_paths)} documents marked for addition, {skipped_docs_count} skipped.")


    if not docs_to_add_paths:
        if db_exists:
            print("No new documents found to add to the existing database.")
            # st.info("No new documents found to add.") # Could add user info message here
            return vector_store # Return the loaded existing DB
        else:
             st.error("No documents found or marked for processing. Cannot create a database.") # Keep user error visible
             print("Error: No documents found or marked for processing.")
             return None

    print(f"Processing {len(docs_to_add_paths)} documents for addition...")

    # --- Process documents to be added ---
    processed_docs_for_addition = []
    for source_path in docs_to_add_paths:
        # Metadata extraction only happens for docs in docs_to_add_paths
        print(f"- Processing: {os.path.basename(source_path)}")
        try:
            loader = get_document_loader(source_path)
            if loader is None:
                warnings.warn(f"  Unsupported file type, skipping: {os.path.basename(source_path)}")
                continue

            documents = loader.load()
            if not documents:
                 warnings.warn(f"  Loader returned no documents for {os.path.basename(source_path)}, skipping.")
                 continue

            doc = documents[0] # Assume one Langchain Document object per file

            # Extract metadata
            metadata = extract_document_metadata(source_path, metadata_llm) # This call is correctly placed

            # Save metadata to JSON
            json_path = save_metadata_json(metadata, source_path, metadata_dir)
            # Logging already inside save_metadata_json

            # Add metadata to document's metadata (ensuring 'source' is included)
            # extract_document_metadata already adds 'source' (absolute path)
            doc.metadata.update(metadata) # Update with extracted metadata

            processed_docs_for_addition.append(doc)

        except Exception as e:
            print(f"  Error processing document {os.path.basename(source_path)}: {str(e)}. Skipping.")
            st.sidebar.error(f"Error processing {os.path.basename(source_path)}: {str(e)}") # Keep critical errors visible


    if not processed_docs_for_addition:
        if not db_exists:
            st.error("No documents were successfully processed to create a new database.") # Keep user error visible
            print("Error: No documents successfully processed.")
            return None
        else:
             print("No *new* documents were successfully processed to add to the database.")
             # st.info("No *new* documents successfully processed to add.") # Could add user info message here
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
    print("Splitting processed documents into chunks...")
    for doc in processed_docs_for_addition:
        try:
            # Split the individual document
            doc_chunks = text_splitter.split_documents([doc])

            # Get the metadata for this document
            doc_metadata = doc.metadata

            # Prepend reference/context to each chunk (only needed for Vector DB use)
            # We still do it here during DB creation/update as these chunks go INTO the DB
            doc_chunks = prepend_reference_to_chunks(doc_chunks, doc_metadata)

            chunks_to_add.extend(doc_chunks)
            print(f"  Split '{os.path.basename(doc.metadata.get('source', 'Unknown'))}' into {len(doc_chunks)} chunks.")
        except Exception as e:
            # Fix the f-string syntax error here
            warnings.warn(f"  Error splitting or prepping chunks for {os.path.basename(doc.metadata.get('source', 'Unknown'))}: {e}. Skipping chunks for this document.")


    if not chunks_to_add:
        if not db_exists:
            st.error("No text chunks were generated to create a new database.") # Keep user error visible
            print("Error: No chunks generated.")
            return None
        else:
            print("No text chunks were generated from new documents to add to the database.")
            # st.info("No text chunks generated from new documents.") # Could add user info message here
            return vector_store # Return existing DB if no new chunks

    # --- Add the chunks to the vector store ---
    batch_size = 100 # Process in batches to avoid potential limits
    total_chunks = len(chunks_to_add)
    total_batches = (total_chunks + batch_size - 1) // batch_size

    if db_needs_full_recreate:
        # Create a new vector store from all chunks
        print(f"Creating a new database and collection '{collection_name}' with {total_chunks} chunks...")
        try:
            vector_store = Chroma.from_documents(
                documents=chunks_to_add, # Use all chunks
                embedding=embeddings,
                persist_directory=db_path,
                collection_name=collection_name
            )
            print("✅ New vector store created successfully!")
            return vector_store
        except Exception as e:
            st.error(f"Error creating new vector store: {str(e)}") # Keep user error visible
            print(f"Error creating new vector store: {str(e)}")
            shutil.rmtree(db_path, ignore_errors=True) # Clean up failed creation
            return None

    elif db_exists and vector_store:
        # Add documents to the existing vector store
        print(f"Adding {total_chunks} new chunks to the existing database...")
        for i in range(0, total_chunks, batch_size):
            batch_num = (i // batch_size) + 1
            end_idx = min(i + batch_size, total_chunks)
            batch = chunks_to_add[i:end_idx]

            print(f"  Adding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

            try:
                # Generate unique IDs if Chroma needs them explicitly for add_documents
                # Langchain's add_documents should handle this, but if issues arise,
                # manually generating IDs might be needed. For now, assume Langchain handles it.
                vector_store.add_documents(documents=batch)
                print(f"  Batch {batch_num} completed.")
            except Exception as e:
                 warnings.warn(f"  Error adding batch {batch_num}: {e}. Attempting to continue with next batch.")
                 # Decide how to handle errors here - continue, break, etc.
                 # For now, we'll just log and continue
                 pass

        print("✅ New documents added to the existing vector store!")
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
             print("Re-loaded vector store instance after adding documents.")
        except Exception as e:
             warnings.warn(f"Could not re-load vector store instance after adding documents: {e}")
             # Keep the potentially old vector_store instance if reload fails


        return vector_store

    else:
         # This case should ideally not be reached if the logic is correct
         st.error("Unexpected state in create_or_update_vector_store. Aborting.") # Keep user error visible
         print("Error: Unexpected state in create_or_update_vector_store.")
         return None


def load_vector_store(db_path: str, embedding_type: str, embedding_model: str, ollama_base_url: str | None = None):
    """Load an existing vector store"""
    print("Attempting to load existing vector database...")
    try:
        if not os.path.exists(db_path):
            warnings.warn(f"Database directory not found at {db_path}.")
            return None

        # Create embeddings
        embeddings = create_embeddings(embedding_type, embedding_model, ollama_base_url)
        if embeddings is None:
             warnings.warn("Failed to create embeddings model for loading.")
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
             print(f"Chroma collection '{collection_name}' found.")
        except InvalidCollectionException:
             st.warning(f"Vector database directory exists at `{db_path}`, but the collection '{collection_name}' was not found. Please rebuild the database.") # Keep user error visible
             print(f"Warning: Chroma collection '{collection_name}' not found.")
             return None
        except Exception as e:
             st.error(f"Error accessing Chroma collection '{collection_name}': {e}. Database might be corrupt. Please rebuild.") # Keep user error visible
             print(f"Error accessing Chroma collection: {e}")
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
                 st.warning(f"Vector database collection '{collection_name}' is empty. Please rebuild the database.") # Keep user error visible
                 print(f"Warning: Collection '{collection_name}' is empty.")
                 return None
            else:
                 print(f"Loaded database with {count} items.")
        except Exception as e:
            warnings.warn(f"Could not get item count from database collection: {e}. Database might need rebuilding.")
            # Continue loading anyway, maybe count failed but DB is usable
            pass # Or return None if count is critical

        print("✅ Existing vector database loaded successfully!")
        return vector_store
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}") # Keep user error visible
        print(f"Error loading vector store: {str(e)}")
        return None

def create_llm(model_type: str, model_name: str, ollama_base_url: str | None = None, temperature: float = 0.2):
    """Create the appropriate LLM based on model type, name, and temperature"""
    try:
        if model_type == "Ollama":
            if not ollama_base_url:
                # warnings.warn("Ollama Base URL is not set for LLM.") # Avoid spamming warnings
                return None
            return ChatOllama(
                base_url=ollama_base_url,
                model=model_name,
                temperature=temperature
            )
        elif model_type == "OpenAI":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                # warnings.warn("OpenAI API key is not set in environment variables.") # Avoid spamming warnings
                return None
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_key=api_key
            )
        elif model_type == "Anthropic":
            api_key = os.getenv("CLAUDE_API_KEY")
            if not api_key:
                 # warnings.warn("Anthropic API key is not set in environment variables.") # Avoid spamming warnings
                 return None
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
                anthropic_api_key=api_key
            )
        elif model_type == "Google":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                 # warnings.warn("Google API key is not set in environment variables.") # Avoid spamming warnings
                 return None
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=api_key
            )
        else:
            warnings.warn(f"Unsupported model type: {model_type}")
            return None
    except Exception as e:
         print(f"Error creating LLM {model_name} ({model_type}): {e}") # Log to console
         st.error(f"Error creating LLM: {e}") # Keep user error visible
         return None

def initialize_conversation(vector_store, model_type: str, model_name: str, k_value: int = 10, ollama_llm_base_url: str | None = None, use_reranking: bool = False, num_chunks_kept: int = 4, llm_temperature: float = 0.2, context_option: str = "Vector Database"):
    """Initialize the conversation chain or base LLM based on context option."""
    print("Initializing conversation/LLM...")
    try:
        # Always create the LLM instance based on current settings
        llm = create_llm(model_type, model_name, ollama_llm_base_url, llm_temperature)
        if llm is None:
             st.warning(f"Cannot initialize LLM: Model creation failed for {model_name} ({model_type}). Check API keys/endpoints.") # Keep user error visible
             print(f"Error: Cannot initialize LLM: Model creation failed for {model_name} ({model_type}).")
             # Clear existing conversation/LLM state if creation fails
             st.session_state.conversation = None
             st.session_state.llm = None
             st.session_state.retriever = None
             return None, None, None # Return None for all

        st.session_state.llm = llm # Store the LLM in session state
        print(f"LLM ({model_name}, Temp={llm_temperature}) created.")

        # Only create the ConversationalRetrievalChain and retriever if using "Vector Database" context
        if context_option == "Vector Database":
            if vector_store is None:
                st.warning("Cannot initialize Vector Database conversation: Vector store is not loaded.") # Keep user error visible
                print("Error: Cannot initialize Vector Database conversation: Vector store is None.")
                # Clear existing chain/retriever state
                st.session_state.conversation = None
                st.session_state.retriever = None
                return None, llm, None # Return LLM but no chain/retriever

            print(f"Creating Vector Database conversation chain (k={k_value}, reranking={use_reranking}, kept={num_chunks_kept})...")

            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )

            # Create retriever
            retriever = vector_store.as_retriever(search_kwargs={"k": k_value})
            print(f"Retriever created (k={k_value}).")

            # Reranking with ContextualCompressionRetriever
            if use_reranking:
                print(f"Using LLM Reranking, keeping top {num_chunks_kept} chunks...")
                # Ensure num_chunks_kept is not greater than k_value
                num_chunks_kept = min(num_chunks_kept, k_value)
                try:
                    compressor = LLMChainExtractor.from_llm(llm)
                    retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever, k=num_chunks_kept)
                    print("Reranking retriever created.")
                except Exception as e:
                    warnings.warn(f"Failed to create Reranking retriever: {e}. Proceeding without reranking.")
                    st.warning(f"Failed to set up LLM Reranking: {e}. Reranking disabled for this session.") # Keep user error visible
                    # Fallback: keep the basic retriever
                    pass # use_reranking flag will be false if checkbox was unchecked

            # Make the chain verbose so we can see prompts in stdout
            conversation = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                verbose=True
            )
            st.session_state.conversation = conversation # Store the chain
            st.session_state.retriever = retriever # Store the retriever

            print("✅ Vector Database Conversation chain initialized.")
            return conversation, llm, retriever # Return all

        else:
            # For other context options, we don't use the ConversationalRetrievalChain
            print(f"Using '{context_option}' context option. Initializing base LLM.")
            # Clear existing chain/retriever state as it's not used
            st.session_state.conversation = None
            st.session_state.retriever = None
            print("✅ Base LLM initialized.")
            return None, llm, None # Return only the LLM


    except Exception as e:
        st.error(f"Error initializing conversation/LLM: {str(e)}") # Keep user error visible
        print(f"Error initializing conversation/LLM: {str(e)}")
        # Clear all related state on failure
        st.session_state.conversation = None
        st.session_state.llm = None
        st.session_state.retriever = None
        return None, None, None # Return None for all

def extract_markdown_content(text: str, type: str = "json") -> str:
        """Extract content from markdown code blocks."""
        start_tag = f"```{type}"
        end_tag = """```"""

        start_idx = text.find(start_tag)
        if start_idx == -1:
             return text.strip() # Return original text if no start tag

        start_idx += len(start_tag)
        # Handle potential newline immediately after start_tag
        if start_idx < len(text) and text[start_idx] == '\n':
             start_idx += 1

        end_idx = text.rfind(end_tag, start_idx) # Search for end tag after start

        if end_idx != -1:
            return (text[start_idx:end_idx]).strip()
        else:
             return text[start_idx:].strip() # Return from start tag to end if no end tag


# Function to execute chain of thought reasoning with preserved context
# (Keep this function as is, it uses the llm passed to it)
# Note: This function is only called when context_option is "Vector Database" due to UI logic (Q4-A)
def execute_chain_of_thought(llm, retriever, prompt, max_steps=5):
    """Executes a chain of thought reasoning process using the LLM and retriever."""
    if llm is None or retriever is None:
        st.error("Chain of Thought requires an initialized LLM and Retriever.") # Keep user error visible
        print("Error: CoT: LLM or Retriever is None.")
        return "Error: LLM or Retriever not initialized for Chain of Thought.", []

    print("Starting Chain of Thought process (Vector DB mode)...")
    # Initial prompt to get chain of thought planning
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

    # Get the relevant documents using the retriever
    print("Retrieving documents for Chain of Thought context...")
    try:
        retrieved_docs = retriever.get_relevant_documents(prompt)
        context = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in retrieved_docs])
        if not retrieved_docs:
             print("No documents retrieved for context.")
             context = "No relevant documents found."
        else:
             print(f"Retrieved {len(retrieved_docs)} documents for context.")

    except Exception as e:
        print(f"Error retrieving documents for CoT: {e}")
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
    print("Generating Chain of Thought plan...")
    plan_response = llm.invoke(initial_prompt)
    plan_text = plan_response.content

    # Extract JSON plan
    try:
        json_str = extract_markdown_content(plan_text, "json")
        # print("plan:", json_str) # Debug print
        plan_json = json.loads(json_str)
        steps = plan_json.get("reasoning_steps", [])
        if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
             st.warning("LLM returned invalid plan format. Using default steps.") # Keep user error visible
             warnings.warn("CoT: LLM returned invalid plan format.")
             raise ValueError("Invalid plan format") # Trigger fallback

    except Exception as e:
        print(f"Error extracting/parsing JSON plan: {e}. LLM Output (first 500 chars): {plan_text[:500]}...") # Debug print
        st.warning("Failed to extract valid plan from LLM. Using default steps.") # Keep user error visible
        warnings.warn("CoT: Failed to parse plan JSON.")
        # If JSON parsing fails, create a default plan
        steps = [
            "Step 1: Analyze the user's question and identify the core concepts and information required.",
            "Step 2: Review the provided context information from the retrieved documents to find relevant facts, data, or descriptions related to the question.",
            "Step 3: Synthesize information from the context and the question, combining relevant pieces and performing any necessary analysis or calculations.",
            "Step 4: Formulate a clear and comprehensive final answer based on the synthesis, directly addressing the user's question.",
        ]


    total_steps = min(len(steps), max_steps)  # Limit to max_steps

    print(f"Executing {total_steps} reasoning steps.")

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
        step_prompt_template = """
        You are executing step {current_step_number} of a chain of thought process to answer the user's question.
        Your goal is to perform the task described in "Current step to execute" using the available information.

        Original User Question: {user_question}

        Accumulated Information (Original context from documents + reasoning from previous steps):
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
            st.warning(f"Error in CoT step {i+1}: {e}") # Keep user error visible
            print(f"Error in CoT step {i+1}: {e}")


        # Add to outputs list (for summary/display later)
        step_output_display.append(f"Step {i+1} ({current_step_description}): {step_result}")

        # Add this step's reasoning to the growing context for the *next* step
        growing_context += f"Reasoning for Step {i+1} ({current_step_description}):\n{step_result}\n\n---\n\n"

        # Display intermediate step in an expander
        with st.expander(f"Step {i+1}: {current_step_description}", expanded=False):
            st.markdown(step_result) # Use markdown for step output display

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
        print("✅ Chain of Thought process completed.")
    except Exception as e:
        final_answer = f"Error generating final answer after steps: {e}"
        st.error(final_answer) # Keep user error visible
        print(final_answer)
        # As a fallback, just concatenate step outputs
        # final_answer = "Could not synthesize final answer. Here are the step results:\n\n" + "\n\n".join(step_output_display)


    progress_bar.empty() # Clear the progress bar

    # Note: We return step_output_display which contains descriptions + results for display,
    # but the final_answer is generated from the `growing_context` which includes all reasoning.
    return final_answer, step_output_display


def convert_chat_to_qmd(chat_history: List[Dict]) -> str:
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

def get_download_link(qmd_content: str, filename: str = "rag_chatbot_conversation.qmd") -> str:
    """
    Generate a download link for the QMD content
    """
    # Encode the content as bytes, then base64
    b64 = base64.b64encode(qmd_content.encode('utf-8')).decode()
    href = f'<a href="data:text/markdown;base64,{b64}" download="{filename}">Download Conversation as Quarto Document</a>'
    return href

# Helper to get document loader based on file extension
def get_document_loader(file_path: str):
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
             print(f"Using generic JSONLoader for {os.path.basename(file_path)}. May need custom schema.")
             return JSONLoader(file_path=file_path, jq_schema=".", text_content=False) # text_content=False loads dict/list
        elif ext == ".csv":
            return CSVLoader(file_path)
        elif ext == ".ipynb":
            return NotebookLoader(file_path)
        else:
            return None # Indicate unsupported file type
    except Exception as e:
         print(f"Error creating loader for {os.path.basename(file_path)} (ext: {ext}): {e}")
         return None

def load_document_content(file_path: str) -> str:
    """Loads the full text content of a document using the appropriate loader."""
    try:
        loader = get_document_loader(file_path)
        if loader:
            docs = loader.load()
            if docs:
                # Concatenate content from all pages/parts of the document
                return "\n".join([doc.page_content for doc in docs])
        return "" # Return empty string if loading fails or no content
    except Exception as e:
        st.warning(f"Error loading full content from {os.path.basename(file_path)}: {e}") # Keep user error visible
        print(f"Error loading full content from {os.path.basename(file_path)}: {e}")
        return ""


def get_documents_with_metadata(docs_dir: str) -> List[Tuple[str, str, Dict]]:
    """
    Lists documents in docs_dir that have a corresponding metadata JSON file.
    Returns a list of tuples: (full_source_path, filename, metadata_dict).
    Uses filename as the display name.
    """
    print(f"Scanning for metadata files in {docs_dir}/metadata...")
    if not docs_dir or not os.path.exists(docs_dir):
        print("Docs directory not found.")
        return []

    metadata_dir = os.path.join(docs_dir, "metadata")
    if not os.path.exists(metadata_dir):
        print("Metadata directory not found.")
        return []

    docs_list = []
    # List all json files in the metadata directory
    metadata_files = glob.glob(os.path.join(metadata_dir, "*.json"))
    print(f"Found {len(metadata_files)} potential metadata files.")

    for meta_json_path in metadata_files:
        # Read metadata directly from the JSON file path
        metadata = read_metadata_from_json_path(meta_json_path)

        if metadata:
            # Get the *actual* source path from the loaded metadata (should be absolute)
            source_path = metadata.get('source')
            if source_path and os.path.exists(source_path):
                # Use the filename as the display name as requested
                filename = os.path.basename(source_path)
                docs_list.append((source_path, filename, metadata))
                # print(f"  Found valid metadata for source: {filename}") # Avoid excessive logging
            else:
                warnings.warn(f"Metadata file found ({os.path.basename(meta_json_path)}), but source file '{source_path}' not found or missing 'source' key.")
        else:
             # Warning already logged by read_metadata_from_json_path
             pass # Skip this metadata file if loading failed

    print(f"Found metadata for {len(docs_list)} documents with existing source files.")
    return docs_list


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

# Context specific session state
if 'context_option' not in st.session_state:
    st.session_state.context_option = "Vector Database" # Default context option
if 'selected_document_paths' not in st.session_state:
    st.session_state.selected_document_paths = [] # For Full Docs/Abstracts mode

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
if 'use_cot_checkbox' not in st.session_state: # Add state for CoT checkbox
     st.session_state.use_cot_checkbox = False


# Flag to track if initialization has happened
if 'initialized' not in st.session_state:
     st.session_state.initialized = False


# Title and description
st.title("🤖 RAG Chatbot")
st.markdown("Ask questions about your documents.")

# Sidebar for configuration
with st.sidebar:
    # --- Documents Directory Input (Moved to top) ---
    docs_dir = st.text_input("Documents Directory", os.getenv("SOURCE_DOC_DIR", "docs"), key='docs_directory_input') # Unique key
    # Ensure docs_dir is created if it doesn't exist
    if docs_dir:
        try:
            os.makedirs(docs_dir, exist_ok=True)
            # print(f"Using documents directory: {docs_dir}") # Avoid spamming
        except OSError as e:
            st.error(f"Error creating/accessing documents directory {docs_dir}: {e}") # Keep user error visible
            print(f"Error creating/accessing documents directory {docs_dir}: {e}")


    # Use expander for Vector DB Settings
    with st.expander("Vector DB Settings", expanded=False): # Collapsed by default
        # Input for documents directory (Removed from here)

        # Chunk size slider
        chunk_size = st.slider(
            "Chunk Size",
            min_value=100, # Smaller min for diverse docs
            max_value=8000,
            value=st.session_state.chunk_size, # Use session state default/current
            step=100,
            help="Size of text chunks when processing documents (in characters)",
            key="chunk_size_slider" # Unique key
        )
        st.session_state.chunk_size = chunk_size

        # Chunk overlap slider
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=1000,
            value=st.session_state.chunk_overlap, # Use session state default/current
            step=50,
            help="Amount of overlap between consecutive chunks (in characters)",
            key="chunk_overlap_slider" # Unique key
        )
        st.session_state.chunk_overlap = chunk_overlap

        # Embedding model selection
        embedding_providers_str = os.getenv("EMBEDDING_PROVIDERS", "Ollama,OpenAI,Google")
        embedding_type = st.selectbox(
            "Select Embedding Model Provider",
            split_csv(embedding_providers_str),
            index=split_csv(embedding_providers_str).index(os.getenv("DEFAULT_EMBEDDING_PROVIDER", "Ollama")) if os.getenv("DEFAULT_EMBEDDING_PROVIDER", "Ollama") in split_csv(embedding_providers_str) else 0,
            key="embedding_provider_selectbox_key" # Unique key
        )

        embedding_model = "nomic-embed-text:latest" # Default placeholder, updated below
        ollama_embedding_base_url = None # Initialize here

        if embedding_type == "Ollama":
            embedding_model = st.text_input("Ollama Embedding Model", os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest"), key="ollama_embedding_model_input") # Unique key
            # Corrected unique key for Ollama Base URL for embeddings
            ollama_embedding_base_url = st.text_input("Ollama Base URL (Embeddings)", os.getenv("OLLAMA_END_POINT", "http://localhost:11434"), key='ollama_embedding_url_key') # Unique key


        elif embedding_type == "OpenAI":
            embedding_model = st.text_input("OpenAI Embedding Model", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"), key="openai_embedding_model_input") # Unique key
        elif embedding_type == "Google":
            embedding_model = st.text_input("Google Embedding Model", os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001"), key="google_embedding_model_input") # Unique key

        db_path = os.path.join(docs_dir, "vectorstore")

        # Database operations button (moved to the bottom of this section)
        build_db_button = st.button("🔨 Create/Update Vector DB")

    # New Context Settings Expander
    with st.expander("Context Settings", expanded=True): # Expanded by default
        context_options = ["Vector Database", "Abstracts Only", "Full Documents", "No Context/Base Model"]
        # Find the current index for the selectbox to maintain state
        current_context_index = context_options.index(st.session_state.context_option) if st.session_state.context_option in context_options else 0
        selected_context_option = st.selectbox(
            "Select Context Option",
            context_options,
            index=current_context_index,
            key="context_option_selectbox_key" # Add a unique key
        )

        # Update session state if the selection changed
        if selected_context_option != st.session_state.context_option:
            st.session_state.context_option = selected_context_option
            st.session_state.chat_history = [] # Clear chat history on context change (Q5-A)
            st.session_state.initialized = False # Mark for re-initialization
            st.session_state.conversation = None # Force re-initialization
            st.session_state.llm = None
            st.session_state.retriever = None
            st.session_state.selected_document_paths = [] # Clear selected docs on context change
            st.session_state.use_cot_checkbox = False # Disable CoT on context change
            st.rerun() # Rerun to update UI based on new context option

        # Conditional UI based on context_option
        if st.session_state.context_option == "Vector Database":
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
            use_reranking = st.checkbox("Use LLM Reranking", value=st.session_state.use_reranking, key="use_reranking_checkbox") # Unique key
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
                    help="Number of chunks to keep after LLM reranking. Must be less than or equal to the number of retrieved documents (K).",
                    key="num_chunks_kept_slider" # Unique key
                )
                st.session_state.num_chunks_kept = num_chunks_kept
            else:
                 # If reranking is off, effectively all K chunks are kept
                 st.session_state.num_chunks_kept = st.session_state.k_value


        elif st.session_state.context_option in ["Full Documents", "Abstracts Only"]:
            st.write(f"Select documents to provide as context ({st.session_state.context_option}):")
            # List documents that have metadata files
            available_docs_tuples = get_documents_with_metadata(docs_dir) # Returns list of (path, filename, metadata)

            if not available_docs_tuples:
                 st.warning("No documents with metadata found in the documents directory. Build/Update DB first to extract metadata.") # Keep user warning visible
                 st.session_state.selected_document_paths = []
            else:
                 # Create lists for filenames and corresponding paths
                 available_filenames = [filename for path, filename, meta in available_docs_tuples]
                 # Map filename to its original path for easy lookup
                 filename_to_path_map = {filename: path for path, filename, meta in available_docs_tuples}

                 # Get the currently selected filenames based on paths in session state
                 # Filter to ensure the path still exists and is in the current available list
                 current_selected_filenames = [os.path.basename(path) for path in st.session_state.selected_document_paths if path in filename_to_path_map.values()]


                 # --- Select All/None Buttons ---
                 col_select_all, col_select_none = st.columns(2)
                 with col_select_all:
                     # Use a button with a callback to update the multiselect's default
                     if st.button("Select All", key="select_all_docs_button"):
                         st.session_state.selected_document_paths = [path for path, name, meta in available_docs_tuples] # Select all original paths
                         st.rerun() # Rerun to update multiselect state
                 with col_select_none:
                      if st.button("Select None", key="select_none_docs_button"):
                           st.session_state.selected_document_paths = [] # Select no paths
                           st.rerun() # Rerun to update multiselect state
                 # --- End Select All/None Buttons ---


                 # Use a multiselect for document selection, displaying filenames
                 selected_filenames = st.multiselect(
                     "Choose documents",
                     available_filenames, # Display filenames
                     default=current_selected_filenames, # Set default from session state using filenames
                     key="document_multiselect_key" # Unique key
                 )
                 # Update selected paths in session state based on chosen filenames
                 # We need to map the selected filenames back to their original full paths
                 st.session_state.selected_document_paths = [filename_to_path_map.get(filename) for filename in selected_filenames if filename in filename_to_path_map] # Ensure path exists


                 if st.session_state.context_option == "Full Documents":
                     st.warning("Using 'Full Documents' can easily exceed LLM context window limits. Response may be truncated by the model API.") # Keep user warning visible
                     # Optional: Add a way to check approximate token count of selected docs
                     total_chars = sum(len(load_document_content(path)) for path in st.session_state.selected_document_paths)
                     if total_chars > 0:
                          st.info(f"Approximate character count of selected documents: {total_chars}") # Keep user info visible
                          if total_chars > 100000: # Higher threshold for a second warning
                               st.error("Warning: Total document size is very large. Expect severe truncation.") # Keep user error visible


        elif st.session_state.context_option == "No Context/Base Model":
            st.info("The LLM will answer questions without any document context.") # Keep user info visible
            st.session_state.selected_document_paths = [] # Clear selected docs in this mode

    # LLM Settings Expander
    with st.expander("LLM Settings", expanded=True): # Expanded by default
        # Temperature Slider (already here, just ensure it's inside the expander)
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
             index=split_csv(llm_providers_str).index(os.getenv("DEFAULT_LLM_PROVIDER", "Ollama")) if os.getenv("DEFAULT_LLM_PROVIDER", "Ollama") in split_csv(llm_providers_str) else 0,
             key="llm_provider_selectbox_key" # Unique key
        )

        # Model selection based on provider
        model_name = "" # Default placeholder
        ollama_llm_base_url = None # Initialize here

        if model_type == "Ollama":
            ollama_model_str = os.getenv("OLLAMA_MODEL", "llama3:latest,mistral:latest")
            model_name = st.selectbox(
                "Select Ollama Model",
                split_csv(ollama_model_str),
                index=0, # Could try to get default from env var if exists
                key="ollama_llm_model_selectbox_key" # Unique key
            )
            ollama_llm_base_url = st.text_input("Ollama Base URL (LLM)", os.getenv("OLLAMA_END_POINT", "http://localhost:11434"), key='ollama_llm_url_key') # Unique key
        elif model_type == "OpenAI":
            openai_model_str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo,gpt-4o")
            model_name = st.selectbox(
                "Select OpenAI Model",
                split_csv(openai_model_str),
                index=0, # Could try to get default from env var if exists
                key="openai_llm_model_selectbox_key" # Unique key
            )
        elif model_type == "Anthropic":
            anthropic_model_str = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307,claude-3-sonnet-20240229")
            model_name = st.selectbox(
                "Select Anthropic Model",
                split_csv(anthropic_model_str),
                index=0, # Could try to get default from env var if exists
                key="anthropic_llm_model_selectbox_key" # Unique key
            )
        elif model_type == "Google":
            google_model_str = os.getenv("GOOGLE_MODEL", "gemini-pro")
            model_name = st.selectbox(
                "Select Google Model",
                split_csv(google_model_str),
                index=0, # Could try to get default from env var if exists
                key="google_llm_model_selectbox_key" # Unique key
            )

        # Add Chain of Thought option
        # Only enable if Vector Database context is selected (Q4-A)
        cot_disabled = st.session_state.context_option != "Vector Database"
        use_cot = st.checkbox(
            "Use Chain of Thought",
            value=st.session_state.use_cot_checkbox, # Use state value
            disabled=cot_disabled,
            help="Enable step-by-step reasoning (only available with 'Vector Database' context)" if cot_disabled else "Enable step-by-step reasoning",
            key="use_cot_checkbox" # Unique key
        )
        # The state is automatically managed by key="use_cot_checkbox"

# Handle database building/updating (placed outside sidebar but triggered by sidebar button)
if build_db_button:
    if not docs_dir:
         st.error("Please specify a documents directory.") # Keep user error visible
    elif not embedding_model:
         st.error("Please specify an embedding model.") # Keep user error visible
    elif embedding_type == "Ollama" and ollama_embedding_base_url is None: # Check the specific URL variable
         st.error("Please specify the Ollama Base URL for embeddings.") # Keep user error visible
    elif not model_name:
         st.error("Please select or specify an LLM model name.") # Keep user error visible
    elif model_type == "Ollama" and ollama_llm_base_url is None: # Check the specific URL variable
         st.error("Please specify the Ollama Base URL for the LLM.") # Keep user error visible
    else:
        with st.spinner("Processing documents and updating vector database..."):
            vector_store = create_or_update_vector_store(
                docs_dir,
                db_path,
                embedding_type,
                embedding_model,
                model_type, # Pass model_type for metadata LLM
                model_name, # Pass model_name for metadata LLM
                ollama_embedding_base_url, # Pass embedding Ollama URL
                ollama_llm_base_url, # Pass LLM Ollama URL
                st.session_state.temperature # Pass the current temperature
            )
            if vector_store:
                st.session_state.vector_store = vector_store
                # Initialize conversation with new/updated vector store and current settings
                st.session_state.chat_history = [] # Clear chat on DB update (Q5-A related behavior)
                st.session_state.initialized = False # Mark for re-initialization below
                # No need to call initialize_conversation here, the block below handles it on rerun
            # else: st.error("Failed to create or update vector database. Check console logs.") # Error shown inside function


# --- Initial Load & Re-initialization Logic ---
# This block runs on initial app load, after reset, after context change, or after DB update
# if the chain/LLM is not initialized.
# Ensure all necessary settings are available before attempting initialization.
# Read URLs directly from sidebar state if they exist
current_ollama_embedding_base_url = st.session_state.get('ollama_embedding_url_key', None)
current_ollama_llm_base_url = st.session_state.get('ollama_llm_url_key', None)
# Read model names and types from sidebar state
current_embedding_type = st.session_state.get('embedding_provider_selectbox_key', split_csv(os.getenv("EMBEDDING_PROVIDERS", "Ollama,OpenAI,Google"))[0]) # Default to first available
# Get the actual embedding model string from the correct input based on selected type
if current_embedding_type == "Ollama":
    current_embedding_model = st.session_state.get('ollama_embedding_model_input', os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest"))
elif current_embedding_type == "OpenAI":
    current_embedding_model = st.session_state.get('openai_embedding_model_input', os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
elif current_embedding_type == "Google":
    current_embedding_model = st.session_state.get('google_embedding_model_input', os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001"))
else:
    current_embedding_model = None


current_model_type = st.session_state.get('llm_provider_selectbox_key', split_csv(os.getenv("LLM_PROVIDERS", "Ollama,OpenAI,Anthropic,Google"))[0]) # Default to first available
# Get the actual LLM model string from the correct input based on selected type
if current_model_type == "Ollama":
    current_model_name = st.session_state.get('ollama_llm_model_selectbox_key', split_csv(os.getenv("OLLAMA_MODEL", "llama3:latest"))[0] if split_csv(os.getenv("OLLAMA_MODEL", "llama3:latest")) else None)
elif current_model_type == "OpenAI":
    current_model_name = st.session_state.get('openai_llm_model_selectbox_key', split_csv(os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))[0] if split_csv(os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")) else None)
elif current_model_type == "Anthropic":
     current_model_name = st.session_state.get('anthropic_llm_model_selectbox_key', split_csv(os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"))[0] if split_csv(os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")) else None)
elif current_model_type == "Google":
    current_model_name = st.session_state.get('google_llm_model_selectbox_key', split_csv(os.getenv("GOOGLE_MODEL", "gemini-pro"))[0] if split_csv(os.getenv("GOOGLE_MODEL", "gemini-pro")) else None)
else:
    current_model_name = None


required_settings_available = (
    docs_dir is not None and os.path.exists(docs_dir) and
    current_embedding_model is not None and current_model_name is not None and
    (current_embedding_type != "Ollama" or current_ollama_embedding_base_url is not None) and # Check embedding URL if Ollama is selected for embeddings
    (current_model_type != "Ollama" or current_ollama_llm_base_url is not None) # Check LLM URL if Ollama is selected for LLM
)

# Check if initialization is needed
if not st.session_state.initialized and required_settings_available:
     # Attempt to load vector store first if context is Vector DB
     vector_store_to_pass = None
     if st.session_state.context_option == "Vector Database":
          db_path = os.path.join(docs_dir, "vectorstore")
          if os.path.exists(db_path):
              if st.session_state.vector_store is None: # Only load if not already loaded
                   with st.spinner("Loading existing vector database..."):
                       vector_store_to_pass = load_vector_store(
                           db_path,
                           current_embedding_type, # Use current selection
                           current_embedding_model, # Use current selection
                           current_ollama_embedding_base_url # Use embedding Ollama URL from state
                       )
                       st.session_state.vector_store = vector_store_to_pass # Store loaded VS (can be None)
              else:
                   vector_store_to_pass = st.session_state.vector_store # Use already loaded VS
          else:
               st.sidebar.warning(f"Vector database directory not found at `{db_path}`. Cannot load for Vector Database context.") # Keep user warning visible
               print(f"Warning: Vector database directory not found at `{db_path}`. Cannot load for Vector Database context.")
               st.session_state.vector_store = None # Ensure state is None if dir doesn't exist
               vector_store_to_pass = None


     # Now, initialize the conversation/LLM based on the loaded VS (if any) and context option
     # Only attempt to initialize the chain/LLM if required settings are complete AND (DB is loaded if using Vector DB context)
     # If context is NOT Vector DB, we only need LLM settings available
     llm_initialization_ok = (current_model_name is not None and (current_model_type != "Ollama" or current_ollama_llm_base_url is not None))

     if llm_initialization_ok and ((st.session_state.context_option == "Vector Database" and st.session_state.vector_store is not None) or st.session_state.context_option != "Vector Database"):
         with st.spinner(f"Initializing LLM for '{st.session_state.context_option}' context..."):
             conversation, llm, retriever = initialize_conversation(
                 vector_store_to_pass, # Pass the potentially loaded vector store
                 current_model_type, # Use current selection
                 current_model_name, # Use current selection
                 st.session_state.k_value, # Use current selection
                 current_ollama_llm_base_url if current_model_type == "Ollama" else None, # Pass LLM ollama URL from state
                 st.session_state.use_reranking, # Use current selection
                 st.session_state.num_chunks_kept, # Use current selection
                 st.session_state.temperature, # Pass the current temperature
                 st.session_state.context_option # Pass the selected context option
             )
             # Storing llm, conversation, retriever is handled inside initialize_conversation now
             # st.session_state.conversation = conversation
             # st.session_state.llm = llm
             # st.session_state.retriever = retriever # This will be None for non-Vector DB options

             if st.session_state.llm:
                  st.session_state.initialized = True # Mark as initialized ONLY if LLM is successfully created
                  if st.session_state.context_option == "Vector Database" and st.session_state.conversation is None and st.session_state.vector_store is not None:
                       st.warning("LLM initialized, but Vector DB chain failed to initialize. Check DB status or console logs.") # Keep user warning visible
                  else:
                       st.success(f"App initialized with '{st.session_state.context_option}' context.") # Keep user success visible
             else:
                  st.error("App initialization failed. Check sidebar settings and API keys/endpoints.") # Keep user error visible
                  print("Error: App initialization failed.")
                  st.session_state.initialized = False # Ensure it's false if LLM creation failed
     elif not llm_initialization_ok:
          st.warning("LLM settings are incomplete. Cannot initialize LLM.") # Keep user warning visible
          print("Warning: LLM settings are incomplete. Cannot initialize LLM.")
          st.session_state.initialized = False
     else:
          # This case happens if context is Vector DB but VS load/init failed
          st.sidebar.warning("Required components not available for initialization.") # Keep user warning visible
          print("Warning: Required components not available for initialization.")
          st.session_state.initialized = False


# Display chat messages
for message in st.session_state.chat_history:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"]) # Use markdown for message display

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message immediately
    with st.chat_message("user", avatar="🧑"):
         st.markdown(prompt)

    # Add user message to chat history immediately
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Check if LLM is initialized before proceeding
    if st.session_state.llm is None:
        st.error("LLM is not initialized. Please check your settings and try building/loading the database.") # Keep user error visible
        with st.chat_message("assistant", avatar="🤖"):
             st.markdown("Sorry, I'm not initialized yet. Please check the sidebar settings.")
        st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, I'm not initialized yet. Please check the sidebar settings."})
    else:
        # --- Process based on Context Option ---
        answer = None
        try:
            with st.spinner("Thinking..."):
                current_context_option = st.session_state.context_option
                llm_instance = st.session_state.llm # Get the initialized LLM

                if current_context_option == "Vector Database":
                    if st.session_state.conversation:
                         # Prepare chat history for the chain
                         langchain_history = []
                         # Iterate in steps of 2 (user, assistant pairs)
                         # Exclude the current user message being processed
                         history_for_chain = st.session_state.chat_history[:-1] if len(st.session_state.chat_history) > 0 else []

                         for i in range(0, len(history_for_chain), 2):
                              if i + 1 < len(history_for_chain): # Ensure there's a pair
                                 user_msg = history_for_chain[i]
                                 ai_msg = history_for_chain[i+1]
                                 if user_msg['role'] == 'user' and ai_msg['role'] == 'assistant':
                                      langchain_history.append((user_msg['content'], ai_msg['content']))

                         # Use Chain of Thought if enabled and in Vector DB mode (Q4-A)
                         if st.session_state.use_cot_checkbox and st.session_state.retriever:
                             answer, step_outputs = execute_chain_of_thought(
                                 llm_instance,
                                 st.session_state.retriever, # CoT uses the retriever directly
                                 prompt
                             )
                         else:
                             # Use the standard ConversationalRetrievalChain
                             response = st.session_state.conversation.invoke({"question": prompt, "chat_history": langchain_history})
                             answer = response["answer"]

                    else:
                         answer = "Vector Database chain is not initialized. Please check settings or rebuild the database." # Keep user error visible
                         st.warning(answer)

                elif current_context_option in ["Full Documents", "Abstracts Only"]:
                    selected_paths = st.session_state.selected_document_paths
                    if not selected_paths:
                        answer = f"No documents selected for '{current_context_option}' context. Please select documents in the sidebar." # Keep user error visible
                        st.warning(answer)
                    else:
                        context_string = ""
                        metadata_dir = os.path.join(docs_dir, "metadata") # Define metadata_dir here as well

                        if current_context_option == "Full Documents":
                            print(f"Loading full content for {len(selected_paths)} documents...")
                            full_contents = []
                            total_chars = 0
                            for path in selected_paths:
                                content = load_document_content(path)
                                full_contents.append(f"--- Document: {os.path.basename(path)} ---\n{content}\n--- End Document ---\n\n")
                                total_chars += len(content)

                            context_string = "".join(full_contents)

                            print(f"Total characters loaded: {total_chars}. Potential token limit issues may occur.")
                            if total_chars > 80000: # Warning threshold (can be adjusted)
                                 st.warning(f"Loaded {total_chars} characters from selected documents. This may exceed the LLM's context window.") # Keep user warning visible


                        elif current_context_option == "Abstracts Only":
                            print(f"Loading abstracts/metadata for {len(selected_paths)} documents...")
                            abstract_contexts = []
                            # metadata_dir defined above

                            for path in selected_paths:
                                # Derive the metadata JSON path from the source file path
                                meta_json_path = os.path.join(metadata_dir, os.path.splitext(os.path.basename(path))[0] + '.json')
                                metadata = read_metadata_from_json_path(meta_json_path) # Load metadata using its own path
                                if metadata:
                                    abstract_context = f"--- Document Metadata: {metadata.get('title', os.path.basename(path))} ---\n"
                                    # Include APA reference if available, otherwise other key fields (Q2-D)
                                    if metadata.get('apa_reference', 'N/A') != "N/A":
                                         abstract_context += f"Reference: {metadata['apa_reference']}\n"
                                    else: # Fallback to title, author, year, journal
                                        if metadata.get('author', 'N/A') != "N/A": abstract_context += f"Author(s): {metadata['author']}\n"
                                        if metadata.get('year', 'N/A') != "N/A": abstract_context += f"Year: {metadata['year']}\n"
                                        if metadata.get('journal', 'N/A') != "N/A": abstract_context += f"Journal: {metadata['journal']}\n"

                                    if metadata.get('abstract', 'N/A') != "N/A":
                                         abstract_context += f"Abstract: {metadata['abstract']}\n"
                                    elif metadata.get('description', 'N/A') != "N/A":
                                         abstract_context += f"Description: {metadata['description']}\n"
                                    abstract_context += "--- End Metadata ---\n\n"
                                    abstract_contexts.append(abstract_context)
                                else:
                                     warnings.warn(f"Could not load metadata for {os.path.basename(path)} from {os.path.basename(meta_json_path)}") # Specify which JSON failed

                            context_string = "".join(abstract_contexts)


                        # Construct the prompt for the LLM
                        # For simplicity, we won't include full chat history in the prompt
                        # for these modes, only the current question and the context.
                        # If chat history is needed, a manual chain/prompt builder would be required.
                        if context_string:
                             # Use SystemMessage for context and instruction
                             messages = [
                                SystemMessage(content=f"""You are a helpful assistant. Use the provided context to answer the user's question.
                                If you cannot answer the question based on the context, state that you cannot find the information in the provided documents.
                                Context:
                                ---
                                {context_string}
                                ---
                                """),
                                HumanMessage(content=prompt) # Just current question
                             ]

                             # Invoke the LLM directly
                             response = llm_instance.invoke(messages)
                             answer = response.content
                        else:
                             answer = f"No context generated from selected documents for '{current_context_option}'. Cannot answer." # Keep user error visible
                             st.warning(answer)


                elif current_context_option == "No Context/Base Model":
                     print("Invoking base model without context...")
                     # Construct prompt with just the user question
                     # Again, simplifying by not including chat history manually
                     messages = [HumanMessage(content=prompt)]
                     response = llm_instance.invoke(messages)
                     answer = response.content


                # Ensure an answer was generated
                if answer is None:
                     answer = "An internal error occurred while processing your request." # Keep user error visible
                     st.error(answer)

        except Exception as e:
            st.error(f"An error occurred during processing: {e}") # Keep user error visible
            print(f"An error occurred during processing: {e}")
            answer = "Sorry, I encountered an unexpected error."
            # Optional: Log the full traceback
            import traceback
            print(traceback.format_exc())

        # Display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(answer)

        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})


else:
    # Message to display if initialization hasn't happened or failed
    docs_dir_exists = os.path.exists(docs_dir) if docs_dir else False
    db_path = os.path.join(docs_dir, "vectorstore") if docs_dir else "path/to/vectorstore"
    db_dir_exists = os.path.exists(db_path) if docs_dir else False

    if not docs_dir or not docs_dir_exists:
        st.info(f"Documents directory `{docs_dir}` not found or not specified. Please create the directory or enter a valid path in the sidebar.") # Keep user info visible
    elif st.session_state.context_option == "Vector Database" and not db_dir_exists:
         st.info(f"Vector database directory not found at `{db_path}`. Please configure settings in the sidebar and click 'Create/Update Vector DB'.") # Keep user info visible
    elif not st.session_state.llm: # Check if LLM was created successfully during init
         st.warning("LLM is not initialized. Please check LLM settings (Provider, Model, API Keys/Endpoints) in the sidebar.") # Keep user warning visible
         if st.session_state.context_option == "Vector Database":
              st.warning("If using 'Vector Database' context, you also need a valid database. Click 'Create/Update Vector DB'.") # Keep user warning visible

    if not st.session_state.initialized and required_settings_available:
         # If settings are available but not initialized, it means initialization block failed or is running
         st.info("Initializing application components...") # Keep user info visible
    elif not st.session_state.initialized:
         st.warning("App not fully initialized. Please check settings in the sidebar.") # Keep user warning visible
    else:
         # This case might be reached if init succeeded but something else is wrong
         st.warning("Please configure settings in the sidebar and ensure the LLM is initialized to start chatting.") # Keep user warning visible


# --- Chat Bottom Buttons ---
col1, col2 = st.columns(2) # Use columns for horizontal layout

with col1:
    # Handle chat reset
    reset_chat = st.button("🔄 Reset Chat")
    if reset_chat:
        st.session_state.chat_history = []
        # Clear initialization state to force re-init with current sidebar settings
        st.session_state.initialized = False
        st.session_state.conversation = None
        st.session_state.llm = None
        st.session_state.retriever = None
        st.session_state.selected_document_paths = [] # Also clear selected docs
        print("Chat reset requested.")
        st.rerun() # Rerun the script to clear chat display and re-initialize


with col2:
    # Add Quarto export button
    if st.button("📥 Download Chat as Quarto (.qmd)"):
        if st.session_state.chat_history:
            qmd_content = convert_chat_to_qmd(st.session_state.chat_history)
            # Display the download link directly after the button click
            st.markdown(get_download_link(qmd_content), unsafe_allow_html=True)
        else:
            st.warning("No chat history to export yet.") # Keep user warning visible
# --- End Chat Bottom Buttons ---