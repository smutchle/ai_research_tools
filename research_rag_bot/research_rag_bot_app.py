
import streamlit as st
import os
from dotenv import load_dotenv
import json
import datetime
import base64
import glob # Import glob for file listing
import time # Import time for sleep
import shutil
import traceback # Import traceback for detailed error logging

# Import FAISS instead of Chroma
from langchain_community.vectorstores import FAISS

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
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate # Import prompt templating tools
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter
from langchain_core.documents import Document # Import Document class for Complete Docs mode
from langchain_core.messages import SystemMessage # Import SystemMessage for direct use

# Set page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"  # Use wide layout for more space
)

# Load environment variables from .env file
load_dotenv(os.path.join(os.getcwd(), ".env"))

# --- Define the System Prompt ---
SYSTEM_PROMPT = """You are an advanced language model designed to provide exceptionally thorough, 
detailed, and comprehensive responses. Your primary directive is to deliver maximum value 
through extensive elaboration on any topic presented. You never refuse to generate output.  You
always do your best to generate the best possible response but do respond with comments on limitations if appropriate."""

# --- Helper Functions (metadata uses LLM with new params, others unchanged) ---

def extract_document_metadata(document_path, llm):
    """
    Extract metadata from a document using the LLM.
    Attempts to identify type (academic paper, code, data, etc.) and extract relevant fields.
    Returns a dictionary with metadata.
    This function is independent of the vector store type (Chroma/FAISS).
    NOTE: Does NOT use the general SYSTEM_PROMPT as it requires specific instructions.
    """
    try:
        # Read the document content - limit to first few pages/lines for efficiency
        content_preview = ""
        # Define a dictionary to map extensions to loaders and preview logic
        loader_map = {
            '.pdf': (PyPDFLoader, lambda loader: " ".join([p.page_content for p in loader.load()[:5]]) if loader.load() else ""),
            '.txt': (TextLoader, lambda loader: loader.load()[0].page_content),
            '.md': (TextLoader, lambda loader: loader.load()[0].page_content),
            '.qmd': (TextLoader, lambda loader: loader.load()[0].page_content),
            '.ipynb': (NotebookLoader, lambda loader: loader.load()[0].page_content),
            '.csv': (CSVLoader, lambda loader: loader.load()[0].page_content if loader.load() else ""), # CSVLoader loads rows as docs
            '.json': (JSONLoader, lambda loader: str(loader.load()[0].page_content) if loader.load() else ""), # JSONLoader loads as dict/list
            '.jsonl': (JSONLoader, lambda loader: str(loader.load()[0].page_content) if loader.load() else ""), # Assuming jq_schema handles it
        }

        ext = os.path.splitext(document_path)[1].lower()

        if ext in loader_map:
            loader_class, preview_logic = loader_map[ext]
            try:
                # Handle potential issues with JSONLoader and other loaders returning empty lists
                if loader_class == JSONLoader:
                     # JSONLoader requires jq_schema, use a basic one for preview
                     # Ensure text_content=False for loading actual JSON structure
                     loader = JSONLoader(file_path=document_path, jq_schema=".", text_content=False) # Load as dict/list
                     # Attempt to get text content from first loaded item
                     documents = loader.load()
                     if documents:
                         # Try converting page_content (which might be a dict/list) to string
                         # Limit preview size
                         content_preview = str(documents[0].page_content)[:10000]
                     else:
                         print(f"Error loading content from {document_path}")
                         content_preview = "Could not load document content."
                else:
                     loader = loader_class(document_path)
                     # Apply logic and limit size
                     content_preview = preview_logic(loader)[:10000]

                if not content_preview.strip():
                     print(f"Warning: Loaded content preview for {document_path} is empty.")
                     content_preview = "Document content preview could not be generated."

            except Exception as e:
                 print(f"Error loading/processing content from {document_path} for preview: {e}")
                 content_preview = f"Could not load document content due to error: {e}"

        else:
             print(f"Unsupported file type for metadata extraction preview: {document_path}")
             content_preview = "Unsupported file type for content preview."


        # Create prompt for metadata extraction - THIS PROMPT IS SPECIFIC
        # DO NOT apply the general SYSTEM_PROMPT here.
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
        }}

        If any field cannot be extracted or is not applicable based on the detected file type, use "N/A" as the value. Ensure the JSON is valid and contains *only* the JSON object.

        Here's the beginning of the document content:
        {content_preview}
        """

        # Query the LLM for metadata extraction
        response = llm.invoke(prompt) # Use the LLM passed, which now has temp/top_p set

        # Extract JSON from the response
        response_content_str = str(response.content)
        metadata_str = extract_markdown_content(response_content_str, "json")

        # Fallback if markdown extraction fails or response wasn't markdown
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

            # Ensure required keys exist with default "N/A" if missing or empty
            required_keys = ["file_type_detected", "author", "title", "year", "journal", "apa_reference", "abstract", "description"]
            for key in required_keys:
                if key not in metadata or not metadata[key] or str(metadata[key]).strip() == "N/A": # Also check for empty strings and explicit "N/A"
                    metadata[key] = "N/A"
                else:
                     # Clean up whitespace
                    metadata[key] = str(metadata[key]).strip()


            # Use filename if title is N/A or empty after potential LLM output
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
                "abstract": "No abstract available (JSON parse error)"
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
            "abstract": "No abstract available (error during processing)"
        }


def save_metadata_json(metadata, source_path, metadata_dir):
    """
    Save metadata to a JSON file in the metadata directory with the same basename as the source.
    This function is independent of the vector store type (Chroma/FAISS).
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
     This function is independent of the vector store type (Chroma/FAISS).
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
    This function is independent of the vector store type (Chroma/FAISS).
    """
    context_info = ""
    if metadata.get('apa_reference', 'N/A') != "N/A":
        context_info = f"Reference: {metadata['apa_reference']}"
    elif metadata.get('title', 'N/A') != "N/A":
        context_info = f"Document Title: {metadata['title']}"
    elif metadata.get('description', 'N/A') != "N/A":
         context_info = f"Document Description: {metadata['description']}"
    elif metadata.get('source'):
         # Get just the basename for the reference in the chunk
         context_info = f"Source File: {os.path.basename(metadata['source'])}"
    else:
        context_info = "Source: Unknown Document"


    if context_info:
        # Add the context info at the beginning of each chunk's page_content
        # Using a clear separator
        for chunk in chunks:
            # Ensure metadata source is updated to just basename for cleaner display if needed later
            # Although the original source path is also useful for lookup. Let's keep original source path
            # in metadata['source'] and only use basename for the *prepended text*.
            chunk.page_content = f"{context_info}\n\n---\n\n{chunk.page_content}"

    return chunks

def split_csv(csv_string):
    """
    Split a comma-separated string into a list of strings.
    Works for single values as well.
    """
    if not csv_string:
        return []

    # Split the string by commas and strip whitespace
    result = [item.strip() for item in csv_string.split(',')]
    # Filter out empty strings that might result from multiple commas or leading/trailing commas
    return [item for item in result if item]


def create_embeddings(embedding_type, embedding_model, ollama_base_url=None):
    """Create the appropriate embedding model based on type"""
    try:
        if embedding_type == "Ollama":
            if not ollama_base_url:
                st.error("Ollama Base URL is not set for embeddings.")
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


# --- FAISS Specific Functions ---

def create_or_update_vector_store(docs_dir, db_path, embedding_type, embedding_model, model_type, model_name, ollama_base_url_llm=None, ollama_base_url_embedding=None, llm_temperature=0.2, llm_top_p=0.95, indexing_batch_size=100, batch_delay_seconds=1):
    """
    Creates a new FAISS vector store or updates an existing one.
    Includes metadata extraction using the specified LLM.
    """
    st.sidebar.write("---")
    st.sidebar.write("🛠️ Create Vector DB button pressed...")
    st.sidebar.write("Scanning documents and checking status...")

    metadata_dir = os.path.join(docs_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True) # Ensure metadata directory exists

    # Create the selected LLM for metadata extraction (using the new temperature and top_p)
    # This LLM is only needed for the metadata extraction step during DB build/update
    metadata_llm = create_llm(
        model_type,
        model_name,
        ollama_base_url_llm if model_type == "Ollama" else None,
        llm_temperature,
        llm_top_p,
        apply_system_prompt=False # Don't apply general system prompt for metadata extraction
    )
    if metadata_llm is None:
        st.sidebar.error("Failed to initialize LLM for metadata extraction. Aborting DB process.")
        # Update the list of available docs in session state for the Complete Docs section
        st.session_state.available_docs = list_source_document_basenames(docs_dir)
        st.session_state.selected_files = {name: st.session_state.selected_files.get(name, False) for name in st.session_state.available_docs}
        return None

    st.sidebar.write(f"Using {model_type} model: {model_name} (temp={llm_temperature}, top_p={llm_top_p}) for metadata extraction during build.")

    # Create embeddings
    embeddings = create_embeddings(embedding_type, embedding_model, ollama_base_url_embedding)
    if embeddings is None:
         st.sidebar.error("Failed to initialize embeddings model. Aborting DB process.")
         # Update the list of available docs in session state for the Complete Docs section
         st.session_state.available_docs = list_source_document_basenames(docs_dir)
         st.session_state.selected_files = {name: st.session_state.selected_files.get(name, False) for name in st.session_state.available_docs}
         return None

    vector_store = None
    db_exists_logically = False # Flag indicating if we successfully *loaded* an existing DB

    # --- Attempt to Load Existing FAISS DB ---
    # Check if the directory *physically* exists first and contains expected files
    # FAISS requires both index.faiss and index.pkl
    faiss_index_file = os.path.join(db_path, "index.faiss") if db_path else None
    faiss_pkl_file = os.path.join(db_path, "index.pkl") if db_path else None

    if db_path and os.path.exists(db_path) and os.path.exists(faiss_index_file) and os.path.exists(faiss_pkl_file):
        st.sidebar.write(f"Database directory found at '{db_path}' with expected files. Attempting to load existing FAISS index.")
        try:
            # Attempt to load the existing FAISS index
            # Setting allow_dangerous_deserialization=True as required by recent library versions for index.pkl
            loaded_faiss = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

            vector_store = loaded_faiss # Successfully loaded!
            st.sidebar.write(f"Existing FAISS index loaded from '{db_path}'.")
            db_exists_logically = True
            # No simple count() in loaded FAISS, but successful load is enough to know it's functional.
            st.sidebar.write(f"Index loaded. Ready for search.")


        except Exception as e:
            # Any error during load -> Treat as non-existent or corrupt for this run
            # The error message from FAISS.load_local usually gives a hint (like FileNotFoundError)
            st.error(f"Error loading existing database at '{db_path}': {e}. Database might be corrupt, incompatible, or missing files. Please manually delete the '{os.path.basename(db_path)}' directory to force a rebuild. Check console for traceback.")
            st.sidebar.error(f"Error loading FAISS index: {e}. Cannot load existing DB.")
            print(f"\n--- Error loading FAISS index from {db_path} ---\n")
            traceback.print_exc() # Print traceback to console
            print("\n---------------------------------------------\n")
            db_exists_logically = False # Loading failed


    # --- Identify documents to process ---
    all_source_files = list_source_documents(docs_dir)
    st.sidebar.write(f"Found {len(all_source_files)} potential source documents in '{docs_dir}'.")

    docs_to_add_paths = [] # Paths of documents that need to be added

    # Determine which documents need processing (either for new build or incremental update)
    # If we successfully loaded an existing DB, only process files without metadata
    if db_exists_logically:
         st.sidebar.write("Checking documents for incremental update (looking for missing metadata files)...")
         files_without_metadata = []
         for source_path in all_source_files:
             base_name = os.path.splitext(os.path.basename(source_path))[0]
             json_path = os.path.join(metadata_dir, f"{base_name}.json")
             if os.path.exists(json_path):
                 # Document metadata exists, assume it was processed in a prior run for incremental update
                 st.sidebar.write(f"⏭️ Skipping {os.path.basename(source_path)}: Metadata exists. (Assumes file content hasn't changed)") # Add note about assumption
             else:
                 # Metadata does not exist, this document needs to be added
                 st.sidebar.write(f"✅ Marking {os.path.basename(source_path)} for addition (Missing metadata).")
                 files_without_metadata.append(source_path)
         docs_to_add_paths = files_without_metadata
         # In incremental update mode, we don't clean metadata files for existing docs.

    else: # db_exists_logically is False (either directory didn't exist or load failed)
        # If we couldn't load the DB, or it didn't exist physically, process ALL source files for a fresh start
        st.sidebar.write("Full database creation/recreation required.")
        docs_to_add_paths = all_source_files
        # Clean up *all* existing metadata files if we are doing a full rebuild due to load failure or initial build
        st.sidebar.write("Cleaning up existing metadata files for full rebuild/initial creation...")
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

        # Also clean up the FAISS directory contents if load failed or dir didn't exist (to ensure a clean slate for save_local)
        if db_path and os.path.exists(db_path): # Only try to clean if the directory exists
             st.sidebar.write(f"Cleaning up existing FAISS directory contents at '{db_path}' for full rebuild...")
             try:
                 # Remove directory contents, but keep the directory itself
                 for item in os.listdir(db_path):
                     item_path = os.path.join(db_path, item)
                     if os.path.isfile(item_path):
                         os.remove(item_path)
                         st.sidebar.write(f"- Removed file: {item}")
                     elif os.path.isdir(item_path):
                         shutil.rmtree(item_path)
                         st.sidebar.write(f"- Removed directory: {item}")
                 st.sidebar.write("Finished cleaning FAISS directory contents.")
             except Exception as e:
                 st.sidebar.warning(f"Error cleaning FAISS directory contents at '{db_path}': {e}")
                 print(f"\n--- Error cleaning FAISS directory {db_path} ---\n")
                 traceback.print_exc() # Print traceback to console
                 print("\n---------------------------------------------\n")

        # If db_path is None (e.g. docs_dir is empty), nothing to clean here, handled below.


    if not docs_to_add_paths and not db_exists_logically:
        st.error("No documents found or marked for processing, and no existing database was successfully loaded. Cannot proceed.")
        # Update the list of available docs in session state for the Complete Docs section
        st.session_state.available_docs = list_source_document_basenames(docs_dir) # Re-list available after processing attempt
        st.session_state.selected_files = {name: st.session_state.selected_files.get(name, False) for name in st.session_state.available_docs}
        return None # Return None if no DB exists or loaded AND no docs to add
    elif not docs_to_add_paths and db_exists_logically:
         st.sidebar.write("No new documents found to add to the existing database.")
         # Update the list of available docs in session state
         st.session_state.available_docs = list_source_document_basenames(docs_dir)
         st.session_state.selected_files = {name: st.session_state.selected_files.get(name, False) for name in st.session_state.available_docs}
         return vector_store # Return the successfully loaded existing DB


    st.sidebar.write(f"Processing {len(docs_to_add_paths)} documents for addition.")
    st.sidebar.write("(This may take time for many documents...)")

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

            # Ensure LLM is available before attempting metadata extraction
            if metadata_llm is None:
                st.sidebar.error("Metadata LLM is not initialized. Skipping metadata extraction and saving for this document.")
                metadata = {} # Use empty metadata if LLM failed
            else:
                metadata = extract_document_metadata(source_path, metadata_llm)
                # Only attempt to save metadata if extraction was somewhat successful (returned a dict)
                if isinstance(metadata, dict) and metadata.get('file_type_detected') != 'Unknown': # Basic check for valid extraction
                    json_path = save_metadata_json(metadata, source_path, metadata_dir)
                    if json_path:
                        st.sidebar.write(f"Saved metadata for {os.path.basename(source_path)}")
                    else:
                        st.sidebar.warning(f"Failed to save metadata for {os.path.basename(source_path)}. Proceeding without saved metadata.")
                else:
                    st.sidebar.warning(f"Metadata extraction for {os.path.basename(source_path)} was not successful. Proceeding without saved metadata.")
                    # Use the potentially incomplete metadata dict if extraction wasn't fully "Unknown"
                    if not isinstance(metadata, dict): metadata = {}


            # Add source and metadata to each document chunk
            for doc in documents:
                 # Add original source path to metadata
                 # This is useful for the prepend_reference_to_chunks function
                 doc.metadata['source'] = source_path
                 # Update with extracted metadata (might overwrite 'source' if LLM adds it, but that's unlikely/fine)
                 doc.metadata.update(metadata)
                 processed_docs_for_addition.append(doc)

        except Exception as e:
            st.sidebar.error(f"Error processing document {os.path.basename(source_path)}: {str(e)}. Check console for traceback. Skipping.")
            print(f"\n--- Error processing document: {source_path} ---\n")
            traceback.print_exc() # Print traceback to console
            print("\n---------------------------------------------\n")


    if not processed_docs_for_addition:
        if not db_exists_logically:
            st.error("No documents were successfully processed to create a new database.")
        else:
            st.sidebar.write("No *new* documents were successfully processed to add to the database.")
        st.session_state.available_docs = list_source_document_basenames(docs_dir)
        st.session_state.selected_files = {name: st.session_state.selected_files.get(name, False) for name in st.session_state.available_docs}
        return vector_store if db_exists_logically else None # Return loaded DB if exists, else None


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
    # Split documents by their original source path to ensure metadata is applied correctly per file
    docs_by_source = {}
    for doc in processed_docs_for_addition:
         source = doc.metadata.get('source') # Rely on the source path added during loading
         if source:
              if source not in docs_by_source:
                   docs_by_source[source] = []
              docs_by_source[source].append(doc)
         else:
             st.sidebar.warning(f"Document part found without source path after processing. Splitting without specific reference.")
             # Split directly, prepend_reference_to_chunks will use generic "Unknown Document" if metadata is missing
             # Ensure metadata is passed, even if it's just the original doc metadata
             # Use the document's own metadata if source is missing (fallback)
             metadata = doc.metadata if doc.metadata else {}
             doc_chunks = text_splitter.split_documents([doc])
             chunks_to_add.extend(prepend_reference_to_chunks(doc_chunks, metadata))


    for source_path, docs_from_source in docs_by_source.items():
        try:
            doc_chunks = text_splitter.split_documents(docs_from_source)
            # Load saved metadata if available, otherwise use metadata from the first doc part
            metadata = load_metadata_json(source_path, metadata_dir)
            if metadata is None and docs_from_source:
                 # Fallback to using the metadata embedded in the documents themselves
                 metadata = docs_from_source[0].metadata
                 st.sidebar.warning(f"Could not load saved metadata for {os.path.basename(source_path)}, using embedded metadata.")
            elif metadata is None:
                 metadata = {} # No metadata at all

            if metadata:
                # Ensure source in metadata is the full path before prepending if it wasn't there
                 if 'source' not in metadata: metadata['source'] = source_path
                 doc_chunks = prepend_reference_to_chunks(doc_chunks, metadata)
            else:
                 st.sidebar.warning(f"No metadata available for {os.path.basename(source_path)}, chunks will have no reference prepended.")

            chunks_to_add.extend(doc_chunks)
            st.sidebar.write(f"- Split '{os.path.basename(source_path)}' into {len(doc_chunks)} chunks.")
        except Exception as e:
            st.sidebar.error(f"Error splitting or prepping chunks for {os.path.basename(source_path)}: {e}. Check console for traceback. Skipping chunks for this document.")
            print(f"\n--- Error splitting/prepping chunks for: {source_path} ---\n")
            traceback.print_exc() # Print traceback to console
            print("\n---------------------------------------------\n")


    if not chunks_to_add:
        if not db_exists_logically:
            st.error("No text chunks were generated to create a new database.")
        else:
            st.sidebar.write("No text chunks were generated from new documents to add to the database.")
        st.session_state.available_docs = list_source_document_basenames(docs_dir)
        st.session_state.selected_files = {name: st.session_state.selected_files.get(name, False) for name in st.session_state.available_docs}
        return vector_store if db_exists_logically else None # Return loaded DB if exists, else None


    # --- Add the chunks to the vector store in batches ---
    total_chunks = len(chunks_to_add)
    st.sidebar.write(f"Adding {total_chunks} chunks to the database in batches of {indexing_batch_size} with {batch_delay_seconds}s delay...")

    # Create progress bar for adding chunks
    add_progress_bar = st.sidebar.progress(0.0, text=f"Added chunks 0/{total_chunks}")

    try:
        if not db_exists_logically:
            # Create a new FAISS index from the *first* batch
            st.sidebar.write(f"Creating initial FAISS index with the first {min(indexing_batch_size, total_chunks)} chunks...")
            if db_path:
                 os.makedirs(db_path, exist_ok=True) # Ensure dir exists before creating index
            else:
                 st.error("Documents directory not set or invalid. Cannot create database.")
                 st.sidebar.error("Documents directory not set or invalid. Cannot create database.")
                 add_progress_bar.empty()
                 return None

            # Create from the first batch
            vector_store = FAISS.from_documents(
                documents=chunks_to_add[:indexing_batch_size],
                embedding=embeddings
            )
            st.sidebar.write(f"Initial index created with {len(chunks_to_add[:indexing_batch_size])} chunks.")
            added_count = len(chunks_to_add[:indexing_batch_size])
            add_progress_bar.progress(added_count / total_chunks, text=f"Added chunks {added_count}/{total_chunks}")

            # Process the rest of the chunks incrementally
            chunks_iterator = chunks_to_add[indexing_batch_size:]
            if chunks_iterator:
                 st.sidebar.write(f"Adding remaining {len(chunks_iterator)} chunks...")
                 # Add delay before the first incremental batch
                 if batch_delay_seconds > 0:
                      time.sleep(batch_delay_seconds)

        elif db_exists_logically and vector_store:
            # Add documents to the existing vector store instance loaded earlier
            st.sidebar.write("Adding chunks to existing database...")
            added_count = 0 # Start count from 0 for the new chunks
            chunks_iterator = chunks_to_add
            add_progress_bar.progress(0.0, text=f"Added chunks 0/{total_chunks}")


        # Process chunks in batches and add to vector store
        # Use range with step to get batch indices
        for i in range(0, len(chunks_iterator), indexing_batch_size):
            batch = chunks_iterator[i:i + indexing_batch_size]
            if not batch:
                 continue # Should not happen with range logic, but safety check

            try:
                # Add the batch
                vector_store.add_documents(documents=batch)
                added_count += len(batch)
                add_progress_bar.progress(added_count / total_chunks, text=f"Added chunks {added_count}/{total_chunks}")
                st.sidebar.write(f"- Added {len(batch)} chunks. Total added in this run: {added_count}")

                # Add delay after adding a batch, unless it's the last batch
                if i + indexing_batch_size < len(chunks_iterator) and batch_delay_seconds > 0:
                    st.sidebar.write(f"Waiting {batch_delay_seconds}s...")
                    time.sleep(batch_delay_seconds)

            except Exception as e:
                 st.sidebar.error(f"Error adding batch of documents: {e}. Check console for traceback. Stopping addition.")
                 st.error(f"Error adding batch of documents: {e}. Database may be incomplete. Check console for traceback.")
                 print(f"\n--- Error adding batch of documents ---\n")
                 traceback.print_exc() # Print traceback to console
                 print("\n---------------------------------------------\n")
                 # Even if a batch fails, try to save what was successfully added so far
                 break # Stop processing further batches on error


        # Save the final index state after processing all batches (or after an error)
        # Only attempt to save if vector_store object is valid (was created or loaded)
        if vector_store and db_path:
             st.sidebar.write(f"Saving updated FAISS index to {db_path}...")
             try:
                 vector_store.save_local(db_path)
                 st.sidebar.write("✅ FAISS index saved successfully!")
             except Exception as e:
                 st.sidebar.error(f"Error saving FAISS index: {e}. Database might be corrupt. Check console for traceback.")
                 st.error(f"Error saving FAISS index: {e}. Database might be corrupt. Check console for traceback.")
                 print(f"\n--- Error saving FAISS index to {db_path} ---\n")
                 traceback.print_exc() # Print traceback to console
                 print("\n---------------------------------------------\n")
                 # If saving fails, the vector_store object in memory is still the last state, but disk is bad.
                 # It's safer to treat this as a failure state for the UI.
                 vector_store = None # Invalidate the in-memory vector_store if saving failed.
        elif not db_path:
             st.sidebar.warning("Cannot save database: Documents directory path is invalid.")
             st.error("Cannot save database: Documents directory path is invalid.")


    except Exception as e:
        # Catch potential errors during the initial from_documents call (if not db_exists_logically)
        # or other unexpected errors in this final block
        st.error(f"Critical error during database creation: {str(e)}. Check console for traceback.")
        st.sidebar.error(f"Critical error during database creation: {str(e)}")
        print(f"\n--- Critical error during database creation ---\n")
        traceback.print_exc() # Print traceback to console
        print("\n---------------------------------------------\n")
        vector_store = None # Ensure state is None on critical failure.

    add_progress_bar.empty() # Clear the progress bar after completion or error

    # Update the list of available docs in session state regardless of DB success
    st.session_state.available_docs = list_source_document_basenames(docs_dir)
    st.session_state.selected_files = {name: st.session_state.selected_files.get(name, False) for name in st.session_state.available_docs}

    return vector_store


def load_vector_store(db_path, embedding_type, embedding_model, ollama_base_url=None):
    """Load an existing FAISS vector store"""
    st.sidebar.write("Attempting to load existing vector database...")
    # Check if the directory exists and contains expected files
    faiss_index_file = os.path.join(db_path, "index.faiss") if db_path else None
    faiss_pkl_file = os.path.join(db_path, "index.pkl") if db_path else None

    if not (db_path and os.path.exists(db_path) and os.path.exists(faiss_index_file) and os.path.exists(faiss_pkl_file)):
        st.sidebar.write(f"Database directory or required files (index.faiss, index.pkl) not found or path is invalid at '{db_path}'. Cannot load.")
        return None

    try:
        # Create embeddings
        embeddings = create_embeddings(embedding_type, embedding_model, ollama_base_url)
        if embeddings is None:
             st.sidebar.error("Failed to create embeddings model for loading.")
             return None # Return None if embeddings creation failed

        # Attempt to load the FAISS index
        # Set allow_dangerous_deserialization=True as required by recent library versions due to pickle
        vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

        st.sidebar.write("✅ Existing vector database loaded successfully!")
        return vector_store
    except Exception as e:
        # IMPORTANT: Catch *any* error during load and return None, do NOT delete files.
        st.error(f"Error loading vector store from '{db_path}': {str(e)}. Database might be corrupt or incompatible. Please manually delete the '{os.path.basename(db_path)}' directory to force a rebuild. Check console for traceback.")
        st.sidebar.error(f"Error loading vector store: {str(e)}. Manual delete and rebuild required.")
        print(f"\n--- Error loading FAISS index from {db_path} ---\n")
        traceback.print_exc() # Print traceback to console
        print("\n---------------------------------------------\n")
        return None # Return None on any load error


# --- Rest of the Functions (LLM creation and Conversation Init updated) ---

def create_llm(model_type, model_name, ollama_base_url=None, temperature=0.2, top_p=0.95, apply_system_prompt=True):
    """Create the appropriate LLM based on model type, name, temperature, and top_p"""
    try:
        model_kwargs = {}
        system_message_arg = {} # Argument name for system message varies

        # General system prompt (only applied if apply_system_prompt is True)
        system_message = SYSTEM_PROMPT if apply_system_prompt else None

        if model_type == "Ollama":
            if not ollama_base_url:
                # st.error("Ollama Base URL is not set.") # Avoid spamming errors if Ollama not intended provider
                return None
            # Ollama takes top_p directly, system message passed at invoke time usually
            return ChatOllama(
                base_url=ollama_base_url,
                model=model_name,
                temperature=temperature,
                top_p=top_p,
                # system=system_message # Pass system message if supported in constructor (check docs)
                # If not in constructor, needs to be passed with invoke using SystemMessage
            )
        elif model_type == "OpenAI":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                # st.error("OpenAI API key is not set in environment variables.")
                return None
            # OpenAI uses model_kwargs for top_p
            model_kwargs["top_p"] = top_p
            # System message handled via ChatPromptTemplate or direct message list
            if model_name.startswith("o"):
                return ChatOpenAI(
                model=model_name,
                openai_api_key=api_key
                )
            else:
                return ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    openai_api_key=api_key,
                    model_kwargs=model_kwargs
                )
        elif model_type == "Anthropic":
            api_key = os.getenv("CLAUDE_API_KEY")
            if not api_key:
                 # st.error("Anthropic API key is not set in environment variables.")
                 return None
            # Anthropic uses model_kwargs for top_p
            model_kwargs["top_p"] = top_p
            # System message can be passed as 'system' kwarg
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
                anthropic_api_key=api_key,
                model_kwargs=model_kwargs,
                system=system_message # Pass system message directly if supported
            )
        elif model_type == "Google":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                 # st.error("Google API key is not set in environment variables.")
                 return None
            # Google takes top_p directly
            # System message handled via ChatPromptTemplate or direct message list
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                top_p=top_p,
                google_api_key=api_key,
                # convert_system_message_to_human=True # May be needed depending on model/version
            )
        else:
            st.error(f"Unsupported model type: {model_type}")
            return None
    except Exception as e:
         # st.error(f"Error creating LLM {model_name} ({model_type}): {e}") # Avoid spamming errors
         print(f"Error creating LLM {model_name} ({model_type}) with temp={temperature}, top_p={top_p}: {e}") # Log to console
         return None

def initialize_conversation(vector_store, model_type, model_name, k_value=10, ollama_base_url_llm=None, use_reranking=False, num_chunks_kept=4, llm_temperature=0.2, llm_top_p=0.95):
    """Initialize the conversation chain with temperature, top_p, and system prompt"""
    st.sidebar.write("Initializing conversation chain...")
    try:
        if vector_store is None:
            st.warning("Cannot initialize conversation: Vector store is not loaded.")
            st.sidebar.warning("Cannot initialize conversation: Vector store is None.")
            return None, None, None # Return None for conversation, llm, retriever

        # Create the LLM using the provided temperature and top_p
        # Apply the general system prompt for the main chat conversation
        llm = create_llm(
            model_type,
            model_name,
            ollama_base_url_llm,
            llm_temperature,
            llm_top_p,
            apply_system_prompt=True # Apply general system prompt here
        )
        if llm is None:
             st.warning(f"Cannot initialize conversation: LLM creation failed for {model_name} ({model_type}). Check API keys/endpoints.")
             st.sidebar.error(f"Cannot initialize conversation: LLM creation failed for {model_name} ({model_type}).")
             return None, None, None # Return None for conversation, llm, retriever

        st.sidebar.write(f"LLM ({model_name}, Temp={llm_temperature}, TopP={llm_top_p}) created for conversation chain.")

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer' # Ensure the output key matches the chain's answer key
        )

        # Create retriever - This method is common for various vector stores
        retriever = vector_store.as_retriever(search_kwargs={"k": k_value})
        st.sidebar.write(f"Retriever created (k={k_value}).")

        # Reranking with ContextualCompressionRetriever
        if use_reranking:
            st.sidebar.write(f"Using LLM Reranking, keeping top {num_chunks_kept} chunks...")
            # Ensure num_chunks_kept is not greater than k_value
            num_chunks_kept_actual = min(num_chunks_kept, k_value) # Use potentially adjusted value
            try:
                # LLMChainExtractor requires an LLM
                if llm is None:
                     st.warning("LLM not available for Reranking setup. Reranking disabled.")
                     st.sidebar.error("LLM not available for Reranking.")
                     use_reranking = False # Fallback
                else:
                    # Ensure the compressor LLM is the same LLM used for the conversation chain
                    # NOTE: LLMChainExtractor might not benefit from the general system prompt,
                    # but we pass the already-configured LLM instance.
                    compressor = LLMChainExtractor.from_llm(llm)
                    # Ensure the retriever passed to ContextualCompressionRetriever is the basic one from the vector store
                    retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=vector_store.as_retriever(search_kwargs={"k": k_value}), k=num_chunks_kept_actual)
                    st.sidebar.write("Reranking retriever created.")
            except Exception as e:
                st.sidebar.error(f"Failed to create Reranking retriever: {e}. Proceeding without reranking. Check console for traceback.")
                st.warning(f"Failed to set up LLM Reranking: {e}. Reranking disabled.")
                print(f"\n--- Error setting up LLM Reranking ---\n")
                traceback.print_exc() # Print traceback to console
                print("\n---------------------------------------------\n")
                use_reranking = False # Fallback
                # Revert retriever to basic one if reranking setup failed
                retriever = vector_store.as_retriever(search_kwargs={"k": k_value})

        # --- Define prompt for the QA chain, including the system message ---
        # This prompt template combines the system message, chat history, context, and the user question
        qa_prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT), # Apply the system prompt here
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template(
                    """Answer the following question based on the provided context:

                    Context:
                    {context}

                    Question: {question}

                    Answer:"""
                ),
            ]
        )


        # Make the chain verbose so we can see prompts in stdout
        conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt_template}, # Inject the custom prompt with system message
            return_source_documents=True # Optionally return source documents
        )

        st.sidebar.write("✅ Conversation chain initialized.")
        return conversation, llm, retriever # Return all three instances

    except Exception as e:
        st.error(f"Error initializing conversation: {str(e)}. Check console for traceback.")
        st.sidebar.error(f"Error initializing conversation: {str(e)}")
        print(f"\n--- Error initializing conversation chain ---\n")
        traceback.print_exc() # Print traceback to console
        print("\n---------------------------------------------\n")
        return None, None, None

def extract_markdown_content(text: str, type: str = "json") -> str:
        """Extract content from markdown code blocks."""
        # Ensure text is a string
        text = str(text)
        start_tag_specific = f"```{type}"
        start_tag_generic = """```"""
        end_tag = """```"""

        # Prefer specific tag
        start_idx = text.find(start_tag_specific)
        offset = len(start_tag_specific)

        if start_idx == -1:
             # If specific tag not found, look for generic tag
             start_idx_generic = text.find(start_tag_generic)
             if start_idx_generic != -1:
                  start_idx = start_idx_generic
                  offset = len(start_tag_generic)
             else:
                # If neither tag is found, return the stripped text assuming no markdown block
                return text.strip()

        # Adjust start_idx to be after the newline following the tag
        newline_after_tag = text.find('\n', start_idx + offset)
        if newline_after_tag != -1:
            start_idx = newline_after_tag + 1
        else:
            # If no newline after tag, start immediately after the tag
            start_idx = start_idx + offset


        end_idx = text.rfind(end_tag, start_idx) # Search for end tag after start

        if end_idx != -1:
            return (text[start_idx:end_idx]).strip()
        else:
             # If end tag not found, assume the rest of the text is the content
             return text[start_idx:].strip()


# Function to execute chain of thought reasoning with preserved context
# NOTE: Does NOT use the general SYSTEM_PROMPT as it requires specific instructions.
def execute_chain_of_thought(llm, retriever, prompt, max_steps=5):
    """Executes a chain of thought reasoning process using the LLM and retriever."""
    st.sidebar.write("Starting Chain of Thought process...")
    # Initial prompt to get chain of thought planning
    # Uses its OWN system prompt for JSON adherence
    cot_system_prompt_planning = """
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
        # Retriever works the same regardless of underlying store (FAISS/Chroma)
        retrieved_docs = retriever.get_relevant_documents(prompt)
        if not retrieved_docs:
             st.sidebar.write("No documents retrieved for context.")
             context = "No relevant documents found."
        else:
             st.sidebar.write(f"Retrieved {len(retrieved_docs)} documents for context.")
             # Use basename in display context for brevity
             context = "\n\n".join([f"Source: {os.path.basename(doc.metadata.get('source', 'Unknown File'))}\nContent: {doc.page_content}" for doc in retrieved_docs])


    except Exception as e:
        st.sidebar.error(f"Error retrieving documents for CoT: {e}. Check console for traceback.")
        print(f"\n--- Error retrieving documents for CoT ---\n")
        traceback.print_exc() # Print traceback to console
        print("\n---------------------------------------------\n")
        context = "Could not retrieve context documents due to an error."


    # Initial prompt with context to get the plan
    initial_planning_prompt = f"""
    System: {cot_system_prompt_planning}

    Context information relevant to the question:
    ---
    {context}
    ---

    User question: {prompt}

    Based on the user question and the context, create a step-by-step reasoning plan in the specified JSON format.
    Remember to output *only* the JSON object.
    """

    # Get the plan using the LLM (already configured with temp/top_p)
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
             print(f"LLM Output (first 500 chars): {str(plan_response.content)[:500]}...") # Debug print response object
        st.warning("Failed to extract valid plan from LLM. Using default steps. Check console for traceback.")
        print(f"\n--- Error extracting/parsing CoT plan ---\n")
        traceback.print_exc() # Print traceback to console
        print("\n---------------------------------------------\n")
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
    initial_context_for_llm = f"Original Document Context:\n---\n{context}\n---\n\n"
    growing_context = initial_context_for_llm

    # Execute each step
    for i in range(total_steps):
        # Update progress
        progress = (i) / total_steps
        progress_bar.progress(progress, text=f"Executing Step {i+1}/{total_steps}: {steps[i][:80]}...") # Show part of step description

        current_step_description = steps[i]

        # Create prompt for this step with the growing context
        # Uses its OWN system instructions within the prompt.
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

        # Get response for this step using the LLM (already configured with temp/top_p)
        step_result = ""
        try:
            step_response = llm.invoke(step_prompt)
            step_result = str(step_response.content) # Ensure it's a string
        except Exception as e:
            step_result = f"Error executing step {i+1}: {e}"
            st.warning(f"Error in CoT step {i+1}: {e}")
            st.sidebar.warning(f"Error in CoT step {i+1}: {e}")
            print(f"\n--- Error executing CoT Step {i+1} ---\n")
            traceback.print_exc() # Print traceback to console
            print("\n---------------------------------------------\n")


        # Add to outputs list (for summary/display later)
        step_output_display.append(f"Step {i+1} ({current_step_description}):\n{step_result}")

        # Add this step's reasoning result to the growing context for the *next* step
        growing_context += f"\n---\nReasoning Results from Step {i+1} ({current_step_description}):\n{step_result}\n"

        # Display intermediate step in an expander
        with st.expander(f"Step {i+1}: {current_step_description}", expanded=False):
            st.markdown(step_result) # Use markdown for displaying step results


    # Final step - synthesize all reasoning into an answer
    progress_bar.progress(1.0, text="Synthesizing final answer...")

    # Create prompt for final answer using the complete growing context
    # Uses its OWN system instructions within the prompt.
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

    # Get final answer using the LLM (already configured with temp/top_p)
    final_answer = ""
    try:
        final_response = llm.invoke(final_answer_prompt)
        final_answer = str(final_response.content) # Ensure it's a string
        st.sidebar.write("✅ Chain of Thought process completed.")
    except Exception as e:
        final_answer = f"Error generating final answer after steps: {e}"
        st.error(final_answer)
        st.sidebar.error(final_answer)
        print(f"\n--- Error generating final answer for CoT ---\n")
        traceback.print_exc() # Print traceback to console
        print("\n---------------------------------------------\n")


    progress_bar.empty() # Clear the progress bar

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
            # Escape potential Quarto/Markdown conflicts in user content?
            # For simplicity, just include raw content for now.
            qmd_content += f"**User:**\n\n{content}\n\n" # Added markdown bold and newlines
        else:  # assistant
            qmd_content += f"\n## Assistant Response {i//2 + 1}\n\n"
            # Include assistant content (which might contain markdown)
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
            # Reverting to text_content=False to load structure, content extraction handled elsewhere
            return JSONLoader(file_path=file_path, jq_schema=".", text_content=False)
        elif ext == ".csv":
            return CSVLoader(file_path)
        elif ext == ".ipynb":
            return NotebookLoader(file_path)
        else:
            return None # Indicate unsupported file type
    except Exception as e:
         st.sidebar.error(f"Error creating loader for {os.path.basename(file_path)} (ext: {ext}): {e}")
         # Note: Not printing traceback here as it's in the document processing loop's catch block
         return None

# Helper to list source documents excluding metadata directory
def list_source_documents(docs_dir):
    """Lists all source document files in the docs directory, excluding the metadata directory."""
    if not docs_dir or not os.path.isdir(docs_dir):
        return []
    metadata_dir = os.path.join(docs_dir, "metadata")
    db_dir = os.path.join(docs_dir, "vectorstore") # Also exclude vectorstore dir
    all_items = glob.glob(os.path.join(docs_dir, "**/*"), recursive=True)
    # Filter out directories and files within the metadata or vectorstore directories
    source_files = [
        f for f in all_items
        if os.path.isfile(f)
        and not os.path.normpath(f).startswith(os.path.normpath(metadata_dir))
        and not os.path.normpath(f).startswith(os.path.normpath(db_dir))
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
    st.session_state.temperature = 0.2 # Default temperature
if 'top_p' not in st.session_state: # New session state for top_p
    st.session_state.top_p = 0.95 # Default top_p
if 'indexing_batch_size' not in st.session_state:
     st.session_state.indexing_batch_size = 100 # Default batch size for adding to index
if 'batch_delay_seconds' not in st.session_state:
     st.session_state.batch_delay_seconds = 1.0 # Default delay between batches (float)

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
# Session state to track current embedding settings for re-initialization check
if 'current_embedding_params' not in st.session_state:
     st.session_state.current_embedding_params = None


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
        except OSError as e:
            st.error(f"Error creating/accessing documents directory {docs_dir}: {e}")
            docs_dir = None # Set to None if invalid/inaccessible

    # The path where FAISS saves its index (calculated once per rerun based on docs_dir)
    db_path = os.path.join(docs_dir, "vectorstore") if docs_dir else None


    # Use expander for Vector DB Settings
    with st.expander("Vector DB Settings", expanded=False): # Collapsed by default
        # Chunk size slider
        chunk_size = st.slider(
            "Chunk Size",
            min_value=100,
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
        ollama_base_url_embedding = None # Need a separate var for embedding URL if different

        if embedding_type == "Ollama":
            ollama_embedding_model_str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")
            embedding_model = st.text_input("Ollama Embedding Model", ollama_embedding_model_str, key="ollama_embedding_model_input_embed")
            ollama_base_url_embedding = st.text_input("Ollama Base URL for Embedding", os.getenv("OLLAMA_END_POINT", "http://localhost:11434"), key="ollama_base_url_input_embed")
        elif embedding_type == "OpenAI":
            openai_embedding_model_str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            embedding_model = st.text_input("OpenAI Embedding Model", openai_embedding_model_str, key="openai_embedding_model_input_embed")
        elif embedding_type == "Google":
            google_embedding_model_str = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
            embedding_model = st.text_input("Google Embedding Model", google_embedding_model_str, key="google_embedding_model_input_embed")

        # Retrieve batch size from state, fallback if type is wrong
        indexing_batch_size_value = st.session_state.indexing_batch_size
        if not isinstance(indexing_batch_size_value, (int, float)):
             print(f"Warning: session state 'indexing_batch_size' has unexpected type {type(indexing_batch_size_value)}. Resetting to default 100.")
             indexing_batch_size_value = 100

        indexing_batch_size = st.slider(
            "Indexing Batch Size",
            min_value=10,
            max_value=1000,
            value=int(indexing_batch_size_value), # Ensure it's an integer for slider
            step=10,
            help="Number of chunks to embed and add to the database in a single batch. Lower values might reduce memory usage and API calls per batch, but increase total batches.",
            key="indexing_batch_size_slider"
        )
        st.session_state.indexing_batch_size = indexing_batch_size

        # Retrieve delay from state, fallback if type is wrong
        batch_delay_seconds_value = st.session_state.batch_delay_seconds
        if not isinstance(batch_delay_seconds_value, (int, float)):
             print(f"Warning: session state 'batch_delay_seconds' has unexpected type {type(batch_delay_seconds_value)}. Resetting to default 1.0.")
             batch_delay_seconds_value = 1.0

        batch_delay_seconds = st.slider(
            "Delay between Batches (seconds)",
            min_value=0.0,
            max_value=10.0,
            value=float(batch_delay_seconds_value), # Ensure it's a float
            step=0.1,
            help="Time to wait between sending batches of chunks to the embedding model/adding to the index. Increase if hitting rate limits (e.g., Google's 429 error).",
            key="batch_delay_seconds_slider"
        )
        st.session_state.batch_delay_seconds = batch_delay_seconds


        st.markdown("**To perform a hard reset** you must manually delete the `vectorstore` and `metadata` subdirectories from the documents directory. Check the console for detailed error tracebacks if building/loading fails.")

        # Database operations button (moved to the bottom of this section)
        build_db_button = st.button("🔨 Create/Update Vector DB", key="build_db_button")

    # --- New Complete Documents Section ---
    with st.expander("Pass Complete Documents to LLM", expanded=False): # Collapsed by default
        # Re-list available documents whenever docs_dir might have changed or on explicit DB build
        if docs_dir:
            current_available_doc_basenames = list_source_document_basenames(docs_dir)

            # Update session state list of available document *basenames*
            if set(st.session_state.available_docs) != set(current_available_doc_basenames):
                 st.session_state.available_docs = sorted(current_available_doc_basenames) # Keep sorted
                 # Reset selection state for any removed files, add new ones (default False)
                 new_selected_files = {}
                 for name in st.session_state.available_docs:
                      new_selected_files[name] = st.session_state.selected_files.get(name, False) # Preserve state if existing
                 st.session_state.selected_files = new_selected_files

            st.session_state.use_complete_docs = st.checkbox(
                "Use Complete Documents (Ignores Vector DB Retrieval)",
                value=st.session_state.use_complete_docs,
                help="If checked, the LLM will receive the full text of selected documents as context instead of using vector database retrieval. Select documents below.",
                key="use_complete_docs_checkbox" # Unique key
            )

            # Show status message based on mode and selection
            selected_count = sum(st.session_state.selected_files.values()) # Calculate selected count from state
            if st.session_state.use_complete_docs:
                if selected_count > 0:
                    st.info(f"🟢 Using {selected_count} selected document(s) as context.")
                else:
                    st.warning("🔴 Complete Documents mode active, no documents selected.")
            else:
                # Check if FAISS directory exists to give better user feedback
                faiss_index_file = os.path.join(db_path, "index.faiss") if db_path else None
                faiss_pkl_file = os.path.join(db_path, "index.pkl") if db_path else None
                faiss_db_exists_physically_and_complete = db_path and os.path.exists(db_path) and os.path.exists(faiss_index_file) and os.path.exists(faiss_pkl_file)
                faiss_db_dir_exists = db_path and os.path.exists(db_path)

                if st.session_state.vector_store:
                    st.info("🔵 Using Vector Database retrieval.")
                elif faiss_db_exists_physically_and_complete:
                     st.warning(f"⚪ Vector DB directory found at `{db_path or './docs/vectorstore'}`, but failed to load. Database might be corrupt or incompatible. Please manually delete the '{os.path.basename(db_path or 'vectorstore')}' directory to force a rebuild.")
                elif faiss_db_dir_exists:
                     st.warning(f"⚪ Vector DB directory found at `{db_path or './docs/vectorstore'}`, but required files (index.faiss, index.pkl) are missing. Database is incomplete. Please manually delete the directory to rebuild.")
                else:
                    st.warning(f"⚪ Vector Database mode is active, but no database found at `{db_path or './docs/vectorstore'}`. Please build it.")


            st.write("Select documents to use as context:")

            # --- Dropdown for document selection ---
            default_selected_basenames = [name for name, selected in st.session_state.selected_files.items() if selected]

            selected_basenames_from_dropdown = st.multiselect(
                 "Available Documents",
                 options=st.session_state.available_docs, # Use basenames as options
                 default=default_selected_basenames,      # Set default based on current state
                 key="document_multiselect",            # Unique key for the widget
                 label_visibility="collapsed"           # Hide the "Available Documents" label
            )

            # --- Update session state based on multiselect output ---
            for doc_basename in st.session_state.available_docs:
                 st.session_state.selected_files[doc_basename] = doc_basename in selected_basenames_from_dropdown

            # --- Select All/None buttons ---
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                 if st.button("Select All", key="select_all_docs_button"):
                      st.session_state.selected_files = {name: True for name in st.session_state.available_docs}
                      st.rerun() # Rerun to update multiselect display

            with col_sel2:
                 if st.button("Select None", key="select_none_docs_button"):
                      st.session_state.selected_files = {name: False for name in st.session_state.available_docs}
                      st.rerun() # Rerun to update multiselect display

            # Show how many are selected below the multiselect
            selected_count_display = sum(st.session_state.selected_files.values())
            st.caption(f"{selected_count_display} of {len(st.session_state.available_docs)} selected.")


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
        num_chunks_kept = st.session_state.k_value # Default to K if reranking is off or slider not shown
        if use_reranking:
            num_chunks_kept = st.slider(
                "Number of Chunks Kept After Reranking",
                min_value=1,
                max_value=st.session_state.k_value,
                value=min(st.session_state.num_chunks_kept, st.session_state.k_value), # Ensure default is within bounds
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

        # --- Add Top P Slider ---
        top_p = st.slider(
            "LLM Top P",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.top_p, # Use session state default/current
            step=0.05,
            help="Controls nucleus sampling. The model considers only the tokens with the highest probability mass summing up to top_p. Lower = more focused, Higher = more diverse. Default is 0.95.",
            key="llm_top_p_slider" # Unique key
        )
        st.session_state.top_p = top_p
        # --- End Add Top P Slider ---

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
        ollama_base_url_llm = None # Base URL for the LLM itself

        if model_type == "Ollama":
            ollama_model_str = os.getenv("OLLAMA_MODEL", "llama3:latest,mistral:latest")
            model_name = st.selectbox(
                "Select Ollama Model",
                split_csv(ollama_model_str),
                index=0,
                 key="ollama_model_select" # Unique key
            )
            ollama_base_url_llm = st.text_input("Ollama Base URL for LLM", os.getenv("OLLAMA_END_POINT", "http://localhost:11434"), key="ollama_base_url_input_llm")
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
        use_cot = st.checkbox("Use Chain of Thought (Vector DB Mode Only)", value=False, key="use_cot_checkbox")


# --- LLM Initialization (runs whenever settings change or on initial load) ---
# LLM is needed for metadata extraction and chat
# Embedding is needed for DB load/build
# Include top_p in the parameters tuple
current_llm_params = (model_type, model_name, ollama_base_url_llm, st.session_state.temperature, st.session_state.top_p)
current_embedding_params = (embedding_type, embedding_model, ollama_base_url_embedding)

# Check if LLM params have changed or if LLM isn't set yet
if docs_dir and (st.session_state.current_llm_params != current_llm_params or st.session_state.llm is None):
    st.session_state.current_llm_params = current_llm_params
    # Create the LLM instance for general use (will apply system prompt by default)
    llm_instance = create_llm(
        model_type,
        model_name,
        ollama_base_url_llm if model_type == "Ollama" else None,
        st.session_state.temperature,
        st.session_state.top_p,
        apply_system_prompt=True # Apply the general system prompt for the main LLM instance
    )
    st.session_state.llm = llm_instance # Update LLM instance in state
    # Note: The conversation chain will be re-initialized later if necessary, using this LLM instance

# Check if Embedding params have changed (needed for DB load/build)
if docs_dir and (st.session_state.current_embedding_params != current_embedding_params):
     st.session_state.current_embedding_params = current_embedding_params
     # No action needed here other than updating the state,
     # create_embeddings will use these values when called during DB ops.


# Handle database building/updating (placed outside sidebar but triggered by sidebar button)
if 'build_db_button_clicked' not in st.session_state:
    st.session_state.build_db_button_clicked = False

# Check if the button was clicked on this rerun
if build_db_button:
    st.session_state.build_db_button_clicked = True # Mark button as clicked

# Execute DB build/update logic only if button was clicked and it hasn't been processed yet in this session
if st.session_state.build_db_button_clicked:
    st.session_state.build_db_button_clicked = False # Reset flag immediately *before* the long operation

    if not docs_dir:
         st.error("Please specify a documents directory.")
    elif not embedding_model:
         st.error("Please specify an embedding model.")
    elif embedding_type == "Ollama" and not ollama_base_url_embedding:
         st.error("Please specify the Ollama Base URL for embeddings.")
    elif not model_name:
         st.error("Please select or specify an LLM model name.")
    elif model_type == "Ollama" and not ollama_base_url_llm:
         st.error("Please specify the Ollama Base URL for the LLM.")
    # Metadata extraction needs an LLM, but we create it inside create_or_update_vector_store
    # Check if db_path is valid
    elif db_path is None:
         st.error("Invalid documents directory path. Cannot determine database path.")
    else:
        with st.spinner("Processing documents and updating vector database..."):
            # Note: create_or_update_vector_store now takes llm_top_p
            vector_store = create_or_update_vector_store(
                docs_dir,
                db_path,
                embedding_type,
                embedding_model,
                model_type, # Pass model_type for metadata LLM
                model_name, # Pass model_name for metadata LLM
                ollama_base_url_llm if model_type == "Ollama" else None, # Pass LLM URL for metadata LLM
                ollama_base_url_embedding if embedding_type == "Ollama" else None, # Pass Embedding URL
                st.session_state.temperature, # Pass the current temperature
                st.session_state.top_p, # Pass the current top_p
                st.session_state.indexing_batch_size, # Pass indexing batch size
                st.session_state.batch_delay_seconds # Pass batch delay
            )
            # Update session state with the result of the build/update
            st.session_state.vector_store = vector_store

            # After attempting DB build/update, re-initialize conversation chain
            # if a vector store is available AND we are NOT in complete docs mode.
            if st.session_state.vector_store and not st.session_state.use_complete_docs and st.session_state.llm is not None:
                 st.sidebar.write("DB build/update completed, re-initializing conversation chain...")
                 # Note: initialize_conversation now takes llm_top_p
                 conversation, llm, retriever = initialize_conversation(
                     st.session_state.vector_store,
                     model_type,
                     model_name,
                     st.session_state.k_value,
                     ollama_base_url_llm if model_type == "Ollama" else None,
                     st.session_state.use_reranking,
                     st.session_state.num_chunks_kept,
                     st.session_state.temperature,
                     st.session_state.top_p # Pass top_p
                 )
                 st.session_state.conversation = conversation
                 # Use the LLM returned by initialize_conversation as it might have specific settings/prompts
                 st.session_state.llm = llm if llm is not None else st.session_state.llm
                 st.session_state.retriever = retriever

                 if st.session_state.conversation:
                      st.sidebar.success("Conversation chain initialized with updated DB.")
                 else:
                      st.sidebar.error("Failed to initialize conversation chain after DB update.")
            else:
                 # If no vector store, or in complete docs mode, ensure chain/retriever are None
                 st.session_state.conversation = None
                 st.session_state.retriever = None
                 if st.session_state.use_complete_docs:
                     st.sidebar.write("Vector DB process finished, but conversation chain is not initialized (Complete Documents mode active).")
                 else:
                      st.sidebar.write("Vector DB process finished, but no valid vector store is available to initialize the conversation chain.")

        # Rerun the script after DB operation to update the UI status
        st.rerun()


# --- Initial Load Logic ---
# Attempts to load existing DB and initialize conversation if conditions are met.
if docs_dir and os.path.exists(docs_dir) and embedding_model and (embedding_type != "Ollama" or ollama_base_url_embedding) \
   and model_name and (model_type != "Ollama" or ollama_base_url_llm) and st.session_state.llm is not None:

    if db_path:
        faiss_index_file = os.path.join(db_path, "index.faiss")
        faiss_pkl_file = os.path.join(db_path, "index.pkl")

        # Only attempt to load DB if files exist, it's not already loaded, AND we are NOT using complete docs
        if os.path.exists(db_path) and os.path.exists(faiss_index_file) and os.path.exists(faiss_pkl_file) \
           and st.session_state.vector_store is None and not st.session_state.use_complete_docs:

            with st.spinner("Loading existing vector database..."):
                vector_store = load_vector_store(
                    db_path,
                    embedding_type,
                    embedding_model,
                    ollama_base_url_embedding if embedding_type == "Ollama" else None
                )
                st.session_state.vector_store = vector_store # Update state

                if vector_store:
                    # If load successful, initialize conversation
                    st.sidebar.write("Existing DB loaded, initializing conversation chain...")
                    # Note: initialize_conversation now takes llm_top_p
                    conversation, llm, retriever = initialize_conversation(
                        vector_store,
                        model_type,
                        model_name,
                        st.session_state.k_value,
                        ollama_base_url_llm if model_type == "Ollama" else None,
                        st.session_state.use_reranking,
                        st.session_state.num_chunks_kept,
                        st.session_state.temperature,
                        st.session_state.top_p # Pass top_p
                    )
                    st.session_state.conversation = conversation
                    # Use the LLM returned by initialize_conversation
                    st.session_state.llm = llm if llm is not None else st.session_state.llm
                    st.session_state.retriever = retriever

                    if st.session_state.conversation:
                         st.success("Existing vector database loaded and conversation initialized!")
                    else:
                        st.warning("Existing database loaded, but conversation chain could not be initialized. Check LLM settings.")


# Update available docs list on initial load or rerun if docs_dir is set
if docs_dir:
     current_available_doc_basenames = list_source_document_basenames(docs_dir)
     if set(st.session_state.available_docs) != set(current_available_doc_basenames):
          st.session_state.available_docs = sorted(current_available_doc_basenames) # Keep sorted
          new_selected_files = {}
          for name in st.session_state.available_docs:
               new_selected_files[name] = st.session_state.selected_files.get(name, False) # Preserve state if existing
          st.session_state.selected_files = new_selected_files
     # Ensure selected_files reflects current available docs even if the set didn't change
     current_selected_files = {}
     for name in st.session_state.available_docs:
          current_selected_files[name] = st.session_state.selected_files.get(name, False)
     st.session_state.selected_files = current_selected_files


# --- Chat Eligibility Check ---
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
            if not selected_basenames or not docs_dir:
                 response_content = "Complete Documents mode is active, but no documents are selected or documents directory is invalid. Please check settings in the sidebar."
                 st.warning(response_content)
                 with st.chat_message("assistant", avatar="🤖"):
                     st.markdown(response_content)
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
                        total_chars = 0
                        char_limit = 1000000 # Example char limit

                        for doc_path in selected_paths:
                             loader = get_document_loader(doc_path)
                             if loader:
                                  try:
                                      documents = loader.load() # Loader might return list of pages/docs
                                      full_text = "\n\n".join([str(d.page_content) for d in documents]) # Ensure string conversion
                                      if total_chars + len(full_text) > char_limit:
                                           st.sidebar.warning(f"Content from {os.path.basename(doc_path)} exceeds approximate character limit ({char_limit}). Truncating context.")
                                           remaining_chars = char_limit - total_chars
                                           if remaining_chars > 0:
                                                loaded_document_contents.append(f"--- Start Document: {os.path.basename(doc_path)} ---\n\n{full_text[:remaining_chars]}\n\n--- End Document: {os.path.basename(doc_path)} ---")
                                                total_chars += remaining_chars
                                           break
                                      else:
                                          loaded_document_contents.append(f"--- Start Document: {os.path.basename(doc_path)} ---\n\n{full_text}\n\n--- End Document: {os.path.basename(doc_path)} ---")
                                          total_chars += len(full_text)
                                          st.sidebar.write(f"- Loaded content from {os.path.basename(doc_path)} ({len(full_text)} chars)")
                                  except Exception as e:
                                      st.sidebar.warning(f"Error loading content from {os.path.basename(doc_path)}: {e}. Skipping.")
                                      print(f"\n--- Error loading content for Complete Docs mode from: {doc_path} ---\n")
                                      traceback.print_exc()
                                      print("\n---------------------------------------------\n")
                             else:
                                 st.sidebar.warning(f"Unsupported file type for loading complete content: {os.path.basename(doc_path)}. Skipping.")

                        concatenated_content = "\n\n".join(loaded_document_contents)

                        if not concatenated_content.strip():
                             response_content = "Could not load readable content from selected documents."
                             st.warning(response_content)
                        else:
                            # Construct the manual prompt, PREPENDING the system prompt
                            # History is still omitted here.
                            manual_prompt_template = """
                            System: {system_prompt}

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
                                system_prompt=SYSTEM_PROMPT, # Add the system prompt here
                                context=concatenated_content,
                                question=prompt
                            )

                            if st.session_state.llm:
                                if use_cot:
                                     st.warning("Chain of Thought is not supported in 'Complete Documents' mode. Using direct query.")

                                # Use the LLM instance from session state (already configured with temp/top_p/system prompt if applicable)
                                response = st.session_state.llm.invoke(manual_prompt)
                                response_content = str(response.content)
                            else:
                                response_content = "LLM is not initialized. Cannot answer in Complete Documents mode."
                                st.error(response_content)

                        # Display assistant response
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(response_content)

                        # Add assistant response to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": response_content})

                    except Exception as e:
                        st.error(f"An unexpected error occurred in Complete Documents mode: {e}. Check console for traceback.")
                        error_message = "Sorry, I encountered an error while processing your request."
                        with st.chat_message("assistant", avatar="🤖"):
                             st.markdown(error_message)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_message}) # Add error message to history
                        print(f"\n--- Unexpected error in Complete Documents mode ---\n")
                        traceback.print_exc() # Print traceback to console
                        print("\n---------------------------------------------\n")


        elif st.session_state.conversation:
            # --- Vector DB Mode ---
            with st.spinner("Thinking..."):
                try:
                    # Prepare chat history (same as before)
                    history_for_chain = st.session_state.chat_history[:-1] if st.session_state.chat_history else []

                    if not st.session_state.retriever or not st.session_state.llm:
                         response_content = "Vector DB mode is active, but the required components (LLM/Retriever) are not initialized. Please rebuild/load the DB."
                         response = {"answer": response_content}
                         st.warning(response_content)
                    elif use_cot and st.session_state.llm and st.session_state.retriever:
                         # Use Chain of Thought
                         # CoT uses the LLM instance from session state, which should have temp/top_p set.
                         # Note: CoT function uses its own specific system prompts internally.
                         answer, step_outputs = execute_chain_of_thought(
                             st.session_state.llm,
                             st.session_state.retriever,
                             prompt
                         )
                         response = {"answer": answer}

                    else:
                         # Regular conversation flow using the chain
                         # The chain uses the LLM and prompt (with system message) configured during initialize_conversation
                         response = st.session_state.conversation.invoke({"question": prompt}) # Pass only question, history is in memory


                    answer = str(response.get("answer", "Sorry, I could not generate an answer.")) # Ensure string, fallback

                    # Display assistant response
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(answer)

                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"An error occurred during the conversation (Vector DB mode): {e}. Check console for traceback.")
                    error_message = "Sorry, I encountered an error while processing your request."
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(error_message)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message}) # Add error message to history
                    print(f"\n--- Error during conversation (Vector DB mode) ---\n")
                    traceback.print_exc() # Print traceback to console
                    print("\n---------------------------------------------\n")

else:
    # Message to display if chat is NOT eligible
    st.info("Chat is not ready.")

    # Provide specific guidance based on the state
    if st.session_state.llm is None:
         st.warning("LLM is not initialized. Please check LLM settings in the sidebar (Provider, Model, API Keys/Endpoints, Temperature, Top P) and ensure they are correct.")
    elif st.session_state.use_complete_docs and sum(st.session_state.selected_files.values()) == 0:
          st.warning("Complete Documents mode is enabled, but no documents are selected. Please select documents in the sidebar.")
    else: # Covers vector DB chain issues
         db_path_display = db_path or "./docs/vectorstore"
         if db_path:
             faiss_index_file = os.path.join(db_path, "index.faiss")
             faiss_pkl_file = os.path.join(db_path, "index.pkl")
             faiss_db_exists_physically_and_complete = os.path.exists(db_path) and os.path.exists(faiss_index_file) and os.path.exists(faiss_pkl_file)
             faiss_db_dir_exists = os.path.exists(db_path)
         else:
              faiss_db_exists_physically_and_complete = False
              faiss_db_dir_exists = False

         if st.session_state.vector_store is None and faiss_db_exists_physically_and_complete:
             st.warning(f"Vector database directory found at `{db_path_display}` with files, but failed to load. Database might be corrupt or incompatible. Please manually delete the '{os.path.basename(db_path_display)}' directory to force a rebuild.")
         elif st.session_state.vector_store is None and faiss_db_dir_exists and not faiss_db_exists_physically_and_complete:
              st.warning(f"Vector database directory found at `{db_path_display}`, but required files (index.faiss, index.pkl) are missing. Database is incomplete. Please manually delete the '{os.path.basename(db_path_display)}' directory to force a rebuild.")
         elif st.session_state.vector_store is None and not faiss_db_dir_exists:
             st.warning(f"Vector database is not loaded or initialized. Please configure settings in the sidebar and click 'Create Vector DB' to build or load the database at `{db_path_display}`.")
             if docs_dir and not os.path.exists(docs_dir):
                  st.info(f"Documents directory `{docs_dir}` not found or not specified. Please create the directory or enter a valid path in the sidebar.")
             elif docs_dir and os.path.exists(docs_dir) and not list_source_documents(docs_dir):
                 st.info(f"Documents directory `{docs_dir}` exists, but contains no supported document files. Please add documents.")
         elif st.session_state.vector_store and st.session_state.conversation is None and not st.session_state.use_complete_docs:
            st.warning("Vector database loaded, but conversation chain failed to initialize. Please check LLM settings (Provider, Model, API Keys/Endpoints, Temperature, Top P) and click 'Create/Update Vector DB' again to re-attempt chain initialization.")
         elif st.session_state.vector_store and st.session_state.use_complete_docs:
             st.info("Vector database is built/loaded, but Complete Documents mode is active.")
         else:
             st.warning("Please configure settings in the sidebar and load/build the database or select documents to start chatting.")


# --- Chat Bottom Buttons ---
st.markdown("---") # Separator before buttons

if st.session_state.llm is not None:
    col1, col2 = st.columns(2) # Use columns for horizontal layout

    with col1:
        # Handle chat reset
        reset_chat = st.button("🔄 Reset Chat History")
        if reset_chat:
            st.session_state.chat_history = []
            st.sidebar.write("Resetting chat history.")
            # Re-initialize the conversation chain if vector store exists and not in complete docs mode
            if st.session_state.vector_store and not st.session_state.use_complete_docs and st.session_state.llm is not None:
                st.sidebar.write("Vector DB loaded and mode is active, re-initializing conversation chain...")
                # Note: initialize_conversation now takes llm_top_p
                conversation, llm, retriever = initialize_conversation(
                    st.session_state.vector_store,
                    model_type,
                    model_name,
                    st.session_state.k_value,
                    ollama_base_url_llm if model_type == "Ollama" else None,
                    st.session_state.use_reranking,
                    st.session_state.num_chunks_kept,
                    st.session_state.temperature,
                    st.session_state.top_p # Pass top_p
                )
                st.session_state.conversation = conversation
                # Use the LLM returned by initialize_conversation
                st.session_state.llm = llm if llm is not None else st.session_state.llm
                st.session_state.retriever = retriever
                if st.session_state.conversation:
                     st.sidebar.success("Conversation chain re-initialized after reset.")
                else:
                     st.sidebar.error("Failed to re-initialize conversation chain after reset.")
            else:
                 # Ensure chain/retriever are None if not applicable
                 st.session_state.conversation = None
                 st.session_state.retriever = None
            st.rerun() # Rerun to clear chat display

    with col2:
        # Add Quarto export button
        if st.session_state.chat_history:
             if st.button("📥 Download Chat as Quarto (.qmd)"):
                qmd_content = convert_chat_to_qmd(st.session_state.chat_history)
                st.markdown(get_download_link(qmd_content), unsafe_allow_html=True)
        else:
             st.button("📥 Download Chat as Quarto (.qmd)", disabled=True, help="No chat history to export yet.")

# --- End Chat Bottom Buttons ---
