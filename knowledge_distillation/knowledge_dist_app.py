import streamlit as st
import os
import json
import shutil
import time
import PyPDF2  # Using PyPDF2 instead of fitz for PDF processing
from OllamaChatBot import OllamaChatBot
from dotenv import load_dotenv

def read_pdf_content(file_path, max_chars=4000):
    """Extract text from a PDF file, limited to max_chars using PyPDF2."""
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                if len(text) >= max_chars:
                    break
        return text[:max_chars]
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def read_markdown_content(file_path, max_chars=4000):
    """Read content from a markdown file, limited to max_chars."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read(max_chars)
        return content
    except Exception as e:
        return f"Error reading Markdown: {str(e)}"

def generate_metadata(content, bot, max_retries=5):
    """Generate metadata using OllamaChatBot with retry logic."""
    prompt = f"""
    Extract or generate the following metadata from the document content below. 
    If the content is not in English, first translate it to English.
    
    Return ONLY a JSON object with the following fields:
    - year: (publication year, numeric)
    - authors: (author names as a string)
    - journal: (journal or publication name)
    - title: (document title)
    - abstract: (the exact content of the Abstract section)
    - reference: (a full APA style reference)
    
    Document content: {content}
    
    """
    
    # Implement retry logic
    attempts = 0
    backoff_factor = 1.5  # Exponential backoff factor
    initial_wait = 1  # Initial wait time in seconds
    
    while attempts < max_retries:
        try:
            # Use JSON completion to ensure proper formatting
            response = bot.completeAsJSON(prompt)
            
            if response:
                try:
                    # Parse the response to ensure it's valid JSON
                    metadata = json.loads(response)
                    return metadata
                except json.JSONDecodeError:
                    # JSON parsing failed, try again
                    attempts += 1
                    if attempts >= max_retries:
                        return {"error": f"Failed to parse metadata response after {max_retries} attempts"}
                    # Wait before retry with exponential backoff
                    wait_time = initial_wait * (backoff_factor ** (attempts - 1))
                    time.sleep(wait_time)
                    continue
            else:
                # No response from the bot
                attempts += 1
                if attempts >= max_retries:
                    return {"error": f"Failed to generate metadata after {max_retries} attempts"}
                # Wait before retry with exponential backoff
                wait_time = initial_wait * (backoff_factor ** (attempts - 1))
                time.sleep(wait_time)
                continue
                
        except Exception as e:
            # Handle any other exceptions
            attempts += 1
            if attempts >= max_retries:
                return {"error": f"Error generating metadata: {str(e)} (after {max_retries} attempts)"}
            # Wait before retry with exponential backoff
            wait_time = initial_wait * (backoff_factor ** (attempts - 1))
            time.sleep(wait_time)
            
    # This point should only be reached if all retries fail
    return {"error": f"Failed to generate metadata after {max_retries} attempts"}

def process_files(folder_path, bot, force_regenerate=False):
    """Process all PDF and Markdown files in the folder.
    
    Args:
        folder_path: Path to the folder containing PDF and Markdown files
        bot: OllamaChatBot instance for metadata generation
        force_regenerate: If True, regenerate all metadata even if files exist
    """
    results = []
    
    # Create metadata folder if it doesn't exist
    metadata_folder = os.path.join(folder_path, "metadata")
    if not os.path.exists(metadata_folder):
        os.makedirs(metadata_folder)
    
    # Get all PDF and Markdown files
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.pdf', '.md'))]
    
    # Calculate files that need processing (if not forcing regeneration)
    files_to_process = []
    skipped_files = []
    
    if not force_regenerate:
        for file in files:
            metadata_file = os.path.join(metadata_folder, f"{file}.json")
            if os.path.exists(metadata_file):
                # Check if the metadata file is valid JSON
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    # If it loaded successfully and doesn't have an error field, skip it
                    if isinstance(metadata, dict) and "error" not in metadata:
                        skipped_files.append(file)
                        results.append({
                            "file": file,
                            "metadata_file": f"{file}.json",
                            "status": "Skipped (already exists)"
                        })
                        continue
                except (json.JSONDecodeError, IOError):
                    # If the file is invalid or can't be read, process it again
                    pass
            
            files_to_process.append(file)
    else:
        files_to_process = files
    
    # Show information about skipped files
    if skipped_files:
        st.info(f"Skipping {len(skipped_files)} files that already have metadata.")
    
    # If no files need processing, return early
    if not files_to_process:
        return results
    
    # Progress bar
    progress_bar = st.progress(0)
    file_status_placeholder = st.empty()
    
    for i, file in enumerate(files_to_process):
        file_path = os.path.join(folder_path, file)
        file_status_placeholder.text(f"Processing: {file} ({i+1}/{len(files_to_process)})")
        
        # Skip if it's a directory
        if os.path.isdir(file_path):
            continue
            
        # Extract content based on file type
        if file.lower().endswith('.pdf'):
            content = read_pdf_content(file_path)
        elif file.lower().endswith('.md'):
            content = read_markdown_content(file_path)
        else:
            continue
            
        # Generate metadata with retry logic
        metadata = generate_metadata(content, bot)
        
        # Save metadata
        metadata_file = os.path.join(metadata_folder, f"{file}.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        results.append({
            "file": file,
            "metadata_file": f"{file}.json",
            "status": "Success" if "error" not in metadata else f"Error: {metadata['error']}"
        })
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(files_to_process))
    
    file_status_placeholder.empty()
    progress_bar.empty()    
    return results

def main():
    load_dotenv(os.path.join(os.getcwd(), ".env"))

    st.title("Document Metadata Generator")

    # Sidebar
    st.sidebar.header("Settings")
    folder_path = st.sidebar.text_input("Enter folder path containing .pdf/.md files:", os.getenv("SOURCE_DOC_DIR"))
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        max_retries = st.number_input("Maximum retries", min_value=1, max_value=10, value=5)
        model = st.text_input("Model name", value=os.getenv("OLLAMA_MODEL"))
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
        force_regenerate = st.checkbox("Force regenerate all metadata", value=False, 
                                       help="If checked, regenerate metadata for all files even if they already exist")

    # Initialize OllamaChatBot
    bot = OllamaChatBot(
        model=model or os.getenv("OLLAMA_MODEL"),
        end_point_url=os.getenv("OLLAMA_END_POINT"),
        temperature=temperature,
        keep_history=False
    )
    
    if st.sidebar.button("Generate Metadata"):
        if not folder_path or not os.path.isdir(folder_path):
            st.error("Please enter a valid folder path.")
        else:
            with st.spinner("Processing files..."):
                results = process_files(folder_path, bot, force_regenerate)
                
            # Display results
            files_processed = sum(1 for r in results if r["status"] != "Skipped (already exists)")
            
            # Count successful, failed, and skipped files
            success_count = sum(1 for r in results if r["status"] == "Success")
            skipped_count = sum(1 for r in results if r["status"] == "Skipped (already exists)")
            error_count = len(results) - success_count - skipped_count
            
            st.success(f"Operation complete: {len(results)} total files")
            
            # Create status summary
            status_html = f"""
            <div style="display: flex; gap: 20px;">
                <div>✅ Success: {success_count}</div>
                <div>⏭️ Skipped: {skipped_count}</div>
                <div>❌ Errors: {error_count}</div>
            </div>
            """
            st.markdown(status_html, unsafe_allow_html=True)
            
            # Create a table of results
            st.subheader("Processing Results")
            result_data = {
                "File": [r["file"] for r in results],
                "Metadata File": [r["metadata_file"] for r in results],
                "Status": [r["status"] for r in results]
            }
            st.dataframe(result_data)
            
            # Display metadata for each file
            st.subheader("Generated Metadata")
            for result in results:
                metadata_path = os.path.join(folder_path, "metadata", result["metadata_file"])
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    with st.expander(f"Metadata for {result['file']}"):
                        st.json(metadata)

if __name__ == "__main__":
    main()