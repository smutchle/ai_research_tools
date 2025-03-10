import streamlit as st
import os
import pandas as pd
import PyPDF2
import re
import json
from collections import Counter
import base64
import csv
import subprocess
import time
import threading
from OllamaChatBot import OllamaChatBot
from dotenv import load_dotenv

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        st.error(f"Error extracting text from {pdf_path}: {e}")
    return text

def extract_references_section(text):
    """Extract the references section from the document text."""
    # Common section titles for references
    patterns = [
        r'(?i)References\s*\n',
        r'(?i)Bibliography\s*\n',
        r'(?i)Works Cited\s*\n',
        r'(?i)Literature Cited\s*\n',
        r'(?i)References Cited\s*\n'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            start_idx = match.start()
            # Look for the next section heading or end of document - make case insensitive
            next_section = re.search(r'(?i)\n\s*[A-Za-z][a-zA-Z]+ *\n', text[start_idx+len(match.group(0)):])
            if next_section:
                end_idx = start_idx + len(match.group(0)) + next_section.start()
                return text[start_idx:end_idx].strip()
            else:
                return text[start_idx:].strip()
    
    return ""

def split_references_into_chunks(references_text, chunk_size=10):
    """Split the references into chunks of approximately chunk_size references."""
    # Common patterns that might indicate the start of a new reference
    ref_patterns = [
        r'\n\[\d+\]', # [1], [2], etc.
        r'\n\d+\.\s', # 1. 2. etc.
        r'\n[A-Z][a-zA-Z]*,\s*[A-Z]\.', # Last, F. format
        r'\n[A-Z][a-zA-Z]*\s*[A-Z][a-zA-Z]*,\s*[A-Z]\.', # Last First, F. format
        r'\n\s*[A-Z][a-zA-Z]*,\s*[A-Z]\.' # Indented Last, F. format
    ]
    
    # Try to find all potential reference start points
    ref_starts = []
    for pattern in ref_patterns:
        matches = list(re.finditer(pattern, '\n' + references_text))
        ref_starts.extend([(m.start() + 1, pattern) for m in matches])  # +1 to account for the added newline
    
    # If we couldn't find reference patterns, fall back to newlines as separators
    if not ref_starts:
        # Just split by double newlines (common in references)
        chunks = re.split(r'\n\s*\n', references_text)
        result = []
        current_chunk = []
        
        for item in chunks:
            if item.strip():  # Ignore empty items
                current_chunk.append(item.strip())
                if len(current_chunk) >= chunk_size:
                    result.append('\n\n'.join(current_chunk))
                    current_chunk = []
        
        if current_chunk:  # Add the last chunk if it has any items
            result.append('\n\n'.join(current_chunk))
        
        return result
    
    # Sort the reference start points by position
    ref_starts.sort(key=lambda x: x[0])
    
    # Create chunks of approximately chunk_size references
    chunks = []
    for i in range(0, len(ref_starts), chunk_size):
        if i + chunk_size < len(ref_starts):
            start_pos = ref_starts[i][0]
            end_pos = ref_starts[i + chunk_size][0]
            chunks.append(references_text[start_pos:end_pos].strip())
        else:
            # Last chunk goes to the end of the text
            start_pos = ref_starts[i][0]
            chunks.append(references_text[start_pos:].strip())
    
    # Handle the case where there are fewer references than chunk_size
    if not chunks and references_text.strip():
        chunks.append(references_text.strip())
    
    return chunks

def reformat_references_with_llm(chatbot, references_text, filename, chunk_size, output_csv_path, log_file=None):
    """Use the OllamaChatBot to reformat references in APA style, processing in chunks and appending to CSV."""
    # Split references into manageable chunks - pass the chunk_size parameter
    reference_chunks = split_references_into_chunks(references_text, chunk_size)
    all_formatted_references = []
    successful_chunks = 0
    failed_chunks = 0
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # Check if output file exists, create with headers if not
    if not os.path.exists(output_csv_path):
        # Create file with just the header
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            f.write('reference\n')
    
    # Log function for background processing
    def log_message(message):
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        else:
            # If no log file is provided, use Streamlit messages
            progress_message = st.empty()
            progress_message.text(message)
            time.sleep(0.1)  # Brief pause to allow UI to update
            progress_message.empty()
    
    # Process each chunk
    for i, chunk in enumerate(reference_chunks):
        log_message(f"Processing chunk {i+1}/{len(reference_chunks)} from {filename}...")
        
        # Skip empty chunks
        if not chunk.strip():
            continue
        
        prompt = f"""
        Below is a partial References section extracted from a PDF document named '{filename}'.
        Please extract each individual reference, reformat it to proper APA style (7th edition), 
        and return the results as a JSON array where each element is an object with:
        1. "reference": the original reference text
        2. "formatted": the APA formatted reference
        
        References chunk:
        {chunk}
        
        Return ONLY valid JSON with this structure:
        [
            {{"reference": "original reference text", "formatted": "APA formatted reference"}},
            ...
        ]
        """

        # Try up to 2 times for each chunk
        max_retries = 2
        chunk_references = []
        for attempt in range(max_retries):
            try:
                result = json.loads(chatbot.completeAsJSON(prompt))
                if result and isinstance(result, list):
                    # Process the references and append directly to the file
                    with open(output_csv_path, 'a', newline='', encoding='utf-8') as f:
                        for ref in result:
                            # Add double quotes around the reference and write directly to the file
                            formatted_ref = f'"{ref["formatted"]}"\n'
                            f.write(formatted_ref)
                            
                            # Also store in memory for the return value
                            all_formatted_references.append({"reference": ref["formatted"]})
                    
                    successful_chunks += 1
                    break  # Success, exit retry loop
                else:
                    if attempt < max_retries - 1:  # Not the last attempt
                        log_message(f"Retry {attempt+1}/{max_retries} for chunk {i+1} from {filename}...")
                    else:
                        log_message(f"Could not process chunk {i+1} from {filename} after {max_retries} attempts")
                        failed_chunks += 1
            except Exception as e:
                if attempt < max_retries - 1:  # Not the last attempt
                    log_message(f"Error in attempt {attempt+1}/{max_retries} for chunk {i+1} from {filename}. Retrying...")
                else:
                    log_message(f"Error reformatting chunk {i+1} from {filename}: {e}")
                    failed_chunks += 1
    
    # Summarize processing results
    if reference_chunks:
        summary = f"References processing summary for {filename}: " + \
                 f"{successful_chunks}/{len(reference_chunks)} chunks processed successfully " + \
                 f"({failed_chunks} failed)"
        log_message(summary)
    
    return all_formatted_references

def get_download_link(df, filename, text):
    """Generate a download link for a dataframe."""
    # Read the file directly to get the content as it was written
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Encode the content for download
        b64 = base64.b64encode(content.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{os.path.basename(filename)}">{text}</a>'
        return href
    except Exception as e:
        st.error(f"Error creating download link: {e}")
        # Fallback to using the dataframe if file reading fails
        content = f"reference\n" + "\n".join([f'"{ref}"' for ref in df['reference']])
        b64 = base64.b64encode(content.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{os.path.basename(filename)}">{text}</a>'
        return href

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

def process_pdfs_in_background(pdf_folder, pdf_files, ollama_model, ollama_endpoint, 
                              temperature, chunk_size, output_csv_path, status_file):
    """
    Process PDFs in a background thread and update status in a file.
    
    This function is designed to be run in a separate thread or process.
    """
    # Initialize status file
    with open(status_file, 'w', encoding='utf-8') as f:
        f.write(f"Started processing {len(pdf_files)} PDFs at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Status: Running\n")
        f.write(f"0/{len(pdf_files)} files processed\n")
    
    try:
        # Initialize the chatbot
        chatbot = OllamaChatBot(
            model=ollama_model,
            end_point_url=ollama_endpoint,
            temperature=temperature
        )
        
        # Clear any existing output file
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            f.write('reference\n')
        
        for i, pdf_file in enumerate(pdf_files):
            # Update status
            with open(status_file, 'w', encoding='utf-8') as f:
                f.write(f"Processing PDF {i+1}/{len(pdf_files)}: {pdf_file}\n")
                f.write("Status: Running\n")
                f.write(f"{i}/{len(pdf_files)} files processed\n")
            
            pdf_path = os.path.join(pdf_folder, pdf_file)
            
            # Extract text from PDF
            text = extract_text_from_pdf(pdf_path)
            
            # Extract references section
            references_section = extract_references_section(text)
            
            if not references_section:
                with open(status_file, 'a', encoding='utf-8') as f:
                    f.write(f"No references section found in {pdf_file}\n")
                continue
            
            # Reformat references using the chatbot and append to CSV
            reformat_references_with_llm(chatbot, references_section, pdf_file, chunk_size, 
                                        output_csv_path, status_file)
        
        # Update status to complete
        with open(status_file, 'w', encoding='utf-8') as f:
            f.write(f"Completed processing {len(pdf_files)} PDFs at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Status: Complete\n")
            f.write(f"{len(pdf_files)}/{len(pdf_files)} files processed\n")
            
    except Exception as e:
        # Log error to status file
        with open(status_file, 'w', encoding='utf-8') as f:
            f.write(f"Error occurred at {time.strftime('%Y-%m-%d %H:%M:%S')}: {str(e)}\n")
            f.write("Status: Error\n")
            f.write(f"Process stopped after {i}/{len(pdf_files)} files\n")

def start_background_process(settings):
    """Start the background processing script as a separate process."""
    # Create a status file to track progress
    status_file = os.path.join(os.path.dirname(settings['output_csv_path']), "process_status.txt")
    
    # Start a background thread
    process_thread = threading.Thread(
        target=process_pdfs_in_background,
        args=(
            settings['pdf_folder'],
            settings['pdf_files'],
            settings['ollama_model'],
            settings['ollama_endpoint'],
            settings['temperature'],
            settings['chunk_size'],
            settings['output_csv_path'],
            status_file
        ),
        daemon=False  # Non-daemon thread to continue after app closes
    )
    
    process_thread.start()
    return status_file

def check_background_status(status_file):
    """Check the status of a background process from its status file."""
    if not os.path.exists(status_file):
        return "Unknown", "Status file not found"
    
    try:
        with open(status_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        status_info = "".join(lines)
        
        # Extract status
        status_match = re.search(r'Status: (\w+)', status_info)
        status = status_match.group(1) if status_match else "Unknown"
        
        return status, status_info
    except Exception as e:
        return "Error", f"Error reading status file: {str(e)}"

def main():
    load_dotenv(os.getcwd() + "/.env")

    st.title("PDF References Extractor and APA Formatter")
    
    # Sidebar configuration
    st.sidebar.header("Settings")
    pdf_folder = st.sidebar.text_input("PDF Directory", os.getenv('SOURCE_PDF_DIR'))
    ollama_endpoint = st.sidebar.text_input("Ollama Endpoint URL", os.getenv('OLLAMA_END_POINT'))
    ollama_model = st.sidebar.selectbox("Ollama Model", split_csv(os.getenv('OLLAMA_MODEL')))
    chunk_size = st.sidebar.slider("References per chunk", 1, 10, 5)
    output_csv_path = st.sidebar.text_input("Output CSV Path", "./output/references.csv")
    temperature = 0.5
    
    # Status file path for background process
    status_file = os.path.join(os.path.dirname(output_csv_path), "process_status.txt")
    
    # Check for running background process
    if os.path.exists(status_file):
        status, status_info = check_background_status(status_file)
        
        st.info("Background Process Status")
        st.code(status_info)
        
        if status in ["Complete", "Error"]:
            if st.button("Clear Status"):
                try:
                    os.remove(status_file)
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error removing status file: {e}")
        
        # If a process is running, show its status and provide option to view results
        if status == "Running":
            st.warning("A background process is currently running. You can close this window and the process will continue.")
            
            if st.button("Refresh Status"):
                st.experimental_rerun()
                
        # If process is complete, offer to view results
        if status == "Complete" and os.path.exists(output_csv_path):
            st.success("Processing complete!")
            
            if st.button("View Results"):
                try:
                    # Read the file directly as it was written
                    with open(output_csv_path, 'r', encoding='utf-8') as f:
                        # Skip header
                        next(f)
                        # Read all lines
                        references = [line.strip() for line in f if line.strip()]
                    
                    # Display results
                    st.subheader("Extracted References")
                    st.write(f"Total references found: {len(references)}")
                    
                    # Create a dataframe for display
                    df_display = pd.DataFrame({'reference': references})
                    
                    # Display table with only reference column
                    st.dataframe(df_display)
                    
                    # Download link (for convenience)
                    st.markdown(
                        get_download_link(df_display, output_csv_path, "Download References as CSV"),
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Error reading output file: {e}")
                    st.warning("No references extracted from any PDF.")
    else:
        # Initialize the chatbot to check connection
        try:
            chatbot = OllamaChatBot(
                model=ollama_model,
                end_point_url=ollama_endpoint,
                temperature=temperature
            )
            st.success(f"Successfully connected to Ollama at {ollama_endpoint}")
            connection_ok = True
        except Exception as e:
            st.error(f"Failed to initialize Ollama chatbot: {e}")
            st.error("Please check your endpoint URL and make sure the Ollama service is running.")
            connection_ok = False
        
        # Check if the directory exists
        if not os.path.exists(pdf_folder):
            if st.button("Create Directory"):
                try:
                    os.makedirs(pdf_folder)
                    st.success(f"Directory {pdf_folder} created successfully!")
                except Exception as e:
                    st.error(f"Error creating directory: {e}")
            else:
                st.warning(f"Directory {pdf_folder} does not exist.")
                return
        
        # Make sure output directory exists
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        # Get list of PDF files
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            st.info(f"No PDF files found in {pdf_folder}. Please add some PDF files to process.")
            return
        
        st.write(f"Found {len(pdf_files)} PDF files.")
        
        col1, col2 = st.columns(2)
        
        # Process button
        if col1.button("Process PDFs") and connection_ok:
            # Display the regular processing interface
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Clear any existing output file
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
                f.write('reference\n')
            
            for i, pdf_file in enumerate(pdf_files):
                pdf_path = os.path.join(pdf_folder, pdf_file)
                status_text.text(f"Processing {pdf_file}...")
                
                # Extract text from PDF
                text = extract_text_from_pdf(pdf_path)
                
                # Extract references section
                references_section = extract_references_section(text)
                
                if not references_section:
                    st.warning(f"No references section found in {pdf_file}")
                    continue
                
                # Reformat references using the chatbot and append to CSV
                reformat_references_with_llm(chatbot, references_section, pdf_file, chunk_size, output_csv_path)
                
                # Update progress
                progress_bar.progress((i + 1) / len(pdf_files))
            
            status_text.text("Processing complete!")
            
            # Read the output file to display
            try:
                # Read the file directly as it was written
                with open(output_csv_path, 'r', encoding='utf-8') as f:
                    # Skip header
                    next(f)
                    # Read all lines
                    references = [line.strip() for line in f if line.strip()]
                
                # Display results
                st.subheader("Extracted References")
                st.write(f"Total references found: {len(references)}")
                
                # Create a dataframe for display
                df_display = pd.DataFrame({'reference': references})
                
                # Display table with only reference column
                st.dataframe(df_display)
                
                # Download link (for convenience)
                st.markdown(
                    get_download_link(df_display, output_csv_path, "Download References as CSV"),
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error reading output file: {e}")
                st.warning("No references extracted from any PDF.")
        
        # Background process button
        if col2.button("Process PDFs in Background") and connection_ok:
            # Package settings
            settings = {
                'pdf_folder': pdf_folder,
                'pdf_files': pdf_files,
                'ollama_model': ollama_model,
                'ollama_endpoint': ollama_endpoint,
                'temperature': temperature,
                'chunk_size': chunk_size,
                'output_csv_path': output_csv_path
            }
            
            # Start the background process
            status_file = start_background_process(settings)
            
            st.success("Background process started! You can close this window and the process will continue.")
            st.info(f"Status will be saved to: {status_file}")
            st.info(f"Output will be saved to: {output_csv_path}")
            
            # Add a refresh button to check status
            if st.button("Check Status"):
                st.experimental_rerun()

if __name__ == "__main__":
    main()