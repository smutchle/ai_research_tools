import streamlit as st
import os
import pandas as pd
import PyPDF2
import re
import json
from collections import Counter
import base64
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

def reformat_references_with_llm(chatbot, references_text, filename, chunk_size):
    """Use the OllamaChatBot to reformat references in APA style, processing in chunks."""
    # Split references into manageable chunks - pass the chunk_size parameter
    reference_chunks = split_references_into_chunks(references_text, chunk_size)
    all_formatted_references = []
    successful_chunks = 0
    failed_chunks = 0
    
    # Process each chunk
    for i, chunk in enumerate(reference_chunks):
        progress_message = st.empty()
        progress_message.text(f"Processing chunk {i+1}/{len(reference_chunks)} from {filename}...")
        
        # Skip empty chunks
        if not chunk.strip():
            progress_message.empty()
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
        for attempt in range(max_retries):
            try:
                result = json.loads(chatbot.completeAsJSON(prompt))
                if result and isinstance(result, list):
                    all_formatted_references.extend(result)
                    successful_chunks += 1
                    break  # Success, exit retry loop
                else:
                    if attempt < max_retries - 1:  # Not the last attempt
                        st.warning(f"Retry {attempt+1}/{max_retries} for chunk {i+1} from {filename}...")
                    else:
                        st.warning(f"Could not process chunk {i+1} from {filename} after {max_retries} attempts")
                        failed_chunks += 1
            except Exception as e:
                if attempt < max_retries - 1:  # Not the last attempt
                    st.warning(f"Error in attempt {attempt+1}/{max_retries} for chunk {i+1} from {filename}. Retrying...")
                else:
                    st.error(f"Error reformatting chunk {i+1} from {filename}: {e}")
                    failed_chunks += 1
        
        progress_message.empty()
    
    # Summarize processing results
    if reference_chunks:
        st.info(f"References processing summary for {filename}: " +
                f"{successful_chunks}/{len(reference_chunks)} chunks processed successfully " +
                f"({failed_chunks} failed)")
    
    return all_formatted_references

def get_download_link(df, filename, text):
    """Generate a download link for a dataframe."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def main():
    load_dotenv(os.getcwd() + "/.env")

    st.title("PDF References Extractor and APA Formatter")
    
    # Sidebar configuration
    st.sidebar.header("Settings")
    pdf_folder = st.sidebar.text_input("PDF Directory", os.getenv('SOURCE_PDF_DIR'))
    ollama_endpoint = st.sidebar.text_input("Ollama Endpoint URL", os.getenv('OLLAMA_END_POINT'))
    ollama_model = st.sidebar.text_input("Ollama Model", os.getenv('OLLAMA_MODEL'))
    chunk_size = st.sidebar.slider("References per chunk", 1, 10, 5)    
    temperature = 1.0
    
    # Initialize the chatbot
    try:
        chatbot = OllamaChatBot(
            model=ollama_model,
            end_point_url=ollama_endpoint,
            temperature=temperature
        )
        st.success(f"Successfully connected to Ollama at {ollama_endpoint}")
    except Exception as e:
        st.error(f"Failed to initialize Ollama chatbot: {e}")
        st.error("Please check your endpoint URL and make sure the Ollama service is running.")
        return
    
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
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        st.info(f"No PDF files found in {pdf_folder}. Please add some PDF files to process.")
        return
    
    st.write(f"Found {len(pdf_files)} PDF files.")
    
    # Process button
    if st.button("Process PDFs"):
        all_references = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
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
            
            # Reformat references using the chatbot with chunking - pass the chunk_size parameter
            formatted_references = reformat_references_with_llm(chatbot, references_section, pdf_file, chunk_size)
            
            # Add file source and append to all references
            for ref in formatted_references:
                ref['source'] = pdf_file
                all_references.append(ref)
            
            # Update progress
            progress_bar.progress((i + 1) / len(pdf_files))
        
        status_text.text("Processing complete!")
        
        if all_references:
            # Convert to DataFrame
            df = pd.DataFrame(all_references)
            
            # Count duplicates
            formatted_refs = df['formatted'].tolist()
            ref_counts = Counter(formatted_refs)
            
            # Add count column
            df['count'] = df['formatted'].map(ref_counts)
            
            # Sort by count (descending)
            df_sorted = df.sort_values('count', ascending=False)
            
            # Display results
            st.subheader("Extracted References")
            st.write(f"Total references found: {len(df_sorted)}")
            st.write(f"Unique references: {len(ref_counts)}")
            
            # Display table
            st.dataframe(df_sorted[['formatted', 'count', 'source']])
            
            # Download link
            st.markdown(
                get_download_link(df_sorted, "references_apa.csv", "Download References as CSV"),
                unsafe_allow_html=True
            )
        else:
            st.warning("No references extracted from any PDF.")

if __name__ == "__main__":
    main()