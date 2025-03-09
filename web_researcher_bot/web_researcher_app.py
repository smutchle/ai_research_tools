import streamlit as st
import requests
import os
import time
import json
import re
import urllib.parse
import base64
import csv
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO
import urllib.robotparser
from OllamaChatBot import OllamaChatBot
import base64
from pathlib import Path

# Load environment variables from .env file
load_dotenv(os.getcwd() + "/.env")
# New function to extract metadata using the LLM
def extract_paper_metadata_with_llm(text, url, llm_model, endpoint_url):
    """
    Use LLM to extract metadata from scientific paper text
    
    Parameters:
    text (str): The text content of the paper
    url (str): URL of the paper
    llm_model (str): The LLM model to use
    endpoint_url (str): Endpoint URL for the LLM service
    
    Returns:
    dict: Metadata including title, authors, journal, year, etc.
    """
    try:
        # Initialize the OllamaChatBot
        bot = OllamaChatBot(
            model=llm_model,
            end_point_url=endpoint_url,
            temperature=0.2  # Lower temperature for more precise extraction
        )
        
        # Create a sample of the text to send to the LLM (first 2000 chars + middle 1000 + last 1000)
        text_length = len(text)
        if text_length > 4000:
            sample_text = text[:2000] + "\n...\n" + text[text_length//2-500:text_length//2+500] + "\n...\n" + text[-1000:]
        else:
            sample_text = text
        
        # Create a prompt for the LLM to extract metadata
        system_prompt = f"""
        You are a scientific research assistant tasked with extracting metadata from a scientific paper.
        Based on the following text excerpt from a scientific paper, extract the following metadata:
        
        1. Title of the paper
        2. Authors (list all authors with proper formatting)
        3. Journal or conference name
        4. Publication year
        5. DOI (Digital Object Identifier) if available
        6. Abstract (first 2-3 sentences only)
        
        Paper text excerpt:
        ```
        {sample_text}
        ```
        
        The paper was found at this URL: {url}
        
        Return your response as a JSON object with the following structure:
        {{"title": "Paper Title", "authors": "Author1, Author2, and Author3", "journal": "Journal Name", "year": "YYYY", "doi": "10.xxxx/xxxxx", "url": "{url}", "abstract": "Brief abstract..."}}
        
        If you cannot find certain information, use empty strings for those fields. Make your best guess for fields where the information is unclear or ambiguous.
        """
        
        # Get JSON response from the LLM
        json_response = bot.completeAsJSON(system_prompt)
        
        # Parse the JSON response
        metadata = json.loads(json_response)
        
        # Ensure all expected fields are present
        expected_fields = ['title', 'authors', 'journal', 'year', 'doi', 'url', 'abstract']
        for field in expected_fields:
            if field not in metadata:
                metadata[field] = ''
                
        # Ensure URL is present
        metadata['url'] = url
                
        return metadata
            
    except Exception as e:
        st.error(f"Error extracting metadata with LLM: {str(e)}")
        # Fall back to regex-based extraction
        return extract_paper_metadata(text, url)

# New function to generate citation using the LLM
def generate_citation_with_llm(metadata, citation_style="APA", llm_model=None, endpoint_url=None):
    """
    Generate a citation in the specified style using LLM
    
    Parameters:
    metadata (dict): Dictionary containing paper metadata
    citation_style (str): Citation style to use (APA, MLA, Chicago, etc.)
    llm_model (str): The LLM model to use
    endpoint_url (str): Endpoint URL for the LLM service
    
    Returns:
    str: Formatted citation string
    """
    try:
        # Initialize the OllamaChatBot
        bot = OllamaChatBot(
            model=llm_model,
            end_point_url=endpoint_url,
            temperature=0.1 
        )
        
        # Create a prompt for the LLM to generate a citation
        system_prompt = f"""
        You are a scientific citation specialist. Generate a proper {citation_style} citation for the following paper metadata:
        
        Title: {metadata.get('title', 'Unknown Title')}
        Authors: {metadata.get('authors', 'Unknown Authors')}
        Journal: {metadata.get('journal', '')}
        Year: {metadata.get('year', 'n.d.')}
        DOI: {metadata.get('doi', '')}
        URL: {metadata.get('url', '')}
        
        Return only the formatted citation with no additional text or explanation. If certain information is missing, adapt the citation format appropriately to handle the missing elements.
        """
        
        # Get response from the LLM
        citation = bot.complete(system_prompt).strip()
        
        return citation
            
    except Exception as e:
        st.error(f"Error generating citation with LLM: {str(e)}")
        # Fall back to traditional citation generator
        return generate_citation(metadata)

# Function to clean CSV data by replacing special characters with spaces
def clean_csv_references(df):
    """
    Clean the references in a DataFrame by replacing special characters with spaces
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing references
    
    Returns:
    pandas.DataFrame: DataFrame with cleaned references
    """
    # Make a copy of the DataFrame to avoid modifying the original
    cleaned_df = df.copy()
    
    # Replace special characters with spaces in the first column (references)
    if cleaned_df.shape[1] > 0:  # Ensure there's at least one column
        # Convert any non-string values to strings first
        cleaned_df.iloc[:, 0] = cleaned_df.iloc[:, 0].astype(str)
        
        # Replace underscore, ampersand, and hyphen with spaces
        cleaned_df.iloc[:, 0] = cleaned_df.iloc[:, 0].str.replace('_', ' ')
        cleaned_df.iloc[:, 0] = cleaned_df.iloc[:, 0].str.replace('&', ' ')
        cleaned_df.iloc[:, 0] = cleaned_df.iloc[:, 0].str.replace('-', ' ')
        
        # Remove extra spaces (multiple spaces -> single space)
        cleaned_df.iloc[:, 0] = cleaned_df.iloc[:, 0].str.replace(r'\s+', ' ', regex=True)
        
        # Trim leading/trailing whitespace
        cleaned_df.iloc[:, 0] = cleaned_df.iloc[:, 0].str.strip()
        
    return cleaned_df

# Add this function after your other utility functions (e.g., after save_binary_pdf)
def display_pdf_in_ui(pdf_path):
    """
    Display a PDF file in the Streamlit UI
    
    Parameters:
    pdf_path (str): Path to the PDF file
    
    Returns:
    None: Displays the PDF in the UI
    """
    try:
        # Check if file exists
        if not Path(pdf_path).is_file():
            st.error(f"PDF file not found: {pdf_path}")
            return
            
        # Open and read the PDF file
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
        # Embed the PDF in the Streamlit app using an iframe
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")

# Function to generate a citation in APA format
def generate_citation(metadata):
    """
    Generate an APA-style citation from paper metadata
    
    Parameters:
    metadata (dict): Dictionary containing paper metadata (title, authors, year, etc.)
    
    Returns:
    str: Formatted citation string
    """
    # Default citation if minimal information
    if not metadata.get('title') and not metadata.get('authors'):
        return f"Retrieved from {metadata.get('url', 'Unknown URL')}"
    
    # Authors
    authors = metadata.get('authors', 'Unknown')
    # If multiple authors are separated by commas, format properly
    if ',' in authors:
        authors = authors.replace(' and ', ', ')
    
    # Year
    year = metadata.get('year', 'n.d.')
    
    # Title
    title = metadata.get('title', 'Untitled document')
    
    # Journal/Source
    journal = metadata.get('journal', '')
    if journal:
        # Add italics markdown for the journal name
        journal = f"*{journal}*"
    
    # DOI
    doi = metadata.get('doi', '')
    doi_text = f"https://doi.org/{doi}" if doi else ""
    
    # Build citation
    citation = f"{authors} ({year}). {title}."
    if journal:
        citation += f" {journal}."
    if doi_text:
        citation += f" {doi_text}"
    
    return citation

# Function to create a CSV download link for the dataframe
def get_csv_download_link(df, filename="data.csv", text="Download CSV"):
    """
    Create a download link for a pandas DataFrame as CSV
    
    Parameters:
    df (pandas.DataFrame): DataFrame to convert to CSV
    filename (str): Name of the file to download
    text (str): Link text to display
    
    Returns:
    str: HTML link element as a string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
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

# Function to check if a URL is allowed by robots.txt
def is_url_allowed(url, respect_robots_txt):
    if not respect_robots_txt:
        return True
    
    try:
        parsed_url = urllib.parse.urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        robots_url = f"{base_url}/robots.txt"
        
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        
        return rp.can_fetch("*", url)
    except Exception as e:
        st.warning(f"Error checking robots.txt for {url}: {e}")
        return True  # Default to allowing if there's an error

# Function to detect if a PDF is likely a scientific paper
def is_scientific_paper(text):
    # Check for common sections in scientific papers
    paper_indicators = [
        r'\babstract\b', r'\bintroduction\b', r'\bmethods\b', r'\bresults\b', 
        r'\bdiscussion\b', r'\bconclusion\b', r'\breferences\b', r'\bcitation\b',
        r'\bjournal\b', r'\bvolume\b', r'\bissue\b', r'\bdoi\b', r'\backnowledgements\b'
    ]
    
    # Count how many indicators are present
    indicator_count = sum(1 for pattern in paper_indicators if re.search(pattern, text.lower()))
    
    # If at least 3 indicators are present, consider it a scientific paper
    return indicator_count >= 3

# Function to extract metadata from a scientific paper
def extract_paper_metadata(text, url):
    metadata = {
        'title': '',
        'authors': '',
        'journal': '',
        'year': '',
        'doi': '',
        'url': url,
        'abstract': ''
    }
    
    # Try to extract title (usually at the beginning, often in larger font)
    title_match = re.search(r'^([^\n]+)', text)
    if title_match:
        metadata['title'] = title_match.group(1).strip()
    
    # Try to extract authors (usually after title, before abstract)
    authors_pattern = r'(?:authors?|by)[:\s]+([^\.]+?)(?:\.|abstract|\n\n)'
    authors_match = re.search(authors_pattern, text.lower())
    if authors_match:
        metadata['authors'] = authors_match.group(1).strip()
    
    # Try to extract year
    year_pattern = r'(?:19|20)\d{2}'
    year_matches = re.findall(year_pattern, text[:1000])  # Look in the first 1000 chars
    if year_matches:
        metadata['year'] = year_matches[0]
    
    # Try to extract DOI
    doi_pattern = r'doi:?\s*([^\s]+)'
    doi_match = re.search(doi_pattern, text.lower())
    if doi_match:
        metadata['doi'] = doi_match.group(1).strip()
    
    # Try to extract journal name
    journal_patterns = [
        r'journal\s+of\s+([^,\.]+)',
        r'published\s+in\s+([^,\.]+)',
        r'proceedings\s+of\s+([^,\.]+)'
    ]
    for pattern in journal_patterns:
        journal_match = re.search(pattern, text.lower())
        if journal_match:
            metadata['journal'] = journal_match.group(1).strip()
            break
    
    # Try to extract abstract
    abstract_pattern = r'abstract[:\s]+([^\.]+(?:\.[^\.]+){0,5})'
    abstract_match = re.search(abstract_pattern, text.lower())
    if abstract_match:
        metadata['abstract'] = abstract_match.group(1).strip()
    
    return metadata

# Function to extract text from PDF
def extract_text_from_pdf(pdf_content):
    try:
        # Use PyPDF2
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

# Function to generate search phrases using the LLM
def generate_search_phrases(research_prompt, num_phrases=5):
    try:
        # Initialize the OllamaChatBot
        bot = OllamaChatBot(
            model=st.session_state.llm_model,
            end_point_url=st.session_state.endpoint_url,
            temperature=0.5
        )
        
        # Create a prompt for the LLM to generate search phrases
        system_prompt = f"""
        You are a research assistant helping to generate effective search phrases for a scientific research topic.
        Based on the following research prompt, generate {num_phrases} specific search phrases that would yield
        relevant scientific information. Format your response as a JSON array of strings.
        
        Research prompt: {research_prompt}
        
        Respond with JSON only, no explanations. Format: ["phrase 1", "phrase 2", ...]
        """
        
        # Get JSON response from the LLM
        json_response = bot.completeAsJSON(system_prompt)
        
        # Parse the JSON response
        search_phrases = json.loads(json_response)
        
        # Ensure we have a list of strings
        if isinstance(search_phrases, list) and all(isinstance(x, str) for x in search_phrases):
            return search_phrases[:num_phrases]  # Limit to requested number
        else:
            raise ValueError("LLM did not return a valid list of search phrases")
            
    except Exception as e:
        st.error(f"Error generating search phrases: {str(e)}")
        return [research_prompt]  # Fall back to the original prompt as a single search phrase

# Function to search Google using the Custom Search JSON API
def search_google(query, api_key, search_engine_id, num_results=10):
    try:
        if not search_engine_id:
            return [], "No search engine ID provided. Please enter your Google Custom Search Engine ID."
        
        # Google Custom Search API has a limit of 10 results per request
        # We need to make multiple requests with different start indices if user wants more
        max_per_request = 10
        all_results = []
        error = None
        
        # Calculate how many requests we need to make
        num_requests = (num_results + max_per_request - 1) // max_per_request  # Ceiling division
        
        for i in range(num_requests):
            # Calculate start index (1-based)
            start_index = i * max_per_request + 1
            
            # Calculate how many results to request in this batch
            results_to_request = min(max_per_request, num_results - len(all_results))
            
            # Use the Google Custom Search JSON API
            url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={urllib.parse.quote(query)}&num={results_to_request}&start={start_index}"
            
            response = requests.get(url)
            if response.status_code == 200:
                results = response.json()
                if "items" in results:
                    batch_results = [(item["title"], item["link"]) for item in results["items"]]
                    all_results.extend(batch_results)
                else:
                    # No more results available
                    break
            else:
                error_message = f"Google API error: {response.status_code}"
                try:
                    error_details = response.json()
                    if "error" in error_details and "message" in error_details["error"]:
                        error_message += f" - {error_details['error']['message']}"
                except:
                    error_message += f" - {response.text}"
                
                error = error_message
                break
            
            # If we've got enough results, stop making more requests
            if len(all_results) >= num_results:
                break
        
        # Check if we got any results
        if all_results:
            return all_results[:num_results], None  # Limit to exactly the number requested
        elif error:
            return [], error
        else:
            return [], f"No search results found for query: {query}"
            
    except Exception as e:
        return [], f"Error searching Google: {str(e)}"

# Modified function to scrape a webpage for PDFs only
def scrape_webpage(url, respect_robots_txt, politeness_delay):
    try:
        # Check if URL is allowed by robots.txt
        if not is_url_allowed(url, respect_robots_txt):
            return f"Skipped {url} due to robots.txt restrictions", None, None
            
        # Add politeness delay
        time.sleep(politeness_delay)
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            return f"Failed to retrieve {url}: HTTP {response.status_code}", None, None
            
        content_type = response.headers.get("Content-Type", "").lower()
        
        # If it's a PDF, return the original PDF content
        if "application/pdf" in content_type or url.lower().endswith('.pdf'):
            return None, None, response.content  # Return no text, no PDF links, just the original PDF content
            
        # Otherwise, parse HTML to find PDF links only
        soup = BeautifulSoup(response.text, "html.parser")
            
        # Check for PDF links in the HTML
        pdf_links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.lower().endswith('.pdf'):
                # Convert relative URLs to absolute
                absolute_link = urllib.parse.urljoin(url, href)
                pdf_links.append((link.text.strip() or "Unnamed PDF", absolute_link))
                
        return None, pdf_links, None  # Return no text content, only PDF links
        
    except Exception as e:
        return f"Error scraping {url}: {str(e)}", None, None

# Function to download a PDF from a URL
def download_pdf(url, respect_robots_txt, politeness_delay):
    try:
        # Check if URL is allowed by robots.txt
        if not is_url_allowed(url, respect_robots_txt):
            return None, f"Skipped {url} due to robots.txt restrictions"
            
        # Add politeness delay
        time.sleep(politeness_delay)
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            return None, f"Failed to retrieve {url}: HTTP {response.status_code}"
            
        content_type = response.headers.get("Content-Type", "").lower()
        
        if "application/pdf" in content_type or url.lower().endswith('.pdf'):
            return response.content, None
        else:
            return None, f"URL {url} did not return a PDF"
            
    except Exception as e:
        return None, f"Error downloading PDF from {url}: {str(e)}"

# Function to save content as PDF
def save_as_pdf(content, output_path, filename):
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Sanitize filename
        filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
        
        # Ensure filename ends with .pdf
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'
            
        # Full path to save
        full_path = os.path.join(output_path, filename)
        
        # Create a new PDF with the text content using PyPDF2
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        
        # Split content into chunks that can fit on pages
        def split_text_into_chunks(text, chunk_size=3000):
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        chunks = split_text_into_chunks(content)
        
        # Create PDF
        doc = SimpleDocTemplate(full_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        for chunk in chunks:
            para = Paragraph(chunk.replace('\n', '<br/>'), styles['Normal'])
            story.append(para)
            story.append(Spacer(1, 12))
            
        doc.build(story)
        
        return full_path
        
    except Exception as e:
        st.error(f"Error saving PDF: {str(e)}")
        return None

# Function to save binary PDF content
def save_binary_pdf(pdf_content, output_path, filename):
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Sanitize filename
        filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
        
        # Ensure filename ends with .pdf
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'
            
        # Full path to save
        full_path = os.path.join(output_path, filename)
        
        with open(full_path, 'wb') as f:
            f.write(pdf_content)
            
        return full_path
        
    except Exception as e:
        st.error(f"Error saving PDF: {str(e)}")
        return None

# New function to format a safe filename from metadata
def format_safe_filename(metadata):
    """
    Create a safe filename from paper metadata
    
    Parameters:
    metadata (dict): Dictionary containing paper metadata
    
    Returns:
    str: Safe filename in format: year_authors_title
    """
    # Extract year, fallback to 'unknown' if not present
    year = metadata.get('year', 'unknown')
    
    # Process authors: convert to lowercase, replace spaces with underscores
    authors = metadata.get('authors', 'unknown_author')
    # Keep only the first author's last name, or first two words
    if ',' in authors:
        first_author = authors.split(',')[0].strip()
    else:
        author_parts = authors.split()
        first_author = author_parts[-1] if len(author_parts) > 0 else "unknown"
    
    # Process title: lowercase, max 50 chars, replace spaces with underscores
    title = metadata.get('title', 'untitled')
    title = title[:50]  # Limit title length
    
    # Combine and sanitize
    filename = f"{year}_{first_author}_{title}"
    # Replace invalid characters and multiple spaces/underscores
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)  # Remove invalid chars
    filename = re.sub(r'\s+', "_", filename)  # Replace spaces with underscores
    filename = re.sub(r'_+', "_", filename)  # Replace multiple underscores with single
    
    return filename.lower() + '.pdf'

# New function to verify if a PDF matches a reference
def verify_pdf_matches_reference(pdf_content, reference, llm_model, endpoint_url):
    """
    Verify if a PDF matches a given reference using LLM
    
    Parameters:
    pdf_content (bytes): The binary content of the PDF
    reference (str): The reference text to match against
    llm_model (str): The LLM model to use
    endpoint_url (str): Endpoint URL for the LLM service
    
    Returns:
    bool: True if the PDF matches the reference, False otherwise
    """
    try:
        # Extract text from the first few pages of the PDF
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
        
        # Get text from first 3 pages or all pages if fewer than 3
        sample_text = ""
        for i in range(min(3, len(pdf_reader.pages))):
            page_text = pdf_reader.pages[i].extract_text() or ""
            sample_text += page_text
            
        # Limit sample text length (first 3000 characters)
        sample_text = sample_text[:3000]
        
        # Initialize the OllamaChatBot
        bot = OllamaChatBot(
            model=llm_model,
            end_point_url=endpoint_url,
            temperature=0.1  # Low temperature for more consistent outputs
        )
        
        # Create a prompt for the LLM to verify the match
        system_prompt = f"""
        You are a scientific reference matching system. Your task is to determine if the provided PDF 
        excerpt matches the given academic reference.
        
        Reference: {reference}
        
        PDF excerpt (first portion of the document):
        ```
        {sample_text}
        ```
        
        Analyze if this PDF is likely the document described in the reference. Consider:
        1. Do author names match or appear in the PDF?
        2. Does the title or key terms from it appear in the PDF?
        3. Is the publication year consistent?
        4. Does the content seem relevant to the reference topic?
        
        Return only a JSON object with this format:
        {{"match": true/false, "confidence": "high"/"medium"/"low", "reason": "brief explanation"}}
        """
        
        # Get JSON response from the LLM
        json_response = bot.completeAsJSON(system_prompt)
        
        # Parse the JSON response
        result = json.loads(json_response)
        
        # Return match status, with a fallback if JSON parsing fails
        return result.get('match', False)
            
    except Exception as e:
        st.error(f"Error verifying PDF match: {str(e)}")
        return False  # By default, don't consider it a match if verification fails

# Function to process a reference CSV
def process_reference_csv(csv_file, api_key, search_engine_id, llm_model, endpoint_url, 
                         respect_robots_txt, politeness_delay, output_path, max_results_per_ref=3):
    """
    Process a CSV file of references to find matching PDFs
    
    Parameters:
    csv_file: The uploaded CSV file
    api_key: Google API key
    search_engine_id: Google Custom Search Engine ID
    llm_model: LLM model to use for verification
    endpoint_url: LLM endpoint URL
    respect_robots_txt: Whether to respect robots.txt
    politeness_delay: Delay between requests
    output_path: Where to save PDFs
    max_results_per_ref: Maximum search results to try per reference
    
    Returns:
    list: List of successful matches with metadata
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Check if the CSV has at least one column
        if df.shape[1] < 1:
            return [], "CSV file must have at least one column with references"

        df = clean_csv_references(df)
        
        # Assume the first column contains the references
        references = df.iloc[:, 0].tolist()
        
        # Process each reference
        results = []
        for i, reference in enumerate(references):
            with st.expander(f"Processing reference {i+1}: {reference[:100]}..."):
                st.write(f"Searching for: {reference[:100]}...")
                
                # Search Google for the reference
                search_results, error = search_google(reference, api_key, search_engine_id, num_results=max_results_per_ref)
                
                if error:
                    st.error(f"Error searching for reference: {error}")
                    continue
                    
                if not search_results:
                    st.warning("No search results found for this reference")
                    continue
                
                # Try each search result until a match is found
                match_found = False
                for title, url in search_results:
                    st.write(f"Trying: {title} ({url})")
                    
                    # Download PDF if direct link
                    if url.lower().endswith('.pdf'):
                        pdf_content, error = download_pdf(url, respect_robots_txt, politeness_delay)
                        
                        if pdf_content:
                            # Verify if the PDF matches the reference
                            if verify_pdf_matches_reference(pdf_content, reference, llm_model, endpoint_url):
                                # Extract metadata from PDF and reference
                                extracted_text = extract_text_from_pdf(pdf_content)
                                metadata = extract_paper_metadata_with_llm(
                                    extracted_text, url, llm_model, endpoint_url
                                )
                                
                                # Create a filename based on the metadata
                                filename = format_safe_filename(metadata)
                                saved_path = save_binary_pdf(pdf_content, output_path, filename)
                                
                                if saved_path:
                                    results.append({
                                        "Reference": reference,
                                        "Title": metadata.get('title', title),
                                        "Authors": metadata.get('authors', ''),
                                        "Year": metadata.get('year', ''),
                                        "File Path": saved_path,
                                        "Source URL": url
                                    })
                                    match_found = True
                                    st.success(f"✅ Match found and saved as: {filename}")
                                    break
                            else:
                                st.info("PDF didn't match the reference, trying next result...")
                                
                    # If not a direct PDF, scrape the page for PDF links
                    else:
                        content, pdf_links, original_pdf = scrape_webpage(url, respect_robots_txt, politeness_delay)
                        
                        # Check if the URL returned a PDF directly
                        if original_pdf:
                            if verify_pdf_matches_reference(original_pdf, reference, llm_model, endpoint_url):
                                extracted_text = extract_text_from_pdf(original_pdf)
                                metadata = extract_paper_metadata_with_llm(
                                    extracted_text, url, llm_model, endpoint_url
                                )
                                
                                filename = format_safe_filename(metadata)
                                saved_path = save_binary_pdf(original_pdf, output_path, filename)
                                
                                if saved_path:
                                    results.append({
                                        "Reference": reference,
                                        "Title": metadata.get('title', title),
                                        "Authors": metadata.get('authors', ''),
                                        "Year": metadata.get('year', ''),
                                        "File Path": saved_path,
                                        "Source URL": url
                                    })
                                    match_found = True
                                    st.success(f"✅ Match found and saved as: {filename}")
                                    break
                        
                        # Check PDF links from the page
                        if isinstance(pdf_links, list) and pdf_links:
                            for pdf_title, pdf_url in pdf_links:
                                st.write(f"Checking PDF link: {pdf_title}")
                                pdf_content, error = download_pdf(pdf_url, respect_robots_txt, politeness_delay)
                                
                                if pdf_content and verify_pdf_matches_reference(pdf_content, reference, llm_model, endpoint_url):
                                    extracted_text = extract_text_from_pdf(pdf_content)
                                    metadata = extract_paper_metadata_with_llm(
                                        extracted_text, pdf_url, llm_model, endpoint_url
                                    )
                                    
                                    filename = format_safe_filename(metadata)
                                    saved_path = save_binary_pdf(pdf_content, output_path, filename)
                                    
                                    if saved_path:
                                        results.append({
                                            "Reference": reference,
                                            "Title": metadata.get('title', pdf_title),
                                            "Authors": metadata.get('authors', ''),
                                            "Year": metadata.get('year', ''),
                                            "File Path": saved_path,
                                            "Source URL": pdf_url
                                        })
                                        match_found = True
                                        st.success(f"✅ Match found and saved as: {filename}")
                                        break
                            
                            # Break the search results loop if we found a match
                            if match_found:
                                break
                
                if not match_found:
                    st.warning("❌ No matching PDF found for this reference")
        
        return results, None
        
    except Exception as e:
        return [], f"Error processing reference CSV: {str(e)}"

# Set page title and configuration
st.set_page_config(
    page_title="Scientific Web Crawler/Scraper",
    page_icon="🔍",
    layout="wide"
)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # LLM Configuration
    st.subheader("🤖 LLM Settings")
    
    st.session_state.llm_model = st.selectbox("LLM Model", split_csv(os.getenv("OLLAMA_MODEL")))
    st.session_state.endpoint_url = st.text_input("Endpoint URL", os.getenv('OLLAMA_END_POINT'))
    
    # Google API Configuration
    st.subheader("🔎 Google API")
    
    # Get API key from .env
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Google API key not found in .env file")
    else:
        st.success("Google API key loaded from .env file")
    
    # Add search engine ID input
    search_engine_id = st.text_input(
        "Google Custom Search Engine ID", 
        value=os.getenv("GOOGLE_SEARCH_ENGINE_ID"),
        help="Create a programmable search engine at https://programmablesearchengine.google.com/"
    )
    
    num_search_results = st.slider("Number of search results per phrase", min_value=1, max_value=20, value=5)
    num_search_phrases = st.slider("Number of search phrases to generate", min_value=1, max_value=10, value=3)
    
    # Scraping Configuration
    st.subheader("🕸️ Scraping Settings")
    respect_robots_txt = st.checkbox("Respect robots.txt exclusions", value=True)
    politeness_delay = st.slider("Politeness delay (seconds)", min_value=0, max_value=10, value=0, step=1)
    
    # Output Configuration
    st.subheader("📁 Output Settings")
    output_path = st.text_input("Output folder", value=os.getenv("OUTPUT_DIR"))

# Main content area
st.title("🔬 Scientific Web Crawler/Scraper 🌐")
st.write("🔍 This tool helps gather information for scientific research by searching the web for relevant content.")

# Create tabs for different search modes
tab1, tab2 = st.tabs(["📝 Search by Topics", "📋 Search by Reference CSV"])

# Initialize results container
if 'results' not in st.session_state:
    st.session_state.results = {}
    
# Initialize scientific papers container
if 'scientific_papers' not in st.session_state:
    st.session_state.scientific_papers = []

# Tab 1: Original search by topics functionality
with tab1:
    research_prompt = st.text_area("Enter your research prompt:", height=150)
    
    if st.button("🚀 Start Topic Research"):
        if not research_prompt:
            st.error("Please enter a research prompt")
        elif not api_key:
            st.error("Google API key not found. Please add it to the .env file.")
        else:
            # Clear previous results
            st.session_state.results = {}
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Generate search phrases
            status_text.text("Generating search phrases...")
            search_phrases = generate_search_phrases(research_prompt, num_phrases=num_search_phrases)
            
            if search_phrases:
                st.write("### 🔤 Search Phrases Generated")
                for i, phrase in enumerate(search_phrases):
                    st.write(f"{i+1}. {phrase}")
                    
                # Step 2: Search Google for each phrase
                all_search_results = []
                search_error = None
                
                for i, phrase in enumerate(search_phrases):
                    status_text.text(f"Searching for: {phrase}...")
                    progress_bar.progress((i / len(search_phrases)) * 0.2)  # First 20% of progress
                    
                    search_results, error = search_google(phrase, api_key, search_engine_id, num_results=num_search_results)
                    
                    if error:
                        search_error = error
                        break
                        
                    if search_results:
                        all_search_results.extend(search_results)
                
                # If we encountered an error during search, show it
                if search_error:
                    st.error(f"Error during Google search: {search_error}")
                else:
                    # Remove duplicates
                    unique_results = []
                    seen_urls = set()
                    for title, url in all_search_results:
                        if url not in seen_urls:
                            unique_results.append((title, url))
                            seen_urls.add(url)
                        
                # Step 3: Scrape each result to find PDFs
                if unique_results:
                    st.write(f"### 🔗 Found {len(unique_results)} unique results")
                    
                    # Create a container for the scraped content
                    content_container = st.container()
                    
                    # Process each result
                    pdf_files_found = []  # Track all PDF files (direct and linked)
                    
                    for i, (title, url) in enumerate(unique_results):
                        status_text.text(f"Processing: {title}...")
                        progress_bar.progress(0.2 + (i / len(unique_results)) * 0.6)  # 20% to 80% of progress
                        
                        # Check if the URL is a direct PDF
                        if url.lower().endswith('.pdf'):
                            status_text.text(f"Downloading PDF: {title}...")
                            pdf_content, error = download_pdf(url, respect_robots_txt, politeness_delay)
                            
                            if pdf_content:
                                # Extract metadata using LLM
                                extracted_text = extract_text_from_pdf(pdf_content)
                                if is_scientific_paper(extracted_text):
                                    metadata = extract_paper_metadata_with_llm(
                                        extracted_text, url, st.session_state.llm_model, st.session_state.endpoint_url
                                    )
                                    
                                    # Create a filename based on the metadata
                                    filename = format_safe_filename(metadata)
                                    saved_path = save_binary_pdf(pdf_content, output_path, filename)
                                    
                                    if saved_path:
                                        pdf_files_found.append((metadata.get('title', title), saved_path, url))
                            continue
                        
                        # Scrape the webpage to find PDF links
                        content, pdf_links, original_pdf = scrape_webpage(url, respect_robots_txt, politeness_delay)
                        
                        # If the URL was a PDF (but didn't have .pdf extension)
                        if original_pdf:
                            extracted_text = extract_text_from_pdf(original_pdf)
                            if is_scientific_paper(extracted_text):
                                metadata = extract_paper_metadata_with_llm(
                                    extracted_text, url, st.session_state.llm_model, st.session_state.endpoint_url
                                )
                                
                                filename = format_safe_filename(metadata)
                                pdf_path = save_binary_pdf(original_pdf, output_path, filename)
                                
                                if pdf_path:
                                    pdf_files_found.append((metadata.get('title', title), pdf_path, url))
                                    
                        # Process any PDF links found
                        if isinstance(pdf_links, list) and pdf_links:
                            for pdf_title, pdf_url in pdf_links:
                                status_text.text(f"Downloading PDF: {pdf_title}...")
                                
                                # Download the PDF
                                pdf_content, error = download_pdf(pdf_url, respect_robots_txt, politeness_delay)
                                
                                if pdf_content:
                                    extracted_text = extract_text_from_pdf(pdf_content)
                                    if is_scientific_paper(extracted_text):
                                        metadata = extract_paper_metadata_with_llm(
                                            extracted_text, pdf_url, st.session_state.llm_model, st.session_state.endpoint_url
                                        )
                                        
                                        filename = format_safe_filename(metadata)
                                        saved_path = save_binary_pdf(pdf_content, output_path, filename)
                                        
                                        if saved_path:
                                            pdf_files_found.append((metadata.get('title', pdf_title), saved_path, pdf_url))
                
                    # Display results
                    status_text.text("Finalizing results...")
                    progress_bar.progress(1.0)
                    
                    # Show final results
                    with content_container:
                        if pdf_files_found:
                            st.write("### 📊 Research Results")
                            st.write(f"**{len(pdf_files_found)} PDFs found and downloaded:**")
                            
                            # Create a dataframe for better display
                            pdf_df = pd.DataFrame(pdf_files_found, columns=["Title", "File Path", "Source URL"])
                            st.dataframe(pdf_df)
                            
                            # Create download button for CSV of PDFs
                            csv = pdf_df.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="found_pdfs.csv">📥 Download PDF list as CSV</a>'
                            st.markdown(href, unsafe_allow_html=True)
                        else:
                            st.warning("No PDF files were found for the given search phrases.")
                                
                    status_text.text("🎉 PDF search completed!")
                else:
                    st.warning("No search results found for the given search phrases.")
            else:
                st.error("Failed to generate search phrases. Please try a different research prompt.")

# Tab 2: New CSV-based reference search functionality
with tab2:
    st.write("Upload a CSV file with references in the first column")
    
    # File uploader for CSV
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Preview the CSV
        try:
            df_preview = pd.read_csv(uploaded_file)
            st.write("Preview of first 5 references:")
            st.dataframe(df_preview.iloc[:5, 0].reset_index(drop=True))
            
            # Reset the file pointer to the beginning
            uploaded_file.seek(0)
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
    
    max_refs_to_process = st.slider("Maximum references to process", min_value=1, max_value=50, value=10, 
                               help="Limit the number of references to process to avoid long execution times")
    
    if st.button("🚀 Start Reference Search"):
        if uploaded_file is None:
            st.error("Please upload a CSV file with references")
        elif not api_key:
            st.error("Google API key not found. Please add it to the .env file.")
        else:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Processing reference CSV...")
            
            # Process the CSV file
            results, error = process_reference_csv(
                uploaded_file, 
                api_key, 
                search_engine_id, 
                st.session_state.llm_model,
                st.session_state.endpoint_url, 
                respect_robots_txt, 
                politeness_delay, 
                output_path,
                max_results_per_ref=3
            )
            
            # Show results
            if error:
                st.error(f"Error processing references: {error}")
            else:
                progress_bar.progress(1.0)
                status_text.text("🎉 Reference search completed!")
                
                if results:
                    st.write(f"### 📊 Found {len(results)} matching PDFs")
                    
                    # Create a dataframe for better display
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)
                    
                    # Create download button for results CSV
                    csv = results_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="reference_matches.csv">📥 Download results as CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    st.warning("No matching PDFs were found for the given references.")

# PDF viewing section
st.markdown("---")
st.write("### 📄 View Saved PDFs")

# Check if output folder exists
output_folder = Path(output_path) if 'output_path' in locals() else Path("output")
if not output_folder.exists():
    st.info(f"Output folder '{output_folder}' does not exist yet. Run a search to generate PDFs.")
else:
    # Get list of PDF files in the output folder
    pdf_files = list(output_folder.glob("*.pdf"))
    
    if not pdf_files:
        st.info(f"No PDF files found in '{output_folder}'. Run a search to generate PDFs.")
    else:
        # Create a dropdown to select a PDF file
        selected_pdf = st.selectbox(
            "Select a PDF file to view:",
            options=pdf_files,
            format_func=lambda x: x.name
        )
        
        if selected_pdf:
            st.write(f"**Viewing:** {selected_pdf.name}")
            
            # Add button to open PDF in new tab (as backup option)
            pdf_data = open(selected_pdf, "rb").read()
            pdf_b64 = base64.b64encode(pdf_data).decode('utf-8')
            href = f'<a href="data:application/pdf;base64,{pdf_b64}" download="{selected_pdf.name}" target="_blank">Open PDF in new tab</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Display the selected PDF
            display_pdf_in_ui(selected_pdf)

# Footer
st.markdown("---")
st.write("🔬 Scientific Web Crawler/Scraper - Research Assistant Tool 📚")