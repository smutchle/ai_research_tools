import streamlit as st
import os
import json
import glob
from typing import List, Dict, Any
import pandas as pd
from LLMReranker import LLMReranker
from OllamaChatBot import OllamaChatBot
from dotenv import load_dotenv

def load_metadata_files(folder_path: str) -> List[Dict[str, Any]]:
    """
    Load all metadata JSON files from the metadata folder.
    
    Args:
        folder_path: Path to the folder containing the metadata subfolder
        
    Returns:
        List of dictionaries containing metadata
    """
    metadata_folder = os.path.join(folder_path, "metadata")
    metadata_files = glob.glob(os.path.join(metadata_folder, "*.json"))
    
    metadata_list = []
    for file_path in metadata_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
                # Skip files with errors
                if "error" in metadata:
                    continue
                    
                # Add source document filename
                source_filename = os.path.basename(file_path).replace(".json", "")
                metadata["source_filename"] = source_filename
                metadata["full_path"] = os.path.join(folder_path, source_filename)
                
                metadata_list.append(metadata)
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
    
    return metadata_list

def read_document_content(file_path: str) -> str:
    """
    Read the content of a document file (PDF or Markdown).
    
    Args:
        file_path: Path to the document file
        
    Returns:
        String containing the document content
    """
    try:
        if file_path.lower().endswith('.pdf'):
            # Use the PDF reading function from your existing code
            from PyPDF2 import PdfReader
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
            return text
        elif file_path.lower().endswith('.md'):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        else:
            return f"Unsupported file format: {os.path.basename(file_path)}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

def get_document_chunk_data(file_path: str, chunk_size: int, chunk_overlap: int, 
                           reranker: LLMReranker, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Read a document, chunk it, and prepare the data for ranking.
    
    Args:
        file_path: Path to the document file
        chunk_size: Size of chunks in characters
        chunk_overlap: Overlap between chunks in characters
        reranker: LLMReranker instance for chunking
        metadata: Document metadata
        
    Returns:
        List of dictionaries with chunk data
    """
    # Read document content
    content = read_document_content(file_path)
    
    # Chunk content based on file type
    if file_path.lower().endswith('.pdf'):
        chunks = reranker.chunk_pdf_text(content, chunk_size, chunk_overlap)
    elif file_path.lower().endswith('.md'):
        chunks = reranker.chunk_markdown(content)
    else:
        chunks = [content]  # Just one chunk for unsupported files
    
    # Create chunk data dictionaries
    chunk_data = []
    reference = metadata.get("reference", f"Source: {os.path.basename(file_path)}")
    
    for i, chunk_text in enumerate(chunks, 1):
        chunk_data.append({
            "chunk": chunk_text,
            "source": os.path.basename(file_path),
            "chunk_number": i,
            "reference": reference,
            "title": metadata.get("title", "Unknown Title"),
            "authors": metadata.get("authors", "Unknown Authors"),
            "year": metadata.get("year", "Unknown Year")
        })
    
    return chunk_data

def main():
    load_dotenv(os.path.join(os.getcwd(), ".env"))

    st.title("Document Chunk Ranker")
    
    # Add imports for PDF generation
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    import io

    # Sidebar for settings and filtering
    st.sidebar.header("Settings")
    folder_path = st.sidebar.text_input("Enter folder path containing documents:", os.getenv("SOURCE_DOC_DIR"))
    
    # Use environment variables directly
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
    ollama_endpoint = os.getenv("OLLAMA_END_POINT", "http://localhost:11434")
    temperature = 0.1  # Default temperature
    
    # Simple settings that users might want to adjust
    chunk_size = 2000  # Default chunk size
    chunk_overlap = 200  # Default chunk overlap
    min_score_threshold = st.sidebar.slider("Min Ranking", min_value=1, max_value=10, value=6, 
                                         help="Only show chunks with this ranking or higher")
    
    # Initialize session state for selected papers
    if "selected_papers" not in st.session_state:
        st.session_state.selected_papers = []
    
    if "ranked_chunks" not in st.session_state:
        st.session_state.ranked_chunks = []
    
    if "selected_chunks" not in st.session_state:
        st.session_state.selected_chunks = []
        
    if "current_query" not in st.session_state:
        st.session_state.current_query = "What are the key findings and methodologies in these papers?"
    
    # Main workflow
    if folder_path and os.path.isdir(folder_path):
        # Step 1: Load and filter papers
        papers = load_metadata_files(folder_path)
        
        if not papers:
            st.warning("No metadata files found. Please run the Document Metadata Generator first.")
        else:
            # Convert to DataFrame for easy filtering
            papers_df = pd.DataFrame(papers)
            
            # Create tabs for document selection, prompt, chunk selection, and output
            tab1, tab2, tab3, tab4 = st.tabs(["Document Selection", "Prompt", "Chunk Selection", "Output"])
            
            # Display paper filtering options in sidebar
            st.sidebar.subheader("Filter Papers")
            
            # Text search filter in sidebar
            search_query = st.sidebar.text_input("Search titles, abstracts, or authors:")
            
            # Year range filter if 'year' is available (in sidebar)
            if 'year' in papers_df.columns:
                min_year, max_year = int(papers_df['year'].min()), int(papers_df['year'].max())
                year_range = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))
                
                # Apply year filter
                filtered_df = papers_df[(papers_df['year'] >= year_range[0]) & 
                                       (papers_df['year'] <= year_range[1])]
            else:
                filtered_df = papers_df
            
            # Apply text search filter
            if search_query:
                # Search in title, abstract, and authors if these columns exist
                mask = pd.Series(False, index=filtered_df.index)
                
                if 'title' in filtered_df.columns:
                    mask |= filtered_df['title'].fillna('').str.contains(search_query, case=False)
                
                if 'abstract' in filtered_df.columns:
                    mask |= filtered_df['abstract'].fillna('').str.contains(search_query, case=False)
                
                if 'authors' in filtered_df.columns:
                    mask |= filtered_df['authors'].fillna('').str.contains(search_query, case=False)
                
                filtered_df = filtered_df[mask]
            
            # TAB 1: Document Selection
            with tab1:
                st.write(f"Found {len(filtered_df)} papers matching your criteria.")
                
                # Create a detailed table with scrollable container
                paper_selection_container = st.container()
                
                with paper_selection_container:
                    # Create a scrollable list with detailed information
                    st.subheader("Available Papers")
                    
                    # Create a selection container with a scrollable height
                    paper_list = st.container()
                    
                    # Display the papers in an expandable format
                    for i, (_, paper) in enumerate(filtered_df.iterrows()):
                        # With this code:
                        # Format header: Year - Authors - Title - Journal
                        year = str(paper.get('year', 'n/a'))
                        # With this code to properly format year values:
                        # Handle year - convert to integer if it's a float
                        year_value = paper.get('year', 'n/a')
                        if isinstance(year_value, float):
                            # Check if it's a whole number
                            if year_value.is_integer():
                                year = str(int(year_value))  # Convert 1970.0 to "1970"
                            else:
                                year = str(year_value)  # Keep decimal if it's not a whole number
                        elif year_value is None:
                            year = 'n/a'
                        else:
                            year = str(year_value)
                        # Handle authors - ensure it's a string
                        authors = paper.get('authors', 'Unknown')
                        if authors is None or not isinstance(authors, str):
                            authors = "Unknown"
                        # Trim authors if too long
                        if len(authors) > 40:
                            authors = authors[:37] + "..."
                        
                        # Ensure title is a string
                        title = str(paper.get('title', 'Unknown Title'))
                        
                        # Handle journal - ensure it's a string
                        journal = paper.get('journal', 'Unknown')
                        if journal is None or not isinstance(journal, str):
                            journal = "Unknown"
                        elif isinstance(journal, str) and len(journal) > 30:
                            journal = journal[:27] + "..."
                        header = f"{year} - {authors} - {title} - {journal}"
                        if len(header) > 120:  # Trim if too long
                            header = header[:117] + "..."
                            
                        with paper_list.expander(header):
                            cols = st.columns([3, 1])
                            with cols[0]:
                                st.markdown(f"**Authors:** {paper.get('authors', 'Unknown')}")
                                st.markdown(f"**Year:** {paper.get('year', 'Unknown')}")
                                st.markdown(f"**Journal:** {paper.get('journal', 'Unknown')}")
                                st.markdown(f"**File:** {paper.get('source_filename', '')}")
                                if 'abstract' in paper and paper['abstract']:
                                    st.markdown("**Abstract:**")
                                    st.markdown(f"{paper['abstract']}")
                            with cols[1]:
                                # Add a select button for each paper
                                if st.button(f"Select", key=f"select_paper_{i}"):
                                    # Check if already selected
                                    already_selected = any(p.get('source_filename') == paper.get('source_filename') 
                                                          for p in st.session_state.selected_papers)
                                    if not already_selected:
                                        st.session_state.selected_papers.append(paper.to_dict())
                                        st.success("Added to selection!")
                                    else:
                                        st.info("Already selected!")
                
                # Display selected papers
                if st.session_state.selected_papers:
                    st.subheader("Selected Papers")
                    for i, paper in enumerate(st.session_state.selected_papers):
                        st.write(f"{i+1}. {paper.get('title')} ({paper.get('year')})")
                    
                    # Allow clearing selection
                    if st.button("Clear Paper Selection"):
                        st.session_state.selected_papers = []
                        st.success("Selection cleared!")
                        st.experimental_rerun()
            
            # TAB 2: Prompt entry
            with tab2:
                st.subheader("Research Question/Prompt")
                
                if not st.session_state.selected_papers:
                    st.warning("Please select papers in the 'Document Selection' tab first.")
                else:
                    st.write("Enter your research question or topic to find relevant chunks in the selected papers:")
                    query = st.text_area("", 
                                       st.session_state.current_query, 
                                       height=150)
                    
                    # Display the selected papers for reference
                    st.subheader("Selected Papers")
                    for i, paper in enumerate(st.session_state.selected_papers):
                        # Format the paper header to match Tab 1
                        year_value = paper.get('year', 'n/a')
                        if isinstance(year_value, float):
                            # Check if it's a whole number
                            if year_value.is_integer():
                                year = str(int(year_value))  # Convert 1970.0 to "1970"
                            else:
                                year = str(year_value)  # Keep decimal if it's not a whole number
                        elif year_value is None:
                            year = 'n/a'
                        else:
                            year = str(year_value)
                            
                        # Handle authors - ensure it's a string
                        authors = paper.get('authors', 'Unknown')
                        if authors is None or not isinstance(authors, str):
                            authors = "Unknown"
                        # Trim authors if too long
                        if len(authors) > 40:
                            authors = authors[:37] + "..."
                        
                        # Ensure title is a string
                        title = str(paper.get('title', 'Unknown Title'))
                        
                        # Handle journal - ensure it's a string
                        journal = paper.get('journal', 'Unknown')
                        if journal is None or not isinstance(journal, str):
                            journal = "Unknown"
                        elif isinstance(journal, str) and len(journal) > 30:
                            journal = journal[:27] + "..."
                            
                        # Create the formatted header
                        header = f"{i+1}. {year} - {authors} - {title} - {journal}"
                        if len(header) > 120:  # Trim if too long
                            header = header[:117] + "..."
                            
                        st.write(header)
                        
                    # Analyze button in the prompt tab
                    if st.button("Analyze Selected Papers", type="primary"):
                        # Store the query in session state to access it later
                        st.session_state.current_query = query
                        
                        # Initialize Ollama chatbot
                        bot = OllamaChatBot(
                            model=ollama_model,
                            end_point_url=ollama_endpoint,
                            temperature=temperature,
                            keep_history=False
                        )
                        
                        # Initialize reranker
                        reranker = LLMReranker(bot)
                        
                        # Collect chunks from all selected papers
                        all_chunks = []
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, paper in enumerate(st.session_state.selected_papers):
                            status_text.text(f"Processing {i+1}/{len(st.session_state.selected_papers)}: {paper.get('title')}")
                            
                            file_path = paper.get('full_path')
                            if file_path and os.path.exists(file_path):
                                paper_chunks = get_document_chunk_data(
                                    file_path, chunk_size, chunk_overlap, reranker, paper
                                )
                                all_chunks.extend(paper_chunks)
                            
                            progress_bar.progress((i + 1) / len(st.session_state.selected_papers))
                        
                        status_text.text(f"Ranking {len(all_chunks)} chunks against your query...")
                        
                        # Rank chunks
                        ranked_chunks = reranker.rank_chunks(
                            query, 
                            all_chunks,
                            progress_callback=lambda p: progress_bar.progress(p)
                        )
                        
                        # Filter by threshold
                        filtered_chunks = reranker.filter_chunks_by_score(ranked_chunks, min_score_threshold)
                        
                        # No limit on chunks - only filter by threshold score
                        
                        # Store in session state
                        st.session_state.ranked_chunks = filtered_chunks
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.success(f"Analysis complete! Found {len(filtered_chunks)} relevant chunks.")
                        
                # Display selected chunks in the prompt tab as well
                if st.session_state.selected_chunks:
                    st.subheader("Selected Chunks")
                    st.write(f"You have selected {len(st.session_state.selected_chunks)} chunks.")
                    
                    if st.button("Clear Selected Chunks"):
                        st.session_state.selected_chunks = []
                        st.success("Cleared all selected chunks!")
                    else:
                        # Display all selected chunks
                        for i, chunk in enumerate(st.session_state.selected_chunks):
                            with st.expander(f"Selected Chunk {i+1}: {chunk['title']}"):
                                st.markdown("**Source:** " + chunk['source'])
                                st.markdown("**Reference:** " + chunk['reference'])
                                st.markdown("### Content")
                                st.markdown(chunk['chunk'])
                                
                                # Option to remove this chunk
                                if st.button(f"Remove Chunk", key=f"remove_chunk_{i}"):
                                    st.session_state.selected_chunks.pop(i)
                                    st.experimental_rerun()
            
            # TAB 3: Chunk Selection
            with tab3:
                if st.session_state.ranked_chunks:
                    st.subheader("Relevant Document Chunks")
                    st.write(f"Displaying chunks with relevance score ≥ {min_score_threshold}")
                    
                    for i, chunk in enumerate(st.session_state.ranked_chunks):
                        with st.expander(f"Chunk {i+1}: {chunk['title']} (Score: {chunk['score']}/10)"):
                            st.markdown("**Source:** " + chunk['source'])
                            st.markdown("**Reference:** " + chunk['reference'])
                            
                            # Display the chunk text
                            st.markdown("### Content")
                            st.markdown(chunk['chunk'])
                            
                            # Add a button to select this chunk
                            if st.button(f"Select Chunk {i+1}", key=f"select_chunk_{i}"):
                                if chunk not in st.session_state.selected_chunks:
                                    st.session_state.selected_chunks.append(chunk)
                                    st.success(f"Added chunk {i+1} to selected chunks!")
                else:
                    st.info("No chunks available yet. Please select documents and run the analysis first.")
            
            # TAB 4: Output
            with tab4:
                st.subheader("Export Selected Chunks")
                
                if st.session_state.selected_chunks:
                    st.write(f"You have selected {len(st.session_state.selected_chunks)} chunks for export.")
                    
                    # Generate PDF button
                    if st.button("Download as PDF"):
                        from reportlab.lib.pagesizes import letter
                        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                        from reportlab.lib import colors
                        import io
                        
                        # Create PDF content
                        st.info("Preparing PDF for download...")
                        
                        # Create a BytesIO buffer to save the PDF to
                        buffer = io.BytesIO()
                        
                        # Create the PDF document
                        doc = SimpleDocTemplate(buffer, pagesize=letter, 
                                             title="Selected Document Chunks")
                        styles = getSampleStyleSheet()
                        
                        # Add custom styles - use unique names to avoid conflicts
                        styles.add(ParagraphStyle(name='ChunkTitle', 
                                               parent=styles['Heading1'],
                                               fontSize=16,
                                               spaceAfter=12))
                        styles.add(ParagraphStyle(name='ChunkSource', 
                                               parent=styles['Normal'],
                                               fontSize=10, 
                                               textColor=colors.gray))
                        styles.add(ParagraphStyle(name='ChunkContent', 
                                               parent=styles['Normal'],
                                               fontSize=11,
                                               leading=14,
                                               spaceAfter=12))
                        
                        # Create PDF content
                        elements = []
                        elements.append(Paragraph("Selected Document Chunks", styles['Heading1']))
                        elements.append(Spacer(1, 12))
                        
                        # Add each chunk to the PDF
                        for i, chunk in enumerate(st.session_state.selected_chunks):
                            # Add chunk title
                            chunk_title = f"Chunk {i+1}: {chunk['title']}"
                            elements.append(Paragraph(chunk_title, styles['Heading2']))
                            
                            # Add source and reference
                            source_text = f"Source: {chunk['source']}"
                            reference_text = f"Reference: {chunk['reference']}"
                            elements.append(Paragraph(source_text, styles['ChunkSource']))
                            elements.append(Paragraph(reference_text, styles['ChunkSource']))
                            elements.append(Spacer(1, 6))
                            
                            # Add chunk content - sanitize HTML tags that cause issues with ReportLab
                            content_text = chunk['chunk']
                            # Fix common HTML issues that cause problems with ReportLab
                            if content_text:
                                # Replace <br> tags with newlines
                                content_text = content_text.replace('<br>', '\n')
                                content_text = content_text.replace('<br/>', '\n')
                                content_text = content_text.replace('<br />', '\n')
                            elements.append(Paragraph(content_text, styles['ChunkContent']))
                            elements.append(Spacer(1, 12))
                        
                        # Build the PDF
                        doc.build(elements)
                        
                        # Get the value of the buffer
                        pdf_bytes = buffer.getvalue()
                        buffer.close()
                        
                        # Provide the PDF for download
                        st.download_button(
                            label="Download PDF",
                            data=pdf_bytes,
                            file_name="selected_chunks.pdf",
                            mime="application/pdf"
                        )
                else:
                    st.info("No chunks have been selected. Please select chunks from the 'Chunk Selection' tab first.")
    else:
        st.warning("Please enter a valid folder path.")

if __name__ == "__main__":
    main()