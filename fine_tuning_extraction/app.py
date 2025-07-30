import streamlit as st
import os
import json
import PyPDF2
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
import time
from typing import List, Dict, Tuple
import traceback

# Load environment variables
load_dotenv()

class QAExtractor:
    def __init__(self):
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.google_model = os.getenv('GOOGLE_MODEL', 'gemini-2.5-flash')
        self.source_dir = os.getenv('SOURCE_DIR', './docs')
        self.output_dir = os.getenv('OUTPUT_DIR', './output')
        self.prompt_dir = os.getenv('PROMPT_DIR', './extraction_prompts')
        
        # Configure Google AI
        if self.google_api_key:
            genai.configure(api_key=self.google_api_key)
            self.model = genai.GenerativeModel(self.google_model)
        else:
            st.error("GOOGLE_API_KEY not found in environment variables!")
            self.model = None
    
    def load_prompt_template(self, prompt_path: str) -> str:
        """Load prompt template from file."""
        try:
            with open(prompt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            st.error(f"Error loading prompt from {prompt_path}: {str(e)}")
            return ""
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text content from PDF or Markdown file."""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.md':
            return self.extract_text_from_markdown(str(file_path))
        elif file_path.suffix.lower() == '.pdf':
            return self.extract_text_from_pdf(str(file_path))
        else:
            print(f"DEBUG: Unsupported file format: {file_path.suffix}")
            return ""
    
    def extract_text_from_markdown(self, md_path: str) -> str:
        """Extract text content from Markdown file."""
        try:
            with open(md_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            if len(text.strip()) < 100:
                print(f"DEBUG: Markdown file appears to be too short (only {len(text.strip())} characters)")
                return ""
            
            print(f"DEBUG: Successfully loaded markdown file with {len(text)} characters")
            return text
            
        except Exception as e:
            st.error(f"Error reading markdown file {md_path}: {str(e)}")
            print(f"DEBUG: Failed to read markdown file: {e}")
            return ""
            
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file."""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    text += page_text + "\n"
            
            # Check if we got meaningful text (not just whitespace/minimal characters)
            if len(text.strip()) < 100:  # Threshold for "empty" text
                print(f"DEBUG: PDF appears to be image-based (only {len(text.strip())} characters extracted)")
                print("DEBUG: Attempting OCR with Google Vision...")
                
                # Try OCR conversion
                ocr_text = self.perform_ocr_on_pdf(pdf_path)
                if ocr_text and len(ocr_text.strip()) > 100:
                    return ocr_text
                else:
                    print("DEBUG: OCR failed or returned insufficient text, skipping document")
                    return ""  # Return empty string to skip this document
            
            return text
            
        except Exception as e:
            st.error(f"Error extracting text from {pdf_path}: {str(e)}")
            print(f"DEBUG: PDF extraction failed, attempting OCR: {e}")
            
            # Try OCR as fallback
            try:
                ocr_text = self.perform_ocr_on_pdf(pdf_path)
                if ocr_text and len(ocr_text.strip()) > 100:
                    return ocr_text
                else:
                    print("DEBUG: OCR also failed or returned insufficient text, skipping document")
                    return ""  # Return empty string to skip this document
            except Exception as ocr_e:
                print(f"DEBUG: OCR also failed: {ocr_e}")
                return ""  # Return empty string to skip this document
    
    def perform_ocr_on_pdf(self, pdf_path: str) -> str:
        """Perform OCR on PDF using Google Gemini Vision and save as markdown."""
        try:
            import fitz  # PyMuPDF
            from PIL import Image
            import io
            import tempfile
            
            # Convert PDF pages to images
            pdf_document = fitz.open(pdf_path)
            all_text = ""
            
            for page_num in range(len(pdf_document)):
                print(f"DEBUG: Processing page {page_num + 1}/{len(pdf_document)} with OCR")
                
                try:
                    # Get page as image
                    page = pdf_document[page_num]
                    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))
                    
                    # Use Gemini Vision for OCR
                    prompt = """Extract ALL text from this image and convert it to clean markdown format. 

IMPORTANT INSTRUCTIONS:
- Preserve mathematical equations as LaTeX (use $...$ for inline, $...$ for display)
- Maintain document structure with proper headers (# ## ###)
- Preserve tables using markdown table syntax
- Include figure captions and references
- Keep scientific notation and formulas intact
- Output ONLY the markdown text, no explanations

If this appears to be a scientific paper, maintain academic formatting."""

                    # Create a temporary file for the image
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                        print(f"DEBUG: Saving temporary image to {tmp_path}")
                        
                    # Save image to the temporary file
                    image.save(tmp_path, format="PNG")
                    
                    try:
                        # Upload the temporary file to Gemini
                        print(f"DEBUG: Uploading file to Gemini API...")
                        uploaded_file = genai.upload_file(path=tmp_path, mime_type="image/png")
                        
                        # Wait for file to be processed
                        import time
                        time.sleep(2)  # Give time for upload to complete
                        
                        # Generate content
                        print(f"DEBUG: Generating content with Gemini Vision...")
                        response = self.model.generate_content([prompt, uploaded_file])
                        
                        if response and response.text:
                            all_text += f"\n\n--- Page {page_num + 1} ---\n\n"
                            all_text += response.text
                            print(f"DEBUG: Successfully extracted text from page {page_num + 1}")
                        else:
                            print(f"DEBUG: No text returned from Gemini for page {page_num + 1}")
                        
                        # Clean up uploaded file from Gemini (if possible)
                        try:
                            genai.delete_file(uploaded_file.name)
                        except:
                            pass  # File cleanup is not critical
                            
                    except Exception as api_error:
                        print(f"DEBUG: Gemini API error for page {page_num + 1}: {api_error}")
                        
                    finally:
                        # Always clean up local temp file
                        try:
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                                print(f"DEBUG: Cleaned up temporary file {tmp_path}")
                        except Exception as cleanup_error:
                            print(f"DEBUG: Failed to cleanup temp file: {cleanup_error}")
                    
                    # Small delay to avoid rate limiting
                    time.sleep(1)
                    
                except Exception as page_error:
                    print(f"DEBUG: Error processing page {page_num + 1}: {page_error}")
                    continue
            
            pdf_document.close()
            
            # Save as markdown file if we got any text
            if all_text.strip():
                try:
                    markdown_path = Path(pdf_path).with_suffix('.md')
                    with open(markdown_path, 'w', encoding='utf-8') as f:
                        f.write(f"# OCR Extraction from {Path(pdf_path).name}\n\n")
                        f.write(all_text)
                    
                    print(f"DEBUG: Saved OCR markdown to {markdown_path}")
                except Exception as save_error:
                    print(f"DEBUG: Failed to save markdown file: {save_error}")
                
                return all_text
            else:
                print("DEBUG: No text was extracted from any pages")
                return ""
            
        except ImportError as e:
            print(f"DEBUG: Missing dependencies for OCR: {e}")
            print("DEBUG: Install with: pip install PyMuPDF Pillow")
            return ""
        except Exception as e:
            print(f"DEBUG: OCR failed with error: {e}")
            import traceback
            print(f"DEBUG: Full traceback: {traceback.format_exc()}")
            return ""
    
    def extract_markdown_content(self, text: str, type: str = "json") -> str:
        """Extract content from markdown code fences."""
        start = f"""```{type}"""
        end = """```"""

        start_idx = text.find(start)
        end_idx = text.rfind(end)

        if start_idx >= 0 and end_idx >= 0:
            start_idx += len(type) + 3
            end_idx -= 1
            return (text[start_idx:end_idx]).strip()

        return text.strip()
    
    def extract_raw_json(self, text: str) -> str:
        """Extract raw JSON from text."""
        # First try to find object notation
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            return text[json_start:json_end]
        
        # If no object found, try array notation
        array_start = text.find('[')
        array_end = text.rfind(']') + 1
        
        if array_start != -1 and array_end > array_start:
            return text[array_start:array_end]
        
        return ""
    
    def clean_json_quotes(self, json_str: str) -> str:
        """Clean unescaped quotes within JSON string values."""
        try:
            # More robust approach to fix quote issues
            # Handle both backslash escapes and quote escapes properly
            
            lines = json_str.split('\n')
            fixed_lines = []
            
            for line in lines:
                # Simple approach: if line contains answer field, be more careful
                if '"answer":' in line:
                    # Find the value part of the answer
                    answer_start = line.find('"answer":')
                    if answer_start != -1:
                        # Find the opening quote of the value
                        quote_start = line.find('"', answer_start + len('"answer":'))
                        if quote_start != -1:
                            prefix = line[:quote_start+1]
                            remainder = line[quote_start+1:]
                            
                            # Find the closing quote (but be careful about escaped quotes)
                            quote_end = -1
                            i = 0
                            while i < len(remainder):
                                if remainder[i] == '"' and (i == 0 or remainder[i-1] != '\\'):
                                    quote_end = i
                                    break
                                i += 1
                            
                            if quote_end != -1:
                                content = remainder[:quote_end]
                                suffix = remainder[quote_end:]
                                
                                # Fix the content by properly escaping
                                # First protect already escaped content
                                content = content.replace('\\\\', '|||DOUBLE_BACKSLASH|||')
                                content = content.replace('\\"', '|||ESCAPED_QUOTE|||')
                                
                                # Now escape unescaped quotes
                                content = content.replace('"', '\\"')
                                
                                # Restore protected content
                                content = content.replace('|||ESCAPED_QUOTE|||', '\\"')
                                content = content.replace('|||DOUBLE_BACKSLASH|||', '\\\\')
                                
                                line = prefix + content + suffix
                
                fixed_lines.append(line)
            
            return '\n'.join(fixed_lines)
            
        except Exception as e:
            print(f"DEBUG: Quote cleaning failed: {e}")
            return json_str
    
    def fix_json_structure(self, json_str: str) -> str:
        """Fix JSON structure if it's missing the 'pairs' wrapper."""
        try:
            # Determine the JSON type by looking at the first non-whitespace character
            stripped = json_str.strip()
            
            if stripped.startswith('['):
                # It's an array - parse it and wrap in 'pairs' structure
                data = json.loads(json_str)
                if isinstance(data, list):
                    wrapped_data = {"pairs": data}
                    return json.dumps(wrapped_data, ensure_ascii=False)
            elif stripped.startswith('{'):
                # Check if it looks like multiple objects separated by commas instead of an array
                if stripped.count('{\n    "question"') > 1:
                    # This looks like multiple objects that should be in an array
                    print("DEBUG: Detected multiple objects that should be wrapped in array")
                    # Try to fix by wrapping in array brackets
                    fixed_json = '[' + json_str + ']'
                    try:
                        data = json.loads(fixed_json)
                        if isinstance(data, list):
                            wrapped_data = {"pairs": data}
                            return json.dumps(wrapped_data, ensure_ascii=False)
                    except:
                        pass
                
                # It's an object - check if it has the right structure
                data = json.loads(json_str)
                if isinstance(data, dict):
                    if "pairs" in data and isinstance(data["pairs"], list):
                        # Already correct structure
                        return json_str
                    else:
                        # Single object - wrap it in pairs array
                        if "question" in data and "answer" in data:
                            wrapped_data = {"pairs": [data]}
                            return json.dumps(wrapped_data, ensure_ascii=False)
                        return json_str
            
            # If it's something else, return as-is
            return json_str
            
        except json.JSONDecodeError:
            # If JSON parsing fails, return as-is for further processing
            return json_str
    
    def strip_markdown_fencing(self, text: str) -> str:
        """Remove markdown code fences and try to extract JSON content."""
        # Remove markdown fencing
        cleaned = text
        
        # Remove ```json and ``` markers
        cleaned = cleaned.replace('```json', '')
        cleaned = cleaned.replace('```', '')
        
        # Try to find JSON content
        json_start = cleaned.find('{')
        json_end = cleaned.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            return cleaned[json_start:json_end].strip()
        
        # If no object found, try array
        array_start = cleaned.find('[')
        array_end = cleaned.rfind(']') + 1
        
        if array_start != -1 and array_end > array_start:
            return cleaned[array_start:array_end].strip()
        
        return cleaned.strip()
    
    def save_error_response(self, pdf_name: str, prompt_name: str, output_dir: Path, paper_text: str, prompt_template: str) -> bool:
        """Save the raw AI response when JSON parsing fails for debugging."""
        try:
            # Make one final attempt to get the raw response
            full_prompt = f"{prompt_template}\n\nPaper Content:\n{paper_text}"
            response = self.model.generate_content(full_prompt)
            raw_response = response.text
            
            # Try to strip markdown fencing and clean up
            cleaned_response = self.strip_markdown_fencing(raw_response)
            
            # Create error file with raw content for inspection
            error_file = output_dir / f"{prompt_name}.error.json"
            
            error_data = {
                "pdf_name": pdf_name,
                "prompt_name": prompt_name,
                "error_type": "JSON parsing failed after all attempts",
                "raw_response": raw_response,
                "cleaned_response": cleaned_response,
                "response_length": len(raw_response),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)
            
            print(f"DEBUG: Saved error response to {error_file}")
            return True
            
        except Exception as e:
            print(f"DEBUG: Failed to save error response: {e}")
            return False
    
    def extract_qa_pairs_with_fallback(self, paper_text: str, prompt_template: str, max_retries: int = 5) -> Dict:
        """Extract Q&A pairs with multiple parsing strategies."""
        if not self.model:
            return None
        
        full_prompt = f"{prompt_template}\n\nPaper Content:\n{paper_text}"
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(full_prompt)
                response_text = response.text
                
                # Try multiple extraction strategies
                strategies = [
                    ("raw_json", self.extract_raw_json(response_text)),
                    ("markdown", self.extract_markdown_content(response_text, "json")),
                    ("cleaned_response", response_text.strip())
                ]
                
                for strategy_name, json_str in strategies:
                    if not json_str or len(json_str) < 10:
                        continue
                        
                    print(f"\n=== DEBUG: Trying strategy: {strategy_name} ===")
                    print(f"Length: {len(json_str)} characters")
                    
                    # Log large responses but don't truncate
                    if len(json_str) > 30000:
                        print(f"DEBUG: Large response detected ({len(json_str)} chars) - processing without truncation")
                        # Count pairs for information
                        pair_count = json_str.count('"question"')
                        print(f"DEBUG: Response contains approximately {pair_count} Q&A pairs")
                    
                    # Clean quotes first
                    cleaned_json = self.clean_json_quotes(json_str)
                    if cleaned_json != json_str:
                        print(f"DEBUG: Cleaned unescaped quotes in JSON")
                    
                    # Fix JSON structure if needed
                    fixed_json = self.fix_json_structure(cleaned_json)
                    if fixed_json != cleaned_json:
                        print(f"DEBUG: JSON structure was fixed - converted array to object with 'pairs' key")
                    
                    try:
                        result = json.loads(fixed_json)
                        
                        # Handle both structures: {"pairs": [...]} and direct array [...]
                        pairs_data = None
                        if isinstance(result, dict) and "pairs" in result and isinstance(result["pairs"], list):
                            pairs_data = result["pairs"]
                            print(f"SUCCESS with strategy: {strategy_name} (object structure) - {len(pairs_data)} pairs extracted")
                        elif isinstance(result, list):
                            # Direct array structure - wrap it
                            pairs_data = result
                            result = {"pairs": result}
                            print(f"SUCCESS with strategy: {strategy_name} (array structure - auto-wrapped) - {len(pairs_data)} pairs extracted")
                        
                        if pairs_data is not None:
                            return result
                    except json.JSONDecodeError as e:
                        print(f"Strategy {strategy_name} failed: {e}")
                        print(f"\n=== RAW JSON OUTPUT THAT FAILED ({strategy_name}) ===")
                        print(json_str[:1000] + "..." if len(json_str) > 1000 else json_str)  # Limit console output
                        print("=" * 80)
                        continue
                
                # If all strategies fail for this attempt, continue to next attempt
                time.sleep(1)
                
            except Exception as e:
                print(f"API error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return None
        
        return None
    
    def process_papers(self, source_dir: str, prompt_dir: str, output_dir: str, 
                      progress_bar, status_text) -> Tuple[int, int, List[str]]:
        """Process all papers with all prompts."""
        # Get all supported files (PDF and Markdown) and prompt files
        supported_files = list(Path(source_dir).glob("*.pdf")) + list(Path(source_dir).glob("*.md"))
        prompt_files = list(Path(prompt_dir).glob("*.txt"))
        
        if not supported_files:
            st.error(f"No PDF or Markdown files found in {source_dir}")
            return 0, 0, []
        
        if not prompt_files:
            st.error(f"No prompt files found in {prompt_dir}")
            return 0, 0, []
        
        total_tasks = len(supported_files) * len(prompt_files)
        completed_tasks = 0
        successful_extractions = 0
        errors = []
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for file_path in supported_files:
            file_name = file_path.stem
            file_output_dir = Path(output_dir) / file_name
            file_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract text from file (PDF or Markdown)
            status_text.text(f"Extracting text from: {file_path.name}")
            paper_text = self.extract_text_from_file(str(file_path))
            
            if not paper_text:
                error_msg = f"Failed to extract text from {file_path.name}"
                errors.append(error_msg)
                print(error_msg)
                completed_tasks += len(prompt_files)  # Skip all prompts for this file
                progress_bar.progress(completed_tasks / total_tasks)
                continue
            
            for prompt_file in prompt_files:
                prompt_name = prompt_file.stem
                status_text.text(f"Processing: {file_path.name} with {prompt_file.name}")
                
                # Load prompt template
                prompt_template = self.load_prompt_template(str(prompt_file))
                
                if not prompt_template:
                    error_msg = f"Failed to load prompt from {prompt_file.name}"
                    errors.append(error_msg)
                    print(error_msg)
                    completed_tasks += 1
                    progress_bar.progress(completed_tasks / total_tasks)
                    continue
                
                # Check if output file already exists
                output_file = file_output_dir / f"{prompt_name}.json"
                if output_file.exists():
                    print(f"SKIPPING: {file_path.name} | {prompt_file.name} | Output file already exists")
                    successful_extractions += 1  # Count as successful since file exists
                    completed_tasks += 1
                    progress_bar.progress(completed_tasks / total_tasks)
                    continue
                
                # Extract Q&A pairs
                result = self.extract_qa_pairs_with_fallback(paper_text, prompt_template)
                
                if result is None:
                    # Try to save the raw response as an error file for inspection
                    error_file_saved = self.save_error_response(file_path.name, prompt_name, file_output_dir, paper_text, prompt_template)
                    
                    error_msg = f"{file_path.name} | {prompt_file.name} | All extraction attempts failed"
                    if error_file_saved:
                        error_msg += f" - Raw response saved as {prompt_name}.error.json"
                    errors.append(error_msg)
                    print(error_msg)
                elif "error" in result:
                    error_msg = f"{file_path.name} | {prompt_file.name} | {result['error']}"
                    errors.append(error_msg)
                    print(error_msg)
                else:
                    # Save results to JSON file
                    try:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(result, f, indent=2, ensure_ascii=False)
                        successful_extractions += 1
                    except Exception as e:
                        error_msg = f"{file_path.name} | {prompt_file.name} | Failed to save JSON: {str(e)}"
                        errors.append(error_msg)
                        print(error_msg)
                
                completed_tasks += 1
                progress_bar.progress(completed_tasks / total_tasks)
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
        
        return successful_extractions, len(errors), errors


def main():
    st.set_page_config(
        page_title="Scientific Paper Q&A Extractor",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Scientific Paper Q&A Extractor")
    st.markdown("Extract question/answer pairs from scientific papers (PDF/Markdown) for LLM fine-tuning")
    
    # Initialize extractor
    extractor = QAExtractor()
    
    # Create input fields with defaults from environment
    col1, col2, col3 = st.columns(3)
    
    with col1:
        source_dir = st.text_input(
            "Papers Directory (PDF/MD)",
            value=extractor.source_dir,
            help="Directory containing PDF and Markdown papers to process"
        )
    
    with col2:
        prompt_dir = st.text_input(
            "Prompt Templates Directory",
            value=extractor.prompt_dir,
            help="Directory containing prompt template files"
        )
    
    with col3:
        output_dir = st.text_input(
            "Output Directory",
            value=extractor.output_dir,
            help="Directory to save extracted JSON files"
        )
    
    # Display current configuration
    st.subheader("Current Configuration")
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.info(f"**Google API Key:** {'✅ Configured' if extractor.google_api_key else '❌ Missing'}")
        st.info(f"**Model:** {extractor.google_model}")
    
    with config_col2:
        # Count files
        pdf_count = len(list(Path(source_dir).glob("*.pdf"))) if Path(source_dir).exists() else 0
        md_count = len(list(Path(source_dir).glob("*.md"))) if Path(source_dir).exists() else 0
        total_files = pdf_count + md_count
        prompt_count = len(list(Path(prompt_dir).glob("*.txt"))) if Path(prompt_dir).exists() else 0
        
        st.info(f"**PDF Files Found:** {pdf_count}")
        st.info(f"**Markdown Files Found:** {md_count}")
        st.info(f"**Total Files:** {total_files}")
        st.info(f"**Prompt Files Found:** {prompt_count}")
        st.info(f"**Total Tasks:** {total_files * prompt_count}")
    
    # Process button
    if st.button("🚀 Process Papers", type="primary", disabled=not extractor.model):
        if not extractor.model:
            st.error("Cannot process: Google API key not configured!")
            return
        
        if total_files == 0:
            st.error("No PDF or Markdown files found in the specified directory!")
            return
        
        if prompt_count == 0:
            st.error("No prompt template files found in the specified directory!")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        
        # Process papers
        successful, error_count, errors = extractor.process_papers(
            source_dir, prompt_dir, output_dir, progress_bar, status_text
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Display results
        status_text.text("Processing complete!")
        
        st.subheader("📊 Processing Results")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.metric("✅ Successful Extractions", successful)
        
        with result_col2:
            st.metric("❌ Errors", error_count)
        
        with result_col3:
            st.metric("⏱️ Processing Time", f"{processing_time:.1f}s")
        
        # Show errors if any
        if errors:
            st.subheader("❌ Errors Encountered")
            for error in errors:
                st.error(error)
        
        if successful > 0:
            st.success(f"Successfully extracted Q&A pairs from {successful} paper-prompt combinations!")
            st.info(f"Output files saved to: {output_dir}")


if __name__ == "__main__":
    main()