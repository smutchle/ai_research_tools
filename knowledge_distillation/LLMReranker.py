import time
import json
from typing import List, Dict, Any, Tuple, Optional
import re

class LLMReranker:
    """
    Class for reranking document chunks using an LLM.
    
    The reranker uses a provided OllamaChatBot instance to score chunks
    on a scale of 1-10 based on their relevance to a query.
    """
    
    def __init__(self, bot, batch_size=5, max_retries=3, backoff_factor=1.5):
        """
        Initialize the LLMReranker.
        
        Args:
            bot: OllamaChatBot instance for ranking
            batch_size: Number of chunks to rank in a single batch
            max_retries: Maximum number of retry attempts on failure
            backoff_factor: Exponential backoff factor for retries
        """
        self.bot = bot
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    def _extract_scores(self, response: str) -> List[int]:
        """
        Extract numeric scores from the LLM response.
        
        Args:
            response: Text response from LLM containing scores
            
        Returns:
            List of integer scores extracted from the response
        """
        # First try to extract JSON
        try:
            # Find JSON-like structures in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                # Extract scores from the JSON data
                if isinstance(data, dict) and "scores" in data:
                    return [int(score) if 0 <= int(score) <= 10 else max(0, min(10, int(score))) 
                            for score in data["scores"]]
                elif isinstance(data, dict) and all(isinstance(k, str) and k.isdigit() for k in data.keys()):
                    # Handle cases where scores are keys in the JSON
                    return [int(score) for score in data.keys()]
                elif isinstance(data, list) and all(isinstance(item, int) for item in data):
                    # Handle cases where scores are a simple list
                    return [max(0, min(10, score)) for score in data]
        except (json.JSONDecodeError, ValueError):
            pass
        
        # If JSON extraction fails, try to find numbers in the text
        # Look for patterns like "Score: 7" or "Chunk 1: 8"
        scores = []
        score_pattern = r'(?:chunk|document|text|passage|score|rating|relevance)?\s*(?:\d+\s*:)?\s*(\d+)(?:\s*\/\s*10)?'
        matches = re.finditer(score_pattern, response.lower(), re.IGNORECASE)
        
        for match in matches:
            try:
                score = int(match.group(1))
                if 0 <= score <= 10:  # Only accept scores in the valid range
                    scores.append(score)
            except (ValueError, IndexError):
                continue
        
        return scores
    
    def _create_ranking_prompt(self, query: str, chunks: List[str]) -> str:
        """
        Create a prompt for the LLM to rank chunks.
        
        Args:
            query: User query to compare chunks against
            chunks: List of text chunks to be ranked
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
You are a document relevance ranker. Your job is to rank document chunks based on their relevance to a query.

Query: {query}

Please evaluate each of the following document chunks and assign a relevance score from 1 to 10, where:
- 0-2: Not relevant or only mentions keywords without addressing the query
- 3-5: Somewhat relevant but doesn't directly answer the query
- 6-8: Relevant and partially answers the query
- 9-10: Highly relevant and directly answers the query

Return ONLY a JSON object with a "scores" array containing the integer scores for each chunk in order.
Example: {{"scores": [7, 2, 9, 4, 6]}}

DO NOT include explanations or text outside of the JSON object.

Document chunks to evaluate:
"""
        
        for i, chunk in enumerate(chunks, 1):
            prompt += f"\n--- Chunk {i} ---\n{chunk}\n"
        
        prompt += "\nOutput only the JSON with scores like {\"scores\": [7, 2, 9, 4, 6]}"
        return prompt
    
    def rank_chunks(self, query: str, chunk_data: List[Dict[str, Any]], 
                  progress_callback=None) -> List[Dict[str, Any]]:
        """
        Rank document chunks based on their relevance to the query.
        
        Args:
            query: User query to compare chunks against
            chunk_data: List of dictionaries containing chunk info including:
                - 'chunk': The text content of the chunk
                - 'source': Source document info
                - Any other metadata fields
            progress_callback: Optional callback function to update progress
                
        Returns:
            List of dictionaries with original chunk data plus a 'score' field
        """
        # Track progress
        total_chunks = len(chunk_data)
        processed_chunks = 0
        
        # Create batches of chunks
        batches = [chunk_data[i:i + self.batch_size] 
                  for i in range(0, len(chunk_data), self.batch_size)]
        
        ranked_chunks = []
        
        for batch_idx, batch in enumerate(batches):
            # Extract just the text content for ranking
            batch_texts = [item['chunk'] for item in batch]
            
            # Create the ranking prompt
            prompt = self._create_ranking_prompt(query, batch_texts)
            
            # Implement retry logic
            attempts = 0
            backoff_time = 1  # Start with 1 second
            scores = None
            
            while attempts < self.max_retries:
                try:
                    # Get ranking from LLM
                    response = self.bot.complete(prompt)
                    scores = self._extract_scores(response)
                    
                    # Validate extracted scores
                    if len(scores) == len(batch):
                        break
                    else:
                        # Wrong number of scores, retry
                        attempts += 1
                        time.sleep(backoff_time)
                        backoff_time *= self.backoff_factor
                except Exception as e:
                    attempts += 1
                    time.sleep(backoff_time)
                    backoff_time *= self.backoff_factor
            
            # If all retries failed, assign default scores
            if scores is None or len(scores) != len(batch):
                scores = [5] * len(batch)  # Default middle score of 5
            
            # Add scores to chunk data
            for i, chunk_item in enumerate(batch):
                chunk_with_score = chunk_item.copy()
                chunk_with_score['score'] = scores[i]
                ranked_chunks.append(chunk_with_score)
            
            # Update progress
            processed_chunks += len(batch)
            if progress_callback:
                progress_callback(processed_chunks / total_chunks)
        
        # Sort chunks by score (descending)
        ranked_chunks.sort(key=lambda x: x['score'], reverse=True)
        
        return ranked_chunks
    
    def filter_chunks_by_score(self, ranked_chunks: List[Dict[str, Any]], 
                              min_score: int = 8) -> List[Dict[str, Any]]:
        """
        Filter chunks by minimum score threshold.
        
        Args:
            ranked_chunks: List of dictionaries with chunk data and scores
            min_score: Minimum score threshold (inclusive)
            
        Returns:
            Filtered list of chunk dictionaries
        """
        return [chunk for chunk in ranked_chunks if chunk['score'] >= min_score]
    
    def chunk_markdown(self, text: str, chunk_size: int) -> List[str]:
        """
        Chunk markdown text based on natural section breaks.
        
        Args:
            text: Markdown text to chunk
            
        Returns:
            List of markdown chunks
        """
        # Split by headers
        header_pattern = r'^#{1,6}\s+.+$'
        chunks = []
        
        # Split by markdown headers
        lines = text.split('\n')
        current_chunk = []
        
        for line in lines:
            if re.match(header_pattern, line) and current_chunk:
                # Start a new chunk at headers
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
            else:
                current_chunk.append(line)
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        # If there were no headers, or the chunks are too large, try other delimiters
        if len(chunks) <= 1 and len(text) > chunk_size * 1.5:
            chunks = []
            
            # Try splitting by double newlines (paragraphs)
            paragraphs = re.split(r'\n\s*\n', text)
            
            current_chunk = []
            current_length = 0
            
            for para in paragraphs:
                para_length = len(para)
                
                if current_length + para_length > chunk_size and current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [para]
                    current_length = para_length
                else:
                    current_chunk.append(para)
                    current_length += para_length
            
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def chunk_pdf_text(self, text: str, chunk_size: int, 
                      chunk_overlap: int) -> List[str]:
        """
        Chunk PDF text based on size and overlap.
        
        Args:
            text: Text extracted from PDF
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        # First try to split by double newlines to preserve paragraph structure
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            # If adding this paragraph exceeds the chunk size
            if current_length + para_length > chunk_size and current_chunk:
                # Save the current chunk
                chunks.append('\n\n'.join(current_chunk))
                
                # Start a new chunk with overlap
                overlap_reached = False
                overlap_length = 0
                new_chunk = []
                
                # Add paragraphs from the end until we reach desired overlap
                for p in reversed(current_chunk):
                    new_chunk.insert(0, p)
                    overlap_length += len(p)
                    if overlap_length >= chunk_overlap:
                        overlap_reached = True
                        break
                
                if overlap_reached:
                    current_chunk = new_chunk
                else:
                    current_chunk = []
                
                # Add the current paragraph
                current_chunk.append(para)
                current_length = sum(len(p) for p in current_chunk)
            else:
                # Add paragraph to current chunk
                current_chunk.append(para)
                current_length += para_length
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        # If the above logic didn't create multiple chunks, fall back to character-based chunking
        if len(chunks) <= 1 and len(text) > chunk_size:
            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunks.append(text[i:i + chunk_size])
        
        return chunks