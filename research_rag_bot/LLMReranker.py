import json
from typing import List, Dict, Any, Tuple, Optional
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

class LLMReranker:
    """
    LLM-based reranker for Retrieval Augmented Generation (RAG).
    Uses an Ollama LLM to rerank chunks based on relevance to a query.
    """
    
    def __init__(
        self, 
        llm_bot,
        prompt_template: str = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        initial_chunks: int = 50,
        top_n: int = 5,
        batch_size: int = 5
    ):
        """
        Initialize the LLM Reranker.
        
        Args:
            llm_bot: An instance of OllamaChatBot for generating relevance scores
            prompt_template: Template for reranking prompt (if None, default will be used)
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between chunks to maintain context
            initial_chunks: Number of initial chunks to consider from retriever
            top_n: Number of top chunks to keep after reranking
            batch_size: Number of chunks to process in parallel
        """
        self.llm_bot = llm_bot
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.initial_chunks = initial_chunks
        self.top_n = top_n
        self.batch_size = batch_size
        
        # Default prompt template if none provided
        if prompt_template is None:
            self.prompt_template = """You are evaluating how useful text is as context to answering a specific question.

Query: {query}

Text passage: {passage}

Rate this passage on a scale from 1 to 10 based on directly useful it is to specifically answering the query.
A score of 0 means not useful, and 10 means it will aide in answering the question directly.

Reply with only a number between 0 and 10."""
        else:
            self.prompt_template = prompt_template
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks of specified size with overlap.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # Avoid splitting words
            if end < text_length:
                # Find the last space within the chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunks.append(text[start:end])
            start = end - self.chunk_overlap if end - self.chunk_overlap > start else end
        
        return chunks
    
    def _get_relevance_score(self, query: str, passage: str) -> Tuple[float, str]:
        """
        Get relevance score for a passage using LLM.
        
        Args:
            query: User query
            passage: Text passage to evaluate
            
        Returns:
            Tuple of (score, reasoning)
        """
        prompt = self.prompt_template.format(query=query, passage=passage)
        
        try:
            # First, try using the completeAsJSON method
            json_response = self.llm_bot.completeAsJSON(prompt)
            
            if json_response:
                try:
                    response_data = json.loads(json_response)
                    
                    # Handle case where response is a list
                    if isinstance(response_data, list) and len(response_data) > 0:
                        response_data = response_data[0]
                    
                    # Handle case where response is a dictionary
                    if isinstance(response_data, dict):
                        score = float(response_data.get("score", 0))
                        reasoning = response_data.get("reasoning", "")
                        return (score, reasoning)
                    else:
                        print(f"Unexpected response format: {type(response_data)}")
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error parsing JSON response, falling back to regular completion: {e}")
                    # Fall through to regular completion
            
            # If JSON parsing failed, try regular completion and extract JSON
            regular_response = self.llm_bot.complete(prompt)
            
            # Try to extract JSON from response text
            match = re.search(r'```json\s*(.*?)\s*```', regular_response, re.DOTALL)
            if match:
                json_str = match.group(1)
                try:
                    response_data = json.loads(json_str)
                    
                    # Handle case where response is a list
                    if isinstance(response_data, list) and len(response_data) > 0:
                        response_data = response_data[0]
                    
                    # Handle case where response is a dictionary
                    if isinstance(response_data, dict):
                        score = float(response_data.get("score", 0))
                        reasoning = response_data.get("reasoning", "")
                        return (score, reasoning)
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error parsing extracted JSON: {e}")
                    
            # Try to directly extract score with regex
            score_match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', regular_response)
            if score_match:
                try:
                    score = float(score_match.group(1))
                    # Try to extract reasoning too
                    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', regular_response)
                    reasoning = reasoning_match.group(1) if reasoning_match else "No reasoning provided"
                    return (score, reasoning)
                except ValueError:
                    print("Error converting score to float")
            
            # Final fallback: try to find just a number in the response
            number_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', regular_response)
            if not number_match:
                number_match = re.search(r'score.*?(\d+(?:\.\d+)?)', regular_response, re.IGNORECASE)
            if number_match:
                try:
                    score = float(number_match.group(1))
                    return (score, "Score extracted from text response")
                except ValueError:
                    print("Error converting extracted number to float")
        
        except Exception as e:
            print(f"Error in relevance scoring: {e}")
        
        return (0, "Failed to get valid response")
    
    def _process_batch(self, query: str, passages: List[Tuple[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of passages in parallel.
        
        Args:
            query: User query
            passages: List of (passage, metadata) tuples
            
        Returns:
            List of scored passages with metadata
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=min(self.batch_size, len(passages))) as executor:
            future_to_passage = {
                executor.submit(self._get_relevance_score, query, passage): (passage, metadata)
                for passage, metadata in passages
            }
            
            for future in as_completed(future_to_passage):
                passage, metadata = future_to_passage[future]
                try:
                    score, reasoning = future.result()
                    results.append({
                        "passage": passage,
                        "metadata": metadata,
                        "score": score,
                        "reasoning": reasoning
                    })
                except Exception as e:
                    print(f"Error processing passage: {e}")
                    results.append({
                        "passage": passage,
                        "metadata": metadata,
                        "score": 0,
                        "reasoning": f"Error: {str(e)}"
                    })
        
        return results
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], 
              metadata_key: str = "metadata") -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: User query
            documents: List of document dictionaries with text and metadata
            metadata_key: Key in document dict that contains metadata
            
        Returns:
            List of top N ranked chunks with scores and metadata
        """
        # Extract text and metadata from documents
        all_chunks = []
        
        for doc in documents[:self.initial_chunks]:  # Limit to initial_chunks
            text = doc.get("text", "")
            metadata = doc.get(metadata_key, {})
            
            # Chunk the document
            chunks = self._chunk_text(text)
            
            for chunk in chunks:
                all_chunks.append((chunk, metadata))
        
        # Rerank chunks in batches
        ranked_chunks = []
        
        for i in range(0, len(all_chunks), self.batch_size):
            batch = all_chunks[i:i + self.batch_size]
            batch_results = self._process_batch(query, batch)
            ranked_chunks.extend(batch_results)
        
        # Sort by score in descending order
        ranked_chunks.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top N results
        return ranked_chunks[:self.top_n]
    
    def clear_llm_history(self):
        """
        Clear the LLM chat history to prevent context pollution.
        """
        self.llm_bot.clear_history()
        
    def set_prompt_template(self, prompt_template: str):
        """
        Set a new prompt template for reranking.
        
        Args:
            prompt_template: New template string with {query} and {passage} placeholders
        """
        self.prompt_template = prompt_template


# Example usage:
if __name__ == "__main__":
    # Import OllamaChatBot from the correct module
    from OllamaChatBot import OllamaChatBot
    
    # Create an instance of OllamaChatBot
    llm_bot = OllamaChatBot(model="phi4:latest", end_point_url="http://localhost:11434", temperature=0.0)
    
    # Set a very simple prompt to test
    simple_prompt = """Rate how relevant this passage is to the query on a scale of 0-10.
Query: {query}
Passage: {passage}
Score (0-10): """
    
    # Initialize reranker with the simple prompt
    reranker = LLMReranker(
        llm_bot=llm_bot,
        prompt_template=simple_prompt,
        chunk_size=500,  # Smaller chunks for testing
        top_n=2  # Only keep top 2 for testing
    )
    
    # Example documents
    documents = [
        {
            "text": "Python is a high-level, interpreted programming language known for its simplicity and readability.",
            "metadata": {"source": "doc1", "page": 1}
        },
        {
            "text": "Machine learning is a subfield of artificial intelligence that focuses on developing algorithms that can learn from data.",
            "metadata": {"source": "doc2", "page": 5}
        }
    ]
    
    # Rerank documents
    query = "What is Python programming language?"
    print(f"Reranking documents for query: {query}")
    
    try:
        ranked_chunks = reranker.rerank(query, documents)
        
        # Print results
        for i, chunk in enumerate(ranked_chunks):
            print(f"Rank {i+1}: Score {chunk['score']}")
            print(f"Passage: {chunk['passage'][:100]}...")
            print(f"Reasoning: {chunk['reasoning'][:100]}...")
            print("-" * 50)
    except Exception as e:
        print(f"Error during reranking: {e}")