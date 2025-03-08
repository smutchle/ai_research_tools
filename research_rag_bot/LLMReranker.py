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

Rate this passage on a scale from 0 to 10 based on how directly useful it is to specifically answering the query.
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
            response = self.llm_bot.complete(prompt).strip()
            
            # First try to find an exact number response
            match = re.search(r'^\s*(\d+(?:\.\d+)?)\s*$', response)
            if match:
                return (float(match.group(1)), "Direct score")
            
            # If exact match failed, try to find any number in the response
            match = re.search(r'(\d+(?:\.\d+)?)', response)
            if match:
                return (float(match.group(1)), f"Extracted score from: {response[:50]}...")
            
            print(f"Could not extract numeric score from response: {response[:100]}...")
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
    
    # Use a prompt that explicitly asks for just a number with clear scoring criteria
    simple_prompt = """Rate how directly relevant this passage is to answering the query on a scale of 0-10.
Query: {query}
Passage: {passage}

Scoring guide:
- 0-2: Not relevant or only mentions keywords without addressing the query
- 3-5: Somewhat relevant but doesn't directly answer the query
- 6-8: Relevant and partially answers the query
- 9-10: Highly relevant and directly answers the query

For example, if the query is about Python and the passage only mentions AI or other programming languages without specifically addressing Python, it should receive a low score (0-2).

Your answer must be ONLY a single number between 0 and 10, with no other text.
Score: """
    
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