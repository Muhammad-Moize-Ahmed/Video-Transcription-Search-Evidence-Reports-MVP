"""
Report Generator Module
Searches ChromaDB for relevant transcript segments and generates
evidence-based reports using Ollama with Gemma3:4b.
"""

import os
import json
import ollama
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load sentence-transformers model for query embedding.

    Args:
        model_name: Name of the sentence-transformers model

    Returns:
        Loaded SentenceTransformer model
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    return model


def setup_chromadb(persist_directory: str) -> chromadb.Collection:
    """
    Setup ChromaDB client and load existing collection.

    Args:
        persist_directory: Directory where ChromaDB data is persisted

    Returns:
        ChromaDB collection
    """
    # Validate that the directory exists and contains ChromaDB data
    if not os.path.exists(persist_directory):
        raise ValueError(
            f"No video index found at {persist_directory}. "
            f"Please process a video first in the 'Upload & Process' tab."
        )
    
    # Check if directory has ChromaDB files
    dir_contents = os.listdir(persist_directory)
    if not dir_contents:
        raise ValueError(
            f"Video index directory is empty at {persist_directory}. "
            f"Please process a video first in the 'Upload & Process' tab."
        )

    # Initialize ChromaDB client
    try:
        client = chromadb.PersistentClient(path=persist_directory)
    except Exception as e:
        raise ValueError(
            f"Invalid or corrupted video index at {persist_directory}. "
            f"Please process a video first in the 'Upload & Process' tab. "
            f"Error: {str(e)}"
        )

    # Get existing collection
    try:
        collection = client.get_collection(name="video_transcripts")
    except Exception as e:
        raise ValueError(
            f"No searchable index found for this video. "
            f"Please process a video first in the 'Upload & Process' tab. "
            f"Error: {str(e)}"
        )
    
    return collection


def search_chromadb(collection: chromadb.Collection,
                   query_text: str,
                   embedding_model: SentenceTransformer,
                   n_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search ChromaDB for relevant segments based on query text.

    Args:
        collection: ChromaDB collection to search
        query_text: Text query to search for
        embedding_model: Model to embed the query
        n_results: Number of results to return

    Returns:
        List of search results with metadata and relevance scores
    """
    # Embed the query
    query_embedding = embedding_model.encode([query_text]).tolist()[0]

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    # Format results
    formatted_results = []
    if results['documents'] and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            formatted_results.append({
                "text": doc,
                "timestamp": results['metadatas'][0][i].get("timestamp", 0),
                "end_time": results['metadatas'][0][i].get("end_time", 0),
                "type": results['metadatas'][0][i].get("type", "unknown"),
                "relevance_score": 1 - results['distances'][0][i]  # Convert distance to similarity
            })

    return formatted_results


def generate_evidence_report(query: str,
                           search_results: List[Dict[str, Any]],
                           model_name: str = "gemma3:4b") -> str:
    """
    Generate evidence-based report using Ollama with search results as context.

    Args:
        query: Original user query
        search_results: List of relevant transcript segments from ChromaDB
        model_name: Ollama model to use for generation

    Returns:
        Formatted evidence report with cited timestamps
    """
    if not search_results:
        return "No relevant information found for the query."

    # Prepare context from search results
    context_parts = []
    for i, result in enumerate(search_results, 1):
        timestamp_str = format_timestamp(result['timestamp'])
        end_time_str = format_timestamp(result.get('end_time', result['timestamp']))
        context_parts.append(
            f"[Source {i} - {timestamp_str} to {end_time_str}]:\n{result['text']}"
        )

    context = "\n\n".join(context_parts)

    # Create prompt for Ollama
    prompt = f"""Based on the following video transcript segments, provide a comprehensive evidence-based answer to the query: "{query}"

Video Evidence:
{context}

Instructions:
1. Provide a clear, concise answer to the query based solely on the provided evidence
2. Cite specific sources using [Source X] notation where information is derived
3. Include relevant timestamps from the video when making claims
4. If the evidence doesn't fully answer the query, state what information is missing
5. Maintain an objective tone and only include information supported by the evidence
6. Format your response as a professional evidence report

Evidence Report:"""

    # Generate response using Ollama
    print(f"Generating report with Ollama model: {model_name}")
    response = ollama.generate(
        model=model_name,
        prompt=prompt,
        stream=False
    )

    return response['response']


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to MM:SS or HH:MM:SS format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def query_and_report(video_chroma_dir: str,
                    query: str,
                    n_results: int = 5) -> Dict[str, Any]:
    """
    Complete pipeline: search ChromaDB and generate evidence report.

    Args:
        video_chroma_dir: Directory containing ChromaDB for the video
        query: User query to search for
        n_results: Number of search results to use as context

    Returns:
        Dictionary containing query, search results, and generated report
    """
    # Setup components
    embedding_model = load_embedding_model()
    collection = setup_chromadb(video_chroma_dir)

    # Search for relevant segments
    print(f"Searching for: '{query}'")
    search_results = search_chromadb(
        collection, query, embedding_model, n_results
    )

    print(f"Found {len(search_results)} relevant segments")

    # Generate evidence report
    report = generate_evidence_report(query, search_results)

    # Clean up models from memory
    del embedding_model

    return {
        "query": query,
        "search_results": search_results,
        "report": report,
        "timestamp": None  # Could add generation timestamp if needed
    }


if __name__ == "__main__":
    # Example usage (for testing)
    import sys

    if len(sys.argv) < 3:
        print("Usage: python report_gen.py <chroma_directory> <query> [n_results]")
        print("Example: python report_gen.py ./output/chroma_db \"What is discussed about climate change?\" 5")
        sys.exit(1)

    chroma_dir = sys.argv[1]
    query = sys.argv[2]
    n_results = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    if not os.path.exists(chroma_dir):
        print(f"Error: ChromaDB directory not found: {chroma_dir}")
        sys.exit(1)

    result = query_and_report(chroma_dir, query, n_results)

    print("\n" + "="*60)
    print("EVIDENCE REPORT")
    print("="*60)
    print(f"Query: {result['query']}")
    print("-"*60)
    print(result['report'])
    print("-"*60)
    print(f"Sources consulted: {len(result['search_results'])}")

    # Optional: Show sources
    if result['search_results']:
        print("\nSources:")
        for i, source in enumerate(result['search_results'], 1):
            timestamp = format_timestamp(source['timestamp'])
            end_time = format_timestamp(source.get('end_time', source['timestamp']))
            print(f"  [{i}] {timestamp}-{end_time} ({source['type']}) - Relevance: {source['relevance_score']:.3f}")