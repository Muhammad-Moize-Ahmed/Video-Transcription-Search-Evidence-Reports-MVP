"""
Indexer Module
Merges audio transcription and vision captions into unified timeline,
creates embeddings, and stores in ChromaDB for search.
"""

import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON file from given path."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_transcription_and_captions(transcription_path: str, captions_path: str) -> List[Dict[str, Any]]:
    """
    Merge transcription and captions into unified timeline entries.

    Args:
        transcription_path: Path to transcription JSON
        captions_path: Path to captions JSON

    Returns:
        List of unified timeline entries with text, timestamp, and type
    """
    # Load both JSON files
    transcription_data = load_json_file(transcription_path)
    captions_data = load_json_file(captions_path)

    unified_entries = []

    # Process transcription segments
    if "segments" in transcription_data:
        for segment in transcription_data["segments"]:
            unified_entries.append({
                "timestamp": segment["start"],  # Use start time
                "end_time": segment["end"],
                "text": segment["text"].strip(),
                "type": "audio",
                "words": segment.get("words", [])
            })

    # Process vision captions
    if "captions" in captions_data:
        for caption_item in captions_data["captions"]:
            unified_entries.append({
                "timestamp": caption_item["timestamp"],
                "text": caption_item["caption"],
                "type": "vision",
                "frame_path": caption_item.get("frame_path", "")
            })

    # Sort by timestamp
    unified_entries.sort(key=lambda x: x["timestamp"])

    return unified_entries


def create_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    """
    Create embeddings for a list of texts using sentence-transformers.

    Args:
        texts: List of text strings to embed
        model_name: Name of the sentence-transformers model

    Returns:
        List of embedding vectors
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Creating embeddings for {len(texts)} text entries...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # Clean up model from memory
    del model

    return embeddings.tolist()


def setup_chromadb(persist_directory: str = "./chroma_db") -> chromadb.Collection:
    """
    Setup ChromaDB client and collection.

    Args:
        persist_directory: Directory to persist ChromaDB data

    Returns:
        ChromaDB collection for storing embeddings
    """
    # Create persist directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=persist_directory)

    # Get or create collection
    collection = client.get_or_create_collection(
        name="video_transcripts",
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def store_in_chromadb(collection: chromadb.Collection,
                     entries: List[Dict[str, Any]],
                     embeddings: List[List[float]]) -> None:
    """
    Store entries and their embeddings in ChromaDB.

    Args:
        collection: ChromaDB collection
        entries: List of unified timeline entries
        embeddings: List of embedding vectors
    """
    # Prepare data for ChromaDB
    ids = [f"entry_{i}_{entry['timestamp']}" for i, entry in enumerate(entries)]
    documents = [entry["text"] for entry in entries]
    metadatas = [
        {
            "timestamp": entry["timestamp"],
            "type": entry["type"],
            "end_time": entry.get("end_time", entry["timestamp"])
        }
        for entry in entries
    ]

    # Add to collection
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print(f"Stored {len(entries)} entries in ChromaDB")


def process_video_to_searchable_index(video_path: str,
                                    output_dir: str,
                                    frame_interval: int = 3,
                                    model_size: str = "tiny") -> str:
    """
    Complete pipeline: process video through audio and vision processors,
    then create searchable index.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save intermediate outputs
        frame_interval: Interval in seconds between frames for vision processing
        model_size: Whisper model size for audio processing

    Returns:
        Path to ChromaDB persistence directory
    """
    # Import processors locally to avoid circular imports
    from audio_processor import process_video_to_transcription
    from vision_processor import process_video_to_captions

    print("Step 1: Processing audio...")
    transcription_path = process_video_to_transcription(
        video_path,
        output_dir,
        model_size
    )

    print("Step 2: Processing vision...")
    captions_path = process_video_to_captions(
        video_path,
        output_dir,
        frame_interval
    )

    print("Step 3: Merging transcription and captions...")
    unified_entries = merge_transcription_and_captions(transcription_path, captions_path)

    if not unified_entries:
        raise ValueError("No entries created from transcription and captions")

    print(f"Created {len(unified_entries)} unified timeline entries")

    print("Step 4: Creating embeddings...")
    texts = [entry["text"] for entry in unified_entries]
    embeddings = create_embeddings(texts)

    print("Step 5: Setting up ChromaDB and storing...")
    persist_dir = os.path.join(output_dir, "chroma_db")
    collection = setup_chromadb(persist_dir)
    store_in_chromadb(collection, unified_entries, embeddings)

    return persist_dir


if __name__ == "__main__":
    # Example usage (for testing)
    import sys

    if len(sys.argv) < 3:
        print("Usage: python indexer.py <video_path> <output_dir> [frame_interval] [model_size]")
        print("Example: python indexer.py video.mp4 ./output 3 tiny")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = sys.argv[2]
    frame_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    model_size = sys.argv[4] if len(sys.argv) > 4 else "tiny"

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    persist_directory = process_video_to_searchable_index(
        video_path,
        output_dir,
        frame_interval,
        model_size
    )

    print(f"\nSearchable index created at: {persist_directory}")
    print("You can now search this index using the query function.")