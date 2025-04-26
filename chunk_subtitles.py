import json
import numpy as np
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
import openai
from dotenv import load_dotenv
from constants import EMBEDEDING_MODEL_NAME
import argparse

# Load environment variables
load_dotenv()

# Explicitly set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_subtitles(subtitle_file="subtitles.json"):
    with open(subtitle_file, "r") as f:
        return json.load(f)

def chunk_by_sliding_window(subtitles, window_size=5, stride=2):
    """
    Create chunks using a sliding window over subtitle segments
    - window_size: number of subtitle segments per chunk
    - stride: how many segments to slide the window by
    """
    chunks = []
    
    for i in range(0, len(subtitles), stride):
        if i + window_size <= len(subtitles):
            window = subtitles[i:i+window_size]
            
            # Combine text from segments
            text = " ".join([segment["text"] for segment in window])
            
            # Store start and end timestamps
            start_time = window[0]["start"]
            end_time = window[-1]["end"]
            
            chunks.append({
                "text": text,
                "start_time": start_time,
                "end_time": end_time,
                "segment_ids": list(range(i, i+window_size))  # Keep track of original segments
            })
    
    return chunks

def chunk_by_time_window(subtitles, time_window=30):
    """
    Create chunks by grouping segments that fall within a specific time window
    - time_window: time in seconds for each chunk
    """
    chunks = []
    current_chunk = []
    
    for segment in subtitles:
        if not current_chunk or segment["start"] - current_chunk[0]["start"] < time_window:
            current_chunk.append(segment)
        else:
            # Combine text from segments
            text = " ".join([s["text"] for s in current_chunk])
            
            chunks.append({
                "text": text,
                "start_time": current_chunk[0]["start"],
                "end_time": current_chunk[-1]["end"],
                "segment_ids": list(range(len(chunks), len(chunks) + len(current_chunk)))
            })
            
            current_chunk = [segment]
    
    # Add the last chunk
    if current_chunk:
        text = " ".join([s["text"] for s in current_chunk])
        chunks.append({
            "text": text,
            "start_time": current_chunk[0]["start"],
            "end_time": current_chunk[-1]["end"],
            "segment_ids": list(range(len(chunks), len(chunks) + len(current_chunk)))
        })
    
    return chunks

def store_in_qdrant(chunks, collection_name="video_subtitles"):
    """Store chunks in Qdrant Vector DB"""
    # Initialize Qdrant client
    # Get Qdrant host and port from environment variables or use defaults
    qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_port = int(os.environ.get("QDRANT_PORT", 6333))
    
    # Create Qdrant client
    print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}")
    qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
    
    print(f"qdrant_client: {qdrant_client}")
    # Extract just the text from chunks for embedding
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate embeddings with OpenAI API
    try:
        # Ensure we're using the proper API call format based on the OpenAI version
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print(f"client: {client}")
        response = client.embeddings.create(
            input=texts,
            model=EMBEDEDING_MODEL_NAME
        )
        # Extract the embedding vectors
        embeddings = [item.embedding for item in response.data]
        print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None
    
    # Define collection - delete if exists
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
    except Exception:
        pass
    
    # Create collection with vector configuration
    vector_size = len(embeddings[0]) if embeddings else 1536  # Default size for OpenAI embeddings
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )
    
    # Prepare data for insertion
    ids = list(range(len(chunks)))
    
    # Prepare payload with metadata
    payloads = [{
        "text": chunk["text"],
        "start_time": float(chunk["start_time"]),
        "end_time": float(chunk["end_time"]),
        "segment_ids": chunk["segment_ids"]
    } for chunk in chunks]
    
    # Add data to collection in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        batch_payloads = payloads[i:i+batch_size]
        
        qdrant_client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=batch_ids,
                vectors=batch_embeddings,
                payloads=batch_payloads
            )
        )
    
    print(f"Stored {len(chunks)} chunks in Qdrant collection '{collection_name}'")
    return qdrant_client

def query_video(query_text, qdrant_client, collection_name="video_subtitles", n_results=3):
    """Query the vector database to find relevant video segments"""
    # Generate query embedding with OpenAI
    try:
        # Ensure we're using the proper API call format based on the OpenAI version
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            input=[query_text],
            model=EMBEDEDING_MODEL_NAME
        )
        query_embedding = response.data[0].embedding
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return []
    
    # Search in Qdrant
    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=n_results
    )
    
    # Format results
    formatted_results = []
    for result in results:
        formatted_results.append({
            "score": result.score,
            "text": result.payload["text"],
            "start_time": result.payload["start_time"],
            "end_time": result.payload["end_time"],
            "segment_ids": result.payload["segment_ids"]
        })
    
    return formatted_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process video subtitles and store in Qdrant.')
    parser.add_argument('--subtitles', '-s', type=str, default='subtitles.json',
                       help='Path to subtitle JSON file (default: subtitles.json)')
    parser.add_argument('--time-window', '-t', type=int, default=45,
                       help='Time window in seconds for chunking (default: 45)')
    parser.add_argument('--collection', '-c', type=str, default='video_subtitles',
                       help='Name of Qdrant collection (default: video_subtitles)')
    parser.add_argument('--method', '-m', type=str, choices=['time', 'sliding'], default='time',
                       help='Chunking method: time-based or sliding window (default: time)')
    parser.add_argument('--window-size', '-w', type=int, default=5,
                       help='Number of segments per window when using sliding window (default: 5)')
    parser.add_argument('--stride', type=int, default=2,
                       help='Stride for sliding window chunking (default: 2)')
    parser.add_argument('--preview', '-p', type=int, default=3,
                        help='Number of chunks to preview (default: 3, 0 for none)')
    parser.add_argument('--dry-run', '-d', action='store_true',
                        help='Generate chunks but do not store in Qdrant')
    
    args = parser.parse_args()
    
    # Load subtitles
    print(f"Loading subtitles from {args.subtitles}")
    try:
        subtitles = load_subtitles(args.subtitles)
    except FileNotFoundError:
        print(f"Error: Subtitle file '{args.subtitles}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: File '{args.subtitles}' is not a valid JSON file.")
        return
    
    # Choose chunking method
    if args.method == 'time':
        print(f"Using time window chunking with window of {args.time_window} seconds")
        chunks = chunk_by_time_window(subtitles, time_window=args.time_window)
    else:
        print(f"Using sliding window chunking with window size {args.window_size} and stride {args.stride}")
        chunks = chunk_by_sliding_window(subtitles, window_size=args.window_size, stride=args.stride)
    
    print(f"Created {len(chunks)} chunks from {len(subtitles)} subtitle segments")
    
    # Preview chunks
    if args.preview > 0:
        for i in range(min(args.preview, len(chunks))):
            print(f"\nChunk {i}:")
            print(f"Time: {chunks[i]['start_time']:.2f}s - {chunks[i]['end_time']:.2f}s")
            print(f"Text: {chunks[i]['text']}")
    
    # Store in Qdrant
    if not args.dry_run:
        print(f"Storing chunks in Qdrant collection '{args.collection}'")
        client = store_in_qdrant(chunks, collection_name=args.collection)
        if client:
            print("Successfully stored chunks in Qdrant")
        else:
            print("Failed to store chunks in Qdrant")
    else:
        print("Dry run complete. Chunks not stored in Qdrant.")

if __name__ == "__main__":
    main()
    
    