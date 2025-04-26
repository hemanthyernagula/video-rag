import json
from qdrant_client import QdrantClient
import openai
from dotenv import load_dotenv
import os
from chunk_subtitles import load_subtitles

load_dotenv()

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai_api_key)

class VideoRAG:
    def __init__(self, collection_name="video_subtitles"):
        """Initialize Video RAG system with Qdrant"""
        # Initialize Qdrant client
        try:
            # Get Qdrant host and port from environment variables or use defaults
            qdrant_host = os.environ.get("QDRANT_HOST")
            qdrant_port = int(os.environ.get("QDRANT_PORT", 6333))
            
            # Create Qdrant client
            print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}")
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
            
            self.collection_exists = self.client.get_collection(collection_name)
            self.collection_name = collection_name
            
            if not self.collection_exists:
                print(f"Collection {collection_name} not found. Please create it first using chunk_subtitles.py")
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}")
            self.client = None
            self.collection_exists = False
        
        # Load original subtitles for context
        try:
            self.subtitles = load_subtitles()
        except Exception as e:
            print(f"Error loading subtitles: {e}")
            self.subtitles = []
    
    def get_all_subtitles_from_vector_db(self, limit=1000):
        """
        Get all subtitles from the vector database
        
        Args:
            limit: Maximum number of points to retrieve per batch (default: 1000)
            
        Returns:
            List of subtitle texts with their metadata
        """
        if not self.client or not self.collection_exists:
            return {"error": "Qdrant collection not initialized"}
        
        try:
            # Use scroll to get all points in batches
            all_subtitles = []
            offset = None
            
            while True:
                # Get a batch of points
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False  # We don't need the vectors
                )
                
                # Extract points and their payloads
                points = results[0]
                
                # Break if no more points
                if not points:
                    break
                
                # Extract subtitle texts and metadata
                for point in points:
                    all_subtitles.append({
                        "id": point.id,
                        "text": point.payload.get("text", ""),
                        "start_time": point.payload.get("start_time", 0),
                        "end_time": point.payload.get("end_time", 0),
                        "segment_ids": point.payload.get("segment_ids", [])
                    })
                
                # Update offset for the next batch
                if len(points) < limit:
                    break
                    
                offset = points[-1].id
            
            return all_subtitles
            
        except Exception as e:
            print(f"Error retrieving subtitles: {e}")
            return []
    
    def query(self, query_text, n_results=3):
        """Query the video database with natural language"""
        if not self.client or not self.collection_exists:
            return {"error": "Qdrant collection not initialized"}
        
        # Generate query embedding with OpenAI
        try:
            response = client.embeddings.create(
                input=[query_text],
                model="text-embedding-3-small"
            )
            query_embedding = response.data[0].embedding
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return {"error": f"Failed to generate embedding: {e}"}
        
        # Search in Qdrant
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=n_results
        )
        
        # Format results
        formatted_results = []
        for result in results:
            # Get data from payload
            start_time = result.payload["start_time"]
            end_time = result.payload["end_time"]
            
            # Format time for display
            start_formatted = self._format_time(start_time)
            end_formatted = self._format_time(end_time)
            
            formatted_results.append({
                "text": result.payload["text"],
                "start_time": start_time,
                "end_time": end_time,
                "time_range": f"{start_formatted} - {end_formatted}",
                "relevance_score": result.score,
                "segment_ids": result.payload["segment_ids"]
            })
        
        return formatted_results
    
    def _format_time(self, seconds):
        """Format seconds as MM:SS"""
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"
    
    def get_video_context(self, start_time, end_time, window_seconds=30):
        """Get wider context from a specific timestamp range"""
        # Expand the window
        context_start = max(0, start_time - window_seconds)
        context_end = end_time + window_seconds
        
        # Find all segments that fall within this expanded window
        context_segments = [
            segment for segment in self.subtitles
            if segment["end"] >= context_start and segment["start"] <= context_end
        ]
        
        return {
            "text": " ".join([segment["text"] for segment in context_segments]),
            "start_time": context_start,
            "end_time": context_end,
            "time_range": f"{self._format_time(context_start)} - {self._format_time(context_end)}",
            "segments": context_segments
        }
    
    def get_video_url_with_timestamp(self, video_url, start_time):
        """Generate a link to the video at a specific timestamp"""
        # YouTube format: https://youtu.be/VIDEO_ID?t=SECONDS
        # Assume video_url is just the ID or full YouTube URL
        if "youtube.com" in video_url or "youtu.be" in video_url:
            if "?" in video_url:
                base_url = video_url.split("?")[0]
                return f"{base_url}?t={int(start_time)}"
            else:
                return f"{video_url}?t={int(start_time)}"
        else:
            # Just return a formatted time if not a recognized URL format
            return f"Jump to {self._format_time(start_time)}"

if __name__ == "__main__":
    # Example usage
    rag = VideoRAG()
    
    # Test query
    query = "What happened with the basketball?"
    results = rag.query(query)
    
    print(f"Query: {query}\n")
    
    if isinstance(results, list):
        for i, result in enumerate(results):
            print(f"Result {i+1} (Score: {result['relevance_score']:.2f}):")
            print(f"Time: {result['time_range']}")
            print(f"Text: {result['text']}")
            print()
            
        # Get wider context for the top result
        if results:
            top_result = results[0]
            context = rag.get_video_context(top_result['start_time'], top_result['end_time'])
            print("Expanded Context:")
            print(f"Time: {context['time_range']}")
            print(f"Text: {context['text']}")
            
    else:
        print(results)  # Print error message 