import json
import openai
from dotenv import load_dotenv
import re
import os
from pathlib import Path
from qdrant_client import QdrantClient
from constants import TEMPERATURE
import argparse
load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")

class CommentarySearch:
    def __init__(self, commentary_file="data/commentry.json"):
        """Initialize Commentary Search with commentary data"""
        self.commentary_file = commentary_file
        self.commentary_data = self._load_commentary()
        
    def _load_commentary(self):
        """Load commentary data from JSON file"""
        try:
            with open(self.commentary_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading commentary file: {e}")
            return []

    def search_commentary(self, query_text, n_results=5):
        """
        Search through commentary data using LLM to find relevant timestamps
        
        Args:
            query_text: User's question about the match
            n_results: Number of results to return
        
        Returns:
            List of relevant moments with timestamps
        """
        try:
            # Prepare commentary context for LLM
            commentary_context = "\n".join([
                f"[{item['timestamp']}s]: {item['analysis']}"
                for item in self.commentary_data
            ])

            # Use LLM to find relevant moments
            response = openai.ChatCompletion.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a badminton match analysis assistant. "
                            "Your task is to find specific moments in the match that are relevant to the user's question. "
                            "For each relevant moment, provide:\n"
                            "1. The timestamp\n"
                            "2. A brief explanation of why this moment is relevant\n"
                            "3. The relevant part of the commentary\n"
                            "Return the results as a JSON object with an array of 'moments'."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"MATCH COMMENTARY:\n{commentary_context}\n\n"
                            f"USER QUESTION: {query_text}\n\n"
                            "Find the most relevant moments that answer the user's question. "
                            "Return a JSON object with this structure:\n"
                            "{\n"
                            "  'moments': [\n"
                            "    {\n"
                            "      'timestamp': number"
                            "    }\n"
                            "  ]\n"
                            "}"
                        )
                    }
                ],
                temperature=TEMPERATURE
            )
            
            # Extract and process results
            content = response.choices[0].message.content
            results = json.loads(content).get('moments', [])
            
            # Sort by relevance and limit results
            results.sort(key=lambda x: x.get('timestamp', 0))
            return results[:n_results]
            
        except Exception as e:
            print(f"Error in commentary search: {e}")
            return []

    def create_clickable_results(self, results, video_path):
        """
        Create results with clickable links to open video at timestamps
        
        Args:
            results: Search results from search_commentary
            video_path: Path to the video file
            
        Returns:
            Results with URLs added
        """
        clickable_results = []
        
        for result in results:
            # Add a URL to each result
            timestamp = result['timestamp']
            video_url = f"file:///{os.path.abspath(video_path).replace(os.sep, '/')}#t={int(timestamp)}"
            
            # Copy the result and add URL
            clickable_result = result.copy()
            clickable_result['video_url'] = video_url
            clickable_results.append(clickable_result)
            
        return clickable_results

def main():
    # Add command-line argument parser
    parser = argparse.ArgumentParser(description='Search through badminton match commentary.')
    parser.add_argument('--query', '-q', type=str, required=True,
                      help='Search query text')
    parser.add_argument('--results', '-n', type=int, default=5,
                      help='Number of results to return (default: 5)')
    parser.add_argument('--video', '-v', type=str, default="data/badminton.mp4",
                      help='Path to the video file (default: data/badminton.mp4)')
    parser.add_argument('--commentary', '-c', type=str, default="data/commentry.json",
                      help='Path to the commentary file (default: data/commentry.json)')
    
    args = parser.parse_args()

    # Initialize the commentary search
    commentary_search = CommentarySearch(commentary_file=args.commentary)
    
    # Search for relevant moments
    print(f"Query: {args.query}\n")
    results = commentary_search.search_commentary(args.query, n_results=args.results)
    
    if results:
        # Add clickable links to the results
        clickable_results = commentary_search.create_clickable_results(results, args.video)
        
        print("\nRelevant Moments:")
        for i, result in enumerate(clickable_results):
            print(f"\nMoment {i+1}:")
            print(f"Time: {result['timestamp']} seconds")
            print(f"Video URL: {result['video_url']}")
    else:
        print("No relevant moments found.")

if __name__ == "__main__":
    main() 