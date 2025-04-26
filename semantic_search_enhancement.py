import json
import openai
from dotenv import load_dotenv
import re
import os
from pathlib import Path
from qdrant_client import QdrantClient
from rag_query import VideoRAG
from constants import TEMPERATURE
import argparse
load_dotenv()

class EnhancedVideoRAG(VideoRAG):
    def __init__(self, collection_name="video_subtitles"):
        """Initialize Enhanced Video RAG with additional search capabilities"""
        super().__init__(collection_name)
        
        # Initialize content classifiers for different video aspects
        self.content_types = {
            "heated_moments": ["argument", "tension", "fight", "confrontation", "technical foul", 
                              "ejection", "angry", "shouting", "upset", "tension", "clash"],
            "funny_moments": ["laugh", "funny", "joke", "amusing", "humor", "blooper", "mistake", 
                             "fail", "embarrassing", "comic", "ridiculous"],
            "amazing_plays": ["dunk", "slam", "block", "steal", "incredible", "amazing", "spectacular", 
                             "highlight", "impressive", "celebration", "cheer", "perfect"],
            "emotional_moments": ["emotional", "touching", "tears", "cry", "heartfelt", "moving", 
                                 "sincere", "genuine", "touching"]
        }

    def expand_query(self, query_text, max_expansions=3):
        """Expand a query to multiple related queries using LLM"""
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a sports video search assistant. Convert the user's query into 3 alternative specific searches that would help find relevant basketball video moments. Return ONLY a JSON array of search terms with no additional text."},
                    {"role": "user", "content": f"User query: '{query_text}'\nCreate 3 alternative specific search terms for finding this in basketball footage."}
                ],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"}
            )
            
            # Extract the JSON from the response
            content = response.choices[0].message.content
            expansions = json.loads(content).get("search_terms", [])
            
            # Ensure we don't exceed the max number of expansions
            return expansions[:max_expansions]
        
        except Exception as e:
            print(f"Error in query expansion: {e}")
            # Return a fallback set of expansions based on predefined categories
            return self._classify_and_expand_query(query_text)
    
    def _classify_and_expand_query(self, query_text):
        """Classify the query into predefined categories and return appropriate expansions"""
        expansions = []
        
        for category, keywords in self.content_types.items():
            if any(keyword in query_text.lower() for keyword in category.split('_')):
                # Select a few representative keywords from this category
                expansions.extend(keywords[:3])
        
        # If no matching category found, return some defaults
        if not expansions:
            expansions = ["intense moment", "exciting play", "player confrontation"]
            
        return expansions
    
    def hybrid_search(self, query_text, n_results=5, include_user_query=True):
        """
        Perform hybrid search - combination of semantic search and keyword matching
        
        Args:
            query_text: The user's query
            n_results: Number of results to return
        
        Returns:
            List of search results with relevance scores
        """
        # Step 1: Perform standard semantic search with expanded queries
        # expanded_queries = [query_text] + self.expand_query(query_text)
        
        # Expand query based on actual video content
        expanded_queries = self.contextualized_query_expansion(query_text, include_user_query=include_user_query)
        print(f"Expanded queries: {expanded_queries}")
        
        all_results = []
        for query in expanded_queries:
            results = super().query(query, n_results=n_results)
            if isinstance(results, list):
                all_results.extend(results)
        
        # Step 2: De-duplicate results by time range
        unique_results = {}
        for result in all_results:
            key = f"{result['start_time']}_{result['end_time']}"
            if key not in unique_results or result['relevance_score'] > unique_results[key]['relevance_score']:
                unique_results[key] = result
        
        # Get the de-duplicated results as a list
        results = list(unique_results.values())
        
        # Sort by relevance score and return top n_results
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:n_results]
    
    def get_video_highlights(self, category, n_results=5):
        """
        Get video highlights based on predefined categories
        
        Args:
            category: One of 'heated_moments', 'funny_moments', 'amazing_plays', 'emotional_moments'
            n_results: Number of results to return
        
        Returns:
            List of relevant moments for the requested category
        """
        if category not in self.content_types:
            return {"error": f"Invalid category. Choose from: {list(self.content_types.keys())}"}
        
        # Get keywords for the category
        keywords = self.content_types[category]
        
        # Create a combined query from the keywords
        combined_query = " ".join(keywords[:5])  # Use first few keywords
        
        # Run hybrid search using the combined query
        results = self.hybrid_search(combined_query, n_results=n_results)
        
        return results

    def contextualized_query_expansion(self, query_text, max_expansions=3, include_user_query=True):
        """Expand query based on actual video content"""
        
        # Get all subtitles from the vector database
        all_subtitles = super().get_all_subtitles_from_vector_db()
        print(f"All subtitles: {all_subtitles}")
        # Create a context summary from the subtitles
        if isinstance(all_subtitles, list) and all_subtitles:
            # Limit the context size to avoid token limits
            sample_size = min(30, len(all_subtitles))
            # Select a diverse sample from different parts of the video
            sample_indices = [int(i * len(all_subtitles) / sample_size) for i in range(sample_size)]
            samples = [all_subtitles[i] for i in sample_indices]
            
            # Create context from the samples
            video_context = "\n".join([
                f"[{sample['start_time']:.1f}s - {sample['end_time']:.1f}s]: {sample['text']}" 
                for sample in samples
            ])
            
            try:
                # Now use GPT with context about the actual video
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a video analysis and search assistant. "
                                    "Your job is to carefully read the provided video content samples (from a basketball video transcript), "
                                    "understand the user's query, and generate **very specific search terms** "
                                    "that would help locate the exact moments matching the query inside this video. "
                                    "You must generate search terms that are grounded in the actual events mentioned in the transcript."
                                    "Avoid generic phrases. Focus on real incidents, actions, or scenes described in the text."
                                )
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"VIDEO CONTENT SAMPLES:\n{video_context}\n\n"
                                    f"USER QUERY: {query_text}\n\n"
                                    "Generate 3-5 very specific and focused search terms or scene descriptions "
                                    "based ONLY on the actual events described in the video content. "
                                    "Return them as a JSON object with a 'search_terms' array. "
                                    "Make sure the search terms are tied to real moments from the transcript."
                                )
                            }
                        ],
                    temperature=TEMPERATURE,
                    response_format={"type": "json_object"}
                )
                
                # Extract the JSON from the response
                content = response.choices[0].message.content
                expansions = json.loads(content).get("search_terms", [])
                
                # Add the original query and ensure we don't exceed max expansions
                expansions = [query_text] + expansions[:max_expansions] if include_user_query else expansions[:max_expansions]
                print(f"Expanded queries: {expansions}======================")
                return expansions
                
            except Exception as e:
                print(f"Error in contextualized query expansion: {e}")
                # Fall back to basic expansion if there's an error
                expansions = self._classify_and_expand_query(query_text)
                return [query_text] + expansions[:max_expansions] if include_user_query else expansions[:max_expansions]
        else:
            # Fall back to basic expansion if no subtitles are available
            expansions = self._classify_and_expand_query(query_text)
            return [query_text] + expansions[:max_expansions] if include_user_query else expansions[:max_expansions]

    def create_clickable_results(self, results, video_path):
        """
        Create results with clickable links to open video at timestamps
        
        Args:
            results: Search results from hybrid_search
            video_path: Path to the video file
            
        Returns:
            Results with URLs added
        """
        clickable_results = []
        
        for result in results:
            # Add a URL to each result
            start_time = result['start_time']
            video_url = f"file:///{os.path.abspath(video_path).replace(os.sep, '/')}#t={int(start_time)}"
            
            # Copy the result and add URL
            clickable_result = result.copy()
            clickable_result['video_url'] = video_url
            clickable_results.append(clickable_result)
            
        return clickable_results

if __name__ == "__main__":
    # Add command-line argument parser
    parser = argparse.ArgumentParser(description='Enhanced Video RAG search with semantic and keyword matching.')
    parser.add_argument('--query', '-q', type=str, default="",
                      help='Search query text')
    parser.add_argument('--category', '-c', type=str, choices=['heated_moments', 'funny_moments', 'amazing_plays', 'emotional_moments'], 
                      help='Predefined category to search for')
    parser.add_argument('--results', '-n', type=int, default=5,
                      help='Number of results to return (default: 5)')
    parser.add_argument('--video', '-v', type=str, default="data/basketball.mp4",
                      help='Path to the video file (default: data/basketball.mp4)')
    parser.add_argument('--include-user-query', '-i', action='store_true', default=False,
                      help='Include the user query in expanded queries')
    parser.add_argument('--collection', type=str, default="video_subtitles",
                      help='Qdrant collection name (default: video_subtitles)')
    
    args = parser.parse_args()

    # Initialize the enhanced RAG
    enhanced_rag = EnhancedVideoRAG(collection_name=args.collection)
    
    if args.category:
        # Use predefined category search
        print(f"Searching for category: {args.category}")
        results = enhanced_rag.get_video_highlights(args.category, n_results=args.results)
    elif args.query:
        # Use hybrid search with query
        print(f"Query: {args.query}\n")
        results = enhanced_rag.hybrid_search(
            args.query, 
            n_results=args.results,
            include_user_query=args.include_user_query
        )
    else:
        print("Error: Either --query or --category must be provided")
        parser.print_help()
        exit(1)
    
    print(f"Results: {results}")
    if isinstance(results, list):
        # Add clickable links to the results
        video_path = args.video
        clickable_results = enhanced_rag.create_clickable_results(results, video_path)
        
        for i, result in enumerate(clickable_results):
            print(f"Result {i+1} (Score: {result['relevance_score']:.2f}):")
            print(f"Time: {result['time_range']}")
            print(f"Text: {result['text']}")
            print(f"Video URL: {result['video_url']}")
            print()
    else:
        print(results)  # Print error message 