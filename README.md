# Video RAG System with Qdrant

This system enables natural language querying of video content by:
1. Extracting audio from videos
2. Transcribing the audio using Whisper
3. Chunking the transcripts into semantically meaningful segments
4. Storing them in a Qdrant vector database for efficient search

## Prerequisites

- Python 3.8+
- FFmpeg installed and in PATH
- Qdrant server running (local or remote)
- OpenAI API key (for embeddings and query expansion)

## Setup

### Local Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Qdrant server:
```bash
# Using Docker
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
```

3. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_HOST=""
```

### Docker Setup

1. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

2. Build and run with Docker Compose:
```bash
docker-compose up --build
```

This will:
- Start a Qdrant server container
- Build and start the Video RAG application container
- Set up networking between the containers

#### Running Commands in Docker

To run specific commands in the Docker container:

```bash
# Run with a specific query
docker-compose run video_rag --query "heated moments" --results 5

# Run with a specific category
docker-compose run video_rag --category funny_moments

# Run in interactive mode
docker-compose run video_rag
```

## Usage

### Process Video and Store in Qdrant

1. First, process your video to extract audio and generate subtitles:
```python
# In setup.py
extract_audio("data/my_video", "data/my_video.mp4")
lang, segments = extract_transcript("data/my_video.wav")
```

2. Once you have the subtitles, chunk them and store in Qdrant:
```python
# In chunk_subtitles.py
subtitles = load_subtitles("subtitles.json")
chunks = chunk_by_time_window(subtitles, time_window=45)
client = store_in_qdrant(chunks)
```

### Command-Line Search App

The easiest way to search and watch video content is using the command-line app:

```bash
# Search with a query
python video_rag_app.py --query "heated moments in the game" --video data/basketball.mp4

# Search using a predefined category
python video_rag_app.py --category heated_moments --video data/basketball.mp4

# Automatically open top result in browser
python video_rag_app.py --query "dunks" --open

# Or run in interactive mode
python video_rag_app.py
```

This app lets you:
- Search for specific content in videos
- View results with timestamps
- Open the video directly at the timestamp in your browser
- Search by either natural language queries or predefined categories

### Basic Query Interface

For programmatic usage, the RAG interface provides direct access:

```python
# In rag_query.py
rag = VideoRAG()
results = rag.query("What happened with the basketball?")

# Results include timestamps, text and relevance scores
for result in results:
    print(f"Time: {result['time_range']}")
    print(f"Text: {result['text']}")
```

### Enhanced Search Capabilities

For more advanced searching, especially for abstract concepts like "heated moments" or "funny moments", use the enhanced search interface:

```python
# In semantic_search_enhancement.py
enhanced_rag = EnhancedVideoRAG()

# Use semantic search with query expansion
results = enhanced_rag.hybrid_search("give me heated moments from the video")

# Or use predefined categories
highlights = enhanced_rag.get_video_highlights("heated_moments")

# Generate clickable URLs and open in browser
clickable_results = enhanced_rag.create_clickable_results(results, "path/to/video.mp4")
enhanced_rag.open_video_at_timestamp("path/to/video.mp4", results[0]['start_time'])
```

The enhanced search provides:

1. **Query Expansion**: Automatically generates alternative queries using GPT to find more relevant content
2. **Semantic Search**: Finds content based on meaning rather than exact keyword matches
3. **Predefined Categories**: Quickly retrieve common types of content:
   - `heated_moments`: Arguments, confrontations, technical fouls
   - `funny_moments`: Bloopers, jokes, amusing incidents
   - `amazing_plays`: Spectacular dunks, blocks, steals
   - `emotional_moments`: Touching or heartfelt sequences
4. **Browser Integration**: Open video at specific timestamps for immediate viewing

## Chunking Strategies

The system offers two chunking methods:

1. **Time Window Chunking** (default): Groups subtitles that occur within a specified time window (e.g., 45 seconds).
   - Pros: Maintains temporal context, good for understanding discussions or scenes
   - Use case: General video content, interviews, conversations

2. **Sliding Window Chunking**: Creates overlapping chunks with a fixed number of subtitle segments.
   - Pros: Ensures consistent chunk sizes, better for factual retrieval
   - Use case: Dense information videos, tutorials, lectures

## Customization

- Adjust time window in `chunk_by_time_window()` to make chunks larger or smaller
- Change the embedding model in `openai.embeddings.create()` for different performance characteristics
- Add new categories in `EnhancedVideoRAG.content_types` to support additional predefined searches

## Troubleshooting Docker

If you encounter issues with Docker:

1. Make sure Docker Desktop is running
2. For WSL users, enable WSL integration in Docker Desktop settings
3. If Qdrant connection fails, check the network settings in docker-compose.yml
4. For permission issues with mounted volumes, check file permissions 