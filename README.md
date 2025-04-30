# Video RAG System with Commentary Search

This system enables natural language querying of video content by:
1. Using pre-generated commentary data for videos
2. Searching through commentary using LLM to find relevant moments
3. Providing timestamps and clickable links to specific video moments

## Prerequisites

- Python 3.8+
- OpenAI API key
- FFmpeg (for video playback)

## Setup

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Searching Through Video Commentary

The system allows you to search through video commentary using natural language queries:

```bash
# Search with a natural language query
python semantic_search_enhancement.py --query "show me when players are arguing" --video data/your_video.mp4 --commentary data/commentry.json

# Or search for specific moments
python semantic_search_enhancement.py --query "what are the heated moments in this game" --video data/your_video.mp4 --commentary data/commentry.json
```

This will:
- Load the commentary data from the JSON file
- Use LLM to find relevant moments based on your query
- Return timestamps and explanations for each moment
- Generate clickable links to open the video at exact timestamps

### Commentary Data Format

The commentary data should be in JSON format with the following structure:
```json
[
    {
        "timestamp": 4,
        "analysis": "Hemanth executes a perfect jump smash..."
    },
    {
        "timestamp": 10,
        "analysis": "Intense rally between the players..."
    }
]
```

### Command-Line Search App

The easiest way to search and watch video content is using the command-line app:

```bash
# Search with a query
python semantic_search_enhancement.py --query "heated moments in the game" --video data/badminton.mp4 --commentary data/commentry.json

# Specify number of results
python semantic_search_enhancement.py --query "dunks" --results 3 --video data/basketball.mp4 --commentary data/commentry.json
```

The app provides:
- Natural language search through video commentary
- Timestamps for relevant moments
- Clickable links to jump to specific moments in the video
- Explanations of why each moment is relevant

### Example Output

```
Query: what are the heated moments in this game

Relevant Moments:

Moment 1:
Time: 4 seconds
Relevance: This moment shows Hemanth's powerful jump smash
Commentary: Hemanth executes a perfect jump smash...
Video URL: file:///path/to/video#t=4

Moment 2:
Time: 10 seconds
Relevance: Intense rally between players
Commentary: Players engage in a long, intense rally...
Video URL: file:///path/to/video#t=10
```

## How It Works

1. **Commentary Loading**:
   - System loads pre-generated commentary from JSON file
   - Each commentary entry includes timestamp and analysis

2. **Search Process**:
   - User provides a natural language query
   - LLM analyzes the commentary to find relevant moments
   - System returns timestamps and explanations

3. **Video Navigation**:
   - Results include clickable links to video timestamps
   - Links open the video at the exact moment of interest

## Customization

- Adjust the number of results using the `--results` parameter
- Modify the LLM prompt in `search_commentary()` for different types of analysis
- Change the video path and commentary file paths as needed

## Troubleshooting

If you encounter issues:

1. Check that your .env file has the correct OPENAI_API_KEY
2. Verify the commentary file exists and is in the correct JSON format
3. Ensure the video file path is correct and accessible
4. Make sure all required dependencies are installed

## Future Enhancements

Planned improvements:
1. Support for multiple commentary files
2. Advanced filtering options
3. Integration with video streaming services
4. Real-time commentary generation
5. Support for multiple languages 