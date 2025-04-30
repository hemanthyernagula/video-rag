import os
import json
import argparse
from pathlib import Path
import openai
from typing import List, Dict
import base64
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

class CommentaryGenerator:
    def __init__(self, api_key: str):
        """Initialize the commentary generator with OpenAI API key."""
        openai.api_key = api_key
        
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def analyze_frame(self, frame_path: str, context: str = "", interval: int = 5) -> str:
        """Analyze a single frame using GPT-4 Vision."""
        base64_image = self.encode_image(frame_path)
        
        response = openai.ChatCompletion.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a professional badminton commentator providing live analysis. Your commentary should be natural, flowing, and engaging, like a real sports broadcast. Include:

1. What's happening in the rally:
   - Player positions and movements
   - Shot types and quality
   - Game situation and score (if visible)
   - Technical aspects
   - Player performance

But present it in a natural, flowing way that sounds like a real commentator. For example:

"Beautiful positioning from Hemanth as he sets up for the serve. He's standing tall at the service line, racket ready. Chanti's waiting at the baseline, looking focused and ready to receive. The crowd's quiet, anticipating the start of this crucial point..."

NOT like this:
"Certainly! Here's the detailed analysis:
1. Player Positions and Movement:
- Player in white (foreground)..."

Keep the analysis detailed but make it sound natural and engaging. Use proper badminton terminology but in a conversational way. Maintain the excitement of the moment while providing insightful analysis."""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Previous context: {context}\n\nProvide natural, flowing commentary for this {interval}-second moment in the match:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def generate_commentary(self, frames_dir: str, output_file: str, interval: int = 5):
        """Generate commentary for a sequence of frames."""
        frames = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_')])
        commentary = []
        context = ""
        
        for i, frame in enumerate(frames):
            frame_path = os.path.join(frames_dir, frame)
            timestamp = i * interval  # Calculate timestamp based on frame number and interval
            
            # Generate detailed analysis
            frame_analysis = self.analyze_frame(frame_path, context, interval)
            
            commentary.append({
                "frame": frame,
                "timestamp": timestamp,
                "analysis": frame_analysis,
                "duration": interval
            })
            context = frame_analysis  # Use analysis as context for next frame
        
        # Save commentary to file
        with open(output_file, 'w') as f:
            json.dump(commentary, f, indent=2)
        
        return commentary

def main():
    parser = argparse.ArgumentParser(description="Generate commentary for video frames using LLM")
    parser.add_argument("--frames_dir", required=True, help="Directory containing extracted frames")
    parser.add_argument("--output_file", required=True, help="Output file for commentary")
    parser.add_argument("--interval", type=int, default=5, help="Interval in seconds between frames")
    
    args = parser.parse_args()
    
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    generator = CommentaryGenerator(api_key)
    commentary = generator.generate_commentary(args.frames_dir, args.output_file, args.interval)
    print(f"Generated commentary saved to {args.output_file}")

if __name__ == "__main__":
    main() 