import os
import argparse
import ffmpeg
from pathlib import Path

def extract_frames(video_path: str, output_dir: str, interval: int = 5):
    """
    Extract frames from video at regular intervals using ffmpeg-python.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save extracted frames
        interval (int): Interval in seconds between frames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Extract frames using ffmpeg-python
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.filter(stream, 'fps', fps=1/interval)
        stream = ffmpeg.output(stream, os.path.join(output_dir, 'frame_%04d.jpg'), q=2)
        ffmpeg.run(stream, overwrite_output=True)
        
        # Count the number of frames extracted
        frame_count = len([f for f in os.listdir(output_dir) if f.startswith('frame_')])
        print(f"Extracted {frame_count} frames to {output_dir}")
        
    except Exception as e:
        print(f"Error extracting frames: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video at regular intervals")
    parser.add_argument("--video_path", required=True, help="Path to input video file")
    parser.add_argument("--output_dir", required=True, help="Directory to save extracted frames")
    parser.add_argument("--interval", type=int, default=5, help="Interval in seconds between frames")
    
    args = parser.parse_args()
    
    extract_frames(args.video_path, args.output_dir, args.interval)

if __name__ == "__main__":
    main() 