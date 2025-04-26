import json
import ffmpeg
from faster_whisper import WhisperModel
import argparse
import os
from pathlib import Path
import subprocess

def extract_audio(input_video_name, video):
    
    extracted_audio = f"{input_video_name}.wav"
    stream = ffmpeg.input(video)
    stream = ffmpeg.output(stream, extracted_audio)
    ffmpeg.run(stream, overwrite_output=True)
    return extracted_audio

def extract_transcript(audio_path, model_size="small", device="cpu", compute_type="int8", output_format="json", output_dir="."):
    print(f"Loading Whisper model '{model_size}' on {device}...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    print(f"Transcribing audio: {audio_path}")
    segments, info = model.transcribe(audio_path)
    
    language = info.language
    print(f"Transcription language: {language}")
    
    segments = list(segments)
    subtitles = []
    
    for segment in segments:
        subtitles.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })
        print("[%.2fs -> %.2fs] %s" %
              (segment.start, segment.end, segment.text))
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Base name for output files (without extension)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Write subtitles in specified format
    if output_format == "json" or output_format == "all":
        json_path = os.path.join(output_dir, f"{base_name}.json")
        print(f"Writing JSON subtitles to: {json_path}")
        with open(json_path, "w") as f:
            json.dump(subtitles, f, indent=2)
    
    if output_format == "srt" or output_format == "all":
        srt_path = os.path.join(output_dir, f"{base_name}.srt")
        print(f"Writing SRT subtitles to: {srt_path}")
        with open(srt_path, "w") as f:
            for i, subtitle in enumerate(subtitles):
                start_time = format_time_srt(subtitle['start'])
                end_time = format_time_srt(subtitle['end'])
                f.write(f"{i+1}\n{start_time} --> {end_time}\n{subtitle['text']}\n\n")
    
    return language, subtitles

def format_time_srt(seconds):
    """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    secs = seconds % 60
    milliseconds = int((secs - int(secs)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{milliseconds:03d}"

def extract_subtitles(video_path, subtitles_path):
    # Ref: https://www.digitalocean.com/community/tutorials/how-to-generate-and-add-subtitles-to-videos-using-python-openai-whisper-and-ffmpeg
    ...    

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract audio and generate transcripts from videos.')
    parser.add_argument('--video', '-v', type=str, required=True,
                       help='Path to the video file')
    parser.add_argument('--output-dir', '-o', type=str, default='data',
                       help='Directory for output files (default: data)')
    parser.add_argument('--audio-only', '-a', action='store_true',
                       help='Extract audio only, skip transcription')
    parser.add_argument('--model-size', '-m', type=str, choices=['tiny', 'base', 'small', 'medium', 'large'], default='small',
                       help='Whisper model size (default: small)')
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'cuda'], default='cpu',
                       help='Device for inference (default: cpu)')
    parser.add_argument('--compute-type', '-c', type=str, choices=['int8', 'float16', 'float32'], default='int8',
                       help='Compute type for inference (default: int8)')
    parser.add_argument('--format', '-f', type=str, choices=['json', 'srt', 'all'], default='json',
                       help='Output format for subtitles (default: json)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get base name of video for output files
    video_path = args.video
    video_base = os.path.splitext(os.path.basename(video_path))[0]
    audio_output = os.path.join(args.output_dir, f"{video_base}")
    
    # Extract audio
    print(f"Extracting audio from: {video_path}")
    audio_path = extract_audio(audio_output, video_path)
    print(f"Audio extracted to: {audio_path}")
    
    # Perform transcription if not audio-only
    if not args.audio_only:
        lang, segments = extract_transcript(
            audio_path, 
            model_size=args.model_size,
            device=args.device,
            compute_type=args.compute_type,
            output_format=args.format,
            output_dir=args.output_dir
        )
        print(f"Transcription complete. Detected language: {lang}")
        print(f"Generated {len(segments)} subtitle segments")
    else:
        print("Audio extraction complete. Skipping transcription.")

if __name__ == "__main__":
    main()
