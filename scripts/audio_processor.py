"""
Audio Processor Module
Extracts audio from video and transcribes using Whisper tiny model.
Saves timestamped transcription to JSON.
"""

import os
import json
import whisper
from moviepy.editor import VideoFileClip
from tqdm import tqdm


def extract_audio(video_path, audio_path):
    """
    Extract audio from video file using moviepy.

    Args:
        video_path (str): Path to input video file
        audio_path (str): Path to save extracted audio
    """
    print(f"Extracting audio from {video_path}...")
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    video.close()
    print(f"Audio saved to {audio_path}")


def transcribe_audio(audio_path, model_size="tiny"):
    """
    Transcribe audio using Whisper model.

    Args:
        audio_path (str): Path to audio file
        model_size (str): Whisper model size (default: "tiny")

    Returns:
        dict: Transcription result with timestamps
    """
    print(f"Loading Whisper {model_size} model...")
    model = whisper.load_model(model_size)

    print("Transcribing audio...")
    result = model.transcribe(audio_path, word_timestamps=False)

    # Clean up model from memory
    del model

    return result


def save_transcription_json(transcription_result, output_path):
    """
    Save transcription result to JSON file with timestamped segments.

    Args:
        transcription_result (dict): Result from whisper.transcribe
        output_path (str): Path to save JSON file
    """
    # Format output for consistency with vision processor
    formatted_result = {
        "text": transcription_result["text"],
        "segments": []
    }

    for segment in transcription_result["segments"]:
        formatted_result["segments"].append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip(),
            "words": [
                {
                    "word": word["word"],
                    "start": word["start"],
                    "end": word["end"]
                }
                for word in segment.get("words", [])
            ]
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_result, f, indent=2, ensure_ascii=False)

    print(f"Transcription saved to {output_path}")


def process_video_to_transcription(video_path, output_dir, model_size="tiny"):
    """
    Complete pipeline: extract audio, transcribe, save JSON.

    Args:
        video_path (str): Path to input video file
        output_dir (str): Directory to save outputs
        model_size (str): Whisper model size

    Returns:
        str: Path to transcription JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define file paths
    audio_path = os.path.join(output_dir, "extracted_audio.wav")
    transcription_path = os.path.join(output_dir, "transcription.json")

    try:
        # Step 1: Extract audio
        extract_audio(video_path, audio_path)

        # Step 2: Transcribe audio
        transcription_result = transcribe_audio(audio_path, model_size)

        # Step 3: Save transcription
        save_transcription_json(transcription_result, transcription_path)

        return transcription_path

    finally:
        # Clean up extracted audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"Cleaned up temporary audio file: {audio_path}")


if __name__ == "__main__":
    # Example usage (for testing)
    import sys

    if len(sys.argv) < 3:
        print("Usage: python audio_processor.py <video_path> <output_dir> [model_size]")
        print("Example: python audio_processor.py video.mp4 ./output tiny")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = sys.argv[2]
    model_size = sys.argv[3] if len(sys.argv) > 3 else "tiny"

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    process_video_to_transcription(video_path, output_dir, model_size)