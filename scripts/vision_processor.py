"""
Vision Processor Module
Samples frames from video and generates captions using BLIP model.
Saves timestamped captions to JSON.
"""

import os
import json
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from tqdm import tqdm


def sample_frames(video_path, output_dir, frame_interval=3):
    """
    Sample frames from video at specified interval (in seconds).

    Args:
        video_path (str): Path to input video file
        output_dir (str): Directory to save sampled frames
        frame_interval (int): Interval in seconds between frames (default: 3)

    Returns:
        list: List of tuples (timestamp, frame_path)
    """
    print(f"Sampling frames from {video_path} every {frame_interval} seconds...")

    # Create frames directory
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video FPS: {fps}, Duration: {duration:.2f}s, Total frames: {total_frames}")

    # Calculate frame interval in frames
    frame_step = int(fps * frame_interval)
    if frame_step < 1:
        frame_step = 1

    sampled_frames = []
    frame_count = 0

    with tqdm(total=total_frames, desc="Sampling frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frame at intervals
            if frame_count % frame_step == 0:
                timestamp = frame_count / fps
                frame_filename = f"frame_{int(timestamp):06d}.jpg"
                frame_path = os.path.join(frames_dir, frame_filename)

                # Save frame as image
                cv2.imwrite(frame_path, frame)
                sampled_frames.append((timestamp, frame_path))

            frame_count += 1
            pbar.update(1)

    cap.release()
    print(f"Sampled {len(sampled_frames)} frames")
    return sampled_frames


def load_blip_model():
    """
    Load BLIP processor and model for image captioning.

    Returns:
        tuple: (processor, model)
    """
    print("Loading BLIP model (Salesforce/blip-image-captioning-base)...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Move to appropriate device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"BLIP model loaded on {device}")

    return processor, model, device


def caption_frame(image_path, processor, model, device):
    """
    Generate caption for a single image using BLIP.

    Args:
        image_path (str): Path to image file
        processor: BLIP processor
        model: BLIP model
        device: torch device

    Returns:
        str: Generated caption
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)

    # Generate caption
    with torch.no_grad():
        out = model.generate(**inputs, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption.strip()


def process_frames_to_captions(frame_list, output_dir):
    """
    Process sampled frames to generate captions.

    Args:
        frame_list (list): List of tuples (timestamp, frame_path)
        output_dir (str): Directory to save outputs

    Returns:
        list: List of caption dictionaries with timestamps
    """
    print("Loading BLIP model for captioning...")
    processor, model, device = load_blip_model()

    captions = []

    try:
        print("Generating captions for frames...")
        for timestamp, frame_path in tqdm(frame_list, desc="Captioning frames"):
            caption = caption_frame(frame_path, processor, model, device)
            captions.append({
                "timestamp": timestamp,
                "caption": caption,
                "frame_path": frame_path
            })
    finally:
        # Clean up model from memory
        del model
        del processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("BLIP model cleared from memory")

    return captions


def save_captions_json(captions, output_path):
    """
    Save captions to JSON file.

    Args:
        captions (list): List of caption dictionaries
        output_path (str): Path to save JSON file
    """
    # Format output for consistency
    formatted_result = {
        "captions": captions
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_result, f, indent=2, ensure_ascii=False)

    print(f"Captions saved to {output_path}")


def process_video_to_captions(video_path, output_dir, frame_interval=3):
    """
    Complete pipeline: sample frames, generate captions, save JSON.

    Args:
        video_path (str): Path to input video file
        output_dir (str): Directory to save outputs
        frame_interval (int): Interval in seconds between frames

    Returns:
        str: Path to captions JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define file paths
    captions_path = os.path.join(output_dir, "captions.json")

    try:
        # Step 1: Sample frames
        frame_list = sample_frames(video_path, output_dir, frame_interval)

        # Step 2: Generate captions
        captions = process_frames_to_captions(frame_list, output_dir)

        # Step 3: Save captions
        save_captions_json(captions, captions_path)

        return captions_path

    finally:
        # Clean up frames directory
        frames_dir = os.path.join(output_dir, "frames")
        if os.path.exists(frames_dir):
            import shutil
            shutil.rmtree(frames_dir)
            print(f"Cleaned up temporary frames directory: {frames_dir}")


if __name__ == "__main__":
    # Example usage (for testing)
    import sys

    if len(sys.argv) < 3:
        print("Usage: python vision_processor.py <video_path> <output_dir> [frame_interval]")
        print("Example: python vision_processor.py video.mp4 ./output 3")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = sys.argv[2]
    frame_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    process_video_to_captions(video_path, output_dir, frame_interval)