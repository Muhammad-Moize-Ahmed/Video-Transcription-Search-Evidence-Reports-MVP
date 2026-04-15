# Multimedia Processing Pipeline

This project processes video files to extract:
1. Audio transcriptions with word-level timestamps (using Whisper)
2. Visual captions with frame-level timestamps (using BLIP)

## Quick Start

### 1. Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Process a Video File

Replace `input_video.mp4` with your video file path:

#### Audio Transcription Only:
```bash
python scripts/audio_processor.py input_video.mp4 ./outputs/audio tiny
```

#### Visual Captioning Only:
```bash
python scripts/vision_processor.py input_video.mp4 ./outputs/vision 3
```

#### Both Pipelines:
```bash
# Audio processing
python scripts/audio_processor.py input_video.mp4 ./outputs/audio tiny

# Vision processing  
python scripts/vision_processor.py input_video.mp4 ./outputs/vision 3
```

### 3. Output Files

After processing, you'll find:
- `./outputs/audio/transcription.json` - Audio transcription with timestamps
- `./outputs/vision/captions.json` - Frame captions with timestamps

## Features

### Audio Processing
- Extracts audio from video using MoviePy
- Transcribes using OpenAI Whisper (tiny model by default)
- Outputs JSON with full text, segments, and word-level timestamps
- Automatic cleanup of temporary audio files

### Vision Processing
- Samples frames at configurable intervals (default: 3 seconds)
- Generates captions using Salesforce BLIP model
- Outputs JSON array with timestamp, caption, and frame path for each sampled frame
- Automatic cleanup of temporary frame directories

## Model Options

### Whisper Models (audio_processor.py)
- `tiny` - Fastest, lowest accuracy
- `base` - Good balance
- `small` - Better accuracy
- `medium` - High accuracy
- `large` - Highest accuracy, slowest

### BLIP Model (vision_processor.py)
- Currently uses: `Salesforce/blip-image-captioning-base`
- To change: Modify the model name in `load_blip_model()`

## Directory Structure
```
multimedia-processing/
├── .venv/                  # Virtual environment
├── data/                   # Place input videos here
├── outputs/                # Processing results go here
├── scripts/
│   ├── audio_processor.py  # Audio transcription pipeline
│   └── vision_processor.py # Visual captioning pipeline
├── requirements.txt        # Python dependencies
├── CLAUDE.md               # AI assistant guidance
└── README.md               # This file
```

## Notes
- First model download may take several minutes (hundreds of MBs)
- GPU acceleration is used automatically if available
- Processing time depends on video length and model selection
- Both pipelines handle errors gracefully and clean up temporary files