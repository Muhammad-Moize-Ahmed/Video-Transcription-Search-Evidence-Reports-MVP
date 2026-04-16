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

#### Full Pipeline (Audio + Vision + Indexing):
```bash
python scripts/indexer.py input_video.mp4 ./outputs 3 tiny
```

#### Generate Report from Existing Index:
```bash
python scripts/report_gen.py ./outputs/chroma_db "Your question here" 5
```

#### Launch Streamlit UI:
```bash
streamlit run scripts/app.py
```

### 3. Output Files

After processing, you'll find:
- `./outputs/audio/transcription.json` - Audio transcription with timestamps
- `./outputs/vision/captions.json` - Frame captions with timestamps
- `./outputs/chroma_db/` - ChromaDB vector database (if using full pipeline or indexer)

## Features

### Audio Processing
- Extracts audio from video using MoviePy
- Transcribes using OpenAI Whisper (tiny model by default)
- Outputs JSON with full text, segments, and word-level timestamps
- **Memory Efficient**: Whisper model is unloaded from memory after transcription
- Automatic cleanup of temporary audio files

### Vision Processing
- Samples frames at configurable intervals (default: 3 seconds)
- Generates captions using Salesforce BLIP model
- Outputs JSON array with timestamp, caption, and frame path for each sampled frame
- **Memory Efficient**: BLIP model and processor are unloaded from memory after captioning, and CUDA cache is cleared if applicable
- Automatic cleanup of temporary frame directories

### Indexing Pipeline
- Combines audio transcriptions and visual captions into a unified timeline
- Creates embeddings using sentence-transformers (all-MiniLM-L6-v2)
- Stores entries with embeddings and metadata in ChromaDB for efficient search
- **Memory Efficient**: Embedding model is unloaded after creating embeddings

### Report Generation
- Searches ChromaDB for relevant segments using query embedding
- Generates evidence-based reports using Ollama (Gemma3:4b by default)
- **Memory Efficient**: Embedding model is unloaded after report generation

### Streamlit UI
- Provides graphical interface for video upload and processing
- Real-time progress tracking through all pipeline stages
- Interactive querying and evidence report generation
- Configuration panel for model selection and processing parameters
- Designed for CPU-only systems with sequential processing to conserve memory

## Model Options

### Whisper Models (audio_processor.py)
- `tiny` - Fastest, lowest accuracy (~75 MB)
- `base` - Good balance (~150 MB)
- `small` - Better accuracy (~240 MB)
- `medium` - High accuracy (~768 MB)
- `large` - Highest accuracy, slowest (~1550 MB)
- **Tip**: For CPU-only systems, `tiny` or `base` models are recommended for faster processing.

### BLIP Model (vision_processor.py)
- Currently uses: `Salesforce/blip-image-captioning-base`
- To change: Modify the model name in `load_blip_model()`

### Embedding Model (indexer.py & report_gen.py)
- Currently uses: `sentence-transformers/all-MiniLM-L6-v2`
- To change: Modify model name in `create_embeddings()` (indexer.py) and `load_embedding_model()` (report_gen.py)

### Report LLM (report_gen.py)
- Currently uses: `gemma3:4b` via Ollama
- To change: Modify model name in `generate_evidence_report()`

## Directory Structure
```
multimedia-processing/
├── .venv/                  # Virtual environment
├── data/                   # Place input videos here
├── outputs/                # Processing results go here
│   ├── audio/              # Audio transcription outputs
│   ├── vision/             # Vision caption outputs
│   └── chroma_db/          # ChromaDB vector database (created by indexer)
├── scripts/
│   ├── audio_processor.py  # Audio transcription pipeline
│   ├── vision_processor.py # Visual captioning pipeline
│   ├── indexer.py          # Full pipeline (audio + vision + indexing)
│   ├── report_gen.py       # Report generation from existing index
│   └── app.py              # Streamlit UI
├── requirements.txt        # Python dependencies
├── CLAUDE.md               # AI assistant guidance
└── README.md               # This file
```

## Notes
- First model download may take several minutes (hundreds of MBs for BLIP, ~75MB for Whisper tiny)
- GPU acceleration is used automatically if available (CUDA-compatible drivers and PyTorch with CUDA support)
- Processing time depends on video length and model selection
- Both pipelines handle errors gracefully and clean up temporary files
- Models are loaded on-demand and cleared from memory after use to conserve resources (especially important for CPU-only or memory-constrained systems)
- For optimal performance on limited RAM, process shorter videos or use smaller models

## Common Development Tasks

### Testing Individual Components
- Test audio extraction: Call `extract_audio()` with a video file path
- Test transcription: Call `transcribe_audio()` on an audio file
- Test frame sampling: Call `sample_frames()` with video path and interval
- Test caption generation: Call `caption_frame()` on an image file
- Test indexing: Process a short video through `process_video_to_searchable_index()`
- Test report generation: Query an existing ChromaDB with `query_and_report()`

### Model Management
- Whisper models: tiny, base, small, medium, large (trade-off between speed and accuracy)
- BLIP model: Uses "Salesforce/blip-image-captioning-base" by default
- Embedding model: Uses "sentence-transformers/all-MiniLM-L6-v2" by default
- Report LLM: Uses "gemma3:4b" via Ollama by default
- All models are loaded on-demand and cleared from memory after use to conserve resources

### Output Format
Both processors produce JSON files with consistent structure:
- Audio transcription: Contains "text" (full transcript) and "segments" array with start/end times
- Vision captions: Contains "captions" array with timestamp, caption text, and frame path
- Unified timeline entries: Contains timestamp, text, type (audio/vision), and metadata
- ChromaDB storage: Stores embeddings with metadata including timestamp and type

### Extensibility
- To add new audio models: Modify `transcribe_audio()` function parameter in audio_processor.py
- To change BLIP model: Update model name in `load_blip_model()` in vision_processor.py
- To change embedding model: Modify model name in `create_embeddings()` in indexer.py and `load_embedding_model()` in report_gen.py
- To change report LLM: Modify model name in `generate_evidence_report()` in report_gen.py
- To adjust processing parameters: Modify function arguments or add CLI options in respective scripts
- Output directories are created automatically if they don't exist
- To add new metadata fields: Update metadata dictionaries in indexer.py and report_gen.py
- To change similarity metric: Modify ChromaDB collection metadata in setup_chromadb() functions

## Troubleshooting

### Common Issues
1. **Ollama not running**:
   - Symptoms: Connection errors in report_gen.py
   - Fix: `ollama serve` in background, then `ollama run gemma3:4b`

2. **GPU memory errors**:
   - Symptoms: CUDA out of memory during BLIP/Whisper loading
   - Fix: Reduce batch size, use smaller models, or disable GPU:
     ```bash
     export CUDA_VISIBLE_DEVICES=""  # CPU-only mode
     ```

3. **Missing dependencies**:
   - Symptoms: ImportError on module not found
   - Fix: `pip install -r requirements.txt --upgrade`

4. **Video format issues**:
   - Symptoms: OpenCV/MoviePy cannot read file
   - Fix: Convert video to MP4/H.264 using ffmpeg:
     ```bash
     ffmpeg -i input.mov -vcodec libx264 output.mp4
     ```

5. **ChromaDB lock errors**:
   - Symptoms: Database access conflicts
   - Fix: Ensure only one process accesses the DB at a time, or use different directories

### Performance Tips
- First model load downloads weights (~1GB for BLIP, ~75MB for Whisper tiny)
- Subsequent runs use cached models
- Process short videos (<5 min) for initial testing
- Use `tiny` Whisper model for fastest iteration
- Increase frame interval for faster vision processing (lower temporal resolution)
- Monitor GPU usage with `nvidia-smi` during processing