# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

1. **Virtual Environment**: The project uses a Python virtual environment located at `.venv/`
   - Activate: `source .venv/bin/activate`
   - Install dependencies: `pip install -r requirements.txt`
   - To recreate venv: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

2. **Ollama Setup** (required for report generation):
   - Install Ollama: https://ollama.com/download
   - Pull required model: `ollama pull gemma3:4b`
   - Verify running: `ollama list` should show gemma3:4b

3. **GPU Acceleration** (optional but recommended):
   - Install CUDA toolkit compatible with your PyTorch version
   - Verify with: `python -c "import torch; print(torch.cuda.is_available())"`
   - The project automatically uses GPU when available

## Running the Application

### Individual Processing Scripts

All scripts accept input video path and output directory as required arguments:

```bash
# Audio transcription only (Whisper)
python scripts/audio_processor.py <video_path> <output_dir> [model_size]
# model_size: tiny, base, small, medium, large (default: tiny)

# Visual captioning only (BLIP)
python scripts/vision_processor.py <video_path> <output_dir> [frame_interval]
# frame_interval: seconds between frames (default: 3)

# Full pipeline (audio + vision + indexing)
python scripts/indexer.py <video_path> <output_dir> [frame_interval] [model_size]

# Generate report from existing index
python scripts/report_gen.py <chroma_directory> <query> [n_results]
# n_results: number of segments to retrieve (default: 5)

# Launch Streamlit UI
streamlit run scripts/app.py
```

### Streamlit UI Features
- Upload video files through drag-and-drop or file selector
- Real-time progress bars for each processing stage
- Configure processing parameters (model size, frame interval)
- Interactive query interface for searching processed content
- Evidence report generation with citations
- Download options for JSON outputs and reports

## Code Architecture

### Core Pipelines
1. **Audio Processing** (`scripts/audio_processor.py`):
   - `extract_audio()`: Uses MoviePy to extract WAV audio from video
   - `transcribe_audio()`: OpenAI Whisper with word-level timestamps
   - `save_transcription_json()`: Structures output with text + segments
   - `process_video_to_transcription()`: Orchestrates pipeline with cleanup

2. **Vision Processing** (`scripts/vision_processor.py`):
   - `sample_frames()`: OpenCV frame extraction at set intervals
   - `load_blip_model()`: Salesforce BLIP model for captioning
   - `caption_frame()`: Generates description for single frame
   - `process_frames_to_captions()`: Batch processing of frames
   - `save_captions_json()`: Outputs timestamped caption array
   - `process_video_to_captions()`: Full pipeline with temp file cleanup

3. **Indexing Pipeline** (`scripts/indexer.py`):
   - `load_json_file()`: Unified loader for transcription/captions JSON
   - `merge_transcription_and_captions()`: Creates timeline-aligned entries
   - `create_embeddings()`: Sentence-transformers (all-MiniLM-L6-v2)
   - `setup_chromadb()`: Initializes persistent ChromaDB client
   - `store_in_chromadb()`: Stores embeddings with metadata
   - `process_video_to_searchable_index()`: End-to-end processing

4. **Report Generation** (`scripts/report_gen.py`):
   - `load_embedding_model()`: Loads same transformer as indexer
   - `setup_chromadb()`: Connects to existing ChromaDB collection
   - `search_chromadb()`: Vector similarity search
   - `generate_evidence_report()`: Constructs prompt for Ollama
   - `query_and_report()`: Search → report generation pipeline

### Data Flow
```
Input Video
    │
    ├──→ Audio Processing → transcription.json
    │
    └──→ Vision Processing → captions.json
            │
            └──→ Indexer → ChromaDB (unified timeline with embeddings)
                    │
                    └──→ Report Generator ← Natural Language Query
```

### Output Formats
- **Transcription JSON**: `{ "text": str, "segments": [{"start": float, "end": float, "text": str, "words": [...] }] }`
- **Captions JSON**: `[{ "timestamp": float, "caption": str, "frame_path": str }, ...]`
- **ChromaDB Metadata**: Includes `timestamp`, `type` (audio/vision), `text`, and processing metadata
- **Temporary Files**: Extracted audio (.wav) and frame images are auto-cleared after processing

## Common Development Tasks

### Testing Components
```python
# Test audio extraction
from scripts.audio_processor import extract_audio
extract_audio("test.mp4", "temp.wav")

# Test transcription
from scripts.audio_processor import transcribe_audio
result = transcribe_audio("temp.wav")

# Test frame sampling
from scripts.vision_processor import sample_frames
frames = sample_frames("test.mp4", 3)  # every 3 seconds

# Test caption generation
from scripts.vision_processor import caption_frame
caption = caption_frame("frame.jpg")

# Test full indexing pipeline
from scripts.indexer import process_video_to_searchable_index
process_video_to_searchable_index("test.mp4", "./outputs", 3, "tiny")
```

### Model Management
- **Whisper**: Change model size arg in audio_processor.py or CLI
- **BLIP**: Modify `model_name` in `load_blip_model()` (vision_processor.py)
- **Embeddings**: Update model name in `create_embeddings()` (indexer.py) and `load_embedding_model()` (report_gen.py)
- **Report LLM**: Change in `generate_evidence_report()` (report_gen.py) - uses Ollama model name
- All models lazy-loaded and cleared from memory after use

### Configuration
- Adjust defaults in function signatures or add CLI args:
  - Audio model size: audio_processor.py line ~20
  - Frame interval: vision_processor.py line ~20
  - Batch processing: Consider adding batch_size to vision processing
  - ChromaDB path: indexer.py/report_gen.py (currently uses output_dir)

### Extensibility Patterns
- To add new metadata: Update dictionaries in indexer.py:L120 and report_gen.py:L80
- To change similarity metric: Modify ChromaDB collection metadata in setup_chromadb() functions
- To add preprocessing: Insert steps in process_video_to_* functions before saving
- To support new video formats: Ensure OpenCV/MoviePy compatibility (most common formats work)

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

## Best Practices

### Code Modifications
- Maintain JSON output format compatibility when modifying processors
- Add error handling with try/finally for resource cleanup
- Keep model loading in functions (not global) for memory efficiency
- Use relative paths from script location for portability
- Add type hints to new functions following existing patterns

### Git Workflow
- Commit frequently with descriptive messages
- Use `.gitignore` to exclude:
  - `__pycache__/` directories
  - `.venv/` directory
  - `outputs/` directory (unless sharing specific results)
  - `.DS_Store` files
- Before PR: Run full pipeline on test video to verify changes

### Resource Management
- Temporary files cleaned automatically in finally blocks
- For long-running services, consider explicit model unloading
- ChromaDB persists data until explicitly deleted:
  ```bash
  rm -rf ./outputs/chroma  # Reset vector database
  ```