# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

1. **Virtual Environment**: The project uses a Python virtual environment located at `.venv/`
   - Activate: `source .venv/bin/activate`
   - Install dependencies: `pip install -r requirements.txt`

2. **Running Scripts**: All processing modules can be run independently
   - Audio processing: `python scripts/audio_processor.py <video_path> <output_dir> [model_size]`
   - Vision processing: `python scripts/vision_processor.py <video_path> <output_dir> [frame_interval]`
   - Indexing (full pipeline): `python scripts/indexer.py <video_path> <output_dir> [frame_interval] [model_size]`
   - Report generation: `python scripts/report_gen.py <chroma_directory> <query> [n_results]`
   - Streamlit UI: `streamlit run scripts/app.py`

3. **Default Parameters**:
   - Audio model size: "tiny" (Whisper model)
   - Frame interval: 3 seconds (for vision processing)
   - Number of search results: 5
   - Report LLM: gemma3:4b (via Ollama)

4. **Prerequisites**:
   - Ollama must be installed and running with gemma3:4b model: `ollama run gemma3:4b`
   - For GPU acceleration (optional): Ensure CUDA-compatible drivers and PyTorch with CUDA support

## Code Architecture

### Audio Processing Pipeline (`scripts/audio_processor.py`)
1. **extract_audio()**: Uses MoviePy to extract audio track from video file as WAV
2. **transcribe_audio()**: Uses OpenAI Whisper to transcribe audio with word-level timestamps
3. **save_transcription_json()**: Formats and saves transcription results to JSON
4. **process_video_to_transcription()**: Main pipeline function combining all steps
   - Creates output directory
   - Extracts audio → Transcribes → Saves JSON → Cleans up temporary audio file

### Vision Processing Pipeline (`scripts/vision_processor.py`)
1. **sample_frames()**: Uses OpenCV to sample video frames at specified intervals
2. **load_blip_model()**: Loads Salesforce BLIP model for image captioning
3. **caption_frame()**: Generates caption for a single frame using BLIP
4. **process_frames_to_captions()**: Processes all sampled frames through BLIP model
5. **save_captions_json()**: Formats and saves captions to JSON
6. **process_video_to_captions()**: Main pipeline function combining all steps
   - Creates output directory
   - Samples frames → Generates captions → Saves JSON → Cleans up temporary frames

### Indexing Pipeline (`scripts/indexer.py`)
1. **load_json_file()**: Loads transcription and captions JSON files
2. **merge_transcription_and_captions()**: Combines both JSONs into unified timeline entries
3. **create_embeddings()**: Generates embeddings using sentence-transformers (all-MiniLM-L6-v2)
4. **setup_chromadb()**: Initializes ChromaDB client and collection
5. **store_in_chromadb()**: Stores entries with embeddings and metadata in ChromaDB
6. **process_video_to_searchable_index()**: Main pipeline function combining all steps
   - Processes audio → Processes vision → Merges → Embeds → Stores

### Report Generation Pipeline (`scripts/report_gen.py`)
1. **load_embedding_model()**: Loads sentence-transformers model for query embedding
2. **setup_chromadb()**: Loads existing ChromaDB collection
3. **search_chromadb()**: Searches for relevant segments using query embedding
4. **generate_evidence_report()**: Creates prompt and generates report using Ollama (gemma3:4b)
5. **query_and_report()**: Main pipeline function combining search and report generation

### Data Flow
- Input: Video file (any format supported by OpenCV/MoviePy)
- Intermediate Outputs: 
  - JSON transcription file with text segments and word-level timestamps
  - JSON captions file with frame-by-frame image descriptions and timestamps
- Final Output: 
  - ChromaDB vector database containing unified timeline with embeddings
  - Evidence reports generated from natural language queries
- Temporary files: Extracted audio WAV and sampled frames are automatically cleaned up

### Streamlit UI (`scripts/app.py`)
- Provides graphical interface for video upload and processing
- Real-time progress tracking through all pipeline stages
- Interactive querying and evidence report generation
- Configuration panel for model selection and processing parameters

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

## Error Handling
- Both scripts validate input video file existence before processing
- Proper resource cleanup occurs in finally blocks (temporary files, model memory)
- Video capture objects are properly released
- CUDA memory is cleared when applicable
- Streamlit UI provides user-friendly error messages for file upload and processing failures

## Extensibility
- To add new audio models: Modify `transcribe_audio()` function parameter in audio_processor.py
- To change BLIP model: Update model name in `load_blip_model()` in vision_processor.py
- To change embedding model: Modify model name in `create_embeddings()` in indexer.py and `load_embedding_model()` in report_gen.py
- To change report LLM: Modify model name in `generate_evidence_report()` in report_gen.py
- To adjust processing parameters: Modify function arguments or add CLI options in respective scripts
- Output directories are created automatically if they don't exist
- To add new metadata fields: Update metadata dictionaries in indexer.py and report_gen.py
- To change similarity metric: Modify ChromaDB collection metadata in setup_chromadb() functions