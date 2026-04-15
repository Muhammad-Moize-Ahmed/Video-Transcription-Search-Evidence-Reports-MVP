"""
Streamlit Application for Video Transcription Search & Evidence Reports MVP
Provides UI to upload video, process through pipeline, search, and view evidence reports.
"""

import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
import time

# Import our processing modules
from audio_processor import process_video_to_transcription
from vision_processor import process_video_to_captions
from indexer import process_video_to_searchable_index
from report_gen import query_and_report

# Page configuration
st.set_page_config(
    page_title="Video Transcription Search & Evidence Reports",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🎥 Video Transcription Search & Evidence Reports MVP")
st.markdown("""
Convert videos into searchable multimodal transcripts (audio + visual) and generate timestamped evidence reports.
**Hardware Note:** Designed for CPU-only systems with sequential processing to conserve memory.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Audio settings
    st.subheader("Audio Processing")
    audio_model = st.selectbox(
        "Whisper Model Size",
        options=["tiny", "base", "small", "medium", "large"],
        index=0,  # Default to tiny
        help="Smaller models are faster but less accurate. 'tiny' recommended for CPU-only systems."
    )

    # Vision settings
    st.subheader("Vision Processing")
    frame_interval = st.slider(
        "Frame Interval (seconds)",
        min_value=1,
        max_value=10,
        value=3,
        help="Interval between sampled frames for visual captioning"
    )

    # Search settings
    st.subheader("Search Settings")
    n_results = st.slider(
        "Number of Search Results",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of relevant segments to retrieve for evidence reports"
    )

    # Model info
    st.subheader("📊 Model Information")
    st.info("""
    **Audio:** OpenAI Whisper (tiny)
    **Vision:** Salesforce BLIP (base)
    **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
    **Vector DB:** ChromaDB
    **Report LLM:** Ollama (gemma3:4b)
    """)

# Main interface
tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "🔍 Search & Report", "📊 About"])

with tab1:
    st.header("Upload Video for Processing")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
        help="Supported formats: MP4, AVI, MOV, MKV, WEBM"
    )

    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "FileSize": f"{uploaded_file.size / (1024*1024):.2f} MB",
            "FileType": uploaded_file.type
        }

        st.write("### File Details")
        for key, value in file_details.items():
            st.write(f"**{key}:** {value}")

        # Process button
        if st.button("🚀 Process Video", type="primary"):
            # Create temporary directory for the video file only
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded file temporarily
                video_path = os.path.join(temp_dir, uploaded_file.name)
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Output directory - persistent in outputs folder
                output_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
                os.makedirs(output_root, exist_ok=True)
                
                # Create unique directory for this video based on filename and timestamp
                sanitized_name = Path(uploaded_file.name).stem.replace(" ", "_")
                output_dir = os.path.join(output_root, f"{sanitized_name}_{int(time.time())}")
                os.makedirs(output_dir, exist_ok=True)

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Step 1: Audio Processing
                    status_text.text("🔊 Step 1/4: Extracting audio and transcribing...")
                    progress_bar.progress(25)

                    transcription_path = process_video_to_transcription(
                        video_path,
                        output_dir,
                        model_size=audio_model
                    )

                    # Step 2: Vision Processing
                    status_text.text("👁️ Step 2/4: Sampling frames and generating captions...")
                    progress_bar.progress(50)

                    captions_path = process_video_to_captions(
                        video_path,
                        output_dir,
                        frame_interval=frame_interval
                    )

                    # Step 3: Indexing
                    status_text.text("🔍 Step 3/4: Creating embeddings and building search index...")
                    progress_bar.progress(75)

                    chroma_dir = process_video_to_searchable_index(
                        video_path,
                        output_dir,
                        frame_interval=frame_interval,
                        model_size=audio_model
                    )

                    # Step 4: Complete
                    status_text.text("✅ Step 4/4: Processing complete!")
                    progress_bar.progress(100)

                    # Store results in session state for search tab
                    st.session_state['video_processed'] = True
                    st.session_state['chroma_directory'] = chroma_dir
                    st.session_state['video_name'] = uploaded_file.name
                    st.session_state['processing_time'] = time.time()

                    st.success(f"🎉 Video '{uploaded_file.name}' processed successfully!")
                    st.info(f"Searchable index saved to: `{chroma_dir}`")
                    st.info("💡 This data is persistent and will remain in your outputs folder. Go to the 'Search & Report' tab to ask questions about your video.")
                    st.info(f"📁 Processed data location: `{output_dir}`")

                except Exception as e:
                    st.error(f"❌ Error processing video: {str(e)}")
                    st.exception(e)

with tab2:
    st.header("Search & Generate Evidence Reports")

    # Check if video has been processed
    if 'video_processed' not in st.session_state or not st.session_state['video_processed']:
        st.warning("⚠️ Please upload and process a video first in the 'Upload & Process' tab.")
    else:
        st.info(f"📹 Currently loaded video: **{st.session_state['video_name']}**")

        # Query input
        query = st.text_input(
            "🔍 Enter your question about the video:",
            placeholder="e.g., What is discussed about climate change? Who are the main speakers?",
            help="Ask any question about the video content. The system will search relevant segments and generate an evidence-based report."
        )

        # Search button
        if st.button("📊 Generate Evidence Report", type="primary") and query:
            with st.spinner("🔍 Searching video content and generating report..."):
                try:
                    # Generate report using our pipeline
                    result = query_and_report(
                        st.session_state['chroma_directory'],
                        query,
                        n_results=n_results
                    )

                    # Display results
                    st.success("✅ Evidence report generated!")

                    # Report header
                    st.subheader("📋 Evidence Report")
                    st.write(f"**Query:** {result['query']}")

                    # Report content
                    st.markdown(result['report'])

                    # Sources consulted
                    with st.expander(f"🔍 View Sources Consulted ({len(result['search_results'])})", expanded=False):
                        for i, source in enumerate(result['search_results'], 1):
                            # Format timestamps
                            def format_timestamp(seconds):
                                hours = int(seconds // 3600)
                                minutes = int((seconds % 3600) // 60)
                                secs = int(seconds % 60)
                                if hours > 0:
                                    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
                                else:
                                    return f"{minutes:02d}:{secs:02d}"

                            start_time = format_timestamp(source['timestamp'])
                            end_time = format_timestamp(source.get('end_time', source['timestamp']))

                            st.write(f"**[{i}]** [{start_time} - {end_time}] ({source['type'].upper()})")
                            st.write(f"> {source['text']}")
                            st.write(f"*Relevance: {source['relevance_score']:.3f}*")
                            st.divider()

                except Exception as e:
                    st.error(f"❌ Error generating report: {str(e)}")
                    st.exception(e)

with tab3:
    st.header("About This MVP")

    st.markdown("""
    ## Video Transcription Search & Evidence Reports

    This Minimum Viable Product (MVP) demonstrates how to convert videos into searchable multimodal transcripts
    and generate timestamped evidence reports using open-source AI models.

    ### 🏗️ Architecture

    The system consists of five modular components:

    1. **Audio Processor** (`audio_processor.py`)
       - Extracts audio from video using MoviePy
       - Transcribes audio using OpenAI Whisper (tiny model)
       - Saves timestamped transcription to JSON

    2. **Vision Processor** (`vision_processor.py`)
       - Samples frames from video using OpenCV (1 frame every 3 seconds by default)
       - Generates captions for each frame using Salesforce BLIP model
       - Saves timestamped captions to JSON

    3. **Indexer** (`indexer.py`)
       - Merges audio transcripts and vision captions into unified timeline
       - Creates embeddings using sentence-transformers (all-MiniLM-L6-v2)
       - Stores in ChromaDB vector database for efficient search

    4. **Report Generator** (`report_gen.py`)
       - Searches ChromaDB for relevant transcript segments based on user query
       - Uses retrieved segments as context for Ollama with Gemma3:4b
       - Generates evidence-based reports with cited timestamps

    5. **Streamlit UI** (`app.py`)
       - Provides user-friendly interface for video upload and processing
       - Displays processing progress and results
       - Enables interactive querying and evidence report generation

    ### ⚙️ Hardware Constraints & Optimizations

    - **Memory Efficient:** Sequential processing - loads one model at a time, processes, then deletes from memory
    - **CPU Optimized:** Uses smaller models (Whisper tiny, BLIP base) suitable for 8GB RAM systems
    - **No GPU Required:** All processing runs on CPU
    - **Temporary Files:** Automatically cleans up extracted audio and sampled frames

    ### 📁 Data Flow

    ```
    Input Video
        → Audio Processor → Transcription JSON
        → Vision Processor → Captions JSON
        → Indexer → Unified Timeline + Embeddings → ChromaDB
        → Query → Report Generator → Evidence Report
    ```

    ### 🔧 Setup Instructions

    1. Install dependencies: `pip install -r requirements.txt`
    2. Ensure Ollama is running with gemma3:4b model: `ollama run gemma3:4b`
    3. Run the application: `streamlit run scripts/app.py`

    ### ⚠️ Limitations

    - Processing time depends on video length and system performance
    - Whisper tiny model trade-offs speed for accuracy
    - BLIP model may generate inaccurate captions for complex scenes
    - Evidence reports are based solely on retrieved video content
    """)

    # System status
    st.subheader("💻 System Status")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("RAM Usage", "Optimized", help="Sequential processing minimizes peak memory usage")
    with col2:
        st.metric("GPU Required", "No", help="All models run on CPU")
    with col3:
        st.metric("Models Loaded", "1 at a time", help="Sequential loading/unloading to conserve memory")

# Footer
st.divider()
st.caption("Video Transcription Search & Evidence Reports MVP • Built with Streamlit • CPU-Optimized Processing")