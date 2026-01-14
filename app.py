import streamlit as st
import os
import tempfile
import time
import logging
from typing import List, Dict, Any
import uuid
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.document_processing.doc_processor import DocumentProcessor
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vector_database.milvus_vector_db import MilvusVectorDB
from src.podcast.script_generator import PodcastScriptGenerator
from src.podcast.text_to_speech import PodcastTTSGenerator

st.set_page_config(
    page_title="Research Paper to Podcast",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 24px;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 20px;
    }
    
    .source-item {
        background: #2d3748;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-left: 3px solid #4299e1;
    }
    
    .source-title {
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 4px;
    }
    
    .source-meta {
        font-size: 12px;
        color: #a0aec0;
    }
    
    .chat-message {
        background: #2d3748;
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
    }
    
    .user-message {
        background: #4299e1;
        margin-left: 20%;
    }
    
    .assistant-message {
        background: #2d3748;
        margin-right: 20%;
        border-left: 3px solid #48bb78;
    }
    
    .citation {
        background: #1a202c;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 11px;
        color: #90cdf4;
        margin: 2px;
        display: inline-block;
    }
    
    .upload-area {
        border: 2px dashed #4a5568;
        border-radius: 12px;
        padding: 40px;
        text-align: center;
        background: #1a202c;
        margin: 20px 0;
    }
    
    .upload-text {
        color: #a0aec0;
        font-size: 16px;
        margin-bottom: 20px;
    }
    
    .stButton > button {
        background: #4299e1;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 8px 24px;
        font-weight: 500;
    }
    
    .source-count {
        background: #4a5568;
        color: #ffffff;
        border-radius: 12px;
        padding: 4px 12px;
        font-size: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'sources' not in st.session_state:
        st.session_state.sources = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'show_source_dialog' not in st.session_state:
        st.session_state.show_source_dialog = False
    if 'pipeline_initialized' not in st.session_state:
        st.session_state.pipeline_initialized = False

def initialize_pipeline():
    if st.session_state.pipeline_initialized:
        return True
    
    try:
        openai_key = os.getenv("OPENAI_API_KEY")

        with st.spinner("Initializing Podcast Generator..."):
            # Core components for podcast generation
            doc_processor = DocumentProcessor()
            podcast_script_generator = PodcastScriptGenerator(openai_key) if openai_key else None

            try:
                podcast_tts_generator = PodcastTTSGenerator()
                logger.info("PodcastTTSGenerator initialized successfully")
            except ImportError:
                logger.warning("Kokoro TTS not available. Podcast audio generation will be disabled.")
                podcast_tts_generator = None
            except Exception as e:
                logger.error(f"Error initializing TTS: {e}")
                podcast_tts_generator = None

            # Optional: Vector DB for future features (not needed for basic podcast generation)
            vector_db = None
            embedding_generator = None
            try:
                embedding_generator = EmbeddingGenerator()
                vector_db = MilvusVectorDB(
                    db_path=f"./milvus_lite_{st.session_state.session_id[:8]}.db",
                    collection_name=f"collection_{st.session_state.session_id[:8]}"
                )
                logger.info("Vector database initialized")
            except Exception as e:
                logger.warning(f"Vector database unavailable (this is expected on Windows): {e}")

            st.session_state.pipeline = {
                'doc_processor': doc_processor,
                'podcast_script_generator': podcast_script_generator,
                'podcast_tts_generator': podcast_tts_generator,
                'embedding_generator': embedding_generator,
                'vector_db': vector_db,
            }
            
            st.session_state.pipeline_initialized = True
            st.success("✅ Pipeline initialized successfully!")
            return True
            
    except Exception as e:
        st.error(f"❌ Failed to initialize pipeline: {str(e)}")
        return False

def process_uploaded_files(uploaded_files):
    if not st.session_state.pipeline:
        return
    
    pipeline = st.session_state.pipeline

    with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
        for uploaded_file in uploaded_files:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    temp_path = tmp_file.name

                # Process document (PDF only now)
                chunks = pipeline['doc_processor'].process_document(temp_path)
                source_type = "Document"

                for chunk in chunks:
                    chunk.source_file = uploaded_file.name
                
                if chunks:
                    # If vector DB is available, use it
                    if pipeline['vector_db']:
                        embedded_chunks = pipeline['embedding_generator'].generate_embeddings(chunks)

                        if len(st.session_state.sources) == 0:
                            pipeline['vector_db'].create_index(use_binary_quantization=False)

                        pipeline['vector_db'].insert_embeddings(embedded_chunks)
                    else:
                        # If vector DB is unavailable, store chunks in session state
                        if 'source_chunks' not in st.session_state:
                            st.session_state.source_chunks = {}
                        st.session_state.source_chunks[uploaded_file.name] = chunks

                    source_info = {
                        'name': uploaded_file.name,
                        'type': source_type,
                        'size': f"{len(uploaded_file.getbuffer()) / 1024:.1f} KB",
                        'chunks': len(chunks),
                        'uploaded_at': time.strftime("%Y-%m-%d %H:%M")
                    }
                    st.session_state.sources.append(source_info)
                    st.success(f"✅ Processed {uploaded_file.name}: {len(chunks)} chunks")
                
                os.unlink(temp_path)
                
            except Exception as e:
                st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
                if 'temp_path' in locals():
                    os.unlink(temp_path)

def process_text(text_content):
    if not st.session_state.pipeline or not text_content.strip():
        return
    
    pipeline = st.session_state.pipeline
    
    with st.spinner("Processing text..."):
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
                tmp_file.write(text_content)
                temp_path = tmp_file.name
            
            chunks = pipeline['doc_processor'].process_document(temp_path)
            
            original_name = f"Pasted Text ({time.strftime('%H:%M')})"
            for chunk in chunks:
                chunk.source_file = original_name
            
            if chunks:
                # If vector DB is available, use it
                if pipeline['vector_db']:
                    embedded_chunks = pipeline['embedding_generator'].generate_embeddings(chunks)

                    if len(st.session_state.sources) == 0:
                        pipeline['vector_db'].create_index(use_binary_quantization=False)

                    pipeline['vector_db'].insert_embeddings(embedded_chunks)
                else:
                    # If vector DB is unavailable, store chunks in session state
                    if 'source_chunks' not in st.session_state:
                        st.session_state.source_chunks = {}
                    st.session_state.source_chunks[original_name] = chunks

                source_info = {
                    'name': original_name,
                    'type': "Text",
                    'size': f"{len(text_content)} chars",
                    'chunks': len(chunks),
                    'uploaded_at': time.strftime("%Y-%m-%d %H:%M")
                }
                st.session_state.sources.append(source_info)
                st.success(f"✅ Processed text: {len(chunks)} chunks")
            
            os.unlink(temp_path)
            
        except Exception as e:
            st.error(f"❌ Failed to process text: {str(e)}")

def render_sources_sidebar():
    with st.sidebar:
        st.markdown('<div class="main-header">📚 Sources</div>', unsafe_allow_html=True)
        
        # Display sources
        if st.session_state.sources:
            st.markdown(f'<div class="source-count">{len(st.session_state.sources)} sources</div>', unsafe_allow_html=True)
            
            for i, source in enumerate(st.session_state.sources):
                with st.container():
                    st.markdown(f'''
                    <div class="source-item">
                        <div class="source-title">{source['name']}</div>
                        <div class="source-meta">{source['type']} • {source['size']} • {source['chunks']} chunks</div>
                        <div class="source-meta">{source['uploaded_at']}</div>
                    </div>
                    ''', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 20px; color: #a0aec0;">
                <p>Your sources will appear here</p>
                <p style="font-size: 14px;">Upload a PDF or paste text in the "Input" tab to get started</p>
            </div>
            """, unsafe_allow_html=True)

def render_source_upload_dialog():
    st.markdown("### 📄 Add Your Research Paper")
    st.markdown("""
    Upload a PDF or paste text from your research paper, abstract, or article.
    We'll transform it into an engaging 2-person podcast conversation!
    """)

    # Create two columns for input methods
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📄 Upload PDF")
        uploaded_files = st.file_uploader(
            "Drag & drop or choose PDF file",
            accept_multiple_files=True,
            type=['pdf'],
            help="Upload PDF documents (research papers, articles, etc.)"
        )

        if uploaded_files:
            if st.button("📤 Process PDF", use_container_width=True):
                process_uploaded_files(uploaded_files)
                st.rerun()

    with col2:
        st.markdown("#### 📋 Paste Text")
        text_content = st.text_area(
            "Paste your research paper or abstract",
            placeholder="Paste your text here...\n\nYou can paste:\n• Research paper abstracts\n• Article excerpts\n• Study findings\n• Any text content",
            height=200
        )
        if st.button("📤 Process Text", key="text_btn", use_container_width=True) and text_content.strip():
            process_text(text_content)
            st.rerun()

def generate_podcast(selected_source: str, podcast_style: str, podcast_length: str):
    if not st.session_state.pipeline or not st.session_state.pipeline['podcast_script_generator']:
        st.error("Podcast generation not available. Please check your OpenAI API key.")
        return

    pipeline = st.session_state.pipeline

    # Store in session state to prevent regeneration on download
    st.session_state.current_podcast = {
        'source': selected_source,
        'style': podcast_style,
        'length': podcast_length,
        'script': None,
        'audio_files': None
    }

    try:
        source_info = None
        for source in st.session_state.sources:
            if source['name'] == selected_source:
                source_info = source
                break
        
        if not source_info:
            st.error(f"Could not find source: {selected_source}")
            return
        
        # Gather content from the selected source
        with st.spinner(f"📚 Gathering content from {selected_source}..."):
            try:
                search_results = []

                # If vector DB is available, use it
                if pipeline['vector_db']:
                    query_embedding = pipeline['embedding_generator'].generate_query_embedding(f"content from {selected_source}")
                    search_results = pipeline['vector_db'].search(
                        query_embedding,
                        limit=50,
                        filter_expr=f'source_file == "{selected_source}"'
                    )

                    if not search_results:
                        st.error(f"Could not find content for {selected_source}. Please try again.")
                        return

                    search_results.sort(key=lambda x: x.get('chunk_index', 0))

                # If vector DB is unavailable, use stored chunks
                else:
                    if 'source_chunks' in st.session_state and selected_source in st.session_state.source_chunks:
                        chunks = st.session_state.source_chunks[selected_source]
                        # Convert chunks to search result format
                        search_results = [{'content': chunk.content if hasattr(chunk, 'content') else str(chunk)} for chunk in chunks]
                    else:
                        st.error(f"Could not find content for {selected_source}. Please upload the source again.")
                        return

            except Exception as e:
                st.error(f"Error retrieving content from {selected_source}: {e}")
                return
        
        with st.spinner("✍️ Generating podcast script..."):
            script_generator = pipeline['podcast_script_generator']
            
            if source_info['type'] == 'Website':
                # For websites, use the specialized website method
                from dataclasses import dataclass
                
                @dataclass
                class ChunkLike:
                    content: str
                
                chunks = [ChunkLike(content=result['content']) for result in search_results]
                
                podcast_script = script_generator.generate_script_from_website(
                    website_chunks=chunks,
                    source_url=selected_source,
                    podcast_style=podcast_style.lower(),
                    target_duration=podcast_length
                )
            else:
                # For documents, audio, text, etc., use the text method
                combined_content = "\n\n".join([result['content'] for result in search_results])
                
                podcast_script = script_generator.generate_script_from_text(
                    text_content=combined_content,
                    source_name=selected_source,
                    podcast_style=podcast_style.lower(),
                    target_duration=podcast_length
                )
            
            st.success(f"✅ Generated podcast script with {podcast_script.total_lines} dialogue segments!")

            # Store script in session state
            st.session_state.current_podcast['script'] = podcast_script
            st.session_state.current_podcast['source_info'] = source_info
            st.session_state.current_podcast_script = podcast_script
        
        # Automatically generate audio if TTS is available
        tts_generator = pipeline.get('podcast_tts_generator')
        if tts_generator:
            with st.spinner("🎵 Generating podcast... This may take several minutes..."):
                try:
                    import tempfile
                    temp_dir = tempfile.mkdtemp(prefix="podcast_")
                    
                    # Generate audio
                    audio_files = tts_generator.generate_podcast_audio(
                        podcast_script=podcast_script,
                        output_dir=temp_dir,
                        combine_audio=True
                    )

                    # Store audio files in session state
                    st.session_state.current_podcast['audio_files'] = audio_files

                    st.success(f"✅ Generated {len(audio_files)} audio files!")

                except Exception as e:
                    st.error(f"❌ Audio generation failed: {str(e)}")
                    logger.error(f"Audio generation error: {e}")

                    if "No module named" in str(e):
                        st.error("🔧 Missing dependency. Please check the installation.")
                    elif "File" in str(e) and "not found" in str(e):
                        st.error("📁 File system error. Check permissions and disk space.")
        else:
            st.warning("⚠️ Audio generation not available - TTS not initialized.")

        # Display the generated podcast using the reusable function
        display_generated_podcast(st.session_state.current_podcast)
    
    except Exception as e:
        st.error(f"❌ Podcast generation failed: {str(e)}")
        logger.error(f"Podcast generation error: {e}")

def display_generated_podcast(podcast_data):
    """Display the generated podcast (audio player, script, and download buttons)"""
    import time
    import random
    from pathlib import Path

    podcast_script = podcast_data.get('script')
    audio_files = podcast_data.get('audio_files')
    source_info = podcast_data.get('source_info')
    source_name = podcast_data.get('source', 'podcast')

    if not podcast_script:
        return

    # Generate unique ID for this display to avoid duplicate keys
    display_id = f"{int(time.time() * 1000)}_{random.randint(1000, 9999)}"

    # Display audio player if available
    if audio_files:
        st.markdown("### 🎙️ Generated Podcast")
        for audio_file in audio_files:
            file_name = Path(audio_file).name

            if "complete_podcast" in file_name:
                st.audio(audio_file, format="audio/wav")

                try:
                    with open(audio_file, "rb") as f:
                        audio_data = f.read()
                        st.download_button(
                            label="📥 Download Complete Podcast",
                            data=audio_data,
                            file_name=f"podcast_{source_name.replace(' ', '_')}_{int(time.time())}.wav",
                            mime="audio/wav",
                            key=f"download_audio_{display_id}"
                        )
                except Exception as e:
                    logger.error(f"Error reading audio file for download: {e}")

    # Display the generated script
    st.markdown("### 📝 Generated Podcast Script")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Lines", podcast_script.total_lines)
    with col2:
        st.metric("⏱️ Est. Duration", podcast_script.estimated_duration)
    with col3:
        if source_info and 'type' in source_info:
            st.metric("📚 Source Type", source_info['type'])
        else:
            st.metric("📚 Source", source_name)

    # Display script content
    with st.expander("👀 View Complete Script", expanded=True):
        for i, line_dict in enumerate(podcast_script.script, 1):
            speaker, dialogue = next(iter(line_dict.items()))

            # Color code speakers
            if speaker == "Speaker 1":
                st.markdown(f'<div style="background: #1e3a8a; padding: 10px; border-radius: 5px; margin: 5px 0;"><strong>👩 {speaker}:</strong> {dialogue}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="background: #166534; padding: 10px; border-radius: 5px; margin: 5px 0;"><strong>👨 {speaker}:</strong> {dialogue}</div>', unsafe_allow_html=True)

    # Script download button
    script_json = podcast_script.to_json()
    st.download_button(
        label="📥 Download Script (JSON)",
        data=script_json,
        file_name=f"script_{source_name.replace(' ', '_')}_{int(time.time())}.json",
        mime="application/json",
        key=f"download_script_{display_id}"
    )

def render_studio_tab():
    st.markdown('<div class="main-header">🎙️ Generate Podcast</div>', unsafe_allow_html=True)

    if not st.session_state.sources:
        st.markdown("""
        <div style="text-align: center; padding: 40px; color: #a0aec0;">
            <p>📄 Upload a PDF or paste text in the "Input" tab first</p>
            <p>Then come back here to generate your podcast!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("#### 🎙️ Generate Podcast")
        st.markdown("Create an AI-generated podcast discussion from your documents")

        source_names = [source['name'] for source in st.session_state.sources]
        selected_source = st.selectbox(
            "Select source for podcast",
            source_names,
            index=0 if source_names else None,
            help="Choose a document to create a podcast discussion about"
        )

        col1, col2 = st.columns(2)
        with col1:
            podcast_style = st.selectbox(
                "Podcast Style",
                ["Conversational", "Interview", "Debate", "Educational"]
            )
        with col2:
            podcast_length = st.selectbox(
                "Duration",
                ["5 minutes", "10 minutes", "15 minutes"]
            )

        if st.button("🎙️ Generate Podcast", use_container_width=True):
            if selected_source:
                generate_podcast(selected_source, podcast_style, podcast_length)
            else:
                st.warning("Please select a source for the podcast")

        # Display previously generated podcast if it exists (so it persists after download)
        if 'current_podcast' in st.session_state and st.session_state.current_podcast.get('script'):
            display_generated_podcast(st.session_state.current_podcast)

def main():
    init_session_state()
    
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 30px;">
        <h1 style="color: #ffffff; margin: 0;">🎙️ Research Paper to Podcast</h1>
    </div>
    <div style="margin-bottom: 20px;">
        <p style="color: #a0aec0; font-size: 16px;">Transform your research papers into engaging 2-person podcast conversations</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not initialize_pipeline():
        st.stop()
    
    render_sources_sidebar()

    tab1, tab2 = st.tabs(["📄 Input", "🎙️ Generate Podcast"])
    with tab1:
        render_source_upload_dialog()
    with tab2:
        render_studio_tab()
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #a0aec0; font-size: 12px;">
        AI-generated podcasts may not be 100% accurate. Please verify important information.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
