import logging
import os
import soundfile as sf
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import wave

try:
    from piper.voice import PiperVoice
except ImportError:
    print("piper-tts not installed. Install with: pip install piper-tts")
    PiperVoice = None

from src.podcast.script_generator import PodcastScript

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """Represents a single audio segment with metadata"""
    speaker: str
    text: str
    audio_data: Any
    duration: float
    file_path: str


class PodcastTTSGenerator:
    def __init__(self, sample_rate: int = 22050):
        if PiperVoice is None:
            raise ImportError("piper-tts not available. Install with: pip install piper-tts")

        voice_dir = 'src/podcast/voices'
        female_voice_path = os.path.join(voice_dir, 'en_US-ljspeech-medium.onnx')
        male_voice_path = os.path.join(voice_dir, 'en_US-lessac-medium.onnx')
        
        if not os.path.exists(female_voice_path) or not os.path.exists(male_voice_path):
            raise FileNotFoundError("Voice models not found. Please download them first.")

        self.speaker_voices = {
            "Speaker 1": PiperVoice.from_onnx(female_voice_path),
            "Speaker 2": PiperVoice.from_onnx(male_voice_path)
        }
        
        self.sample_rate = sample_rate
        logger.info(f"Piper TTS initialized with sample_rate={sample_rate}")
    
    def generate_podcast_audio(
        self, 
        podcast_script: PodcastScript,
        output_dir: str = "outputs/podcast_audio",
        combine_audio: bool = True
    ) -> List[str]:

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating podcast audio for {podcast_script.total_lines} segments")
        logger.info(f"Output directory: {output_dir}")
        
        audio_segments = []
        output_files = []
        
        for i, line_dict in enumerate(podcast_script.script):
            speaker, dialogue = next(iter(line_dict.items()))
            
            logger.info(f"Processing segment {i+1}/{podcast_script.total_lines}: {speaker}")
            
            try:
                segment_filename = f"segment_{i+1:03d}_{speaker.replace(' ', '_').lower()}.wav"
                segment_path = os.path.join(output_dir, segment_filename)

                self._generate_single_segment(speaker, dialogue, segment_path)

                segment_audio, sr = sf.read(segment_path, dtype='float32')
                
                output_files.append(segment_path)
                
                if combine_audio:
                    audio_segment = AudioSegment(
                        speaker=speaker,
                        text=dialogue,
                        audio_data=segment_audio,
                        duration=len(segment_audio) / self.sample_rate,
                        file_path=segment_path
                    )
                    audio_segments.append(audio_segment)
                
                logger.info(f"✓ Generated segment {i+1}: {segment_filename}")
                
            except Exception as e:
                logger.error(f"✗ Failed to generate segment {i+1}: {str(e)}")
                continue
        
        if combine_audio and audio_segments:
            combined_path = self._combine_audio_segments(audio_segments, output_dir)
            output_files.append(combined_path)
        
        logger.info(f"Podcast generation complete! Generated {len(output_files)} files")
        return output_files
    
    def _generate_single_segment(self, speaker: str, text: str, output_path: str):
        voice = self.speaker_voices.get(speaker)
        if not voice:
            raise ValueError(f"No voice found for speaker: {speaker}")

        clean_text = self._clean_text_for_tts(text)

        with wave.open(output_path, 'wb') as wav_file:
            voice.synthesize(clean_text, wav_file)
    
    def _clean_text_for_tts(self, text: str) -> str:
        import re

        clean_text = text.strip()

        # Remove laughter and other markers (but keep the natural flow)
        clean_text = re.sub(r'[[laughs?]]', '', clean_text)
        clean_text = re.sub(r'[[chuckles?]]', '', clean_text)
        clean_text = re.sub(r'[[both laugh]]', '', clean_text)
        clean_text = re.sub(r'[[giggles?]]', '', clean_text)
        clean_text = re.sub(r'[[.*?]]', '', clean_text)  # Remove any other [markers]

        # Remove standalone filler sounds that sound terrible when spoken by TTS
        # These are removed when they appear as standalone words or at the start of sentences
        clean_text = re.sub(r'\b(um|uh|ah|er|hmm)\b[,.]?\s*', '', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'^(um|uh|ah|er|hmm)[,.]?\s+', '', clean_text, flags=re.IGNORECASE)

        # Clean up multiple punctuation
        clean_text = clean_text.replace("...", ",")  # Convert trailing off to brief pause
        clean_text = clean_text.replace("!!", "!")
        clean_text = clean_text.replace("??", "?")

        # Clean up extra spaces
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        # Ensure proper ending punctuation
        if not clean_text.endswith(('.', '!', '?', ',')):
            clean_text += '.'

        return clean_text
    
    def _combine_audio_segments(
        self,
        segments: List[AudioSegment],
        output_dir: str
    ) -> str:
        logger.info(f"Combining {len(segments)} audio segments")

        try:
            import numpy as np
            import random

            combined_audio = []
            for i, segment in enumerate(segments):
                combined_audio.append(segment.audio_data)

                if i < len(segments) - 1:
                    # More natural pauses - shorter and more varied
                    # Check if next speaker is different
                    next_segment = segments[i + 1]
                    speaker_change = segment.speaker != next_segment.speaker

                    if speaker_change:
                        # Very brief pause between speakers (0.05-0.12 seconds) for natural flow
                        pause_duration = random.uniform(0.05, 0.12)
                    else:
                        # Almost no pause when same speaker continues (0.02-0.06 seconds)
                        pause_duration = random.uniform(0.02, 0.06)

                    pause_samples = int(pause_duration * self.sample_rate)
                    pause_audio = np.zeros(pause_samples, dtype=np.float32)
                    combined_audio.append(pause_audio)
            
            final_audio = np.concatenate(combined_audio)
            
            combined_filename = "complete_podcast.wav"
            combined_path = os.path.join(output_dir, combined_filename)
            sf.write(combined_path, final_audio, self.sample_rate)
            
            duration = len(final_audio) / self.sample_rate
            logger.info(f"✓ Combined podcast saved: {combined_path} (Duration: {duration:.1f}s)")
            
            return combined_path
            
        except Exception as e:
            logger.error(f"✗ Failed to combine audio segments: {str(e)}")
            raise


if __name__ == "__main__":
    import json
    
    try:
        tts_generator = PodcastTTSGenerator()
        
        sample_script_data = {
            "script": [
                {"Speaker 1": "Welcome everyone to our podcast! Today we're exploring the fascinating world of artificial intelligence."},
                {"Speaker 2": "Thanks for having me! AI is indeed one of the most exciting technological developments of our time."},
                {"Speaker 1": "Let's start with machine learning. Can you explain what makes it so revolutionary?"},
                {"Speaker 2": "Absolutely! Machine learning allows computers to learn from data without being explicitly programmed for every single task."},
                {"Speaker 1": "That's incredible! And deep learning takes this even further, doesn't it?"},
                {"Speaker 2": "Exactly! Deep learning uses neural networks with multiple layers, revolutionizing computer vision and natural language processing."}
            ]
        }
        
        from src.podcast.script_generator import PodcastScript
        test_script = PodcastScript(
            script=sample_script_data["script"],
            source_document="AI Overview Test",
            total_lines=len(sample_script_data["script"]),
            estimated_duration="2 minutes"
        )
        
        print("Generating podcast audio...")
        output_files = tts_generator.generate_podcast_audio(
            test_script,
            output_dir="./podcast_output",
            combine_audio=True
        )
        
        print(f"\nGenerated files:")
        for file_path in output_files:
            print(f"  - {file_path}")
        
        print("\nPodcast TTS test completed successfully!")
        
    except (ImportError, ValueError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
