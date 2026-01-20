import logging
import os
import time
import re
import random
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from openai import OpenAI

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
    """
    Podcast TTS Generator using OpenAI Text-to-Speech API

    Generates natural-sounding podcast audio with dual speakers (male/female voices)
    using OpenAI's TTS models. Supports segment generation and audio combining with
    natural pauses.
    """

    def __init__(self, sample_rate: int = 22050):
        """Initialize OpenAI TTS Generator

        Args:
            sample_rate: Audio sample rate (default 22050)

        Raises:
            ValueError: If OPENAI_API_KEY environment variable is not set
        """
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=self.openai_api_key)

        # Voice mapping for NotebookLM-style dual speakers
        self.speaker_voices = {
            "Speaker 1": "nova",   # Female voice (warm, conversational)
            "Speaker 2": "onyx"    # Male voice (deep, professional)
        }

        self.sample_rate = sample_rate
        self.model = "tts-1"  # Standard quality (tts-1-hd for higher quality)
        self.response_format = "wav"

        # Rate limiting configuration
        self.max_retries = 3
        self.retry_delay = 2  # seconds

        logger.info(f"OpenAI TTS initialized: model={self.model}, voices={self.speaker_voices}")

    def generate_podcast_audio(
        self,
        podcast_script: PodcastScript,
        output_dir: str = "outputs/podcast_audio",
        combine_audio: bool = True
    ) -> List[str]:
        """Generate podcast audio from script

        Args:
            podcast_script: PodcastScript object containing dialogue
            output_dir: Directory to save audio files
            combine_audio: Whether to combine segments into final podcast

        Returns:
            List of generated audio file paths
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating podcast audio for {podcast_script.total_lines} segments")
        logger.info(f"Output directory: {output_dir}")

        audio_segments = []
        output_files = []

        for i, line_dict in enumerate(podcast_script.script):
            speaker, dialogue = next(iter(line_dict.items()))

            logger.info(f"Processing segment {i+1}/{podcast_script.total_lines}: {speaker}")

            try:
                segment_filename = f"segment_{i+1:03d}_{speaker.replace(' ', '_').lower()}.{self.response_format}"
                segment_path = os.path.join(output_dir, segment_filename)

                # Generate with retry logic for rate limiting
                self._generate_single_segment_with_retry(speaker, dialogue, segment_path)

                # Read audio for combining
                segment_audio, sr = sf.read(segment_path, dtype='float32')

                # Resample if OpenAI returns different sample rate
                if sr != self.sample_rate:
                    logger.warning(f"Resampling from {sr} Hz to {self.sample_rate} Hz")
                    # Simple linear resampling
                    resample_ratio = self.sample_rate / sr
                    new_length = int(len(segment_audio) * resample_ratio)
                    segment_audio = np.interp(
                        np.linspace(0, len(segment_audio), new_length),
                        np.arange(len(segment_audio)),
                        segment_audio
                    )

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

    def _generate_single_segment_with_retry(
        self,
        speaker: str,
        text: str,
        output_path: str,
        attempt: int = 1
    ):
        """Generate segment with automatic retry on rate limit errors

        Args:
            speaker: Speaker identifier
            text: Text to synthesize
            output_path: Path to save audio file
            attempt: Current attempt number (for retry logic)

        Raises:
            Exception: If all retry attempts fail
        """
        try:
            self._generate_single_segment(speaker, text, output_path)
        except Exception as e:
            error_msg = str(e).lower()

            # Retry on rate limit errors
            if ("rate limit" in error_msg or "429" in error_msg) and attempt < self.max_retries:
                wait_time = self.retry_delay * attempt
                logger.warning(f"Rate limit hit. Retrying in {wait_time}s (attempt {attempt}/{self.max_retries})")
                time.sleep(wait_time)
                return self._generate_single_segment_with_retry(speaker, text, output_path, attempt + 1)
            else:
                raise

    def _generate_single_segment(self, speaker: str, text: str, output_path: str):
        """Generate audio for a single segment using OpenAI TTS API

        Args:
            speaker: Speaker identifier (e.g., "Speaker 1")
            text: Text to synthesize
            output_path: Path to save the audio file

        Raises:
            ValueError: If speaker not recognized or API key invalid
            Exception: If API call fails
        """
        voice = self.speaker_voices.get(speaker)
        if not voice:
            raise ValueError(f"No voice found for speaker: {speaker}")

        clean_text = self._clean_text_for_tts(text)

        # Validate text length (OpenAI has a 4096 character limit)
        if len(clean_text) > 4096:
            logger.warning(f"Text too long ({len(clean_text)} chars), truncating to 4096")
            clean_text = clean_text[:4093] + "..."

        try:
            # Call OpenAI TTS API
            response = self.client.audio.speech.create(
                model=self.model,
                voice=voice,
                input=clean_text,
                response_format=self.response_format
            )

            # Stream response to file
            response.stream_to_file(output_path)

        except Exception as e:
            logger.error(f"OpenAI TTS API call failed for {speaker}: {str(e)}")

            # Provide helpful error messages
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                raise ValueError(
                    "Invalid OpenAI API key. Please check OPENAI_API_KEY environment variable"
                )
            elif "quota" in str(e).lower() or "billing" in str(e).lower():
                raise ValueError(
                    "OpenAI API quota exceeded. Please check your account billing"
                )
            elif "rate_limit" in str(e).lower():
                raise  # Will be caught by retry logic
            else:
                raise

    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for TTS processing

        Removes markers, filler words, and cleans punctuation for natural speech.

        Args:
            text: Raw text from script

        Returns:
            Cleaned text suitable for TTS synthesis
        """
        clean_text = text.strip()

        # Remove laughter and other markers (but keep the natural flow)
        clean_text = re.sub(r'\[laughs?\]', '', clean_text)
        clean_text = re.sub(r'\[chuckles?\]', '', clean_text)
        clean_text = re.sub(r'\[both laugh\]', '', clean_text)
        clean_text = re.sub(r'\[giggles?\]', '', clean_text)
        clean_text = re.sub(r'\[.*?\]', '', clean_text)  # Remove any other [markers]

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
        """Combine multiple audio segments with natural pauses

        Args:
            segments: List of AudioSegment objects to combine
            output_dir: Directory to save the combined audio

        Returns:
            Path to the combined audio file

        Raises:
            Exception: If combining fails
        """
        logger.info(f"Combining {len(segments)} audio segments")

        try:
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
        # Ensure API key is set
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable not set")
            print("Please set it in your .env file or environment")
            exit(1)

        tts_generator = PodcastTTSGenerator()

        sample_script_data = {
            "script": [
                {"Speaker 1": "Welcome everyone to our podcast! Today we're exploring the fascinating world of artificial intelligence."},
                {"Speaker 2": "Thanks for having me! AI is indeed one of the most exciting technological developments of our time."},
                {"Speaker 1": "Let's start with machine learning. Can you explain what makes it so revolutionary?"},
                {"Speaker 2": "Absolutely! Machine learning allows computers to learn from data without being explicitly programmed for every single task."}
            ]
        }

        from src.podcast.script_generator import PodcastScript
        test_script = PodcastScript(
            script=sample_script_data["script"],
            source_document="AI Overview Test",
            total_lines=len(sample_script_data["script"]),
            estimated_duration="1 minute"
        )

        print("Generating podcast audio with OpenAI TTS...")
        output_files = tts_generator.generate_podcast_audio(
            test_script,
            output_dir="./podcast_output",
            combine_audio=True
        )

        print(f"\nGenerated files:")
        for file_path in output_files:
            print(f"  - {file_path}")

        print("\nOpenAI TTS test completed successfully!")

    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
