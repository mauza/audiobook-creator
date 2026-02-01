"""Text-to-speech conversion module using Qwen3-TTS."""

import tempfile
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from audiobook_creator.file_handlers import extract_text_from_file


class TTSConverter:
    """Convert text to speech using Qwen3-TTS."""

    # Constants for file size management
    MAX_FILE_SIZE_MB = 100
    BYTES_PER_SAMPLE = 2  # 16-bit audio = 2 bytes

    def __init__(
        self,
        output_dir: str = "output",
        voice: str = "Ryan",
        language: str = "English",
        chunk_size: int = 1000,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device: str = "cuda:0",
    ):
        """Initialize the TTS converter.

        Args:
            output_dir: Directory to save audio files
            voice: Speaker name (Vivian, Serena, Dylan, Eric, Ryan, Aiden, etc.)
            language: Language name (English, Chinese, Japanese, Korean, etc.)
            chunk_size: Number of characters to process at once
            model_name: Qwen3-TTS model name or path
            device: Device to run the model on (e.g. "cuda:0" or "cpu")
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.voice = voice
        self.language = language
        self.chunk_size = chunk_size
        self.sample_rate = None  # Will be set after first generation

        if not torch.cuda.is_available() and device.startswith("cuda"):
            device = "cpu"

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            dtype=dtype,
            attn_implementation="sdpa",
        )

    def _calculate_max_samples_per_file(self) -> int:
        """Calculate maximum number of samples per file based on MAX_FILE_SIZE_MB.

        Returns:
            Maximum number of samples that can fit in MAX_FILE_SIZE_MB
        """
        max_bytes = self.MAX_FILE_SIZE_MB * 1024 * 1024
        return max_bytes // self.BYTES_PER_SAMPLE

    def _split_text_into_chunks(self, text: str) -> Generator[str, None, None]:
        """Split text into chunks of approximately equal size.

        Args:
            text: Text to split

        Yields:
            Chunks of text
        """
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = []
        current_size = 0

        for paragraph in paragraphs:
            # If a single paragraph is larger than chunk_size, split it by sentences
            if len(paragraph) > self.chunk_size:
                sentences = paragraph.split('. ')
                for sentence in sentences:
                    if current_size + len(sentence) > self.chunk_size and current_chunk:
                        yield ' '.join(current_chunk) + '. '  # Add period and space
                        current_chunk = [sentence]
                        current_size = len(sentence)
                    else:
                        current_chunk.append(sentence)
                        current_size += len(sentence)
            else:
                if current_size + len(paragraph) > self.chunk_size and current_chunk:
                    yield ' '.join(current_chunk) + '. '  # Add period and space
                    current_chunk = [paragraph]
                    current_size = len(paragraph)
                else:
                    current_chunk.append(paragraph)
                    current_size += len(paragraph)

        # Ensure the last chunk is properly yielded with proper punctuation
        if current_chunk:
            last_chunk = ' '.join(current_chunk)
            if not last_chunk.endswith(('.', '!', '?')):
                last_chunk += '.'
            yield last_chunk

    def _process_audio_chunk(
        self,
        text_chunk: str,
        temp_dir: Path,
        chunk_index: int,
    ) -> Tuple[Path, float]:
        """Process a single chunk of text and save it to a temporary file.

        Args:
            text_chunk: Text chunk to process
            temp_dir: Directory to save temporary files
            chunk_index: Index of the current chunk

        Returns:
            Tuple of (path to temporary file, duration of audio)
        """
        temp_file = temp_dir / f"chunk_{chunk_index:04d}.wav"

        # Generate audio for the chunk
        wavs, sr = self.model.generate_custom_voice(
            text=text_chunk,
            language=self.language,
            speaker=self.voice,
        )

        audio = wavs[0]

        if self.sample_rate is None:
            self.sample_rate = sr

        if len(audio) == 0:
            raise ValueError(f"No audio was generated for chunk {chunk_index}")

        # Save to temporary file
        sf.write(temp_file, audio, self.sample_rate)

        # Calculate duration
        duration = len(audio) / self.sample_rate

        return temp_file, duration

    def _write_audio_file(
        self,
        audio_data: np.ndarray,
        output_path: Path,
        metadata: Optional[dict] = None,
    ) -> None:
        """Write audio data to a file with metadata.

        Args:
            audio_data: Audio data to write
            output_path: Path to write the file
            metadata: Optional metadata to include in the file
        """
        sf.write(output_path, audio_data, self.sample_rate, format='WAV')
        if metadata:
            # TODO: Add metadata support if needed
            pass

    def convert_text(
        self,
        text: str,
        output_filename: str,
        split_pattern: str = r"\n+",
    ) -> List[Path]:
        """Convert text to speech and save as audio files.

        Args:
            text: Text to convert
            output_filename: Base name for output files (without extension)
            split_pattern: Pattern to split text into chunks

        Returns:
            List of paths to the generated audio files
        """
        # Create output directory for this book
        book_dir = self.output_dir / output_filename
        book_dir.mkdir(parents=True, exist_ok=True)

        # Create temporary directory for chunks
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            chunk_files = []
            total_duration = 0.0

            # Process text in chunks
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ) as progress:
                # First pass: process all chunks and collect metadata
                task = progress.add_task("Converting text to speech...", total=None)

                for i, chunk in enumerate(self._split_text_into_chunks(text)):
                    chunk_file, duration = self._process_audio_chunk(chunk, temp_dir, i)
                    chunk_files.append(chunk_file)
                    total_duration += duration

                # Second pass: combine chunks into files of appropriate size
                progress.update(task, total=len(chunk_files))

                max_samples = self._calculate_max_samples_per_file()
                current_audio = None
                current_samples = 0
                file_index = 1
                output_files = []

                for i, chunk_file in enumerate(chunk_files):
                    audio, _ = sf.read(chunk_file)

                    if current_audio is None:
                        current_audio = audio
                        current_samples = len(audio)
                    else:
                        # Check if adding this chunk would exceed the size limit
                        if current_samples + len(audio) > max_samples:
                            # Write current file
                            output_path = book_dir / f"{output_filename}_part{file_index:03d}.wav"
                            self._write_audio_file(current_audio, output_path)
                            output_files.append(output_path)

                            # Start new file
                            current_audio = audio
                            current_samples = len(audio)
                            file_index += 1
                        else:
                            # Add to current file
                            current_audio = np.concatenate([current_audio, audio])
                            current_samples += len(audio)

                    progress.update(task, completed=i + 1)

                # Write the last file if there's any remaining audio
                if current_audio is not None:
                    output_path = book_dir / f"{output_filename}_part{file_index:03d}.wav"
                    self._write_audio_file(current_audio, output_path)
                    output_files.append(output_path)

        return output_files

    def convert_file(
        self,
        input_file: str,
        output_filename: Optional[str] = None,
        encoding: str = "utf-8",
    ) -> List[Path]:
        """Convert a file to speech and save as audio files.

        Args:
            input_file: Path to the input file
            output_filename: Base name for output files (without extension)
            encoding: File encoding for text files

        Returns:
            List of paths to the generated audio files
        """
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Use input filename without extension if output_filename not provided
        if output_filename is None:
            output_filename = input_path.stem

        # Extract text from the file
        text = extract_text_from_file(input_path, encoding=encoding)

        # Convert the extracted text
        return self.convert_text(text, output_filename)
