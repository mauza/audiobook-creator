"""Text-to-speech conversion module using Qwen3-TTS."""

import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

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

from audiobook_creator.file_handlers import (
    Chapter,
    extract_text_from_file,
    _normalize_for_tts,
    _split_into_sentences,
)


class _TTSWorker:
    """A single TTS model instance bound to one device with its own CUDA stream."""

    def __init__(
        self,
        model_name: str,
        device: str,
        voice: str,
        language: str,
        voice_clone_prompt=None,
    ):
        self.device = device
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        self.voice = voice
        self.language = language
        self.voice_clone_prompt = voice_clone_prompt
        self.lock = threading.Lock()

        # Each worker gets its own CUDA stream for true parallel execution on the same GPU
        if device.startswith("cuda"):
            dev_idx = int(device.split(":")[1]) if ":" in device else 0
            self.stream = torch.cuda.Stream(device=dev_idx)
        else:
            self.stream = None

    def generate(self, text: str) -> Tuple[np.ndarray, int]:
        """Generate audio for a text chunk. Uses a dedicated CUDA stream."""
        with self.lock:
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    result = self._run_inference(text)
                self.stream.synchronize()
                return result
            else:
                return self._run_inference(text)

    def _run_inference(self, text: str) -> Tuple[np.ndarray, int]:
        if self.voice_clone_prompt:
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=self.language,
                voice_clone_prompt=self.voice_clone_prompt,
            )
        else:
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language=self.language,
                speaker=self.voice,
            )
        return wavs[0], sr


class TTSConverter:
    """Convert text to speech using Qwen3-TTS with multi-GPU support."""

    # Constants for file size management
    MAX_FILE_SIZE_MB = 100
    BYTES_PER_SAMPLE = 2  # 16-bit audio = 2 bytes

    def __init__(
        self,
        output_dir: str = "output",
        voice: str = "Ryan",
        language: str = "English",
        chunk_size: int = 500,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device: str = "auto",
        output_format: str = "wav",
        voice_sample: Optional[str] = None,
        voice_text: Optional[str] = None,
        workers_per_device: int = 2,
    ):
        """Initialize the TTS converter.

        Args:
            output_dir: Directory to save audio files
            voice: Speaker name (Vivian, Serena, Dylan, Eric, Ryan, Aiden, etc.)
            language: Language name (English, Chinese, Japanese, Korean, etc.)
            chunk_size: Number of characters to process at once
            model_name: Qwen3-TTS model name or path
            device: Device spec. "auto" to use all GPUs, "cuda:0", "cuda:0,cuda:1", or "cpu"
            output_format: Output audio format ("wav" or "mp3")
            voice_sample: Path to a reference audio file for voice cloning
            voice_text: Transcript of the voice sample (recommended for quality)
            workers_per_device: Number of model instances per device (default 2)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.voice = voice
        self.language = language
        self.chunk_size = chunk_size
        self.sample_rate = None
        self.output_format = output_format

        # Resolve device list
        devices = self._resolve_devices(device)

        # Build voice clone prompt on first device if needed
        voice_clone_prompt = None
        if voice_sample:
            sample_path = Path(voice_sample)
            if not sample_path.exists():
                raise FileNotFoundError(f"Voice sample not found: {voice_sample}")
            dtype = torch.float16 if devices[0].startswith("cuda") else torch.float32
            tmp_model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map=devices[0],
                dtype=dtype,
                attn_implementation="sdpa",
            )
            voice_clone_prompt = tmp_model.create_voice_clone_prompt(
                ref_audio=str(sample_path),
                ref_text=voice_text,
                x_vector_only_mode=voice_text is None,
            )
            # Reuse this as the first worker instead of reloading
            first_worker = _TTSWorker.__new__(_TTSWorker)
            first_worker.device = devices[0]
            first_worker.model = tmp_model
            first_worker.voice = voice
            first_worker.language = language
            first_worker.voice_clone_prompt = voice_clone_prompt
            first_worker.lock = threading.Lock()
            if devices[0].startswith("cuda"):
                dev_idx = int(devices[0].split(":")[1]) if ":" in devices[0] else 0
                first_worker.stream = torch.cuda.Stream(device=dev_idx)
            else:
                first_worker.stream = None
        else:
            first_worker = None

        # Create workers: workers_per_device instances on each device
        self.workers: List[_TTSWorker] = []
        for dev in devices:
            for w_idx in range(workers_per_device):
                # Reuse the first_worker we already loaded (if applicable)
                if first_worker is not None and dev == devices[0] and w_idx == 0:
                    self.workers.append(first_worker)
                    continue
                worker = _TTSWorker(
                    model_name=model_name,
                    device=dev,
                    voice=voice,
                    language=language,
                    voice_clone_prompt=voice_clone_prompt,
                )
                self.workers.append(worker)

    @staticmethod
    def _resolve_devices(device: str) -> List[str]:
        """Resolve device specification to a list of device strings."""
        if device == "auto":
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                if count > 0:
                    return [f"cuda:{i}" for i in range(count)]
            return ["cpu"]

        if "," in device:
            devices = [d.strip() for d in device.split(",")]
            # Validate CUDA devices exist
            if torch.cuda.is_available():
                return devices
            return ["cpu"]

        if device.startswith("cuda") and not torch.cuda.is_available():
            return ["cpu"]

        return [device]

    def _calculate_max_samples_per_file(self) -> int:
        max_bytes = self.MAX_FILE_SIZE_MB * 1024 * 1024
        return max_bytes // self.BYTES_PER_SAMPLE

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks of approximately equal size."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            sentences = _split_into_sentences(paragraph)

            for sentence in sentences:
                sentence_len = len(sentence)
                if current_size + sentence_len > self.chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_size = sentence_len
                else:
                    current_chunk.append(sentence)
                    current_size += sentence_len

        if current_chunk:
            last_chunk = ' '.join(current_chunk)
            if not last_chunk.endswith(('.', '!', '?')):
                last_chunk += '.'
            chunks.append(last_chunk)

        return chunks

    def _process_chunk_on_worker(
        self,
        worker: _TTSWorker,
        text_chunk: str,
        temp_dir: Path,
        chunk_index: int,
    ) -> Tuple[int, Path, float]:
        """Process a single chunk using a specific worker.

        Returns:
            Tuple of (chunk_index, path to temp file, duration)
        """
        temp_file = temp_dir / f"chunk_{chunk_index:04d}.wav"
        audio, sr = worker.generate(text_chunk)

        if self.sample_rate is None:
            self.sample_rate = sr

        if len(audio) == 0:
            raise ValueError(f"No audio was generated for chunk {chunk_index}")

        sf.write(temp_file, audio, sr)
        duration = len(audio) / sr
        return chunk_index, temp_file, duration

    def _process_chunks_parallel(
        self,
        text_chunks: List[str],
        temp_dir: Path,
        progress: Progress,
        task_id,
        completed_offset: int = 0,
    ) -> List[Path]:
        """Process text chunks in parallel across all workers.

        Args:
            completed_offset: Number of already-completed chunks (for progress tracking
                across multiple calls sharing the same progress bar)

        Returns chunk files in original order.
        """
        num_workers = len(self.workers)
        results: dict[int, Path] = {}
        completed = 0

        if num_workers == 1:
            for i, chunk in enumerate(text_chunks):
                _, chunk_file, _ = self._process_chunk_on_worker(
                    self.workers[0], chunk, temp_dir, i
                )
                results[i] = chunk_file
                completed += 1
                progress.update(task_id, completed=completed_offset + completed)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {}
                for i, chunk in enumerate(text_chunks):
                    worker = self.workers[i % num_workers]
                    future = executor.submit(
                        self._process_chunk_on_worker,
                        worker, chunk, temp_dir, i,
                    )
                    futures[future] = i

                for future in as_completed(futures):
                    idx, chunk_file, duration = future.result()
                    results[idx] = chunk_file
                    completed += 1
                    progress.update(task_id, completed=completed_offset + completed)

        return [results[i] for i in range(len(text_chunks))]

    def _write_audio_file(
        self,
        audio_data: np.ndarray,
        output_path: Path,
        metadata: Optional[dict] = None,
    ) -> None:
        sf.write(output_path, audio_data, self.sample_rate, format='WAV')

    def _maybe_convert_to_mp3(self, wav_path: Path) -> Path:
        if self.output_format != "mp3":
            return wav_path

        try:
            from pydub import AudioSegment
        except ImportError:
            raise RuntimeError(
                "pydub is required for MP3 output. Install it with: "
                "pip install pydub\n"
                "You also need ffmpeg installed on your system."
            )

        mp3_path = wav_path.with_suffix('.mp3')
        AudioSegment.from_wav(str(wav_path)).export(
            str(mp3_path), format="mp3", bitrate="192k"
        )
        wav_path.unlink()
        return mp3_path

    def _combine_chunks_to_files(
        self,
        chunk_files: List[Path],
        book_dir: Path,
        output_filename: str,
        file_suffix: str = "",
    ) -> List[Path]:
        max_samples = self._calculate_max_samples_per_file()
        silence_duration = 0.5
        silence_samples = int(self.sample_rate * silence_duration)
        silence = np.zeros(silence_samples)

        current_audio = None
        current_samples = 0
        file_index = 1
        output_files = []
        ext = ".wav"

        for chunk_file in chunk_files:
            audio, _ = sf.read(chunk_file)

            if current_audio is None:
                current_audio = audio
                current_samples = len(audio)
            else:
                combined_len = current_samples + silence_samples + len(audio)
                if combined_len > max_samples:
                    output_path = book_dir / f"{output_filename}{file_suffix}_part{file_index:03d}{ext}"
                    self._write_audio_file(current_audio, output_path)
                    output_path = self._maybe_convert_to_mp3(output_path)
                    output_files.append(output_path)

                    current_audio = audio
                    current_samples = len(audio)
                    file_index += 1
                else:
                    current_audio = np.concatenate([current_audio, silence, audio])
                    current_samples = len(current_audio)

        if current_audio is not None:
            output_path = book_dir / f"{output_filename}{file_suffix}_part{file_index:03d}{ext}"
            self._write_audio_file(current_audio, output_path)
            output_path = self._maybe_convert_to_mp3(output_path)
            output_files.append(output_path)

        return output_files

    def convert_text(
        self,
        text: str,
        output_filename: str,
        split_pattern: str = r"\n+",
    ) -> List[Path]:
        """Convert text to speech and save as audio files.

        Writes output files incrementally as chunks are combined,
        so partial results are available on disk during processing.
        """
        text = _normalize_for_tts(text)

        book_dir = self.output_dir / output_filename
        book_dir.mkdir(parents=True, exist_ok=True)

        text_chunks = self._split_text_into_chunks(text)
        num_workers = len(self.workers)

        max_samples = None
        silence = None
        current_audio = None
        current_samples = 0
        file_index = 1
        output_files = []

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task(
                    f"Converting text to speech ({num_workers} worker{'s' if num_workers > 1 else ''})...",
                    total=len(text_chunks),
                )

                if num_workers == 1:
                    for i, chunk in enumerate(text_chunks):
                        _, chunk_file, _ = self._process_chunk_on_worker(
                            self.workers[0], chunk, temp_dir, i
                        )

                        # Initialize audio params on first chunk
                        if max_samples is None:
                            max_samples = self._calculate_max_samples_per_file()
                            silence_samples = int(self.sample_rate * 0.5)
                            silence = np.zeros(silence_samples)

                        audio, _ = sf.read(chunk_file)
                        current_audio, current_samples, file_index = self._accumulate_and_flush(
                            audio, current_audio, current_samples, silence,
                            max_samples, book_dir, output_filename, "", file_index, output_files,
                        )
                        progress.update(task, completed=i + 1)
                else:
                    # Process in parallel but flush as chunks complete in order
                    next_to_flush = 0
                    ready: dict[int, Path] = {}

                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        futures = {}
                        for i, chunk in enumerate(text_chunks):
                            worker = self.workers[i % num_workers]
                            future = executor.submit(
                                self._process_chunk_on_worker,
                                worker, chunk, temp_dir, i,
                            )
                            futures[future] = i

                        completed_count = 0
                        for future in as_completed(futures):
                            idx, chunk_file, _ = future.result()
                            ready[idx] = chunk_file
                            completed_count += 1
                            progress.update(task, completed=completed_count)

                            # Flush all consecutive chunks that are ready
                            while next_to_flush in ready:
                                if max_samples is None:
                                    max_samples = self._calculate_max_samples_per_file()
                                    silence_samples = int(self.sample_rate * 0.5)
                                    silence = np.zeros(silence_samples)

                                audio, _ = sf.read(ready.pop(next_to_flush))
                                current_audio, current_samples, file_index = self._accumulate_and_flush(
                                    audio, current_audio, current_samples, silence,
                                    max_samples, book_dir, output_filename, "", file_index, output_files,
                                )
                                next_to_flush += 1

            # Write remaining audio
            if current_audio is not None:
                output_path = book_dir / f"{output_filename}_part{file_index:03d}.wav"
                self._write_audio_file(current_audio, output_path)
                output_path = self._maybe_convert_to_mp3(output_path)
                output_files.append(output_path)

        return output_files

    def _accumulate_and_flush(
        self,
        audio: np.ndarray,
        current_audio: Optional[np.ndarray],
        current_samples: int,
        silence: np.ndarray,
        max_samples: int,
        book_dir: Path,
        output_filename: str,
        file_suffix: str,
        file_index: int,
        output_files: List[Path],
    ) -> Tuple[Optional[np.ndarray], int, int]:
        """Add audio to the accumulator buffer, flushing to disk when full.

        Returns:
            Tuple of (current_audio, current_samples, file_index)
        """
        silence_samples = len(silence)

        if current_audio is None:
            return audio, len(audio), file_index

        combined_len = current_samples + silence_samples + len(audio)
        if combined_len > max_samples:
            # Flush current buffer to disk
            output_path = book_dir / f"{output_filename}{file_suffix}_part{file_index:03d}.wav"
            self._write_audio_file(current_audio, output_path)
            output_path = self._maybe_convert_to_mp3(output_path)
            output_files.append(output_path)
            return audio, len(audio), file_index + 1

        current_audio = np.concatenate([current_audio, silence, audio])
        return current_audio, len(current_audio), file_index

    def convert_chapters(
        self,
        chapters: List[Chapter],
        output_filename: str,
    ) -> List[Path]:
        """Convert chapters to speech, flushing each chapter to disk as it completes."""
        book_dir = self.output_dir / output_filename
        book_dir.mkdir(parents=True, exist_ok=True)

        all_output_files: List[Path] = []

        # Pre-compute all chapter metadata and total chunk count for progress
        chapter_info: List[Tuple[str, List[str]]] = []
        total_chunks = 0
        for ch_num, (title, text) in enumerate(chapters, 1):
            text = _normalize_for_tts(text)
            chunks = self._split_text_into_chunks(text)
            slug = re.sub(r'[^\w]+', '_', title.lower()).strip('_')[:30]
            suffix = f"_ch{ch_num:02d}_{slug}"
            chapter_info.append((suffix, chunks))
            total_chunks += len(chunks)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                f"Converting chapters ({len(self.workers)} worker{'s' if len(self.workers) > 1 else ''})...",
                total=total_chunks,
            )
            global_completed = 0

            for suffix, chunks in chapter_info:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir = Path(temp_dir)

                    chunk_files = self._process_chunks_parallel(
                        chunks, temp_dir, progress, task,
                        completed_offset=global_completed,
                    )
                    global_completed += len(chunks)

                    chapter_files = self._combine_chunks_to_files(
                        chunk_files, book_dir, output_filename, file_suffix=suffix,
                    )
                    all_output_files.extend(chapter_files)

        return all_output_files

    def convert_file(
        self,
        input_file: str,
        output_filename: Optional[str] = None,
        encoding: str = "utf-8",
        chapter_aware: bool = False,
        skip_front_matter: bool = True,
    ) -> List[Path]:
        """Convert a file to speech and save as audio files."""
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        if output_filename is None:
            output_filename = input_path.stem

        result = extract_text_from_file(
            input_path,
            encoding=encoding,
            chapter_aware=chapter_aware,
            skip_front_matter=skip_front_matter,
        )

        if chapter_aware and isinstance(result, list):
            return self.convert_chapters(result, output_filename)

        text = result if isinstance(result, str) else result[0][1]
        return self.convert_text(text, output_filename)
