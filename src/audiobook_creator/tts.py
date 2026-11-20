"""Text-to-speech conversion with Phonon (HTTP), Kokoro or Qwen3-TTS."""

import io
import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from audiobook_creator.file_handlers import (
    Chapter,
    _normalize_for_tts,
    _split_into_sentences,
    extract_epub_metadata,
    extract_text_from_file,
)


class _GPUWorker:
    """A single TTS model instance bound to one GPU."""

    def __init__(
        self,
        model_name: str,
        device: str,
        voice: str,
        language: str,
        voice_clone_prompt=None,
    ):
        import torch
        from qwen_tts import Qwen3TTSModel

        self.device = device
        dtype = torch.float16 if device.startswith("cuda") or device == "mps" else torch.float32
        self.model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        self.voice = voice
        self.language = language
        self.voice_clone_prompt = voice_clone_prompt

    def generate_batch(self, texts: List[str]) -> Tuple[List[np.ndarray], int]:
        """Generate audio for a batch of text chunks in a single forward pass."""
        if self.voice_clone_prompt:
            wavs, sr = self.model.generate_voice_clone(
                text=texts,
                language=self.language,
                voice_clone_prompt=self.voice_clone_prompt,
            )
        else:
            wavs, sr = self.model.generate_custom_voice(
                text=texts,
                language=self.language,
                speaker=self.voice,
            )
        return wavs, sr


class _KokoroWorker:
    """A Kokoro TTS model instance."""

    SAMPLE_RATE = 24000
    LANG_CODE_MAP = {
        "English": "a",
        "American": "a",
        "British": "b",
        "Spanish": "e",
        "French": "f",
        "Japanese": "j",
        "Chinese": "z",
        "Mandarin": "z",
        "Korean": "a",  # Fallback to American English
        "German": "a",
        "Russian": "a",
        "Portuguese": "a",
        "Italian": "a",
    }

    def __init__(self, voice: str, language: str):
        from kokoro import KPipeline

        lang_code = self.LANG_CODE_MAP.get(language, "a")
        self.pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice

    def generate_batch(self, texts: List[str]) -> Tuple[List[np.ndarray], int]:
        """Generate audio for texts (processes sequentially via generator)."""
        wavs = []
        for text in texts:
            audio_chunks = []
            for _, _, audio in self.pipeline(text, voice=self.voice, speed=1):
                audio_chunks.append(audio)
            wavs.append(np.concatenate(audio_chunks) if audio_chunks else np.array([]))
        return wavs, self.SAMPLE_RATE


class _PhononWorker:
    """Phonon, or any server with OpenAI's /v1/audio/speech API, over HTTP.

    Requests go one at a time: the Phonon pod is CPU-bound, and on its 2-core
    limit two concurrent streams finish later than one (31x vs 42x realtime).
    """

    RETRIES = 6

    def __init__(self, url: str, voice: str):
        self.url = url.rstrip("/") + "/v1/audio/speech"
        self.voice = voice

    def _speak(self, text: str) -> Tuple[np.ndarray, int]:
        body = json.dumps(
            {"input": text, "voice": self.voice, "response_format": "wav"}
        ).encode()
        request = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"}
        )
        for attempt in range(self.RETRIES):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    audio, sr = sf.read(io.BytesIO(response.read()), dtype="float32")
                return audio, sr
            except urllib.error.HTTPError as e:
                # 429 means the server is busy with other callers (Jarvis).
                if e.code != 429 and e.code < 500:
                    detail = e.read().decode(errors="replace")[:500]
                    raise RuntimeError(f"TTS server rejected the request: {detail}")
                error = e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                error = e
            time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"TTS server failed after {self.RETRIES} tries: {error}")

    def generate_batch(self, texts: List[str]) -> Tuple[List[np.ndarray], int]:
        results = [self._speak(text) for text in texts]
        return [audio for audio, _ in results], results[0][1]


def _trim_silence(audio: np.ndarray, sr: int, keep: float = 0.03) -> np.ndarray:
    """Cut leading and trailing silence, so pauses come only from the plan."""
    loud = np.flatnonzero(np.abs(audio) > 0.01)
    if len(loud) == 0:
        return audio
    margin = int(sr * keep)
    return audio[max(loud[0] - margin, 0): loud[-1] + margin + 1]


def _spoken_title(title: str) -> str:
    """'1: You Believe ...' -> 'Chapter 1. You Believe ...'."""
    title = re.sub(r'^(\d+)\s*[:.]\s*', r'Chapter \1. ', title.strip())
    title = re.sub(r'\s*:\s+', '. ', title)
    return title if title.endswith(('.', '!', '?')) else title + '.'


def _ffmetadata_value(value: str) -> str:
    return re.sub(r'([=;#\\\n])', r'\\\1', value)


class TTSConverter:
    """Convert text to speech using Qwen3-TTS or Kokoro with multi-GPU support."""

    # Constants for file size management
    MAX_FILE_SIZE_MB = 100
    BYTES_PER_SAMPLE = 2  # 16-bit audio = 2 bytes

    # Default voices per engine
    DEFAULT_VOICES = {
        "qwen3": "Ryan",
        "kokoro": "af_heart",
        # Settled, kind, middle-aged US voice; the best UTMOS (4.47) and the
        # calmest pace of the US voices on a self-help passage (2026-11-20).
        "phonon": "Hadley",
    }

    # Silence between chunks, in seconds. Chunks are trimmed first.
    SENTENCE_GAP = 0.35
    PARAGRAPH_GAP = 0.75
    TITLE_GAP = 1.2
    CHAPTER_GAP = 2.0
    LOUDNESS_LUFS = -19

    def __init__(
        self,
        output_dir: str = "output",
        voice: Optional[str] = None,
        language: str = "English",
        chunk_size: int = 500,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device: str = "auto",
        output_format: str = "wav",
        voice_sample: Optional[str] = None,
        voice_text: Optional[str] = None,
        batch_size: int = 4,
        tts_engine: str = "phonon",
        tts_url: str = "http://phonon-tts.phonon-tts:8880",
    ):
        """Initialize the TTS converter.

        Args:
            output_dir: Directory to save audio files
            voice: Speaker name (engine-specific; None uses engine default)
            language: Language name (English, Chinese, Japanese, Korean, etc.)
            chunk_size: Number of characters to process at once
            model_name: Qwen3-TTS model name or path (ignored for Kokoro)
            device: Device spec. "auto" to use all GPUs, "cuda:0", "cuda:0,cuda:1", or "cpu"
            output_format: Output audio format ("wav" or "mp3")
            voice_sample: Path to a reference audio file for voice cloning (qwen3 only)
            voice_text: Transcript of the voice sample (qwen3 only)
            batch_size: Number of chunks to process per inference call (default 4)
            tts_engine: TTS engine to use ("phonon", "qwen3" or "kokoro")
            tts_url: Base URL of the speech server (phonon only)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tts_engine = tts_engine
        self.voice = voice if voice else self.DEFAULT_VOICES.get(tts_engine, "Ryan")
        self.language = language
        self.chunk_size = chunk_size
        self.sample_rate = None
        self.output_format = output_format
        self.batch_size = batch_size

        if tts_engine == "phonon":
            self.workers = [_PhononWorker(url=tts_url, voice=self.voice)]
        elif tts_engine == "kokoro":
            # Kokoro: single worker instance (handles its own device placement)
            self.workers: List[Union[_GPUWorker, _KokoroWorker]] = [
                _KokoroWorker(voice=self.voice, language=language)
            ]
        else:
            # Qwen3-TTS: multi-GPU support
            import torch
            from qwen_tts import Qwen3TTSModel

            devices = self._resolve_devices(device)

            # Build voice clone prompt on first device if needed
            voice_clone_prompt = None
            if voice_sample:
                sample_path = Path(voice_sample)
                if not sample_path.exists():
                    raise FileNotFoundError(f"Voice sample not found: {voice_sample}")
                dtype = torch.float16 if devices[0].startswith("cuda") or devices[0] == "mps" else torch.float32
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
                first_worker = _GPUWorker.__new__(_GPUWorker)
                first_worker.device = devices[0]
                first_worker.model = tmp_model
                first_worker.voice = self.voice
                first_worker.language = language
                first_worker.voice_clone_prompt = voice_clone_prompt
            else:
                first_worker = None

            # Create one worker per GPU
            self.workers = []
            for i, dev in enumerate(devices):
                if first_worker is not None and i == 0:
                    self.workers.append(first_worker)
                    continue
                worker = _GPUWorker(
                    model_name=model_name,
                    device=dev,
                    voice=self.voice,
                    language=language,
                    voice_clone_prompt=voice_clone_prompt,
                )
                self.workers.append(worker)

    @staticmethod
    def _resolve_devices(device: str) -> List[str]:
        """Resolve device specification to a list of device strings."""
        import torch

        if device == "auto":
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                if count > 0:
                    return [f"cuda:{i}" for i in range(count)]
            if torch.backends.mps.is_available():
                return ["mps"]
            return ["cpu"]

        if "," in device:
            devices = [d.strip() for d in device.split(",")]
            if torch.cuda.is_available():
                return devices
            return ["cpu"]

        if device.startswith("cuda") and not torch.cuda.is_available():
            return ["cpu"]

        return [device]

    def _calculate_max_samples_per_file(self) -> int:
        max_bytes = self.MAX_FILE_SIZE_MB * 1024 * 1024
        return max_bytes // self.BYTES_PER_SAMPLE

    def _normalize(self, text: str) -> str:
        # Phonon reads digits, dates and percentages correctly on its own.
        return _normalize_for_tts(text, spell_numbers=self.tts_engine != "phonon")

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks of approximately equal size."""
        return [chunk for chunk, _ in self._chunk_plan(text)]

    def _chunk_plan(self, text: str) -> List[Tuple[str, float]]:
        """Split text into (chunk, pause after it) pairs.

        A chunk holds whole sentences, up to chunk_size characters, and never
        crosses a paragraph, so a paragraph ends with a longer pause.
        """
        plan: List[Tuple[str, float]] = []
        for paragraph in text.split('\n\n'):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            current: List[str] = []
            size = 0
            for sentence in _split_into_sentences(paragraph):
                if current and size + len(sentence) > self.chunk_size:
                    plan.append((' '.join(current), self.SENTENCE_GAP))
                    current, size = [], 0
                current.append(sentence)
                size += len(sentence)
            if current:
                chunk = ' '.join(current).rstrip(',;:')
                if not chunk.endswith(('.', '!', '?', '"', "'", ')')):
                    chunk += '.'
                plan.append((chunk, self.PARAGRAPH_GAP))
        return plan

    def _process_batch_on_worker(
        self,
        worker: _GPUWorker,
        text_chunks: List[str],
        temp_dir: Path,
        chunk_indices: List[int],
    ) -> List[Tuple[int, Path, float]]:
        """Process a batch of chunks using a specific worker.

        Returns:
            List of (chunk_index, path to temp file, duration) tuples
        """
        wavs, sr = worker.generate_batch(text_chunks)

        if self.sample_rate is None:
            self.sample_rate = sr

        results = []
        for i, (idx, audio) in enumerate(zip(chunk_indices, wavs)):
            if len(audio) == 0:
                raise ValueError(f"No audio was generated for chunk {idx}")
            temp_file = temp_dir / f"chunk_{idx:04d}.wav"
            sf.write(temp_file, audio, sr)
            duration = len(audio) / sr
            results.append((idx, temp_file, duration))

        return results

    def _process_chunks_parallel(
        self,
        text_chunks: List[str],
        temp_dir: Path,
        progress: Progress,
        task_id,
        completed_offset: int = 0,
    ) -> List[Path]:
        """Process text chunks in parallel across all workers using batched inference.

        Args:
            completed_offset: Number of already-completed chunks (for progress tracking
                across multiple calls sharing the same progress bar)

        Returns chunk files in original order.
        """
        num_gpus = len(self.workers)
        results: dict[int, Path] = {}
        completed = 0

        if num_gpus == 1:
            # Single GPU: process in batches sequentially
            worker = self.workers[0]
            for batch_start in range(0, len(text_chunks), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(text_chunks))
                batch_texts = text_chunks[batch_start:batch_end]
                batch_indices = list(range(batch_start, batch_end))

                batch_results = self._process_batch_on_worker(
                    worker, batch_texts, temp_dir, batch_indices,
                )
                for idx, chunk_file, _ in batch_results:
                    results[idx] = chunk_file
                    completed += 1
                    progress.update(task_id, completed=completed_offset + completed)
        else:
            # Multi-GPU: split chunks into contiguous blocks per GPU
            # Each GPU processes its block in batches via a thread
            chunks_per_gpu = len(text_chunks) // num_gpus
            remainder = len(text_chunks) % num_gpus

            gpu_assignments = []  # (worker, start_idx, end_idx)
            offset = 0
            for gpu_idx in range(num_gpus):
                count = chunks_per_gpu + (1 if gpu_idx < remainder else 0)
                gpu_assignments.append((self.workers[gpu_idx], offset, offset + count))
                offset += count

            def process_gpu_block(worker, start, end):
                block_results = []
                for batch_start in range(start, end, self.batch_size):
                    batch_end = min(batch_start + self.batch_size, end)
                    batch_texts = text_chunks[batch_start:batch_end]
                    batch_indices = list(range(batch_start, batch_end))
                    batch_results = self._process_batch_on_worker(
                        worker, batch_texts, temp_dir, batch_indices,
                    )
                    block_results.extend(batch_results)
                return block_results

            with ThreadPoolExecutor(max_workers=num_gpus) as executor:
                futures = {}
                for worker, start, end in gpu_assignments:
                    if start == end:
                        continue
                    future = executor.submit(process_gpu_block, worker, start, end)
                    futures[future] = (start, end)

                for future in as_completed(futures):
                    batch_results = future.result()
                    for idx, chunk_file, _ in batch_results:
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
        text = self._normalize(text)

        book_dir = self.output_dir / output_filename
        book_dir.mkdir(parents=True, exist_ok=True)

        text_chunks = self._split_text_into_chunks(text)
        num_gpus = len(self.workers)

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
                    f"Converting text to speech ({num_gpus} device{'s' if num_gpus > 1 else ''}, batch size {self.batch_size})...",
                    total=len(text_chunks),
                )

                if num_gpus == 1:
                    # Single GPU: process batches sequentially, flush incrementally
                    worker = self.workers[0]
                    completed_count = 0
                    for batch_start in range(0, len(text_chunks), self.batch_size):
                        batch_end = min(batch_start + self.batch_size, len(text_chunks))
                        batch_texts = text_chunks[batch_start:batch_end]
                        batch_indices = list(range(batch_start, batch_end))

                        batch_results = self._process_batch_on_worker(
                            worker, batch_texts, temp_dir, batch_indices,
                        )

                        for idx, chunk_file, _ in batch_results:
                            if max_samples is None:
                                max_samples = self._calculate_max_samples_per_file()
                                silence_samples = int(self.sample_rate * 0.5)
                                silence = np.zeros(silence_samples)

                            audio, _ = sf.read(chunk_file)
                            current_audio, current_samples, file_index = self._accumulate_and_flush(
                                audio, current_audio, current_samples, silence,
                                max_samples, book_dir, output_filename, "", file_index, output_files,
                            )
                            completed_count += 1
                            progress.update(task, completed=completed_count)
                else:
                    # Multi-GPU: each GPU gets contiguous block, flush in-order
                    chunks_per_gpu = len(text_chunks) // num_gpus
                    remainder = len(text_chunks) % num_gpus

                    gpu_assignments = []
                    offset = 0
                    for gpu_idx in range(num_gpus):
                        count = chunks_per_gpu + (1 if gpu_idx < remainder else 0)
                        gpu_assignments.append((self.workers[gpu_idx], offset, offset + count))
                        offset += count

                    def process_gpu_block(worker, start, end):
                        block_results = []
                        for batch_start in range(start, end, self.batch_size):
                            batch_end = min(batch_start + self.batch_size, end)
                            batch_texts = text_chunks[batch_start:batch_end]
                            batch_indices = list(range(batch_start, batch_end))
                            batch_results = self._process_batch_on_worker(
                                worker, batch_texts, temp_dir, batch_indices,
                            )
                            block_results.extend(batch_results)
                        return block_results

                    next_to_flush = 0
                    ready: dict[int, Path] = {}
                    completed_count = 0

                    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
                        futures = {}
                        for worker, start, end in gpu_assignments:
                            if start == end:
                                continue
                            future = executor.submit(process_gpu_block, worker, start, end)
                            futures[future] = (start, end)

                        for future in as_completed(futures):
                            batch_results = future.result()
                            for idx, chunk_file, _ in batch_results:
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
            text = self._normalize(text)
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
                f"Converting chapters ({len(self.workers)} device{'s' if len(self.workers) > 1 else ''}, batch size {self.batch_size})...",
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

    def convert_book_m4b(
        self,
        chapters: List[Chapter],
        output_filename: str,
        metadata: Optional[dict] = None,
    ) -> List[Path]:
        """Write one M4B with a chapter marker per chapter, plus tags and cover.

        Each chapter starts with its spoken title. Audio streams to a raw PCM
        file as it is made, so a long book never sits in memory.
        """
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg is required for M4B output")
        metadata = metadata or {}
        title = metadata.get("title") or output_filename
        author = metadata.get("author") or ""

        intro = title + "." + (f" Written by {author}." if author else "")
        outro = f"This concludes {title}" + (f", by {author}." if author else ".")
        plans: List[Tuple[str, List[Tuple[str, float]]]] = [("Opening Credits", [(intro, self.CHAPTER_GAP)])]
        for chapter_title, text in chapters:
            heading = self._normalize(_spoken_title(chapter_title))
            plan = [(heading, self.TITLE_GAP)] + self._chunk_plan(self._normalize(text))
            plan[-1] = (plan[-1][0], self.CHAPTER_GAP)
            plans.append((chapter_title, plan))
        plans.append(("End Credits", [(outro, 0.5)]))

        book_dir = self.output_dir / output_filename
        book_dir.mkdir(parents=True, exist_ok=True)
        output_path = book_dir / f"{output_filename}.m4b"
        worker = self.workers[0]
        markers: List[Tuple[str, int, int]] = []  # title, start, end (samples)
        written = 0
        sr = None

        with tempfile.TemporaryDirectory(dir=book_dir) as temp_dir:
            pcm_path = Path(temp_dir) / "book.pcm"
            with open(pcm_path, "wb") as pcm, Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task(
                    f"Narrating {len(chapters)} chapters with {self.voice}...",
                    total=sum(len(plan) for _, plan in plans),
                )
                for chapter_title, plan in plans:
                    start = written
                    for i in range(0, len(plan), self.batch_size):
                        batch = plan[i:i + self.batch_size]
                        wavs, sr = worker.generate_batch([chunk for chunk, _ in batch])
                        for (chunk, gap), audio in zip(batch, wavs):
                            if len(audio) == 0:
                                raise ValueError(f"No audio was generated for: {chunk[:80]}")
                            audio = np.concatenate(
                                [_trim_silence(audio, sr), np.zeros(int(sr * gap), np.float32)]
                            )
                            pcm.write((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
                            written += len(audio)
                            progress.advance(task)
                    markers.append((chapter_title, start, written))

            lines = [";FFMETADATA1", f"title={_ffmetadata_value(title)}", "genre=Audiobook"]
            if author:
                lines += [f"artist={_ffmetadata_value(author)}", f"album_artist={_ffmetadata_value(author)}"]
            lines.append(f"album={_ffmetadata_value(title)}")
            if metadata.get("year"):
                lines.append(f"date={metadata['year']}")
            lines.append(f"comment={_ffmetadata_value(f'Narrated by {self.voice} ({self.tts_engine} TTS)')}")
            for chapter_title, start, end in markers:
                lines += ["[CHAPTER]", "TIMEBASE=1/%d" % sr, f"START={start}", f"END={end}",
                          f"title={_ffmetadata_value(chapter_title)}"]
            meta_path = Path(temp_dir) / "metadata.txt"
            meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            cmd = ["ffmpeg", "-loglevel", "error", "-y",
                   "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", str(pcm_path),
                   "-i", str(meta_path)]
            maps = ["-map", "0:a", "-map_metadata", "1", "-map_chapters", "1"]
            if metadata.get("cover"):
                cover_path = Path(temp_dir) / "cover.jpg"
                cover_path.write_bytes(metadata["cover"])
                cmd += ["-i", str(cover_path)]
                maps += ["-map", "2:v", "-c:v", "mjpeg", "-disposition:v", "attached_pic"]
            # Phonon comes out near -23.5 LUFS, quieter than most audiobooks.
            # loudnorm raises it and limits the peaks; it resamples to 192 kHz
            # internally, so -ar sets the rate back.
            cmd += maps + ["-af", f"loudnorm=I={self.LOUDNESS_LUFS}:TP=-2:LRA=11", "-ar", str(sr),
                           "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", str(output_path)]
            subprocess.run(cmd, check=True)

        return [output_path]

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
            chapter_aware=chapter_aware or self.output_format == "m4b",
            skip_front_matter=skip_front_matter,
        )

        if self.output_format == "m4b":
            chapters = result if isinstance(result, list) else [(output_filename, result)]
            metadata = (
                extract_epub_metadata(input_path)
                if input_path.suffix.lower() == ".epub"
                else {"title": output_filename}
            )
            return self.convert_book_m4b(chapters, output_filename, metadata)

        if chapter_aware and isinstance(result, list):
            return self.convert_chapters(result, output_filename)

        text = result if isinstance(result, str) else result[0][1]
        return self.convert_text(text, output_filename)
