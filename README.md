# Audiobook Creator

Turns an EPUB, PDF, Markdown or text file into an audiobook. The default
engine is [Phonon](https://github.com/gradium-ai/xn-ptts), reached over HTTP;
Kokoro and Qwen3-TTS can also run in-process.

The best output is `--format m4b` from an EPUB: one file with a chapter marker
per TOC entry, the book's title, author, year and cover, and spoken chapter
titles and opening and closing credits. Audiobookshelf reads all of it.

## Usage

```bash
uv venv && uv pip install -e ".[dev]"

# Phonon in the home cluster (phonon-tts.phonon-tts:8880), through a port-forward:
kpf -n phonon-tts svc/phonon-tts 8880 -- \
  'PHONON_URL=$URL .venv/bin/audiobook-creator convert-file book.epub -f m4b -o audiobooks'
```

Inside the cluster, the default `--tts-url` works without a port-forward. Any
server with OpenAI's `/v1/audio/speech` API works too.

### Options

- `--tts-engine`: `phonon` (default), `kokoro` or `qwen3`. The last two need
  `uv pip install -e ".[local]"` (torch, a GPU helps).
- `--voice`, `-v`: defaults to `Hadley` for Phonon, `af_heart` for Kokoro and
  `Ryan` for Qwen3. A wrong Phonon voice name fails with the list of voices.
- `--format`, `-f`: `wav`, `mp3` (both split into files of about 100 MB) or
  `m4b`.
- `--chapters`: one set of files per chapter (wav/mp3 only; m4b always uses
  chapters).
- `--include-front-matter`: also read the cover, copyright and contents
  pages, and TOC entries such as Notes, Index and Bibliography.
- `--clean-language`: read swear words as milder ones (shit as crap, ass as
  butt, hell as heck). Whole words only, so "class" and "hello" stay as they
  are. The list is `CLEAN_WORDS` in `file_handlers.py`.
- `--voice-sample`, `--voice-text`: voice cloning (qwen3 only).

## How a book becomes audio

- **EPUB chapters** follow the table of contents. A spine document without a
  TOC entry continues the chapter before it. Headings that repeat the chapter
  title are dropped, because the title is spoken from the TOC label
  ("1: Thoughts Pass" is read as "Chapter 1. Thoughts Pass.").
- **Paragraphs** are kept. A chunk of text sent to the engine never crosses a
  paragraph, and each chunk is trimmed of silence, so pauses come from one
  plan: 0.35 s between sentences, 0.75 s between paragraphs, 1.2 s after a
  chapter title and 2 s between chapters.
- **Text** is straightened (curly quotes, soft hyphens, odd spaces), and an em
  dash becomes a comma pause. Roman numerals after "Part" or "Chapter" and
  slashes between words are read as words. For Kokoro and Qwen3, numbers,
  money and percentages are spelled out. Phonon reads digits correctly, so it
  gets them as they are.

## Choosing Phonon and the Hadley voice (2026-11-20)

- On the Artificial Analysis open-weights speech arena, Phonon 31M ranks above
  Kokoro (Elo 1108 against 1060). The only open models ranked higher (Fish
  Audio Drama 3, Breeze TTS 2, Ming-3.0) are 0.5B to 35B parameters and run
  at about 3x realtime or less. Phonon runs at about 42x realtime on the
  speech 3090.
- Run requests one at a time. The pod is CPU-bound at its 2-core limit:
  1 stream gave 42x realtime, 2 gave 31x and 4 gave 28x.
- Hadley was the US voice with the best UTMOS (4.47) and the calmest pace
  (182 words per minute) on a 450-word passage. Parakeet transcribed every
  candidate at 1.1% WER. Phonon accepts only `speed=1.0`. Slowing it by 10%
  with ffmpeg `atempo` kept UTMOS the same, but players have a speed control,
  so the book keeps the natural pace.

## Tests

```bash
.venv/bin/python -m pytest tests
```

The M4B test needs `ffmpeg` and `ffprobe`.
