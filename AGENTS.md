# Audiobook Creator

Python CLI (`uv`, `src/`, `tests/`) that turns an EPUB, PDF, Markdown, or text file into a chaptered `.m4b` (or `wav`/`mp3`) audiobook. Default TTS engine is Phonon over HTTP (`Hadley` voice), with optional local Kokoro/Qwen3-TTS.

## Usage & checks
- Setup: `uv venv && uv pip install -e ".[dev]"`
- Tests: `.venv/bin/python -m pytest tests` (M4B tests require `ffmpeg` and `ffprobe`).
- Against home cluster Phonon (`phonon-tts.phonon-tts:8880`):
  ```bash
  kpf -n phonon-tts svc/phonon-tts 8880 -- \
    'PHONON_URL=$URL .venv/bin/audiobook-creator convert-file book.epub -f m4b -o audiobooks'
  ```

## Operational rules
- **Run Phonon requests serially** (1 stream ~42x realtime vs 2 streams ~31x; the pod is CPU-bound at 2 cores). Phonon accepts only `speed=1.0`.
- **Audiobookshelf delivery:** copy finished `.m4b` to `unraid` under `/mnt/user/Media/Audio Books/<Author - Title>/` (watch spaces in `scp`/remote paths; quote properly).
- Never commit generated files under `audiobooks/` or `stories/`.
