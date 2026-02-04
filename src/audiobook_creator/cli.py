"""Command-line interface for the audiobook creator."""

import sys
from pathlib import Path

import click
from rich.console import Console

from audiobook_creator.tts import TTSConverter

console = Console()


@click.group()
def cli():
    """Convert text to audiobooks using Qwen3-TTS."""
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    "-o",
    default="output",
    help="Directory to save the audio file",
    type=click.Path(),
)
@click.option(
    "--tts-engine",
    type=click.Choice(["qwen3", "kokoro"]),
    default="kokoro",
    help="TTS engine to use (default: kokoro)",
)
@click.option(
    "--voice",
    "-v",
    default=None,
    help="Speaker voice (qwen3: Ryan, Vivian, etc.; kokoro: af_heart, af_bella, etc.)",
)
@click.option(
    "--language",
    "-l",
    default="English",
    help="Language (English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian)",
)
@click.option(
    "--model",
    "-m",
    default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    help="Qwen3-TTS model name or path",
)
@click.option(
    "--device",
    "-d",
    default="auto",
    help="Device(s): auto (all GPUs), cuda:0, cuda:0,cuda:1, or cpu",
)
@click.option(
    "--encoding",
    "-e",
    default="utf-8",
    help="Input file encoding",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["wav", "mp3"]),
    default="wav",
    help="Output audio format",
)
@click.option(
    "--chapters/--no-chapters",
    default=False,
    help="Split output by chapters",
)
@click.option(
    "--skip-front-matter/--include-front-matter",
    default=True,
    help="Skip front matter in EPUB files (cover, TOC, copyright, etc.)",
)
@click.option(
    "--voice-sample",
    type=click.Path(exists=True),
    default=None,
    help="Path to a reference audio file for voice cloning (10-30s, clean audio; qwen3 only)",
)
@click.option(
    "--voice-text",
    default=None,
    help="Transcript of the voice sample (recommended for better quality; qwen3 only)",
)
@click.option(
    "--batch-size",
    "-b",
    default=4,
    type=int,
    help="Number of text chunks per batched inference call (default 4)",
)
def convert_file(
    input_file: str,
    output_dir: str,
    tts_engine: str,
    voice: str,
    language: str,
    model: str,
    device: str,
    encoding: str,
    output_format: str,
    chapters: bool,
    skip_front_matter: bool,
    voice_sample: str,
    voice_text: str,
    batch_size: int,
):
    """Convert a text file to an audiobook."""
    try:
        if tts_engine == "kokoro" and (voice_sample or voice_text):
            console.print(
                "[yellow]Warning:[/yellow] Voice cloning (--voice-sample, --voice-text) "
                "is not supported with Kokoro. These options will be ignored."
            )
            voice_sample = None
            voice_text = None

        converter = TTSConverter(
            output_dir=output_dir,
            voice=voice,
            language=language,
            model_name=model,
            device=device,
            output_format=output_format,
            voice_sample=voice_sample,
            voice_text=voice_text,
            batch_size=batch_size,
            tts_engine=tts_engine,
        )

        output_path = converter.convert_file(
            input_file=input_file,
            encoding=encoding,
            chapter_aware=chapters,
            skip_front_matter=skip_front_matter,
        )

        console.print(f"[green]Successfully created audiobook:[/green] {output_path}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument("text")
@click.argument("output_filename")
@click.option(
    "--output-dir",
    "-o",
    default="output",
    help="Directory to save the audio file",
    type=click.Path(),
)
@click.option(
    "--tts-engine",
    type=click.Choice(["qwen3", "kokoro"]),
    default="kokoro",
    help="TTS engine to use (default: kokoro)",
)
@click.option(
    "--voice",
    "-v",
    default=None,
    help="Speaker voice (qwen3: Ryan, Vivian, etc.; kokoro: af_heart, af_bella, etc.)",
)
@click.option(
    "--language",
    "-l",
    default="English",
    help="Language (English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian)",
)
@click.option(
    "--model",
    "-m",
    default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    help="Qwen3-TTS model name or path",
)
@click.option(
    "--device",
    "-d",
    default="auto",
    help="Device(s): auto (all GPUs), cuda:0, cuda:0,cuda:1, or cpu",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["wav", "mp3"]),
    default="wav",
    help="Output audio format",
)
@click.option(
    "--voice-sample",
    type=click.Path(exists=True),
    default=None,
    help="Path to a reference audio file for voice cloning (10-30s, clean audio; qwen3 only)",
)
@click.option(
    "--voice-text",
    default=None,
    help="Transcript of the voice sample (recommended for better quality; qwen3 only)",
)
@click.option(
    "--batch-size",
    "-b",
    default=4,
    type=int,
    help="Number of text chunks per batched inference call (default 4)",
)
def convert_text(
    text: str,
    output_filename: str,
    output_dir: str,
    tts_engine: str,
    voice: str,
    language: str,
    model: str,
    device: str,
    output_format: str,
    voice_sample: str,
    voice_text: str,
    batch_size: int,
):
    """Convert text to an audiobook."""
    try:
        if tts_engine == "kokoro" and (voice_sample or voice_text):
            console.print(
                "[yellow]Warning:[/yellow] Voice cloning (--voice-sample, --voice-text) "
                "is not supported with Kokoro. These options will be ignored."
            )
            voice_sample = None
            voice_text = None

        converter = TTSConverter(
            output_dir=output_dir,
            voice=voice,
            language=language,
            model_name=model,
            device=device,
            output_format=output_format,
            voice_sample=voice_sample,
            voice_text=voice_text,
            batch_size=batch_size,
            tts_engine=tts_engine,
        )

        output_path = converter.convert_text(
            text=text,
            output_filename=output_filename,
        )

        console.print(f"[green]Successfully created audiobook:[/green] {output_path}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
