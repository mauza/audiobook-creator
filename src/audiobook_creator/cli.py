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
    "--voice",
    "-v",
    default="Ryan",
    help="Speaker voice (Vivian, Serena, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee, Uncle_Fu)",
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
    default="cuda:0",
    help="Device to run the model on (e.g. cuda:0, cpu)",
)
@click.option(
    "--encoding",
    "-e",
    default="utf-8",
    help="Input file encoding",
)
def convert_file(
    input_file: str,
    output_dir: str,
    voice: str,
    language: str,
    model: str,
    device: str,
    encoding: str,
):
    """Convert a text file to an audiobook."""
    try:
        converter = TTSConverter(
            output_dir=output_dir,
            voice=voice,
            language=language,
            model_name=model,
            device=device,
        )

        output_path = converter.convert_file(
            input_file=input_file,
            encoding=encoding,
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
    "--voice",
    "-v",
    default="Ryan",
    help="Speaker voice (Vivian, Serena, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee, Uncle_Fu)",
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
    default="cuda:0",
    help="Device to run the model on (e.g. cuda:0, cpu)",
)
def convert_text(
    text: str,
    output_filename: str,
    output_dir: str,
    voice: str,
    language: str,
    model: str,
    device: str,
):
    """Convert text to an audiobook."""
    try:
        converter = TTSConverter(
            output_dir=output_dir,
            voice=voice,
            language=language,
            model_name=model,
            device=device,
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
