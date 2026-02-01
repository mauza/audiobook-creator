# Audiobook Creator

A Python tool that converts text to audiobooks using the Qwen3-TTS model. This tool is designed to help create audiobooks from text files, with configurable voice and language options.

## Features

- Convert text files to audiobooks
- Support for multiple languages and voices
- Progress tracking during conversion
- Docker support for easy deployment

## Installation

### Using pip

```bash
pip install audiobook-creator
```

### Using Docker

```bash
docker build -t audiobook-creator .
```

## Usage

### Command Line Interface

The tool provides two main commands:

1. Convert a text file to an audiobook:
```bash
audiobook-creator convert-file input.txt --output-dir output --voice Ryan --language English
```

2. Convert text directly to an audiobook:
```bash
audiobook-creator convert-text "Your text here" output_filename --output-dir output --voice Ryan --language English
```

### Options

- `--output-dir`, `-o`: Directory to save the audio file (default: "output")
- `--voice`, `-v`: Speaker voice (default: "Ryan")
- `--language`, `-l`: Language (default: "English")
- `--model`, `-m`: Qwen3-TTS model name or path (default: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
- `--device`, `-d`: Device to run the model on (default: "cuda:0")
- `--encoding`, `-e`: Input file encoding (default: "utf-8", convert-file only)

### Available Speakers

Vivian, Serena, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee, Uncle_Fu

### Available Languages

Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

### Using Docker

```bash
# Convert a text file
docker run -v $(pwd):/app audiobook-creator convert-file input.txt

# Convert text directly
docker run -v $(pwd):/app audiobook-creator convert-text "Your text here" output_filename
```

## Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/audiobook-creator.git
cd audiobook-creator
```

2. Install development dependencies:
```bash
uv pip install -e ".[dev]"
```

3. Run tests:
```bash
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) for the text-to-speech model
- [Click](https://click.palletsprojects.com/) for the CLI framework
- [Rich](https://github.com/Textualize/rich) for the terminal formatting
