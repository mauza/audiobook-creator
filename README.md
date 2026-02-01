# Audiobook Creator

A Python tool that converts text to audiobooks using the Kokoro TTS model. This tool is designed to help create audiobooks from text files, with configurable voice and language options.

## Features

- Convert text files to audiobooks
- Support for multiple languages and voices
- Configurable speech speed
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
audiobook-creator convert-file input.txt --output-dir output --voice af_heart --language a --speed 1.0
```

2. Convert text directly to an audiobook:
```bash
audiobook-creator convert-text "Your text here" output_filename --output-dir output --voice af_heart --language a --speed 1.0
```

### Options

- `--output-dir`, `-o`: Directory to save the audio file (default: "output")
- `--voice`, `-v`: Voice to use for synthesis (default: "af_heart")
- `--language`, `-l`: Language code (default: "a" for American English)
- `--speed`, `-s`: Speech speed multiplier (default: 1.0)
- `--encoding`, `-e`: Input file encoding (default: "utf-8")

### Language Codes

- `a`: American English
- `b`: British English
- `e`: Spanish
- `f`: French
- `h`: Hindi
- `i`: Italian
- `j`: Japanese
- `p`: Brazilian Portuguese
- `z`: Mandarin Chinese

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

- [Kokoro TTS](https://github.com/hexgrad/kokoro) for the text-to-speech model
- [Click](https://click.palletsprojects.com/) for the CLI framework
- [Rich](https://github.com/Textualize/rich) for the terminal formatting 