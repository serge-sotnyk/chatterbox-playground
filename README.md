# Chatterbox TTS CLI

CLI tool for text-to-speech synthesis using Chatterbox models. Supports both English and multilingual text-to-speech generation.

## Features

- **English TTS**: Uses the optimized English Chatterbox model for best quality English speech
- **Multilingual TTS**: Supports 23+ languages using the multilingual Chatterbox model
- **Voice Cloning**: Support for custom voice prompts
- **Flexible Input**: Accept text directly or from files
- **Easy to Use**: Simple command-line interface

## Installation

1. Install the project using uv:
```bash
uv sync
```

2. The CLI tool will be available as `chatterbox-tts` after installation.

## Usage

### Basic Examples

```bash
# English text (uses English model)
chatterbox-tts "Hello world, this is a test!" -o output.wav

# French text (uses multilingual model)
chatterbox-tts "Bonjour le monde!" -l fr -o french.wav

# Chinese text (uses multilingual model)
chatterbox-tts "你好，今天天气真不错。" -l zh -o chinese.wav

# Read text from file
chatterbox-tts -i example_en.txt -o from_file.wav

# Use custom voice prompt
chatterbox-tts "Hello world" -o custom_voice.wav --audio-prompt your_voice.wav
```

### Command Line Options

```
positional arguments:
  text                  Text to synthesize (use quotes for multi-word text)

options:
  -h, --help            show this help message and exit
  -i INPUTFILE, --inputfile INPUTFILE
                        Read text from file instead of command line argument
  -l LANG, --lang LANG  Language code (default: en). Use 'en' for English model, 
                        or language codes like 'fr', 'zh', 'es', etc. for multilingual model
  -o OUTPUTFILE, --outputfile OUTPUTFILE
                        Output audio file path (e.g., output.wav)
  --audio-prompt AUDIO_PROMPT
                        Path to audio file to use as voice prompt for different voice synthesis
```

### Supported Languages

- **English** (`en`) - Uses specialized English model
- **Multilingual** - Supports 23+ languages including:
  - French (`fr`)
  - Spanish (`es`) 
  - German (`de`)
  - Chinese (`zh`)
  - Japanese (`ja`)
  - Korean (`ko`)
  - Portuguese (`pt`)
  - Russian (`ru`)
  - Italian (`it`)
  - And many more...

## Examples

The repository includes example text files:
- `example_en.txt` - English example
- `example_fr.txt` - French example

Try them out:
```bash
chatterbox-tts -i example_en.txt -o test_english.wav
chatterbox-tts -i example_fr.txt -l fr -o test_french.wav
```

## Requirements

- Python 3.11-3.12
- CUDA-capable GPU (recommended) or CPU
- Dependencies are automatically installed via uv

## How It Works

The CLI automatically selects the appropriate model:
- For English text (`--lang en`): Uses `ChatterboxTTS` model for optimal English quality
- For other languages: Uses `ChatterboxMultilingualTTS` model with specified language ID

Both models support voice cloning via the `--audio-prompt` parameter.
