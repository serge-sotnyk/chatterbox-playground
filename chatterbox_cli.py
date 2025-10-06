#!/usr/bin/env python3
"""
Chatterbox TTS Command Line Interface

A simple CLI tool for text-to-speech synthesis using Chatterbox models.
Supports both English and multilingual models.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import torchaudio as ta

# Monkey patch torch.load to handle CUDA unavailability
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    """Patched torch.load that automatically adds map_location='cpu' when CUDA is unavailable."""
    if not torch.cuda.is_available() and 'map_location' not in kwargs:
        kwargs['map_location'] = torch.device('cpu')
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# Fix for perth watermarker import issue
try:
    import perth
    from perth.perth_net import PerthImplicitWatermarker
    perth.PerthImplicitWatermarker = PerthImplicitWatermarker
except ImportError:
    print("Warning: Could not import perth watermarker, continuing anyway...")

from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


def read_text_from_file(file_path: str) -> str:
    """Read text from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Input file '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        sys.exit(1)


def generate_speech(text: str, language: str, output_file: str, audio_prompt_path: Optional[str] = None):
    """Generate speech using appropriate model based on language."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")

    try:
        if language == "en":
            print("Loading English TTS model...")
            model = ChatterboxTTS.from_pretrained(device=device)
            print("Generating speech...")

            if audio_prompt_path:
                wav = model.generate(text, audio_prompt_path=audio_prompt_path)
            else:
                wav = model.generate(text)
        else:
            print(f"Loading Multilingual TTS model for language: {language}...")
            model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            print("Generating speech...")

            if audio_prompt_path:
                wav = model.generate(text, language_id=language, audio_prompt_path=audio_prompt_path)
            else:
                wav = model.generate(text, language_id=language)

        print(f"Saving audio to: {output_file}")
        ta.save(output_file, wav, model.sr)
        print("Speech generation completed successfully!")

    except Exception as e:
        import traceback
        print(f"Error during speech generation: {e}")
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main CLI function."""

    parser = argparse.ArgumentParser(
        description="Generate speech from text using Chatterbox TTS models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Hello world" -o output.wav
  %(prog)s "Bonjour le monde" -l fr -o french.wav
  %(prog)s -i input.txt -l zh -o chinese.wav
  %(prog)s "Hello" -o output.wav --audio-prompt voice.wav
        """
    )

    # Text input (positional or from file)
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to synthesize (use quotes for multi-word text)"
    )

    parser.add_argument(
        "-i", "--inputfile",
        help="Read text from file instead of command line argument"
    )

    # Language selection
    parser.add_argument(
        "-l", "--lang",
        default="en",
        help="Language code (default: en). Use 'en' for English model, or language codes like 'fr', 'zh', 'es', etc. for multilingual model"
    )

    # Output file
    parser.add_argument(
        "-o", "--outputfile",
        required=True,
        help="Output audio file path (e.g., output.wav)"
    )

    # Audio prompt for voice cloning
    parser.add_argument(
        "-p", "--audio-prompt",
        help="Path to audio file to use as voice prompt for different voice synthesis"
    )

    args = parser.parse_args()

    # Validate input
    if not args.text and not args.inputfile:
        parser.error("Either provide text as argument or specify --inputfile/-i")

    if args.text and args.inputfile:
        parser.error("Cannot specify both text argument and --inputfile/-i")

    # Get text content
    if args.inputfile:
        text = read_text_from_file(args.inputfile)
    else:
        text = args.text

    if not text:
        print("Error: No text provided for synthesis")
        sys.exit(1)

    # Validate audio prompt file if provided
    if args.audio_prompt and not Path(args.audio_prompt).exists():
        print(f"Error: Audio prompt file '{args.audio_prompt}' not found.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_path = Path(args.outputfile)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"Language: {args.lang}")
    print(f"Output: {args.outputfile}")
    if args.audio_prompt:
        print(f"Audio prompt: {args.audio_prompt}")

    # Generate speech
    generate_speech(text, args.lang, args.outputfile, args.audio_prompt)


if __name__ == "__main__":
    main()
