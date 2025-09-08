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

    # Determine device (prefer CUDA if available, but fallback to CPU for problematic models)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")

    try:
        if language == "en":
            # Use English model
            print("Loading English TTS model...")
            model = ChatterboxTTS.from_pretrained(device=device)
            print("Generating speech...")

            if audio_prompt_path:
                wav = model.generate(text, audio_prompt_path=audio_prompt_path)
            else:
                wav = model.generate(text)
        else:
            # Use multilingual model - force CPU mode due to CUDA compatibility issues
            print("Loading Multilingual TTS model (using CPU mode due to compatibility)...")
            print("Note: Multilingual model may have CUDA compatibility issues, forcing CPU mode...")

            # Force CPU mode for multilingual model to avoid CUDA deserialization issues
            model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")

            print(f"Generating speech in language: {language}")

            if audio_prompt_path:
                wav = model.generate(text, language_id=language, audio_prompt_path=audio_prompt_path)
            else:
                wav = model.generate(text, language_id=language)

        print(f"Generated audio tensor shape: {wav.shape if hasattr(wav, 'shape') else 'Unknown'}")
        print(f"Model sample rate: {model.sr}")

        # Save the generated audio
        print(f"Saving audio to: {output_file}")
        ta.save(output_file, wav, model.sr)
        print("Speech generation completed successfully!")

    except Exception as e:
        import traceback
        print(f"Error during speech generation: {e}")
        print("Full traceback:")
        traceback.print_exc()

        # If multilingual model fails, provide helpful message
        if language != "en":
            print("\nNote: The multilingual model may have compatibility issues.")
            print("This is a known issue with the chatterbox-tts package on some systems.")
            print("Try using the English model instead by omitting the -l parameter or using -l en")

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
        "--audio-prompt",
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
