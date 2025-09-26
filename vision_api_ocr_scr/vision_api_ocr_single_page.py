"""
OpenAI Vision API OCR Demo
==========================

This script demonstrates how to use OpenAI's Vision API to directly transcribe
text from images, providing an alternative to traditional OCR + AI correction.

Usage:
    python vision_api_ocr.py <image_path> [--api-key <key>]

Requirements:
    - OpenAI API key (set as environment variable OPENAI_API_KEY or pass via --api-key)
    - Image file (JPG, PNG, etc.)
"""

import argparse
import os
import sys
from pathlib import Path
import base64
from openai import OpenAI
import glob

"""
Load API key
"""
from dotenv import load_dotenv

load_dotenv()


def encode_image(image_path):
    """
    Converts images to base64 for OpenAI Vision API.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None


def transcribe_with_vision_api(image_path, api_key):
    """
    Send image to OpenAI Vision API for text transcription.
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Encode the image
        base64_image = encode_image(image_path)
        if not base64_image:
            return None, None

        # Get image format (e.g., 'jpg')
        image_format = Path(image_path).suffix.lower()
        if image_format.startswith('.'):
            image_format = image_format[1:]

        # Send request to OpenAI Vision API
        response = client.chat.completions.create(
            model="gpt-4o",  # <-- Switched model here
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
    "Please transcribe all visible text from this image. "
    "Preserve spelling, punctuation, and formatting. "
    "Maintain all original line breaks and paragraph structure as closely as possible. "
    "If a word is unreadable, use [illegible]. Return only the transcribed text."
)
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4000,
            temperature=0.1
        )
        
        # Extract usage information
        usage = response.usage
        usage_info = {
            'prompt_tokens': usage.prompt_tokens,
            'completion_tokens': usage.completion_tokens,
            'total_tokens': usage.total_tokens
        }
        
        transcribed_text = response.choices[0].message.content.strip()
        return transcribed_text, usage_info
        
    except Exception as e:
        print(f"Error calling OpenAI Vision API: {e}")
        if hasattr(e, 'response'):
            print(e.response.text)
        return None, None


def calculate_cost(usage_info, model="gpt-4o"):
    """
    Calculate estimated cost based on token usage.
    
    Args:
        usage_info (dict): Token usage information
        model (str): Model used for the API call
        
    Returns:
        dict: Cost breakdown
    """
    # Pricing per 1K tokens (as of 2024)
    pricing = {
        "gpt-3.5-turbo": {
            "input": 0.0005,   # $0.50 per 1M tokens
            "output": 0.0015   # $1.50 per 1M tokens
        },
        "gpt-4o": {
            "input": 0.005,    # $5.00 per 1M tokens
            "output": 0.015    # $15.00 per 1M tokens
        }
    }
    
    if model not in pricing:
        model = "gpt-4o"  # Default fallback
    
    input_cost = (usage_info['prompt_tokens'] / 1000) * pricing[model]["input"]
    output_cost = (usage_info['completion_tokens'] / 1000) * pricing[model]["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "model": model
    }


def print_usage_summary(usage_info, cost_info):
    """
    Print token usage and cost summary to terminal.
    
    Args:
        usage_info (dict): Token usage information
        cost_info (dict): Cost breakdown
    """
    print("\n" + "=" * 50)
    print("OPENAI VISION API USAGE SUMMARY")
    print("=" * 50)
    print(f"Model: {cost_info['model']}")
    print(f"Prompt Tokens:  {usage_info['prompt_tokens']:,}")
    print(f"Output Tokens:  {usage_info['completion_tokens']:,}")
    print(f"Total Tokens:   {usage_info['total_tokens']:,}")
    print("-" * 50)
    print(f"Input Cost:     ${cost_info['input_cost']:.4f}")
    print(f"Output Cost:    ${cost_info['output_cost']:.4f}")
    print(f"Total Cost:     ${cost_info['total_cost']:.4f}")
    print("=" * 50)

def save_transcription(text, image_path, output_dir="vision_results"):
    """
    Save transcribed text to a file in vision_results directory,
    appending _vision_api_ocr_single_page and a unique number if needed.
    Cleans up common model intro and horizontal rules.
    """
    # Clean unwanted intro and horizontal rules
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        # Remove intro line and horizontal rules
        if line.strip().lower().startswith("sure, here is the transcribed text"):
            continue
        if line.strip() == "---":
            continue
        cleaned_lines.append(line)
    cleaned_text = "\n".join(cleaned_lines).strip()

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get base filename without extension
    base_name = Path(image_path).stem
    base_output = f"{base_name}_vision_api_ocr_single_page"
    output_file = Path(output_dir) / f"{base_output}.txt"
    
    # If file exists, append _1, _2, etc.
    count = 1
    while output_file.exists():
        output_file = Path(output_dir) / f"{base_output}_{count}.txt"
        count += 1

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    print(f"Vision API transcription saved to: {output_file}")


def main():
    """Main function to run the Vision API OCR workflow on a single image."""
    parser = argparse.ArgumentParser(description="OCR using OpenAI Vision API on a single image")
    parser.add_argument(
        "image_path",
        help="Path to the image file to process"
    )
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY environment variable)")
    parser.add_argument("--output-dir", default="vision_results", help="Directory to save results (default: vision_results)")
    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key required.")
        print("Set OPENAI_API_KEY environment variable or use --api-key option.")
        sys.exit(1)

    # Check if image exists
    if not os.path.isfile(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found.")
        sys.exit(1)

    print(f"Processing image: {args.image_path}")
    print("-" * 50)

    transcribed_text, usage_info = transcribe_with_vision_api(args.image_path, api_key)

    if not transcribed_text:
        print("Error: Vision API transcription failed for this image.")
        sys.exit(1)

    print(f"Vision API transcription completed. Transcribed {len(transcribed_text)} characters.")

    if usage_info:
        cost_info = calculate_cost(usage_info, "gpt-4o")
        print_usage_summary(usage_info, cost_info)

    save_transcription(transcribed_text, args.image_path, args.output_dir)

    print("\nProcessing completed successfully!")


if __name__ == "__main__":
    main()