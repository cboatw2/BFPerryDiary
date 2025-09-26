#!/usr/bin/env python3
"""
OCR vs AI-Enhanced OCR Comparison Demo
======================================

This script demonstrates the difference between basic OCR and AI-enhanced OCR
by processing the same image with both methods and showing the results side by side.

Usage:
    python demo_comparison.py <image_path> [--api-key <key>]
"""

import argparse
import os
import sys
from pathlib import Path
import pytesseract
from PIL import Image
from openai import OpenAI


def perform_ocr(image_path):
    """Perform basic OCR using Tesseract."""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, config='--psm 3')
        return text.strip()
    except Exception as e:
        print(f"Error performing OCR: {e}")
        return None


def correct_with_ai(text, api_key):
    """Correct OCR text using OpenAI API."""
    try:
        client = OpenAI(api_key=api_key)
        
        prompt = f"""Please correct the following OCR text from a historical document. 
Fix obvious OCR errors, add punctuation, and improve formatting while preserving the original meaning.

Original OCR text:
{text}

Corrected text:"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at correcting OCR text from historical documents."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None


def main():
    """Main function to run the comparison demo."""
    parser = argparse.ArgumentParser(description="Compare OCR vs AI-enhanced OCR")
    parser.add_argument("image_path", help="Path to the image file to process")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY environment variable)")
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found.")
        sys.exit(1)
    
    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key required.")
        print("Set OPENAI_API_KEY environment variable or use --api-key option.")
        sys.exit(1)
    
    print("OCR vs AI-Enhanced OCR Comparison")
    print("=" * 50)
    print(f"Processing image: {args.image_path}")
    print()
    
    # Step 1: Basic OCR
    print("Step 1: Performing basic OCR with Tesseract...")
    ocr_text = perform_ocr(args.image_path)
    
    if not ocr_text:
        print("Error: OCR failed to extract text.")
        sys.exit(1)
    
    print("✓ OCR completed")
    print()
    
    # Step 2: AI Enhancement
    print("Step 2: Enhancing with AI...")
    ai_text = correct_with_ai(ocr_text, api_key)
    
    if not ai_text:
        print("Error: AI enhancement failed.")
        sys.exit(1)
    
    print("✓ AI enhancement completed")
    print()
    
    # Step 3: Show comparison
    print("COMPARISON RESULTS")
    print("=" * 50)
    
    print("BASIC OCR RESULT:")
    print("-" * 30)
    print(ocr_text)
    print("-" * 30)
    print()
    
    print("AI-ENHANCED RESULT:")
    print("-" * 30)
    print(ai_text)
    print("-" * 30)
    print()
    
    # Step 4: Save both results
    base_name = Path(args.image_path).stem
    
    # Save OCR result
    ocr_file = f"{base_name}_basic_ocr.txt"
    with open(ocr_file, 'w', encoding='utf-8') as f:
        f.write(ocr_text)
    
    # Save AI result
    ai_file = f"{base_name}_ai_enhanced.txt"
    with open(ai_file, 'w', encoding='utf-8') as f:
        f.write(ai_text)
    
    print("FILES SAVED:")
    print(f"  Basic OCR: {ocr_file}")
    print(f"  AI Enhanced: {ai_file}")
    print()
    
    # Show improvement statistics
    ocr_chars = len(ocr_text)
    ai_chars = len(ai_text)
    
    print("STATISTICS:")
    print(f"  Basic OCR characters: {ocr_chars}")
    print(f"  AI Enhanced characters: {ai_chars}")
    print(f"  Character difference: {ai_chars - ocr_chars:+d}")
    print()
    
    print("Demo completed! Compare the two text files to see the improvements.")


if __name__ == "__main__":
    main()