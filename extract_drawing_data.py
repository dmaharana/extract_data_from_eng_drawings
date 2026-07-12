#!/usr/bin/env python3
import os
import sys
import json
import csv
import argparse
from typing import List, Optional
from PIL import Image
from pydantic import BaseModel, Field
import google.generativeai as genai

# Define the Pydantic models for structured output constraint from Gemini
# Note: We avoid optional fields and defaults here because the Gemini API schema builder
# in python-generativeai rejects schema definitions containing a 'default' field.
# The prompt instructs the model to use empty string "" for any missing or non-applicable values.
class TableData(BaseModel):
    name: str
    headers: List[str]
    rows: List[List[str]]

class DimensionData(BaseModel):
    balloon_number: str
    component_or_feature: str
    nominal_value: str
    unit: str
    tolerance: str
    type: str
    notes: str

class DrawingExtractionResult(BaseModel):
    tables: List[TableData]
    dimensions: List[DimensionData]

def sanitize_filename(name: str) -> str:
    """Sanitize the string for safe use in file names."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("_")

def main():
    parser = argparse.ArgumentParser(description="Extract tabular data and dimensions from engineering drawings.")
    parser.add_argument("--image", required=True, help="Path to the engineering drawing image file")
    parser.add_argument("--output-dir", default="output", help="Directory where CSV and JSON results will be written")
    
    args = parser.parse_args()

    # Verify input image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' does not exist.", file=sys.stderr)
        sys.exit(1)
        
    # Verify API key is present
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    print(f"Configuring Gemini API...")
    genai.configure(api_key=api_key)
    
    # Use gemini-3.5-flash as the default model
    model_name = "gemini-3.5-flash"
    print(f"Initializing model: {model_name}...")
    model = genai.GenerativeModel(model_name)
    
    # Load the image
    print(f"Loading image '{args.image}'...")
    try:
        img = Image.open(args.image)
    except Exception as e:
        print(f"Error opening image: {e}", file=sys.stderr)
        sys.exit(1)

    prompt = (
        "Analyze this engineering drawing and extract all structured data.\n\n"
        "1. Extract all tables found in the drawing (e.g. Parts List, BOM, Title Block/Schriftfeld, Revision History, technical specification tables).\n"
        "   - Give each table a descriptive name.\n"
        "   - Populate headers and rows correctly.\n"
        "2. Extract all dimensions, tolerances, surface finishes, geometric tolerances, and annotation notes from drawing elements.\n"
        "   - balloon_number: Use the numbered balloon/callout pointing to the dimension/feature (e.g. '1', '12') if visible. Use empty string \"\" if not present.\n"
        "   - component_or_feature: Describe the part feature being dimensioned (e.g., 'Flange outer diameter', 'Shaft undercut depth', 'Hollow shaft length', 'Left threaded hole').\n"
        "   - nominal_value: The nominal dimension value or text (e.g., '40', 'M4', '1.6', '0.02', '1x45°').\n"
        "   - unit: The unit of measurement (e.g., 'mm', 'degrees') if indicated or standard. Use empty string \"\" if implicit/not shown.\n"
        "   - tolerance: The limit/tolerance notation (e.g., '+0.1', 'H7 (+0.015)', '±0.05') if annotated. Use empty string \"\" if none.\n"
        "   - type: Categorize the dimension as one of: 'diameter', 'length', 'thread', 'chamfer', 'radius', 'pitch_circle_diameter', 'runout', 'roughness', 'angle', 'other'.\n"
        "   - notes: Any extra notes, repetitions (e.g., 'x4'), or annotations (e.g., '2 PLACES'). Use empty string \"\" if none.\n\n"
        "Important: Since all fields are required in the schema, use empty string \"\" if a specific attribute is missing or does not apply."
    )
    
    print("Calling Gemini API for structured extraction (this may take a few seconds)...")
    try:
        response = model.generate_content(
            contents=[prompt, img],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=DrawingExtractionResult,
                temperature=0.1  # Low temperature for deterministic/factual extraction
            )
        )
    except Exception as e:
        print(f"Error calling Gemini API: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse JSON output
    try:
        extracted_data = json.loads(response.text)
    except Exception as e:
        print(f"Error parsing JSON response from API: {e}", file=sys.stderr)
        print("Raw response:", response.text, file=sys.stderr)
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get base filename of the image for outputs
    image_base = os.path.splitext(os.path.basename(args.image))[0]

    # Process Tables -> CSV
    extracted_tables = extracted_data.get("tables", [])
    print(f"\nFound {len(extracted_tables)} table(s). Saving to CSV...")
    for idx, table in enumerate(extracted_tables):
        table_name = table.get("name", f"table_{idx}")
        sanitized_name = sanitize_filename(table_name)
        csv_filename = os.path.join(args.output_dir, f"{image_base}_{sanitized_name}.csv")
        
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        
        try:
            with open(csv_filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if headers:
                    writer.writerow(headers)
                writer.writerows(rows)
            print(f"  Saved: {csv_filename}")
        except Exception as e:
            print(f"  Error writing CSV '{csv_filename}': {e}", file=sys.stderr)

    # Process Dimensions -> JSON
    extracted_dims = extracted_data.get("dimensions", [])
    print(f"\nFound {len(extracted_dims)} dimension(s). Saving to JSON...")
    
    json_filename = os.path.join(args.output_dir, f"{image_base}_dimensions.json")
    try:
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump({"dimensions": extracted_dims}, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {json_filename}")
    except Exception as e:
        print(f"  Error writing JSON '{json_filename}': {e}", file=sys.stderr)

    print("\nExtraction complete!")

if __name__ == "__main__":
    main()
