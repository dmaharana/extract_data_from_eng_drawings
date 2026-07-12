#!/usr/bin/env python3
import os
import sys
import json
import csv
import base64
import argparse
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI

# Define Pydantic models for local validation
class TableData(BaseModel):
    name: str = Field(description="Name or identifier of the table (e.g., 'Bill of Materials', 'Title Block')")
    headers: List[str] = Field(description="List of column headers/names in order")
    rows: List[List[str]] = Field(description="List of rows, where each row is a list of column cell values as strings")

class DimensionData(BaseModel):
    balloon_number: str = Field(description="The balloon/callout identifier number (e.g. '1', '14') pointing to this dimension, if present. Empty string if not present.")
    component_or_feature: str = Field(description="The component, feature, or part element associated with this dimension (e.g., 'Flange outer diameter', 'Left bearing shaft', 'Thread depth')")
    nominal_value: str = Field(description="The nominal dimension value or text (e.g., '40', 'M4', '1.6', '0.02', '1x45°')")
    unit: str = Field(description="The unit of the dimension (e.g., 'mm', 'degrees') or empty string if not specified/implicit")
    tolerance: str = Field(description="The tolerance or limit notation if specified (e.g., '+0.1', 'H7 (+0.015)', '±0.1') or empty string if none")
    type: str = Field(description="The dimension category (e.g., 'diameter', 'length', 'thread', 'chamfer', 'radius', 'pitch_circle_diameter', 'runout', 'roughness', 'angle', 'other')")
    notes: str = Field(description="Any extra metadata, notes, standard reference (e.g., DIN 509), datum references, repetition counts (e.g., 'x4', '2 places') or empty string if none")

class DrawingExtractionResult(BaseModel):
    tables: List[TableData] = Field(description="All tabular data blocks and metadata tables found in the drawing image")
    dimensions: List[DimensionData] = Field(description="All annotated dimensions, limits of size, geometric tolerances, and surface finishes shown on drawing elements")

def sanitize_filename(name: str) -> str:
    """Sanitize the string for safe use in file names."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("_")

def encode_image_to_base64(image_path: str) -> str:
    """Read image file and encode it to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def main():
    parser = argparse.ArgumentParser(description="Extract tabular data and dimensions from engineering drawings using an OpenAI-compliant LLM service.")
    parser.add_argument("--image", required=True, help="Path to the engineering drawing image file")
    parser.add_argument("--output-dir", default="output", help="Directory where CSV and JSON results will be written")
    parser.add_argument("--api-key", default=None, help="API key for the service (defaults to OPENAI_API_KEY env var)")
    parser.add_argument("--api-base", default=None, help="Base URL for the OpenAI-compliant service (defaults to OPENAI_API_BASE env var or OpenAI default)")
    parser.add_argument("--model", default="gpt-4o", help="Model name to use (default: gpt-4o)")
    
    args = parser.parse_args()

    # Verify input image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' does not exist.", file=sys.stderr)
        sys.exit(1)
        
    # Get API credentials
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    api_base = args.api_base or os.environ.get("OPENAI_API_BASE")
    
    # Handle environment check fallback for OpenRouter or other common services
    if not api_key:
        api_key = os.environ.get("OPEN_ROUTER_KEY")
        if api_key and not api_base:
            # If OpenRouter key is set but no base URL is specified, default to OpenRouter base URL
            api_base = "https://openrouter.ai/api/v1"
            
    if not api_key:
        print("Error: API Key is not provided. Set the OPENAI_API_KEY environment variable or pass --api-key.", file=sys.stderr)
        sys.exit(1)

    # Encode image to base64
    print(f"Encoding image '{args.image}' to base64...")
    try:
        base64_image = encode_image_to_base64(args.image)
        # Simple detection of mime type based on extension
        ext = os.path.splitext(args.image)[1].lower().strip(".")
        mime_type = f"image/{ext}" if ext in ["png", "jpg", "jpeg", "webp"] else "image/png"
    except Exception as e:
        print(f"Error reading image: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Initializing OpenAI-compliant client (Base URL: {api_base or 'Default OpenAI'})...")
    client = OpenAI(api_key=api_key, base_url=api_base)

    prompt = (
        "Analyze this engineering drawing and extract all structured data in JSON format.\n\n"
        "Your output must follow this JSON schema exactly:\n"
        "{\n"
        "  \"tables\": [\n"
        "    {\n"
        "      \"name\": \"string (e.g. 'Bill of Materials', 'Title Block')\",\n"
        "      \"headers\": [\"string\", ...],\n"
        "      \"rows\": [[\"string\", ...], ...]\n"
        "    }\n"
        "  ],\n"
        "  \"dimensions\": [\n"
        "    {\n"
        "      \"balloon_number\": \"string (numbered callout number if visible, otherwise empty string \\\"\\\")\",\n"
        "      \"component_or_feature\": \"string (feature being dimensioned, e.g. 'Flange outer diameter')\",\n"
        "      \"nominal_value\": \"string (nominal dimension value or text, e.g. '40', 'M4', '1.6', '1x45°')\",\n"
        "      \"unit\": \"string (unit of measurement if shown, e.g. 'mm', 'degrees', otherwise \\\"\\\")\",\n"
        "      \"tolerance\": \"string (tolerance/limit notation if visible, e.g. 'H7 (+0.015)', '+0.1', otherwise \\\"\\\")\",\n"
        "      \"type\": \"string (must be one of: 'diameter', 'length', 'thread', 'chamfer', 'radius', 'pitch_circle_diameter', 'runout', 'roughness', 'angle', 'other')\",\n"
        "      \"notes\": \"string (extra metadata, e.g., 'x4', '2 places', or standard DIN reference, otherwise \\\"\\\")\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Important instructions:\n"
        "1. Extract all tables found in the drawing (e.g. Parts List, BOM, Schriftfeld/Title Block, Revision History).\n"
        "2. Extract all dimensions, tolerances, surface finishes, geometric tolerances, and annotations on component drawing views.\n"
        "3. Since all schema fields are required in the output, use empty string \"\" if a specific attribute is missing or does not apply."
    )

    print(f"Calling LLM service (Model: {args.model})...")
    try:
        # Standard chat completions call with json_object format
        response = client.chat.completions.create(
            model=args.model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1
        )
        response_text = response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM service: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse and validate JSON response
    try:
        extracted_data = json.loads(response_text)
        # Optional validation using Pydantic to ensure schema correctness
        DrawingExtractionResult.model_validate(extracted_data)
    except Exception as e:
        print(f"Warning: Response validation/parsing failed: {e}", file=sys.stderr)
        print("Falling back to raw dictionary parsing...", file=sys.stderr)
        try:
            extracted_data = json.loads(response_text)
        except Exception as json_err:
            print(f"Error: Response is not valid JSON: {json_err}", file=sys.stderr)
            print("Raw response:", response_text, file=sys.stderr)
            sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
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
