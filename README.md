# Engineering Drawings Data Extraction

An automated toolset to extract tabular data (BOMs, parts lists, title blocks) and drawing annotations (dimensions, tolerances, types, balloon numbers) from engineering drawings. 

The project offers two complementary extraction methodologies:
1. **Multimodal AI Approach (`extract_drawing_data.py`)**: Leverages Google Gemini 3.5 Flash for advanced visual/spatial understanding and structured Pydantic-constrained JSON parsing.
2. **Offline OCR Approach (`extract_drawing_data_ocr.py`)**: Uses a lightweight, local, and offline RapidOCR (ONNX Runtime) pipeline with proximity heuristics and grid clustering to reconstruct tables and map dimensions.

---

## Key Features

- **Table Reconstruction**: Automatically detects, groups, and aligns technical specifications, Bill of Materials (BOMs), revisions, or title blocks into aligned columns and exports them as CSVs.
- **Dimension & Tolerance Association**: Extracts dimensional details, nominal values, units (e.g., `mm`, `degrees`), tolerances (e.g., `H7`, `±0.05`), feature descriptions, and types (e.g., `diameter`, `length`, `radius`, `thread`).
- **Balloon Mapping**: Correlates and links dimensions to their respective drawing balloon numbers/callouts (e.g., `1`, `12`).
- **High Portability**: Zero-cloud dependency option available via the offline OCR script, or highly accurate LLM understanding with Gemini.

---

## Directory Structure

```text
eng-drawings/
├── extract_drawing_data.py       # Gemini-based visual & structured data extractor
├── extract_drawing_data_ocr.py   # Offline OCR-based grid & proximity data extractor
├── output_gearbox/               # Extracted outputs for the gearbox drawing
├── output_shaft/                 # Extracted outputs for the shaft drawing
├── Screenshot from ...-01.png    # Input sample drawing: Shaft
└── Screenshot from ...-58.png    # Input sample drawing: Gearbox
```

---

## Prerequisites & Installation

### 1. System Requirements
- **Python**: 3.8 or higher.
- **C++ Compiler & ONNX Dependencies** (Required for `RapidOCR` offline runtime).

### 2. Install Required Packages
Create a virtual environment and install the required dependencies:

```bash
# Optional: Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install google-generativeai Pillow pydantic rapidocr_onnxruntime
```

---

## Usage Guide

### Method A: Multimodal Gemini AI Extraction (Recommended)
This script utilizes the `gemini-3.5-flash` model to analyze the image and generate precise structured output conforming to a Pydantic schema.

#### 1. Set up your Gemini API Key
```bash
export GEMINI_API_KEY="your_actual_gemini_api_key_here"
```

#### 2. Run the Extractor
```bash
python3 extract_drawing_data.py \
    --image "/path/to/drawing.png" \
    --output-dir "output_directory"
```

---

### Method B: Offline OCR Extraction
This script runs entirely locally. It identifies text elements using `RapidOCR`, divides the drawing into "view regions" versus "table/title block regions", clusters aligned elements into tables, and maps geometric proximity of labels to detect dimensions and matching balloon callouts.

#### 1. Run the OCR Extractor
```bash
python3 extract_drawing_data_ocr.py \
    --image "/path/to/drawing.png" \
    --output-dir "output_directory"
```

---

## Output Formats

All scripts save structured outputs into the specified `--output-dir` folder:

### 1. Tabular Data (CSV)
Tables, BOMs, and title blocks are exported to CSV files named `{image_base}_{table_name}.csv`.
Example excerpt from a Bill of Materials (`output_gearbox/..._table_1.csv`):
```csv
Stck.,Logerdeckel klein,,,S235JR
Stck.,[Rillekugll.oger,DIN 625  6009,,
Stck.,Kegelrdllenloger,DIN 720  30203,,
Stck.,Passfeder gro?,DIN 6885  B 12 x 8 × 22,,
```

### 2. Dimensional Data (JSON)
Dimensions and annotations are saved as a structured JSON array (`{image_base}_dimensions.json`).
Each dimension object includes:
- `balloon_number`: The callout index identifier (e.g., `12`, `3`).
- `component_or_feature`: Description of the item or feature.
- `nominal_value`: The standard dimension value (e.g., `5`, `M4`, `1.6`).
- `unit`: The unit of measurement (e.g., `mm`, `degrees`).
- `tolerance`: Standard fit class or upper/lower bound limits.
- `type`: Category (e.g., `diameter`, `length`, `thread`, `chamfer`, `radius`, `runout`, `roughness`).
- `notes`: Extraneous drawing labels or repetitions.

Example format:
```json
{
  "dimensions": [
    {
      "balloon_number": "12",
      "component_or_feature": "Shaft keyway depth",
      "nominal_value": "5",
      "unit": "mm",
      "tolerance": "+0.1",
      "type": "length",
      "notes": "5 +0.1"
    }
  ]
}
```

---

## Example Datasets Included

This repository contains two pre-analyzed examples in their respective folders:
- **`output_shaft/`**: Reconstructed from the lathe sample shaft drawing (`Screenshot from ...-01.png`).
- **`output_gearbox/`**: Reconstructed from the gearbox drawing (`Screenshot from ...-58.png`) which features complex multi-item BOM tables.
