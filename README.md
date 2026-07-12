# Engineering Drawing Data Extraction Toolkit

This toolkit provides two methods to scan engineering drawing images and extract:
1. **Tabular Data** (such as Bill of Materials, title blocks, revision history) into CSV files.
2. **Dimension Annotations** (such as diameters, tolerances, surface finishes) mapped to their corresponding balloon callout numbers into a JSON file.

---

## Extraction Methods

### Method 1: Local Offline Extraction (`extract_drawing_data.py`)
This method runs **entirely offline** on your local machine. It uses the `rapidocr-onnxruntime` library to perform OCR without needing external cloud APIs or Tesseract. It then applies deterministic spatial heuristics to align and cluster table cells and pair dimensions with balloon numbers.

* **Best for**: Environments without API access, sensitive offline data, and strict deterministic layout parsing.
* **Usage**:
  ```bash
  python3 extract_drawing_data.py --image <path_to_image> --output-dir <output_directory>
  ```
* **How it works**:
  - **Spatial Partitioning**: Automatically separates drawing views from table borders.
  - **Regular Grid-Fitting**: Matches table rows to an aligned vertical grid to avoid rows splitting on skewed images.
  - **Order-Preserving Column Alignment**: Clusters cell x-coordinates globally and maps cell values horizontally to preserve column order (even when cells are missing).
  - **Proximity Balloon Pairing**: Automatically associates dimensions with the closest standalone balloon number.

---

### Method 2: OpenAI-Compliant LLM Service (`extract_drawing_data_openai.py`)
This method sends the image to a multimodal Large Language Model (like OpenAI GPT-4o, OpenRouter models, LocalAI, vLLM, or Ollama) using the standard OpenAI python client.

* **Best for**: Leverages vision-language models for semantic comprehension of complex annotations and tables.
* **Usage**:
  ```bash
  python3 extract_drawing_data_openai.py \
    --image <path_to_image> \
    --output-dir <output_directory> \
    --model <model_name> \
    --api-key <api_key> \
    --api-base <custom_base_url>
  ```
* **Parameters**:
  - `--model`: Model name (default: `gpt-4o`).
  - `--api-key`: API key (defaults to `OPENAI_API_KEY` or `OPEN_ROUTER_KEY` environment variables).
  - `--api-base`: Endpoint base URL (defaults to `OPENAI_API_BASE` or OpenRouter defaults).

---

## Output File Structures

### 1. CSV Tables (`*_table_*.csv`)
For each table detected in the drawing, the scripts output a separate CSV file containing the binned headers and aligned rows. For example, a Bill of Materials table outputs columns aligned as:
```csv
Menge,Benennung,Sachnummer / Norm,Bemerkung
Stck.,[Rillekugll.oger,DIN 625  6009,
S+ck.,ist anzring,,S235JR
```

### 2. JSON Dimensions (`*_dimensions.json`)
Dimension callouts are exported to a structured JSON file mapping dimension values to their corresponding balloon callout number:
```json
{
  "dimensions": [
    {
      "balloon_number": "15",
      "component_or_feature": "Feature near balloon 15",
      "nominal_value": "76",
      "unit": "mm",
      "tolerance": "±0.1",
      "type": "length",
      "notes": "76±0.1"
    },
    {
      "balloon_number": "16",
      "component_or_feature": "Feature near balloon 16",
      "nominal_value": "10",
      "unit": "mm",
      "tolerance": "H7(0.05",
      "type": "other",
      "notes": "10H7(0.05"
    }
  ]
}
```
