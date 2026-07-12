#!/usr/bin/env python3
import os
import sys
import json
import csv
import re
import math
import argparse
import itertools
from rapidocr_onnxruntime import RapidOCR

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def sanitize_filename(name: str) -> str:
    """Sanitize the string for safe use in file names."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("_")

def align_rows_to_columns(rows):
    """
    Align elements in rows to a consistent set of columns using 1D clustering on x-centers.
    """
    if not rows:
        return []
        
    # Gather all items to determine global column centers
    all_items = [item for row in rows for item in row]
    if not all_items:
        return []
        
    cxs = sorted([item["cx"] for item in all_items])
    
    # Greedy 1D clustering of x-coordinates with a 15-pixel gap threshold
    clusters = []
    current_cluster = [cxs[0]]
    for x in cxs[1:]:
        if x - current_cluster[-1] > 15:
            clusters.append(current_cluster)
            current_cluster = [x]
        else:
            current_cluster.append(x)
    clusters.append(current_cluster)
    
    # Column centers and count
    col_centers = [sum(c) / len(c) for c in clusters]
    max_len = len(col_centers)
    
    # Align each row's items to these centers
    aligned_table = []
    for row in rows:
        aligned_row = [""] * max_len
        n_items = len(row)
        
        # Find the best assignment of items to column indices that preserves left-to-right order
        best_comb = None
        min_dist = float('inf')
        
        # Generate all combinations of column indices of size n_items
        for comb in itertools.combinations(range(max_len), n_items):
            dist = sum(abs(row[i]["cx"] - col_centers[c_idx]) for i, c_idx in enumerate(comb))
            if dist < min_dist:
                min_dist = dist
                best_comb = comb
                
        if best_comb:
            for i, c_idx in enumerate(best_comb):
                aligned_row[c_idx] = row[i]["text"]
        else:
            # Fallback direct mapping
            for i, item in enumerate(row):
                if i < max_len:
                    aligned_row[i] = item["text"]
                    
        aligned_table.append(aligned_row)
        
    return aligned_table

def main():
    parser = argparse.ArgumentParser(description="Extract tabular data and dimensions from engineering drawings (Offline OCR).")
    parser.add_argument("--image", required=True, help="Path to the engineering drawing image file")
    parser.add_argument("--output-dir", default="output", help="Directory where CSV and JSON results will be written")
    
    args = parser.parse_args()

    # Verify input image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Initializing RapidOCR engine...")
    try:
        engine = RapidOCR()
    except Exception as e:
        print(f"Error initializing RapidOCR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Running OCR on '{args.image}'...")
    try:
        result, elapse = engine(args.image)
    except Exception as e:
        print(f"Error running OCR: {e}", file=sys.stderr)
        sys.exit(1)
        
    if not result:
        print("No text elements detected in the drawing.", file=sys.stderr)
        sys.exit(0)
        
    print(f"Successfully detected {len(result)} text elements.")
    
    # Process text boxes
    items = []
    for idx, item in enumerate(result):
        box, text, conf = item
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        width = max_x - min_x
        height = max_y - min_y
        
        items.append({
            "text": text.strip(),
            "cx": cx,
            "cy": cy,
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "width": width,
            "height": height,
            "box": box
        })

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    image_base = os.path.splitext(os.path.basename(args.image))[0]

    # Partition elements: Table regions vs Drawing view regions
    # Table/Metadata regions are typically at the bottom of the sheet or on the right side
    table_items = []
    drawing_items = []
    
    for item in items:
        # Standard table region heuristics: bottom half of sheet (cy > 500) OR right hand side (cx > 660 and cy > 350)
        if item["cy"] > 500 or (item["cx"] > 660 and item["cy"] > 350):
            table_items.append(item)
        else:
            drawing_items.append(item)
            
    print(f"Partitioned: {len(table_items)} table elements, {len(drawing_items)} drawing view elements.")

    # 1. Reconstruct Tables
    if table_items:
        # Group items into rows using a y-threshold of 8 pixels
        sorted_table_ys = sorted(table_items, key=lambda x: x["cy"])
        rows = []
        for item in sorted_table_ys:
            placed = False
            for row in rows:
                row_y = sum(r["cy"] for r in row) / len(row)
                if abs(item["cy"] - row_y) < 8:
                    row.append(item)
                    placed = True
                    break
            if not placed:
                rows.append([item])
                
        # Sort each row horizontally
        for row in rows:
            row.sort(key=lambda x: x["cx"])
            
        # Sort rows vertically by average y
        rows.sort(key=lambda r: sum(item["cy"] for item in r) / len(r))
        
        # Identify separate tables (vertical sequences of contiguous rows spaced by < 35 pixels)
        table_groups = []
        current_table = []
        for row in rows:
            if len(row) >= 2 or (current_table and len(row) >= 1):
                row_y = sum(item["cy"] for item in row) / len(row)
                if not current_table:
                    current_table.append(row)
                else:
                    prev_y = sum(item["cy"] for item in current_table[-1]) / len(current_table[-1])
                    if (row_y - prev_y) < 35:
                        current_table.append(row)
                    else:
                        if len(current_table) >= 3:  # Min 3 rows to be a table
                            table_groups.append(current_table)
                        current_table = [row]
            else:
                if len(current_table) >= 3:
                    table_groups.append(current_table)
                current_table = []
        if len(current_table) >= 3:
            table_groups.append(current_table)
            
        print(f"Reconstructing {len(table_groups)} table(s) from table elements...")
        for t_idx, table in enumerate(table_groups):
            # Flatten table items
            flat_table_items = [item for r in table for item in r]
            ys = [item["cy"] for item in flat_table_items]
            min_y = min(ys)
            
            # Grid-fit vertical spacing to align rows mathematically
            best_err, best_h, best_y0 = float('inf'), 14.5, min_y
            for h in [x/10.0 for x in range(120, 250)]:
                for y0 in [x/2.0 for x in range(int(min_y*2)-10, int(min_y*2)+10)]:
                    err = sum(abs(y - (y0 + round((y - y0)/h)*h)) for y in ys)
                    if err < best_err:
                        best_err, best_h, best_y0 = err, h, y0
                        
            # Group into regular grid rows
            grid_rows = {}
            for item in flat_table_items:
                row_idx = round((item["cy"] - best_y0) / best_h)
                if row_idx not in grid_rows:
                    grid_rows[row_idx] = []
                grid_rows[row_idx].append(item)
                
            # Sort each row horizontally
            aligned_rows = []
            for r_idx in sorted(grid_rows.keys()):
                aligned_rows.append(sorted(grid_rows[r_idx], key=lambda x: x["cx"]))
                
            # Align row items into columns
            csv_rows = align_rows_to_columns(aligned_rows)
            
            # Save table to CSV
            csv_path = os.path.join(args.output_dir, f"{image_base}_table_{t_idx+1}.csv")
            try:
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerows(csv_rows)
                print(f"  Saved Table {t_idx+1}: {csv_path}")
            except Exception as e:
                print(f"  Error saving Table {t_idx+1} to CSV: {e}", file=sys.stderr)
    else:
        print("No table elements found.")

    # 2. Extract Dimensions and Balloon Numbers
    balloon_candidates = []
    dimension_candidates = []
    
    # Regex to identify dimensions, limits of size, geometric tolerances, and surface finishes
    dim_regex = re.compile(
        r'(?i)'
        r'(\b\d+([.,]\d+)?\b)|'            # Numbers (integers or floats)
        r'([RMØoOp]\d+)|'                   # R19, M4, Ø3, SR3
        r'(\b\d+\s*x\s*45[°*"]\b)|'         # Chamfers (1x45°)
        r'(\b\d+\s*PLACES\b)|'             # PLACES count
        r'(\b\d+H7\b)'                      # Tolerances (10H7)
    )
    
    # Standalone balloon number pattern (1 to 2 digits)
    balloon_regex = re.compile(r'^\d{1,2}$')
    
    for item in drawing_items:
        text = item["text"]
        
        # Check if it is a balloon candidate
        is_balloon = False
        if balloon_regex.match(text):
            val = int(text)
            if 1 <= val <= 50:
                balloon_candidates.append(item)
                is_balloon = True
                
        # Check if it is a dimension candidate
        if not is_balloon and dim_regex.search(text):
            dimension_candidates.append(item)
            
    print(f"Found {len(balloon_candidates)} balloon candidates and {len(dimension_candidates)} dimension candidates in drawing views.")
    
    # Map dimension candidates to their nearest balloon numbers
    dimensions_json = []
    for d_item in dimension_candidates:
        d_center = (d_item["cx"], d_item["cy"])
        
        # Proximity search for nearest balloon candidate
        closest_balloon = None
        min_dist = float('inf')
        for b_item in balloon_candidates:
            b_center = (b_item["cx"], b_item["cy"])
            dist = distance(d_center, b_center)
            if dist < min_dist:
                min_dist = dist
                closest_balloon = b_item
                
        # Only associate if balloon is within a reasonable proximity threshold (e.g. 150 pixels)
        balloon_num = ""
        if closest_balloon and min_dist < 150:
            balloon_num = closest_balloon["text"]
            
        d_text = d_item["text"]
        
        # Classify dimension category
        d_type = "other"
        if "R" in d_text.upper():
            d_type = "radius"
        elif "M" in d_text.upper():
            d_type = "thread"
        elif any(sym in d_text for sym in ["Ø", "o", "p", "Ø"]):
            # Check if letter is diameter symbol
            d_type = "diameter"
        elif "45" in d_text:
            d_type = "chamfer"
        elif any(sym in d_text for sym in ["±", "+", "-"]):
            d_type = "length"
            
        # Parse nominal value (first numerical match)
        num_match = re.search(r'\d+([.,]\d+)?', d_text)
        nominal = num_match.group(0) if num_match else d_text
        
        # Parse tolerance representation
        tolerance = d_text.replace(nominal, "").strip()
        
        dimensions_json.append({
            "balloon_number": balloon_num,
            "component_or_feature": f"Feature near balloon {balloon_num}" if balloon_num else "Annotated Feature",
            "nominal_value": nominal,
            "unit": "mm" if nominal.replace(",", "").replace(".", "").isdigit() else "",
            "tolerance": tolerance,
            "type": d_type,
            "notes": d_text
        })
        
    # Save dimensions to JSON
    json_path = os.path.join(args.output_dir, f"{image_base}_dimensions.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"dimensions": dimensions_json}, f, indent=2, ensure_ascii=False)
        print(f"Saved dimensions to {json_path}")
    except Exception as e:
        print(f"Error saving dimensions to JSON: {e}", file=sys.stderr)

    print("\nExtraction complete!")

if __name__ == "__main__":
    main()
