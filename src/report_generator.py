import json
import os
import re
from collections import defaultdict
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font
import pandas as pd
import sys
import traceback

def get_num_chips(tpu_type):
    """Extracts the number of chips from the TPU type string."""
    if not tpu_type:
        raise ValueError("TPU_TYPE is not provided")
    match = re.search(r'v\d+[a-z]?-(\d+)', tpu_type)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract number of chips from TPU_TYPE: {tpu_type}")

def generate_excel_report(jsonl_path, xlsx_path, tpu_type):
    """
    Generates an Excel report from the benchmark metrics JSONL file.

    Args:
        jsonl_path (str): Path to the input metrics_report.jsonl file.
        xlsx_path (str): Path to save the output .xlsx file.
        tpu_type (str): The TPU type used for the run (e.g., 'v5p-256').
    """
    print(f"[Report Generator] Starting Excel generation. Input: {jsonl_path}, Output: {xlsx_path}, TPU Type: {tpu_type}", file=sys.stderr)

    if not os.path.exists(jsonl_path):
        print(f"[Report Generator] Error: Metrics file not found at {jsonl_path}", file=sys.stderr)
        return

    try:
        num_chips = get_num_chips(tpu_type)
        print(f"[Report Generator] Detected {num_chips} chips for {tpu_type}", file=sys.stderr)
    except ValueError as e:
        print(f"[Report Generator] Error getting num_chips: {e}", file=sys.stderr)
        return

    data_by_test = defaultdict(list)
    try:
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                try:
                    record = json.loads(line)
                    test_name = record.get("metadata", {}).get("benchmark_name", "Unknown")
                    data_by_test[test_name].append(record)
                except json.JSONDecodeError as je:
                    print(f"[Report Generator] Warning: Skipping invalid JSON line {i+1}: {line.strip()} - Error: {je}", file=sys.stderr)
                    continue
        print(f"[Report Generator] Loaded data for tests: {list(data_by_test.keys())}", file=sys.stderr)
    except Exception as e:
        print(f"[Report Generator] Error reading JSONL file: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return

    if not data_by_test:
        print("[Report Generator] No data found in JSONL file to generate report.", file=sys.stderr)
        return

    try:
        wb = openpyxl.Workbook()
        wb.remove(wb.active)  # Remove default sheet

        metric_keys = [
            "ici_bandwidth_gbyte_s_p50",
            "ici_bandwidth_gbyte_s_p90",
            "ici_bandwidth_gbyte_s_p95",
            "ici_bandwidth_gbyte_s_p99",
            "ici_bandwidth_gbyte_s_avg",
        ]

        for test_name, records in data_by_test.items():
            if not records:
                continue
            print(f"[Report Generator] Processing sheet for test: {test_name}", file=sys.stderr)

            sheet_name = re.sub(r'[\\/*?[\]:]', '_', test_name)[:31]
            if sheet_name in wb.sheetnames:
                 # Avoid duplicate sheet names
                for i in range(1, 10):
                    new_name = f"{sheet_name[:29]}_{i}"
                    if new_name not in wb.sheetnames:
                        sheet_name = new_name
                        break
            sheet = wb.create_sheet(title=sheet_name)

            flat_data = []
            for record in records:
                row = record.get("metadata", {}).copy()
                row.update(record.get("metrics", {}))
                flat_data.append(row)

            if not flat_data:
                print(f"[Report Generator] No flat data for test: {test_name}", file=sys.stderr)
                continue
            df = pd.DataFrame(flat_data)

            metadata_keys = list(records[0].get("metadata", {}).keys())

            # Determine the main dimension key for rows
            potential_dimensions = ["matrix_dimension", "buffer_size", "size", "dimension"] + metadata_keys
            main_dimension = None
            for dim in potential_dimensions:
                if dim in df.columns and df[dim].nunique() > 1:
                    main_dimension = dim
                    break

            if not main_dimension:
                 print(f"[Report Generator] Warning: Could not determine main dimension for test {test_name}", file=sys.stderr)
                 # attempt to use the first metadata key as dimension
                 if metadata_keys:
                     main_dimension = metadata_keys[0]
                 else:
                     continue # Skip sheet if no dimension can be found

            print(f"[Report Generator] Using dimension '{main_dimension}' for test {test_name}", file=sys.stderr)

            if main_dimension not in df.columns:
                print(f"[Report Generator] Error: Dimension key '{main_dimension}' not found in data for {test_name}", file=sys.stderr)
                continue

            try:
                # Attempt to sort dimension values, handling mixed types
                unique_dims = df[main_dimension].unique()
                dimension_values = sorted(unique_dims, key=lambda x: (isinstance(x, str), x))
            except TypeError:
                 dimension_values = sorted(df[main_dimension].unique(), key=str)
            print(f"[Report Generator] Dimensions for {test_name} ({main_dimension}): {dimension_values}", file=sys.stderr)

            col_offset = 0
            for metric in metric_keys:
                if metric not in df.columns:
                    continue

                # Create a small table for this metric
                metric_df = df[[main_dimension, metric]].copy()
                metric_df = metric_df.drop_duplicates(subset=[main_dimension])
                metric_df = metric_df.set_index(main_dimension)
                metric_df = metric_df.reindex(dimension_values)
                metric_df = metric_df.rename(columns={metric: num_chips})
                metric_df.index.name = f"{main_dimension}\\TPUs"

                # Write Metric Name Header
                sheet.cell(row=1, column=1 + col_offset, value=metric).font = Font(bold=True)
                sheet.merge_cells(start_row=1, start_column=1 + col_offset, end_row=1, end_column=2 + col_offset)

                # Write DataFrame to sheet
                rows = dataframe_to_rows(metric_df, index=True, header=True)
                for r_idx, row in enumerate(rows):
                    for c_idx, value in enumerate(row):
                        cell = sheet.cell(row=r_idx + 2, column=c_idx + 1 + col_offset, value=value)
                        if r_idx == 0: # Header row
                            cell.font = Font(bold=True)

                col_offset += (metric_df.shape[1] + 2)  # metric columns + index col + space

            print(f"[Report Generator] Finished sheet for {test_name}", file=sys.stderr)

        if len(wb.sheetnames) > 0:
            print(f"[Report Generator] Saving Excel file to {xlsx_path}", file=sys.stderr)
            wb.save(xlsx_path)
            print(f"[Report Generator] Excel report generated at {xlsx_path}", file=sys.stderr)
            if os.path.exists(xlsx_path):
                 print(f"[Report Generator] Confirmed file exists at {xlsx_path}", file=sys.stderr)
            else:
                 print(f"[Report Generator] Error: File not found after save at {xlsx_path}", file=sys.stderr)
        else:
            print("[Report Generator] No data to write to Excel report.", file=sys.stderr)

    except Exception as e:
        print(f"[Report Generator] Unhandled exception during Excel generation: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

