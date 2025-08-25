import json
import os
import re
from collections import defaultdict
import openpyxl
import pandas as pd

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
    if not os.path.exists(jsonl_path):
        print(f"Warning: Metrics file not found at {jsonl_path}")
        return

    try:
        num_chips = get_num_chips(tpu_type)
    except ValueError as e:
        print(f"Error: {e}")
        return

    data_by_test = defaultdict(list)
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                test_name = record.get("metadata", {}).get("benchmark_name", "Unknown")
                data_by_test[test_name].append(record)
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line.strip()}")
                continue

    if not data_by_test:
        print("No data found in JSONL file to generate report.")
        return

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

        sheet_name = re.sub(r'[\\/*?[\]:]', '_', test_name)[:31]
        sheet = wb.create_sheet(title=sheet_name)

        flat_data = []
        for record in records:
            row = record.get("metadata", {}).copy()
            row.update(record.get("metrics", {}))
            flat_data.append(row)

        if not flat_data:
            continue
        df = pd.DataFrame(flat_data)

        metadata_keys = records[0].get("metadata", {}).keys()
        
        # Determine the main dimension key for rows
        if "matrix_dimension" in metadata_keys:
            main_dimension = "matrix_dimension"
        else:
            dimension_cols = []
            for key in metadata_keys:
                if key not in ["benchmark_name", "test_name", "dtype"]:
                    if df[key].nunique() > 1:
                        dimension_cols.append(key)
            if not dimension_cols:
                print(f"Warning: No varying dimensions found for test {test_name}")
                continue
            main_dimension = dimension_cols[0]
            if len(dimension_cols) > 1:
                print(f"Warning: Multiple varying dimensions found for {test_name}: {dimension_cols}. Using '{main_dimension}'.")

        if main_dimension not in df.columns:
            print(f"Error: Dimension key '{main_dimension}' not found in data for {test_name}")
            continue

        # Sort dimension values
        def sort_key(val):
            try:
                return float(val)
            except (ValueError, TypeError):
                return str(val)
        
        try:
            dimension_values = sorted(df[main_dimension].unique(), key=sort_key)
        except TypeError:
             # Fallback if sorting keys are not comparable
             dimension_values = sorted(df[main_dimension].unique(), key=str)


        col_offset = 0
        for metric in metric_keys:
            if metric not in df.columns:
                continue

            # Metric Name Header
            sheet.cell(row=1, column=1 + col_offset, value=metric)
            sheet.merge_cells(start_row=1, start_column=1 + col_offset, end_row=1, end_column=2 + col_offset)

            # Table Headers
            sheet.cell(row=2, column=1 + col_offset, value=f"{main_dimension}\\TPUs")
            sheet.cell(row=2, column=2 + col_offset, value=num_chips)

            # Data Rows
            for i, dim_val in enumerate(dimension_values):
                row_num = 3 + i
                sheet.cell(row=row_num, column=1 + col_offset, value=dim_val)

                metric_val_series = df[df[main_dimension] == dim_val][metric]
                metric_val = metric_val_series.iloc[0] if not metric_val_series.empty else None
                
                if metric_val is not None:
                    try:
                        sheet.cell(row=row_num, column=2 + col_offset, value=float(metric_val))
                    except (ValueError, TypeError):
                        sheet.cell(row=row_num, column=2 + col_offset, value=str(metric_val))

            col_offset += 3  # Add spacing between metric tables

    if len(wb.sheetnames) > 0:
        wb.save(xlsx_path)
        print(f"Excel report generated at {xlsx_path}")
    else:
        print("No data to write to Excel report.")

