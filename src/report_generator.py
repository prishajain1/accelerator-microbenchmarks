import json
import os
from collections import defaultdict
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def generate_local_excel_report(jsonl_local_path, excel_local_path):
    logging.info(f"Generating report from local file: {jsonl_local_path}")
    try:
        with open(jsonl_local_path, 'r') as f:
            jsonl_content = f.read()
    except FileNotFoundError:
        logging.error(f"Local JSONL file not found: {jsonl_local_path}")
        return
    except Exception as e:
        logging.error(f"Could not read local JSONL file: {e}")
        return

    data_by_test = defaultdict(list)
    for line in jsonl_content.strip().split('\n'):
        if not line: continue
        try:
            record = json.loads(line)
            if 'metrics' in record and 'dimensions' in record:
                metrics = record['metrics']
                dims = record['dimensions']
                test_name = dims.get('test_name')
                matrix_dim = dims.get('matrix_dim')
                ici_size = dims.get('ici_size')

                # Skip records missing essential fields for this report
                is_valid = metrics.get('ici_bandwidth_gbyte_s_avg') is not None
                if test_name and matrix_dim is not None and ici_size is not None and is_valid:
                    try:
                        data_by_test[test_name].append({
                            'matrix_dim': int(matrix_dim),
                            'ici_size': int(ici_size),
                            'metrics': metrics  # Keep all metrics
                        })
                    except ValueError:
                         logging.warning(f"Warning: Could not convert matrix_dim '{matrix_dim}' or ici_size '{ici_size}' to int for test '{test_name}'. Skipping record.")
                         continue
        except json.JSONDecodeError:
            logging.warning(f"Warning: Could not decode a line: {line}")
            continue

    if not data_by_test:
        logging.info("No valid data found to generate report.")
        return

    wb = Workbook()
    if "Sheet" in wb.sheetnames:
        wb.remove(wb["Sheet"])

    metrics_keys = [
        "ici_bandwidth_gbyte_s_p50", "ici_bandwidth_gbyte_s_p90",
        "ici_bandwidth_gbyte_s_p95", "ici_bandwidth_gbyte_s_p99",
        "ici_bandwidth_gbyte_s_avg"
    ]

    for test_name, records in data_by_test.items():
        if not records: continue

        safe_test_name = "".join(c for c in test_name if c.isalnum() or c in (' ', '_')).rstrip()
        safe_test_name = safe_test_name[:31] # Excel sheet name limit
        ws = wb.create_sheet(title=safe_test_name)

        ici_sizes = sorted(list(set(r['ici_size'] for r in records)))
        matrix_dims = sorted(list(set(r['matrix_dim'] for r in records)))

        # Create a map for easy lookup: (matrix_dim, ici_size) -> metrics_dict
        data_map = {}
        for record in records:
            data_map[(record['matrix_dim'], record['ici_size'])] = record['metrics']

        current_col = 1
        for metric in metrics_keys:
            num_ici_sizes = len(ici_sizes)
            block_width = num_ici_sizes + 1

            # Metric Header (e.g., A1)
            ws.cell(row=1, column=current_col, value=metric)
            # Optional: Merge cells for the header
            if num_ici_sizes > 0:
                 ws.merge_cells(start_row=1, start_column=current_col, end_row=1, end_column=current_col + num_ici_sizes)

            # Sub Header (e.g., A2, B2, C2)
            ws.cell(row=2, column=current_col, value="dimensions\\TPUs")
            for i, size in enumerate(ici_sizes):
                ws.cell(row=2, column=current_col + 1 + i, value=size)

            # Data Rows
            for row_idx, dim in enumerate(matrix_dims):
                ws.cell(row=3 + row_idx, column=current_col, value=dim)
                for col_idx, size in enumerate(ici_sizes):
                    cell_data = data_map.get((dim, size))
                    metric_val = cell_data.get(metric) if cell_data else None
                    ws.cell(row=3 + row_idx, column=current_col + 1 + col_idx, value=metric_val if metric_val is not None else "")

            # Move to the next block of columns for the next metric
            current_col += block_width + 1 # Add 1 for the spacer column

    try:
        wb.save(excel_local_path)
        logging.info(f"Excel report saved locally to {excel_local_path}")
    except Exception as e:
        logging.error(f"Failed to save Excel file {excel_local_path}: {e}")
