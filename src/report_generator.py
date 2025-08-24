import json
import os
from collections import defaultdict
from openpyxl import Workbook
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Map ici_size to number of cores
CORE_MAP = {
    64: 128,  # ici_size 64 -> 128 cores
    128: 256   # ici_size 128 -> 256 cores
}
# The fixed column headers for core counts in the report
CORE_COLUMNS = sorted(CORE_MAP.values()) # Results in [128, 256]

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
    processed_ici_sizes = set()

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

                is_valid = metrics.get('ici_bandwidth_gbyte_s_avg') is not None
                if test_name and matrix_dim is not None and ici_size is not None and is_valid:
                    try:
                        ici_size_val = int(ici_size)
                        core_count = CORE_MAP.get(ici_size_val)
                        if core_count:
                            data_by_test[test_name].append({
                                'matrix_dim': int(matrix_dim),
                                'core_count': core_count, # Store the mapped core count
                                'metrics': metrics
                            })
                            processed_ici_sizes.add(ici_size_val)
                        else:
                            logging.warning(f"Warning: Unexpected ici_size '{ici_size_val}' in data, not in CORE_MAP.")
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
        safe_test_name = safe_test_name[:31]
        ws = wb.create_sheet(title=safe_test_name)

        matrix_dims = sorted(list(set(r['matrix_dim'] for r in records)))

        # Create a map for easy lookup: (matrix_dim, core_count) -> metrics_dict
        data_map = {}
        for record in records:
            data_map[(record['matrix_dim'], record['core_count'])] = record['metrics']

        current_col = 1
        for metric in metrics_keys:
            # Metric Header
            ws.cell(row=1, column=current_col, value=metric)

            # Sub Header - Always show both 128 and 256 core columns
            ws.cell(row=2, column=current_col, value="dimensions\\TPUs")
            ws.cell(row=2, column=current_col + 1, value=CORE_COLUMNS[0]) # 128
            ws.cell(row=2, column=current_col + 2, value=CORE_COLUMNS[1]) # 256

            # Data Rows
            for row_idx, dim in enumerate(matrix_dims):
                ws.cell(row=3 + row_idx, column=current_col, value=dim)
                for col_idx, core_count in enumerate(CORE_COLUMNS):
                    cell_data = data_map.get((dim, core_count))
                    metric_val = cell_data.get(metric) if cell_data else None
                    ws.cell(row=3 + row_idx, column=current_col + 1 + col_idx, value=metric_val if metric_val is not None else "")

            # Move to the next block: 1 (dim col) + len(CORE_COLUMNS) (data cols) + 2 (spacer cols)
            current_col += 1 + len(CORE_COLUMNS) + 2

    try:
        os.makedirs(os.path.dirname(excel_local_path), exist_ok=True)
        wb.save(excel_local_path)
        logging.info(f"Excel report saved locally to {excel_local_path}")
    except Exception as e:
        logging.error(f"Failed to save Excel file {excel_local_path}: {e}")

