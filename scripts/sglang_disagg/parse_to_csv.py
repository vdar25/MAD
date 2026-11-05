#!/usr/bin/env python3
"""
Parse SGLang benchmark log file and save results to CSV.
Extracts: Concurrency, Input tokens, Output tokens, Total Token throughput (tok/s)
For each configuration, takes the MAX Total Token throughput across all iterations.
"""

import re
import csv
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


def parse_benchmark_log(log_file: str) -> Dict[Tuple[int, int, int], Dict]:
    """Parse benchmark log file and extract results, keeping max throughput per configuration."""
    results = defaultdict(lambda: {'concurrency': None, 'input_tokens': None, 
                                    'output_tokens': None, 'max_throughput': 0.0})
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Find the start of the first iteration (ignore warmup)
    first_iter_match = re.search(r'RUNNING: the benchserving script for iter: 1', content)
    if not first_iter_match:
        print("Warning: No iteration 1 found. Processing entire file.")
        start_pos = 0
    else:
        start_pos = first_iter_match.start()
    
    # Process only from first iteration onwards
    content = content[start_pos:]
    
    # Split by benchmark result sections
    sections = re.split(r'============ Serving Benchmark Result ============', content)
    
    for i in range(1, len(sections)):
        # A result belongs to the last RUNNING line before it. A result with none
        # (the warmup, when the whole file is parsed) is skipped, and a run that
        # failed without a result does not shift the ones after it.
        configs = re.findall(r'RUNNING: prompts\s+(?:\d+\s+)?isl\s+(\d+)\s+osl\s+(\d+)\s+con\s+(\d+)',
                             sections[i - 1])
        if not configs:
            continue
        current_input_seq_len, current_output_seq_len, current_concurrency = map(int, configs[-1])
        
        # Extract Total token throughput (tok/s) from benchmark result section
        throughput_match = re.search(r'Total token throughput \(tok/s\):\s+([\d.]+)', sections[i])
        throughput = float(throughput_match.group(1)) if throughput_match else None
        
        if throughput is not None:
            config_key = (current_input_seq_len, current_output_seq_len, current_concurrency)
            
            # Update results for this configuration
            # Always use values from RUNNING line (isl, osl, con)
            results[config_key]['concurrency'] = current_concurrency
            results[config_key]['input_tokens'] = current_input_seq_len
            results[config_key]['output_tokens'] = current_output_seq_len
            
            # Keep the maximum throughput
            if throughput > results[config_key]['max_throughput']:
                results[config_key]['max_throughput'] = throughput
    
    return results


def save_to_csv(results: Dict[Tuple[int, int, int], Dict], output_file: str):
    """Save results to CSV file with specified columns."""
    if not results:
        print("No results to save.")
        return
    
    # Define column order
    fieldnames = ['Concurrency', 'Input tokens', 'Output tokens', 'Total Token throughput (tok/s)']
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Sort by concurrency, then input tokens, then output tokens
        for (input_tokens, output_tokens, concurrency), data in sorted(results.items(), 
                                                                         key=lambda x: (x[0][2], x[0][0], x[0][1])):
            row = {
                'Concurrency': data['concurrency'],
                'Input tokens': data['input_tokens'],
                'Output tokens': data['output_tokens'],
                'Total Token throughput (tok/s)': f"{data['max_throughput']:.2f}"
            }
            writer.writerow(row)
    
    print(f"Saved {len(results)} benchmark configurations to {output_file}")


def _get_run_metadata(pipeline: str = "sglang"):
    """Collect run metadata from environment variables."""
    import os
    xP = os.environ.get('xP', '1')
    yD = os.environ.get('yD', '1')
    gpus = os.environ.get('GPUS_PER_NODE', '8')
    mode = os.environ.get('PARALLEL_MODE') or ('wide_ep' if os.environ.get('DP_MODE') == '1' else 'tp')
    transfer = os.environ.get('KV_TRANSFER_BACKEND') or (
        'mooncake' if os.environ.get('RUN_MORI') == '0' else 'mori')

    # Tag names match the models.json card variants.
    if mode in ('wide_ep', 'dp'):
        backend = f'{transfer}_dp'
    elif transfer == 'mori':
        backend = 'mori_io'
    else:
        backend = transfer

    return {
        'pipeline': pipeline,
        'deployment_type': f'disagg_{xP}P{yD}D',
        'tags': f'{pipeline}_disagg,{backend}',
        'n_gpus': str(int(xP) * int(gpus) + int(yD) * int(gpus)),
        'nnodes': str(int(xP) + int(yD)),
        'gpus_per_node': gpus,
        'docker_image': os.environ.get('DOCKER_IMAGE_NAME', ''),
        'machine_name': os.environ.get('SELECTED_NODES') or os.environ.get('SLURM_JOB_NODELIST', ''),
        'launcher': 'slurm_multi',
        'gpu_architecture': os.environ.get('MAD_GPU_ARCH') or _gpu_arch(),
    }


def _gpu_arch() -> str:
    """The first GPU's gfx target, from rocminfo; empty if it cannot be read."""
    import subprocess
    try:
        out = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return ''
    match = re.search(r'\bgfx[0-9a-f]+\b', out)
    return match.group(0) if match else ''


def save_perf_csv(results: Dict[Tuple[int, int, int], Dict], output_file: str,
                  model_name: str = "", pipeline: str = "sglang"):
    """Save results in madengine perf.csv format."""
    if not results:
        print("No results to save to perf.csv.")
        return

    meta = _get_run_metadata(pipeline)

    fieldnames = [
        'model', 'n_gpus', 'nnodes', 'gpus_per_node', 'training_precision',
        'pipeline', 'args', 'tags', 'docker_file', 'base_docker', 'docker_sha',
        'docker_image', 'git_commit', 'machine_name', 'deployment_type', 'launcher',
        'gpu_architecture', 'performance', 'metric', 'relative_change', 'status',
        'build_duration', 'test_duration', 'dataname', 'data_provider_type',
        'data_size', 'data_download_duration', 'build_number',
        'additional_docker_run_options',
    ]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (input_tokens, output_tokens, concurrency), data in sorted(
            results.items(), key=lambda x: (x[0][2], x[0][0], x[0][1])
        ):
            row = {
                'model': model_name,
                'performance': f"{data['max_throughput']:.2f}",
                'metric': f"tok/s (isl={data['input_tokens']} osl={data['output_tokens']} con={data['concurrency']})",
                'status': 'SUCCESS',
            }
            row.update(meta)
            writer.writerow(row)

    print(f"Saved {len(results)} rows to perf.csv: {output_file}")


def main():
    """Main function."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Parse SGLang benchmark log file and save results to CSV')
    parser.add_argument('log_file', type=str, help='Path to benchmark log file')
    parser.add_argument('-o', '--output', type=str, help='Output CSV file name (default: <log_file>_results.csv)')
    parser.add_argument('--perf-csv', type=str, help='Also generate madengine perf.csv at this path')
    parser.add_argument('--model-name', type=str, default='', help='Model name for perf.csv')

    args = parser.parse_args()

    log_file = args.log_file

    # Check if file exists
    if not Path(log_file).exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)

    print(f"Parsing log file: {log_file}")

    # Parse the log file
    results = parse_benchmark_log(log_file)

    if not results:
        print("No benchmark results found in log file.")
        sys.exit(1)

    # Generate output filename
    if args.output:
        output_file = args.output
    else:
        output_file = Path(log_file).stem + '_results.csv'

    # Save to CSV
    save_to_csv(results, output_file)

    # Save madengine perf.csv if requested
    if args.perf_csv:
        save_perf_csv(results, args.perf_csv, args.model_name)

    print(f"\nSummary:")
    print(f"  Total unique configurations: {len(results)}")
    print(f"  Output file: {output_file}")


if __name__ == '__main__':
    main()
