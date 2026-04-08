import sys
import os
from pathlib import Path
from spatialstencil.syntax.stencil_ir.parser import Parser
from spatialstencil.syntax.stencil_ir.flop_counter import FLOPCounter, MemoryCounter


def find_spst_files(directory: str) -> list[Path]:
    """
    Find all .spst files in the given directory.
    
    Args:
        directory: Path to the directory to search
        
    Returns:
        List of Path objects for all .spst files found
    """
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    
    # Find all .spst files recursively
    spst_files = list(directory_path.rglob("*.spst"))
    
    return sorted(spst_files)


def analyze_file(
    filepath: Path,
    parser: Parser,
    flop_counter: FLOPCounter,
    mem_counter: MemoryCounter,
) -> tuple[str, int, int, int, int, float, bool, str]:
    """
    Analyze a single .spst file and return results.

    Returns:
        (filename, flops, loads, stores, bytes_transferred,
         arithmetic_intensity, success, error_message)
    """
    try:
        with open(filepath, 'r') as f:
            code = f.read()

        program = parser.parse(code)

        flops = flop_counter.count(program)
        mem = mem_counter.count(program)

        loads = mem['loads']
        stores = mem['stores']
        total_bytes = mem['bytes']
        ai = flops / total_bytes if total_bytes > 0 else float('nan')

        return (str(filepath), flops, loads, stores, total_bytes, ai, True, "")

    except Exception as e:
        return (str(filepath), 0, 0, 0, 0, float('nan'), False, str(e))


def print_header():
    """Print a nice header for the analysis."""
    print("=" * 80)
    print(" " * 20 + "FLOP Analysis for Spatial Stencil Programs")
    print("=" * 80)
    print()


def print_summary(results: list[tuple]):
    """
    Print a summary of all analysis results and write flops.csv.

    Each result tuple:
        (filename, flops, loads, stores, bytes_transferred,
         arithmetic_intensity, success, error_message)
    """
    successful = [r for r in results if r[6]]
    failed = [r for r in results if not r[6]]

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    import pandas as pd
    rows = []
    for filename, flops, loads, stores, total_bytes, ai, _, _ in successful:
        rows.append({
            'Program': Path(filename).name[:-5],
            'Flop': flops,
            'Loads': loads,
            'Stores': stores,
            'Bytes': total_bytes,
            'ArithmeticIntensity': ai,
        })
    df = pd.DataFrame(rows)
    df.to_csv("flops.csv", index=False)

    if successful:
        print(f"\nSuccessfully analyzed {len(successful)} file(s):")
        print("-" * 100)
        print(f"{'File':<45} {'FLOPs':>14} {'Loads':>12} {'Stores':>10} {'Bytes':>14} {'AI (F/B)':>12}")
        print("-" * 100)

        total_flops = total_loads = total_stores = total_bytes_sum = 0
        for filename, flops, loads, stores, total_bytes, ai, _, _ in successful:
            display_name = Path(filename).name
            ai_str = f"{ai:.4f}" if ai == ai else "n/a"  # nan check
            print(f"{display_name:<45} {flops:>14,} {loads:>12,} {stores:>10,} {total_bytes:>14,} {ai_str:>12}")
            total_flops += flops
            total_loads += loads
            total_stores += stores
            total_bytes_sum += total_bytes

        print("-" * 100)
        overall_ai = total_flops / total_bytes_sum if total_bytes_sum > 0 else float('nan')
        overall_ai_str = f"{overall_ai:.4f}" if overall_ai == overall_ai else "n/a"
        print(f"{'TOTAL':<45} {total_flops:>14,} {total_loads:>12,} {total_stores:>10,} {total_bytes_sum:>14,} {overall_ai_str:>12}")
        print("-" * 100)

    if failed:
        print(f"\n\nFailed to analyze {len(failed)} file(s):")
        print("-" * 100)
        for filename, _, _, _, _, _, _, error in failed:
            display_name = Path(filename).name
            print(f"\n{display_name}:")
            print(f"  Error: {error}")
        print("-" * 100)

    print(f"\n\nTotal files processed: {len(results)}")
    print(f"  Success: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print()


def main():
    """Main entry point for the FLOP analysis script."""
    # Check command line arguments
    if len(sys.argv) != 2:
        print("USAGE: python flop_analysis.py <directory>")
        print("\nAnalyzes all .spst files in the given directory and reports FLOP counts.")
        sys.exit(1)
    
    directory = sys.argv[1]
    
    print_header()
    
    try:
        # Find all .spst files
        spst_files = find_spst_files(directory)
        
        if not spst_files:
            print(f"No .spst files found in directory: {directory}")
            sys.exit(0)
        
        print(f"Found {len(spst_files)} .spst file(s) in {directory}\n")
        
        # Create parser and counter instances
        print("Initializing parser...")
        parser = Parser()
        flop_counter = FLOPCounter()
        mem_counter = MemoryCounter()

        print("Analyzing files...\n")

        # Analyze each file
        results = []
        for i, filepath in enumerate(spst_files, 1):
            print(f"[{i}/{len(spst_files)}] Processing {filepath.name}...", end=" ")

            result = analyze_file(filepath, parser, flop_counter, mem_counter)
            results.append(result)

            if result[6]:  # Success
                ai_str = f"{result[5]:.4f}" if result[5] == result[5] else "n/a"
                print(f"✓ {result[1]:,} FLOPs  {result[4]:,} B  AI={ai_str}")
            else:  # Failed
                print(f"✗ ERROR")
        
        # Print summary
        print_summary(results)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except NotADirectoryError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
        
if __name__ == "__main__":
    main()