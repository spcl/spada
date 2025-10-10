import sys
import os
from pathlib import Path
from spatialstencil.syntax.stencil_ir.parser import Parser
from spatialstencil.syntax.stencil_ir.flop_counter import FLOPCounter


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


def analyze_file(filepath: Path, parser: Parser, counter: FLOPCounter) -> tuple[str, int, bool, str]:
    """
    Analyze a single .spst file and return results.
    
    Args:
        filepath: Path to the .spst file
        parser: Parser instance
        counter: FLOPCounter instance
        
    Returns:
        Tuple of (filename, flop_count, success, error_message)
    """
    try:
        # Read and parse the file
        with open(filepath, 'r') as f:
            code = f.read()
        
        program = parser.parse(code)
        
        # Count FLOPs
        flop_count = counter.count(program)
        
        return (str(filepath), flop_count, True, "")
        
    except Exception as e:
        return (str(filepath), 0, False, str(e))


def print_header():
    """Print a nice header for the analysis."""
    print("=" * 80)
    print(" " * 20 + "FLOP Analysis for Spatial Stencil Programs")
    print("=" * 80)
    print()


def print_summary(results: list[tuple[str, int, bool, str]]):
    """
    Print a summary of all analysis results.
    
    Args:
        results: List of (filename, flop_count, success, error_message) tuples
    """
    successful = [r for r in results if r[2]]
    failed = [r for r in results if not r[2]]
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    import pandas as pd
    df = pd.DataFrame([{'Program': Path(filename).name[:-5], "Flop": flop_count} for filename, flop_count, _, _, in successful])
    df.to_csv("flops.csv")

    if successful:
        print(f"\nSuccessfully analyzed {len(successful)} file(s):")
        print("-" * 80)
        print(f"{'File':<50} {'FLOPs':>20}")
        print("-" * 80)
        
        total_flops = 0
        for filename, flop_count, _, _ in successful:
            # Shorten filename for display
            display_name = Path(filename).name
            print(f"{display_name:<50} {flop_count:>20,}")
            total_flops += flop_count
        
        print("-" * 80)
        print(f"{'TOTAL':<50} {total_flops:>20,}")
        print("-" * 80)
    
    if failed:
        print(f"\n\nFailed to analyze {len(failed)} file(s):")
        print("-" * 80)
        for filename, _, _, error in failed:
            display_name = Path(filename).name
            print(f"\n{display_name}:")
            print(f"  Error: {error}")
        print("-" * 80)
    
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
        counter = FLOPCounter()
        
        print("Analyzing files...\n")
        
        # Analyze each file
        results = []
        for i, filepath in enumerate(spst_files, 1):
            print(f"[{i}/{len(spst_files)}] Processing {filepath.name}...", end=" ")
            
            result = analyze_file(filepath, parser, counter)
            results.append(result)
            
            if result[2]:  # Success
                print(f"✓ {result[1]:,} FLOPs")
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