#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import traceback
from spatialstencil.syntax.gt4py import parser
from spatialstencil.lowering import gt4py_to_stencil_ir
from spatialstencil.lowering.stencil_to_spatial import lower_stencil_to_spatial
from spatialstencil.syntax.stencil_ir import type_inference

def parse_domain_size(domain_str):
    """Parse domain_size string in format 'x,y,z' and return tuple of ints."""
    try:
        parts = domain_str.split(',')
        if len(parts) != 3:
            raise ValueError(f"Domain size must have exactly 3 values, got {len(parts)}")
        
        x, y, z = map(int, parts)
        return (x, y, z)
    except ValueError as e:
        if "invalid literal for int()" in str(e):
            raise argparse.ArgumentTypeError(f"Domain size values must be integers: {domain_str}")
        else:
            raise argparse.ArgumentTypeError(str(e))


def validate_path(path_str, must_exist=False):
    """Convert string to Path object and optionally validate existence."""
    path = Path(path_str)
    if must_exist and not path.exists():
        raise argparse.ArgumentTypeError(f"Path does not exist: {path_str}")
    return path


def lower_function(input_file: Path,
                           function_name: str,
                           domain_size: tuple[int, int, int],
                           output_dir: Path,
                           gtfuncs: dict):
    """Process a single function."""
    print(f"Processing function: {function_name}")
    
    if function_name not in gtfuncs:
        raise ValueError(f"Function {function_name} not found in {input_file}")
    
    program = gtfuncs[function_name]
    irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program, domain=domain_size)
    
    # Save
    output = irprogram.as_ir()
    output_file = output_dir / f"{function_name}_{domain_size[0]}_{domain_size[1]}_{domain_size[2]}.spst"
    with open(output_file, mode="w") as f:
        f.write(output)
    
    print(f"  Saved stencil to: {output_file}")
    
    type_inference.infer_field_extents(irprogram)
    type_inference.infer_field_domains(irprogram)
    spatial_program = lower_stencil_to_spatial(irprogram)
    
    # Save
    output = spatial_program.as_ir()
    output_file = output_dir / f"{function_name}_{domain_size[0]}_{domain_size[1]}_{domain_size[2]}.sptl"
    with open(output_file, mode="w") as f:
        f.write(output)
    
    print(f"  Saved SpaDa to: {output_file}")


def lower_gt4py_to_sptl(input_file: Path,
                 function_name: str | None,
                 domain_size: tuple[int, int, int],
                 output_dir: Path):
    """
    Args:
        input_file (Path): Path to input file
        function_name (str | None): Name of function to use, or None to process all functions
        domain_size (tuple): Tuple of (x, y, z) integers
        output_dir (Path): Path to output directory
    """
    print("Processing with the following parameters:")
    print(f"  Input file: {input_file}")
    print(f"  Function name: {function_name or 'ALL FUNCTIONS'}")
    print(f"  Domain size: {domain_size}")
    print(f"  Output directory: {output_dir}")
    
    gtfuncs = parser.parse_file(str(input_file))
    
    if function_name is None:
        # Process all functions
        if not gtfuncs:
            print("No functions found in the input file.")
            return
        
        print(f"Found {len(gtfuncs)} function(s): {list(gtfuncs.keys())}")
        
        for func_name in gtfuncs.keys():
            try:
                lower_function(input_file, func_name, domain_size, output_dir, gtfuncs)
            except Exception as e:
                print(f"Exception occured during lowering of function {func_name}: ")
                print(traceback.format_exc())
    else:
        # Process single function
        lower_function(input_file, function_name, domain_size, output_dir, gtfuncs)
    
    print("Lowering complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Process data with specified function and domain size",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_file",
        type=lambda x: validate_path(x, must_exist=True),
        help="Path to input file"
    )
    
    parser.add_argument(
        "--function-name",
        type=str,
        default=None,
        help="Name of function to use for processing (optional - if not provided, all functions will be processed)"
    )
    
    parser.add_argument(
        "domain_size",
        type=parse_domain_size,
        help="Domain size in format 'x,y,z' where x,y,z are integers"
    )
    
    parser.add_argument(
        "output_dir",
        type=validate_path,
        help="Path to output directory"
    )
    
    try:
        args = parser.parse_args()
        
        # Ensure output directory exists
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Call the stub function with validated arguments
        lower_gt4py_to_sptl(
            input_file=args.input_file,
            function_name=args.function_name,
            domain_size=args.domain_size,
            output_dir=args.output_dir
        )
        
    except argparse.ArgumentTypeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()