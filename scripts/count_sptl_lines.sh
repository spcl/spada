#!/bin/bash

# Check if directory argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

DIR="$1"

# Check if directory exists
if [ ! -d "$DIR" ]; then
    echo "Error: Directory '$DIR' does not exist"
    exit 1
fi

# Check if sloccount is installed
if ! command -v sloccount &> /dev/null; then
    echo "Error: sloccount is not installed"
    echo "Install it with: sudo apt-get install sloccount (Debian/Ubuntu)"
    echo "             or: brew install sloccount (macOS)"
    exit 1
fi

# Create temporary directory for converted files
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "=========================================="
echo "  SPTL to CPP Converter & SLOC Counter"
echo "=========================================="
echo ""
echo "Processing directory: $DIR"
echo ""

# Counter for converted files
COUNT=0
TOTAL_SLOC=0

# Array to store results
declare -a RESULTS

# Find all .sptl files and process them individually
while IFS= read -r -d '' file; do
    filename=$(basename "$file")
    newname="${filename%.sptl}.cpp"
    
    # Create a temporary subdirectory for this file
    FILE_TEMP_DIR="$TEMP_DIR/file_$COUNT"
    mkdir -p "$FILE_TEMP_DIR"
    
    # Copy file with new extension
    cp "$file" "$FILE_TEMP_DIR/$newname"
    
    # Run sloccount and capture the output
    SLOC_OUTPUT=$(sloccount "$FILE_TEMP_DIR" 2>/dev/null)
    
    # Extract the total SLOC count (line that starts with "Total Physical Source Lines")
    SLOC=$(echo "$SLOC_OUTPUT" | grep "^Total Physical Source Lines" | awk -F'=' '{print $2}' | tr -d ' ' | sed 's/[^0-9]//g')
    
    # If SLOC is empty, try alternative parsing
    if [ -z "$SLOC" ]; then
        SLOC=$(echo "$SLOC_OUTPUT" | grep "cpp:" | awk '{print $2}')
    fi
    
    # Default to 0 if still empty
    if [ -z "$SLOC" ]; then
        SLOC=0
    fi
    
    RESULTS[$COUNT]="$filename|$SLOC"
    TOTAL_SLOC=$((TOTAL_SLOC + SLOC))
    ((COUNT++))
    
done < <(find "$DIR" -type f -name "*.sptl" -print0)

# Check if any files were found
if [ $COUNT -eq 0 ]; then
    echo "No .sptl files found in directory"
    exit 0
fi

echo "=========================================="
echo "  SLOC Count by File"
echo "=========================================="
echo ""
printf "%-50s %10s\n" "File" "SLOC"
printf "%-50s %10s\n" "----" "----"

# Print results sorted by filename
for result in "${RESULTS[@]}"; do
    IFS='|' read -r filename sloc <<< "$result"
    printf "%-50s %10s\n" "$filename" "$sloc"
done

echo ""
echo "=========================================="
printf "%-50s %10s\n" "TOTAL" "$TOTAL_SLOC"
echo "=========================================="
echo ""
echo "Total files processed: $COUNT"
echo ""