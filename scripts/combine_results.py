import os
import csv
import argparse

def combine_device_results(results_folder, output_filename):
    """
    Scans the specified results folder, finds all .csv files within its subdirectories,
    and concatenates them into a single CSV file.

    A new 'device' column is added to the output, with the value taken
    from the name of the subdirectory where the CSV was found.
    """
    all_rows = []
    header = []
    header_written = False

    print(f"Scanning for CSV files in '{results_folder}'...")

    # Check if the results directory exists
    if not os.path.isdir(results_folder):
        print(f"Error: The specified results directory '{results_folder}' does not exist.")
        return

    # Walk through the directory tree starting from the results_folder
    for dirpath, _, filenames in os.walk(results_folder):
        for filename in filenames:
            if filename.endswith('.csv'):
                # The device name is the last part of the directory path
                device_name = os.path.basename(dirpath)
                file_path = os.path.join(dirpath, filename)
                
                # Check if the file is not empty or a placeholder
                try:
                    if os.path.getsize(file_path) > 0:
                        with open(file_path, 'r', encoding='utf-8') as f:
                             # Simple check for placeholder content
                            if '[place-holder]' in f.read(50):
                                print(f"Skipping placeholder file: {file_path}")
                                continue
                except (IOError, OSError):
                    continue # Skip if file size can't be read

                print(f"Processing '{file_path}' for device '{device_name}'...")

                with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile)
                    try:
                        # Read the header from the first file we process
                        if not header_written:
                            original_header = next(reader)
                            header = ['device'] + original_header
                            header_written = True
                        else:
                            # For subsequent files, just skip the header
                            next(reader)

                        # Read the data rows, add the device column, and append
                        for row in reader:
                            if row: # Ensure the row is not empty
                                all_rows.append([device_name] + row)
                    except StopIteration:
                        # This happens if the CSV file is empty
                        print(f"Warning: '{file_path}' is empty or contains only a header.")
                        continue
    
    # Write the combined header and data to the new CSV file
    if all_rows:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        with open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)
            writer.writerows(all_rows)
        print(f"\n✅ Successfully combined {len(all_rows)} rows into '{output_filename}'.")
    else:
        print("\nNo valid CSV data found to combine.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Combine device benchmark CSVs into a single file."
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help="The directory containing device-specific subfolders with CSV files. (Default: 'results')"
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='combined_results.csv',
        help="The path for the combined output CSV file. (Default: 'combined_results.csv')"
    )
    args = parser.parse_args()
    
    combine_device_results(args.results_dir, args.output_file)

