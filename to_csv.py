import csv
import re

def fasta_to_csv(input_file, output_file):
    """
    Convert a FASTA file to a CSV file with 'Sequence' and 'Label' columns.

    Parameters:
    input_file (str): Path to the input FASTA file
    output_file (str): Path to the output CSV file
    """
    sequences = []

    # Read the FASTA file
    with open(input_file, 'r') as fasta_file:
        current_sequence = ""
        current_label = None

        for line in fasta_file:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Header line with metadata
            if line.startswith('>'):
                # Save the previous sequence if it exists
                if current_sequence and current_label is not None:
                    sequences.append({
                        'Sequence': current_sequence,
                        'Label': current_label
                    })

                # Parse the header to extract metadata
                # Format: >seqX|Y|testing where Y is the label (0 or 1)
                header_match = re.match(r'>seq\d+\|(\d+)\|testing', line)
                if header_match:
                    current_label = int(header_match.group(1))
                    current_sequence = ""
                else:
                    current_label = None
                    current_sequence = ""
            else:
                # Sequence line
                current_sequence += line

        # Add the last sequence
        if current_sequence and current_label is not None:
            sequences.append({
                'Sequence': current_sequence,
                'Label': current_label
            })

    # Write the CSV file
    with open(output_file, 'w', newline='') as csv_file:
        fieldnames = ['Sequence', 'Label']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for seq in sequences:
            writer.writerow(seq)

    print(f"Conversion completed. {len(sequences)} sequences written to {output_file}")

# Usage
input_file = "./data/test.fasta"
output_file = "./data/test.csv"

fasta_to_csv(input_file, output_file)