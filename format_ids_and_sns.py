def transform_and_overwrite(input_file):
    # Read lines from the input file
    with open(input_file, 'r') as f:
        lines = f.read().strip().splitlines()
    
    # Ensure the number of lines is even
    # if len(lines) % 2 != 0:
    #     raise ValueError("Input file must have an even number of lines.")
    
    # Prepare pairs of lines with a tab separator
    pairs = ['\t'.join(lines[i:i+2]) for i in range(0, len(lines), 2)]
    
    # Write the transformed pairs back to the same input file
    with open(input_file, 'w') as f:
        f.write('\n'.join(pairs) + '\n')

# Example usage:
input_file = '/home/canyon/Test_Equipment/crispy-garland/QA_ids.txt'

transform_and_overwrite(input_file)
