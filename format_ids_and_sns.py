def transform_and_overwrite(input_file):
    """Reformats scanned QA_ids for Gsheet"""
    with open(input_file, 'r') as f:
        lines = f.read().strip().splitlines()

    pairs = ['\t'.join(lines[i:i+2]) for i in range(0, len(lines), 2)]
    
    with open(input_file, 'w') as f:
        f.write('\n'.join(pairs) + '\n')

input_file = '/home/canyon/Test_Equipment/crispy-garland/QA_ids.txt'
transform_and_overwrite(input_file)
