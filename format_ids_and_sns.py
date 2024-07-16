def overwrite(input_file):
    """Reformats scanned QA_ids for Gsheet"""
    with open(input_file, 'r') as f:
        lines = f.read().strip().splitlines()

    pairs = transform(lines)

    with open(input_file, 'w') as f:
        f.write('\n'.join(pairs) + '\n')

def transform(lines):
    return ['\t'.join(lines[i:i+2]) for i in range(0, len(lines), 2)]

if __name__ == "__main__":
    input_file = '/home/canyon/Test_Equipment/crispy-garland/QA_ids.txt'
    overwrite(input_file)
