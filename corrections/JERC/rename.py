import os

for fname in os.listdir('.'):
    print(fname)
    if '.jec' in fname or '.junc' in fname:
        continue
    elif fname.endswith('.txt'):
        if 'Uncertainty' in fname:
            os.rename(fname, fname[:-4] + '.junc.txt')
        else:
            os.rename(fname, fname[:-4] + '.jec.txt')
