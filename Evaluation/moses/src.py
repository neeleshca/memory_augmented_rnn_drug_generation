from pathlib import Path
import moses
import pprint
import os
from random import sample
import numpy as np
from rdkit import Chem
import random 
sample_size = 30000
current_paths = set()
for path in Path('.').glob('*.txt'):
    current_paths.add(path)
    print(path)

paths = []
for path in Path('.').rglob('*.txt'):
    paths.append(path)
    print(path)

cleaned_paths = []
for i in paths:
    if i not in current_paths:
        cleaned_paths.append(i)
final_paths = cleaned_paths

filtered_paths = []
filter_string = "30k"
filters = ["30k"]
for i in cleaned_paths:
    for j in filters:
        if j in str(i):
            filtered_paths.append(i)
            break
final_paths = filtered_paths
print("final paths")
for i in final_paths:
    print(i)


def write_now(filep, msg):
    """Write msg to the file given by filep, forcing the msg to be written to the filesystem immediately (now).
    Without this, if you write to files, and then execute programs
    that should read them, the files will not show up in the program
    on disk.
    """
    filep.write(msg)
    filep.flush()
    # The above call to flush is not enough to write it to disk *now*;
    # according to https://stackoverflow.com/a/41506739/257924 we must
    # also call fsync:
    os.fsync(filep)

precalculatedStats = None
pathToStats = 'stats.npy'
# pathToStats = None
test_strings = open('test.txt','r').readlines()
test = []
if (pathToStats != None):
    precalculatedStats = np.load(pathToStats, allow_pickle=True).item()
else:
    for i in test_strings:
        mol = Chem.MolFromSmiles(i)
        if mol == None:
            continue
        test.append(i)

final = open("results.txt", "w")
for i in final_paths:
    smile_file = open(i, 'r')
    smiles = sample(smile_file.readlines(), sample_size)
    cleaned_smiles = []
    for j in smiles:
        if(j.find('>') != -1):
            cleaned_smiles.append(j[1:j.find('>')])
    metrics = moses.get_all_metrics(cleaned_smiles, test= test,
     ptest = precalculatedStats, train = open('train.txt','r').read().splitlines())
    write_now(final, str(i))
    write_now(final, "\n")
    y = pprint.pformat(metrics)
    write_now(final, y)
    write_now(final, "\n")

write_now(final, str(i))
write_now(final, "\n")
y = pprint.pformat(metrics)
write_now(final, y)
write_now(final, "\n")
# print(metrics)
