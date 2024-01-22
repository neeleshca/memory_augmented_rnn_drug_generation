'''
    From Memory Augmented Recurrent Neural Networks For De-novo Drug Generation by Naveen Suresh, Neelesh CA, Srikumar Subramanian and Dr. Gowri Srinivasa.
'''

from numpy.lib.function_base import disp
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Fragments import fr_benzene
from rdkit import Chem
import statistics
import pickle 
from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from enum import Enum 

from PIL import Image
from io import BytesIO

class PropertyEnum(Enum):
    logP = 1
    Benzene = 2

if (sys.argv[3] == "Benzene"):
    property_option = PropertyEnum.Benzene
elif (sys.argv[3] == "logP"):
    property_option = PropertyEnum.logP

dataset_file = "set_of_chembl22.pickle"
def load_set():
    with open(dataset_file, 'rb') as handle:
        return pickle.load(handle)

rnn_option = f"{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}"
num_runs = int(sys.argv[4])
good_strings_option = sys.argv[5]

set1 = load_set()

if good_strings_option == "yes_good":
    files_names = [f'new_strings/{rnn_option}-{run_number}.txt' for run_number in range(1, num_runs+1)]
elif good_strings_option == "no_good":
    files_names = [f'one_sample_gen_strings/{rnn_option}-{run_number}.txt' for run_number in range(1, num_runs+1)]

for file_name in files_names:
    lines = []
    len_mol = []
    if (file_name.endswith(".pickle")):
        lines = list(set1)
        for line in lines:
            len_mol.append(len(line))
    else:
        with open(file_name) as f:
            lines = f.read().splitlines()
        for i in range(len(lines)):
            if lines[i]:
                len_mol.append(len(lines[i]))
    values = []
    sas_values = []
    valid_strings = 0
    common_strings = 0
    for i in lines:
        molecule = Chem.MolFromSmiles(i)
        if molecule is not None:
            valid_strings+=1
            if (property_option == PropertyEnum.logP):
                values.append(MolLogP(molecule))
            elif (property_option == PropertyEnum.Benzene):
                values.append(fr_benzene(molecule))
            #print(i)
            sas_values.append(sascorer.calculateScore(molecule))
            if i in set1:
                common_strings+=1
    valid_strings_percent = valid_strings*100/len(lines)
    common_strings_percent = common_strings*100/valid_strings
    values = sorted(values)
    print("\n\n******************\n")
    print(file_name)
    print(len(values))
    print("Property values")
    print("Q1 ", np.percentile(values, 25))
    print("Q2 ", np.percentile(values, 50))
    print("Q3 ", np.percentile(values, 75))
    print("Mean ", np.mean(values))
    print("Valid strings percent ", valid_strings_percent)
    print("Common strings percent ", common_strings_percent)
    print("SAS median ", np.median(sas_values))
    print("SAS mean", np.mean(sas_values))
    set_values = set(lines)
    print("Total", len(set_values))
    print("Unique", len(set_values.difference(set1)))
    print("\nLength of strings metrics")
    print("Average length of strings: ",sum(len_mol)/len(len_mol))
    print("Median length of strings: ",statistics.median(len_mol))
    print("Q1 length of strings: ", np.percentile(len_mol, 25))
    print("Q3 length of strings: ", np.percentile(len_mol, 75))
    print("Min length of strings: ", min(len_mol))
    print("Max length of strings: ", max(len_mol))
