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

property_option = PropertyEnum.Benzene

dataset_file = Path("set_of_chembl22.pickle")
def load_set():
    with open(Path("..", dataset_file), 'rb') as handle:
        return pickle.load(handle)

min_benzene = 0
max_benzene = 6
benzene_counts = dict()

base_folder = Path('biased', property_option.name)
base_folder_unbiased = Path('unbiased')
ntm_biased_min_file = Path('ntm_min.txt')
ntm_biased_max_file = Path('ntm_max.txt')
dnc_biased_min_file = Path('dnc_min.txt')
dnc_biased_max_file = Path('dnc_max.txt')
stack_biased_min_file = Path('stack_min.txt')
stack_biased_max_file = Path('stack_max.txt')
ntm_unbiased_file = Path('ntm.txt')
dnc_unbiased_file = Path('dnc.txt')
stack_unbiased_file = Path('stack.txt')

option = 'unbiased'
if option == 'ntm':
    files_names = [ Path(base_folder, ntm_biased_min_file),
                    Path(base_folder_unbiased, ntm_unbiased_file),
                    Path(base_folder, ntm_biased_max_file)]
elif option == 'dnc':
    files_names = [ Path(base_folder, dnc_biased_min_file),
                    Path(base_folder_unbiased, dnc_unbiased_file),
                    Path(base_folder, dnc_biased_max_file)]
elif option == 'stack':
    files_names = [ Path(base_folder, stack_biased_min_file),
                    Path(base_folder_unbiased, stack_unbiased_file),
                    Path(base_folder, stack_biased_max_file)]
elif option == 'unbiased':
    files_names = [ Path(base_folder_unbiased, stack_unbiased_file), 
                    Path(base_folder_unbiased, ntm_unbiased_file),
                    Path(base_folder_unbiased, dnc_unbiased_file),
                    Path("..", dataset_file)]  
elif option == 'minimized':
    files_names = [ Path(base_folder, stack_biased_min_file),
                    Path(base_folder, ntm_biased_min_file),
                    Path(base_folder, dnc_biased_min_file)]
elif option == 'maximized':
    files_names = [ Path(base_folder, stack_biased_max_file),
                    Path(base_folder, ntm_biased_max_file),
                    Path(base_folder, dnc_biased_max_file)]                  

display_name_map = dict()
if option in ['minimized', 'maximized']:
    display_name_map = {
        str(ntm_biased_min_file) : f'NTM minimized',
        str(ntm_biased_max_file) : f'NTM maximized',
        str(dnc_biased_min_file) : f'DNC minimized',
        str(dnc_biased_max_file) : f'DNC maximized',
        str(stack_biased_min_file) : f'Stack RNN minimized',
        str(stack_biased_max_file) : f'Stack RNN maximized',
        str(ntm_unbiased_file) : 'NTM unbiased',
        str(dnc_unbiased_file) : 'DNC unbiased',
        str(stack_unbiased_file) : 'Stack RNN unbiased'
    }
else:
    display_name_map = {
        str(ntm_biased_min_file) : f'{property_option.name} minimized',
        str(ntm_biased_max_file) : f'{property_option.name} maximized',
        str(dnc_biased_min_file) : f'{property_option.name} minimized',
        str(dnc_biased_max_file) : f'{property_option.name} maximized',
        str(stack_biased_min_file) : f'{property_option.name} minimized',
        str(stack_biased_max_file) : f'{property_option.name} maximized',
        str(ntm_unbiased_file) : 'NTM unbiased',
        str(dnc_unbiased_file) : 'DNC unbiased',
        str(stack_unbiased_file) : 'Stack RNN unbiased',
        str(dataset_file) : 'Training dataset'
    }


if option == 'unbiased':
    max_benzene = 7
for file_name in files_names:
    set1 = load_set()
    lines = []
    len_mol = []
    if (file_name.parts[-1].endswith(".pickle")):
        lines = list(set1)
        for line in lines:
            len_mol.append(len(line))
    else:
        with open(file_name) as f:
            lines = f.read().splitlines()
        for i in range(len(lines)):
            if(lines[i].find('>') != -1):
                lines[i] = lines[i][1:lines[i].find('>')]
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
            sas_values.append(sascorer.calculateScore(molecule))
            if i in set1:
                common_strings+=1
    valid_strings_percent = valid_strings*100/len(lines)
    common_strings_percent = common_strings*100/valid_strings
    values = sorted(values)
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

    if property_option == PropertyEnum.Benzene:
        count_vals = [0] * (max_benzene + 1)
        for i in values:
            count_vals[i]+=1
        for i in range(max_benzene + 1):
            count_vals[i]/=len(values)
        benzene_counts[file_name.parts[-1]] = count_vals
    else:
        sns.distplot(values, hist = False, kde_kws = {'shade': True, 'linewidth': 3},label = display_name_map[file_name.parts[-1]])

if property_option == PropertyEnum.Benzene:
    benzene_values = []
    bar_plot_index = []
    bar_plot_label = []
    for benzene_num in range(max_benzene + 1):
        for file_name in files_names:
            benzene_values+=[benzene_counts[file_name.parts[-1]][benzene_num]]
            bar_plot_index+=[benzene_num]
            bar_plot_label+=[display_name_map[file_name.parts[-1]]]
    df = pd.DataFrame({"benzene_values" : benzene_values, "bar_plot_index": bar_plot_index, "bar_plot_label" : bar_plot_label})
    s = sns.barplot(x ="bar_plot_index", y = 'benzene_values', data = df, hue = "bar_plot_label")
    axes = plt.gca()
    axes.set_xlim([-1, 7])
    plt.legend(prop={'size': 12}, title = 'Model')
    plt.title(f'Density Bar Plot of Generated {property_option.name} Values', fontsize=10)
    plt.xlabel(f'{property_option.name}', fontsize=10); plt.ylabel('Density', fontsize=10)
    plt.show()

else:
    axes = plt.gca()
    axes.set_xlim([-5, 10])
    # axes.set_xlim([-4, 10])
    plt.legend(prop={'size': 12}, title = 'Model')
    plt.title(f'Shaded Density Plot of Generated {property_option.name} Values', fontsize=10)
    plt.xlabel(f'{property_option.name}', fontsize=10); plt.ylabel('Density', fontsize=10)
    plt.show()
