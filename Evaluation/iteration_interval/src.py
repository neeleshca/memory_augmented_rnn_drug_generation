from rdkit import Chem
import pickle 
import pandas as pd
from pathlib import Path

def load_set():
    with open('../set_of_chembl22.pickle', 'rb') as handle:
        return pickle.load(handle)

set1 = load_set()

train_file_number_strings = len(set1)
path_to_results = Path('generated_strings')
generator = ["dnc", "ntm", "snn"]
headers = ['valid strings %', 'common strings %', 'jaccard index', 'iteration_no', 'generator']
results = []
for model in generator:
    files = ['5000', '10000', '15000', '20000', '25000', '30000', '35000', '40000', '45000']
    for f in files:
        generated_file_number_strings = 0
        path_to_file = Path(path_to_results, str(model), f'generated_file_{f}.txt') 
        file1 = open(path_to_file, 'r')
        valid_strings = 0
        common_strings = 0
        for i in file1:
            i = i.rstrip()
            i = i[1:]
            if(i.find('>') != -1):
                i = i[:i.find('>')]
            generated_file_number_strings+=1
            m = Chem.MolFromSmiles(i)
            if m is not None:
                valid_strings+=1
                if i in set1:
                    common_strings+=1
            else:
                pass
        valid_strings_percent = valid_strings*100/generated_file_number_strings
        common_strings_percent = common_strings*100/valid_strings
        jaccard_index = common_strings/(
            train_file_number_strings + generated_file_number_strings - common_strings)
        print("valid_strings is ", valid_strings)
        print("common_strings is ", common_strings)
        print("valid_strings is ", valid_strings_percent)
        print("common_strings is ", common_strings_percent)
        print("Jaccard index is ", jaccard_index)
        results.append([valid_strings_percent, common_strings_percent,  jaccard_index, f, model])

df = pd.DataFrame(results, columns=headers)
df.to_csv('results.csv', index=False)
