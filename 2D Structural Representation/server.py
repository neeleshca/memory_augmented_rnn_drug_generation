import json
import os
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Fragments import fr_benzene
from rdkit.Chem.Descriptors import MolWt
from io import BytesIO
import base64
from flask import Flask, request, Response, url_for
from flask import render_template, send_from_directory
from flask_cors import CORS, cross_origin
from pathlib import Path
import numpy as np  
import random
import pickle 
import time
from datetime import datetime

from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer

app = Flask(__name__)
cors = CORS(app)

IMAGES_FOLDER = './images'

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = IMAGES_FOLDER
unbiased_val = "unbiased"
maximized_val = "maximized"
minimized_val = "minimized"
ntm_val = "ntm"
dnc_val = "dnc"
stack_rnn_val = "stackrnn"
logp_val = "logp"
benzene_val = "benzene"

def load_set():
    with open(Path("strings", "set_of_chembl22.pickle"), 'rb') as handle:
        return pickle.load(handle)

training_dataset = load_set()


def get_file_contents(model, bias, molpropType):
    extension = ".txt"
    suffix = ""
    
    if(molpropType == logp_val):
        prefix = "biased/logP/"
    elif(molpropType == benzene_val):
        prefix = "biased/Benzene/"

    if(bias == minimized_val):
        suffix = "_min"
    elif(bias == maximized_val):
        suffix = "_max"
    else:
        prefix = "unbiased/"

    if(model == ntm_val):
        prefix += "ntm"
    elif(model == dnc_val):
        prefix += "dnc"
    elif(model == stack_rnn_val):
        prefix += "stack"
    path_to_file = Path("../scripts/Evaluation/", prefix+suffix+extension)
    print(path_to_file)
    with open(path_to_file) as f:
        content = f.read().splitlines()
    return content

def clean_strings(sampled_strings):
    newList = []
    for i in sampled_strings:
        if(i.find('>') != -1):
            newList.append(i[1:i.find('>')])
    return newList

def get_logP(sampled_strings):
    vals = []
    for i in sampled_strings:
        molecule = Chem.MolFromSmiles(i)
        if molecule is not None:
            vals.append(MolLogP(molecule))
    return vals

def get_benzene(sampled_strings):
    vals = []
    for i in sampled_strings:
        molecule = Chem.MolFromSmiles(i)
        if molecule is not None:
            vals.append(fr_benzene(molecule))
    return vals

def is_novel_molecule(smile):
    if(smile not in training_dataset):
        return "novel"
    else:
        return "not novel"

@app.route('/')
def home_page():
    return render_template('home_page.html')

@app.route('/mol_gen_page')
def mol_page():
    print("hello world")
    return render_template('mol_page.html')
    

@app.route('/api/config/getconfig', methods = ['GET'])
def send_config():
    # Sends a JSON object with options for generators and biases
    info_dict = {}
    info_dict['generators'] = {
        dnc_val: {'id': dnc_val ,'name': 'DNC (Differentiable Neural Computer)', 'option': 'DNC', 'description': 'An enhancement over the Neural Turing machine that has \
         ability to recover sequences'},
        ntm_val: {'id': ntm_val, 'name': 'NTM (Neural Turing Machine)', 'option': 'NTM', 'description': 'A memory augmented neural network where memory can be read from and written \
            to selectively'},
        stack_rnn_val: {'id': stack_rnn_val, 'name': 'Stack Augmented RNN', 'option': 'StackRNN', 'description': 'A neural network that uses a pushdown stack as external memory'}
    }
    info_dict['biases'] = {
        unbiased_val: {'id': unbiased_val, 'name': 'Unbiased', 'description': 'Not biased towards any desired property'},
        maximized_val: {'id': maximized_val, 'name': 'Maximized', 'description': 'Maximizing value of desired property'},
        minimized_val: {'id': minimized_val , 'name': 'Minimized', 'description': 'Minimizing value of desired property'}
    }
    info_dict['molproperties'] = {
        logp_val: {'id': logp_val, 'name': 'ALogP Value', 'description': 'AlogP is the log of octanol water partition coefficient'},
        benzene_val: {'id': benzene_val, 'name': 'Benzene Rings', 'description': 'The number of Benzene rings in the compound'}
    }
    
    resp = Response(json.dumps(info_dict))
    resp.headers['Content-type'] = 'application/json'

    return resp
    

@app.route('/api/structure/showimg/<nnType>/<biasType>/<molpropType>/<numStrings>', methods = ['GET'])
def show_images(nnType, biasType, molpropType, numStrings):
    print(nnType)
    print(biasType)
    print(molpropType)
    allStrings = get_file_contents(nnType, biasType, molpropType)
    sampled_strings = random.sample(allStrings, int(numStrings))    
    sampled_strings = random.sample(allStrings, int(numStrings))    
    cleaned_strings = clean_strings(sampled_strings)

    i = 0
    invalid_molecules = 0
    json_list = []
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H-%M-%S")


    for smiles_string in cleaned_strings:
        mol_obj = Chem.MolFromSmiles(smiles_string)
        if mol_obj is not None:
            logp = MolLogP(mol_obj)
            benzenering = fr_benzene(mol_obj)
            mol_weight = round(MolWt(mol_obj), 2)
            img = Draw.MolToImage(mol_obj)
            imgLoc = f'images/{i}_{timestampStr}_pic.jpg'
            img.save(imgLoc, format='JPEG')
            imgURL = f'/fetch_image/{i}_{timestampStr}_pic.jpg'
            novelty = is_novel_molecule(smiles_string)
            print(novelty)
            sa_score = round(sascorer.calculateScore(mol_obj), 2)
            print(sa_score)
            json_list.append({'imgURL': imgURL, 'prop': molpropType, 'logp': logp, 'benzene': benzenering, 'Mol_weight': mol_weight, 'SA_score': sa_score, 'Novel molecule': novelty, 'SMILES String': smiles_string})
            i += 1
        else:
            invalid_molecules += 1

    res_dict = {'invalid_molecules': invalid_molecules, 'mol_list': json_list}

    resp = Response(json.dumps(res_dict))
    resp.headers['Content-type'] = 'application/json'

    return resp

@app.route('/fetch_image/<filename>')
def fetch_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    

if __name__=='__main__':
    app.run(host='0.0.0.0')
