import torch

import transformers
from transformers import (WEIGHTS_NAME,BertConfig, BertForMaskedLM, BertTokenizer)

from flask import Flask, jsonify, request, render_template
from flask_restful import Api, Resource
from flask_cors import CORS, cross_origin
import flask

import json
import pandas as pd
import numpy as np
import nmslib
import os

from utils import *

# Create Flask app
app = Flask(__name__)
cors = CORS(app)

## Getting files from google drive:
functions = ["1-Ks6uc_67FwGcIun9cXOb9Pwfe5icyMK", "data/functions.csv"]
docstrings = ["1-D4s2DtE2-V4sqlmkP4wjgRzTseBbzLW", "data/docstrings.csv"]
lineage = ["1-SpoXT2xdzNZ8vqRoLVO-xkv4K0EFGrB", "data/lineage.csv"]
docstrings_vecs = ["1TCiOTxkOEkC0nmKjOpf6VS_MCho5SxdX", "data/docstrings_avg_vecs.npy"]
model_path = ["1kc9qnH3AYvtITpCpnE87_Otd8Az7Gslz", "model"]

if not os.path.isfile(functions[1]):
    download_file_from_google_drive(functions[0], functions[1])
    download_file_from_google_drive(docstrings[0], docstrings[1])
    download_file_from_google_drive(lineage[0], lineage[1])
    download_file_from_google_drive(docstrings_vecs[0], docstrings_vecs[1])
    download_file_from_google_drive(model_path[0], model_path[1])

print("Files exist")

df_function = pd.read_csv(functions[1])[:20000]
df_docstring = pd.read_csv(docstrings[1],sep='\t', names=["docstring"])[:20000]
df_lineage = pd.read_csv(lineage[1],sep='\t', names=["Repo"])[:20000]

print("CSVs loaded")


docstrings_avg_vec = np.load(docstrings_vecs[1],allow_pickle=True)

config = BertConfig.from_json_file(model_path[1]+'/config.json')
config.output_hidden_states = True

print("Tokenizer and model initialized")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

device = torch.device('cpu')

model = BertForMaskedLM.from_pretrained("bert-base-uncased",config=config)
model.load_state_dict(torch.load(model_path[1]+"/pytorch_model.bin",map_location=device))
model.eval()


# Initialize a new index, using a HNSW index on Cosine Similarity
index = nmslib.init(method='hnsw', space='cosinesimil')
index.addDataPointBatch(docstrings_avg_vec)
index.createIndex({'post': 2}, print_progress=True)

print("Index made")

# Routes:
@app.route('/hello')
def hello_world():
    return 'Hello, From DiscoverCode engine!'

@app.route('/', methods=['POST', 'GET'])
def post():
    if flask.request.method == 'GET':
        return render_template('index.html')
    
    print(request.get_data())
    print(request.form["input"])
    user_input = request.form['input']

    input_ids = torch.tensor(tokenizer.encode(user_input, add_special_tokens=True)).unsqueeze(0)

    outputs = model(input_ids, masked_lm_labels=input_ids)

    embeddings = outputs[2][-1].detach().numpy()[0]

    size = embeddings.shape[0]
    sum_array = [sum(x) for x in zip(*embeddings)]
    avg_array = [sum_array[i]/size for i in range(len(sum_array))]

    ids, distances = index.knnQuery(avg_array, k=10)

    functions = []
    sources = []

    for elem in ids:
        functions.append(df_function['0'][int(elem)])
        sources.append(df_lineage['Repo'][int(elem)])
        
    return render_template('index2.html', functions=functions, sources=sources)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
