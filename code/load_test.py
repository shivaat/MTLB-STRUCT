'''
This script receives the path to the model directory which includes the PyTorch model and 
the saved config file. It loads the model and get the prediction for DEV or TEST data based on Config.
'''
import sys
import logging
import json
import numpy as np

import torch
from transformers import BertTokenizer, AdamW

from torch.utils.data import TensorDataset, DataLoader

from preprocessing import DataProcessor, MweDataProcessor, NERDataSet
from evaluation import labels2Parsemetsv
from model import *
from berteval import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

if len(sys.argv)>1:
    SAVED_PATH = sys.argv[1]
else:
    print("Missing Argument: please specify the path to the saved model (the directory contains PyTorch .bin file)")
    exit()

with open(SAVED_PATH + "config_saved.json") as f:
    config = json.load(f)

DEVorTEST = config["mode"]  # 'DEV' or 'TEST'
LANG = config["data"]["language"]
BATCH_SIZE = config["training"]["batch_size"]  
PRETRAIN_MODEL = config["model"]["pretrained_model_name"]
MULTI_TASK = config["model"]["multi_task"]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_path = config["data"]["data_path"]+ config["data"]["language"]+"/"   

tokenizer = BertTokenizer.from_pretrained(PRETRAIN_MODEL) #, do_lower_case=True)

### LOADING DATA ###
data_processor = MweDataProcessor(data_path)

#train_examples = data_processor.get_train_examples()
if DEVorTEST == "DEV":
    val_examples = data_processor.get_dev_examples()
else:
    val_examples = data_processor.get_test_examples()

max_len = max([len(tokenizer.tokenize(x.text)) for x in val_examples])
max_len = max_len+2
print('max len based on data', max_len)

MAX_LEN = config["data"]["max_len"]
max_len = MAX_LEN
print('trained max len', max_len)

print('data size:', len(val_examples))


poses_dict = {"[CLS]": 0, "[SEP]": 1, "X": 2}  #data_processor.get_pos_dict()  
# We don't use this so it can be an empty dictionary
deprels_dict = {"[CLS]": 0, "[SEP]": 1, "X": 2} # data_processor.get_deprels_dict()  
# We don't use this so it can be an empty dictionary

# This is not being used.
idx2tags = np.load(SAVED_PATH+"Idx2Tags.npy", allow_pickle=True).item() 
tags2idx = {idx2tags[t]: t for t in idx2tags}
print('# of labels:',len(tags2idx))
print('tags2idx:', tags2idx)

tags2idx = config["data"]["tags2idx"] # {}
idx2tags = {tags2idx[t]: t for t in tags2idx}
print('# of labels:',len(tags2idx))
print('tags2idx:', tags2idx)

eval_dataset = NERDataSet(data_list=val_examples, tokenizer=tokenizer,
                          label_map=tags2idx, pos_map=poses_dict,
                          deptype_map=deprels_dict, max_len=max_len)

eval_iter = DataLoader(dataset=eval_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=4)


###  CREATING AN OBJECT OF THE MODEL  ###
if not MULTI_TASK:
    model = CoNLLClassifier.from_pretrained(PRETRAIN_MODEL,
                                        num_labels=len(tags2idx)).to(device)
else:
    model = DepMultiTaskClassify(PRETRAIN_MODEL, len(tags2idx)).to(device)
    
if "saved_model_name" in config["model"]:
    model.load_state_dict(torch.load(SAVED_PATH + config["model"]["saved_model_name"]))
else:
    model.load_state_dict(torch.load(SAVED_PATH + "multitask_tagger.torch"))

model = model.eval()

prediction_file_name = SAVED_PATH + '/'
if DEVorTEST == "DEV":
    labels, probs, fscore = eval(eval_iter, model, tags2idx, device, MULTI_TASK)
    labels2Parsemetsv(labels, data_path+'dev.cupt', prediction_file_name+'dev.system.cupt')
else:
    labels, probs = eval_blind(eval_iter, model, tags2idx, device, MULTI_TASK)
    labels2Parsemetsv(labels, data_path+'test.blind.cupt', prediction_file_name+'test.system.cupt')
