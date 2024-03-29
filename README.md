# Overview
This repository contains the code for the system MTLB-STRUCT participated in the [PARSEME 1.2 shaed task on semi-supervised identification of verbal MWEs](http://multiword.sourceforge.net/PHITE.php?sitesig=CONF&page=CONF_02_MWE-LEX_2020___lb__COLING__rb__&subpage=CONF_40_Shared_Task). 
The system is based on pre-trained BERT masked language modelling and jointly learns VMWE tags and dependency parse trees. 
The system ranked first in the open track of the shared task.

### Project Structure
```
.
├── README.md
├── code
│   ├── berteval.py
│   ├── config
│   ├── corpus.py
│   ├── corpus_reader.py
│   ├── evaluation.py
│   ├── load_test.py
│   ├── main.py
│   ├── model.py
│   └──  preprocessing.py
└── requirements.txt
```

The requirements as listed in `requirements.txt` are:
- PyTorch
- [Transformers](https://github.com/huggingface/transformers)
- [Torch_Struct](https://github.com/harvardnlp/pytorch-struct)

## How to Run the System

1. Copy the data files from [PARSEME 1.2 repository](https://gitlab.com/parseme/sharedtask-data/-/tree/master/1.2) under the directory data, in the path data/1.2/{language}.
2. Choose the configuration file from the /code/config/ directory or make your own config file with the same fileds as in the files in the config directory.
3. Run `main.py config/{config.json}`

This performs training the model based on the config file you passed as the argument. As a result of this, the trained model will be saved in a directory called `saved` and then it can be used for testing the model.

You can get the predictions on dev/test data by running `load_test.py [PATH TO THE DIRECTORY OF SAVED MODEL]`. This saves the prediction `.cupt` file in the saved directory.

Note that the evaluation performance results that you see after running `load_test.py` for development sets, are based on seqeval NER metrics and not the PARSEME evaluation measures. 
We evaluate the performance of our predictions using [PARSEME evaluation script](https://gitlab.com/parseme/sharedtask-data/-/tree/master/1.2/bin/) `evaluate.py`.

### Reference
```
@article{Taslimipoor2020,
  author    = {Shiva Taslimipoor and
               Sara Bahaadini and
               Ekaterina Kochmar},
  title     = {MTLB-STRUCT @PARSEME 2020: Capturing Unseen Multiword Expressions 
               Using Multi-task Learning and Pre-trained Masked Language Models},
  year      = {2020},
  eprint={2011.02541},
  archivePrefix={arXiv},
  url       = {}
}
```
