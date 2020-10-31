
# modified version of the code at https://github.com/chnsh/BERT-NER-CoNLL/blob/master/data_set.py

import os
import time
import torch
from torch.utils import data
from corpus_reader import *
#UNIQUE_LABELS = {'X', '[CLS]', '[SEP]'}


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, pos=None, depTag=None, depType=None, label=None, segment_ids=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.pos = pos
        self.depTag = depTag
        self.depType = depType
        self.label = label
        self.segment_ids = segment_ids


def readfile(filename):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line.startswith('# ') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = line.strip().split('\t')
        if splits[0].isdigit():  # Here, we ignore those tokens of conllu that represent the cobinations of two
                                    # two following sub-tokens with a '-'
            sentence.append([splits[0], splits[1], splits[2], splits[3],splits[6], splits[7]])
                            # ID, WORD, LEMMA, POS, DEP-HEAD, DEP-REL
        label.append(splits[-1])   # This is not going to be used for this project
                                    # what we care about is dependency tags

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []
    return data

class MweDataProcessor(object):
    """Processor for parseme formated (.cupt) data."""
    def __init__(self, data_dir):
        self.corpus = Corpus_reader(data_dir)

    def get_train_examples(self):
        train = self.corpus.read(self.corpus.train_sents)
        train_x = [[x[0] for x in elem] for elem in train]
        train_pos = [[x[2] for x in elem] for elem in train]
        train_depTag = [[x[3] for x in elem] for elem in train]
        train_depType = [[x[4] for x in elem] for elem in train]
        train_y = [[x[5] for x in elem] for elem in train]
        train_data = [(x,pos, depTag,depType,y) for x,pos, depTag,depType,y in zip(train_x, train_pos, train_depTag,train_depType, train_y)]
        return self._create_examples(train_data, "train")

    def get_dev_examples(self):
        dev = self.corpus.read(self.corpus.dev_sents)
        dev_x = [[x[0] for x in elem] for elem in dev]
        dev_pos = [[x[2] for x in elem] for elem in dev]
        dev_depTag = [[x[3] for x in elem] for elem in dev]
        dev_depType = [[x[4] for x in elem] for elem in dev]
        dev_y = [[x[5] for x in elem] for elem in dev]
        dev_data = [(x,pos, depTag,depType,y) for x,pos, depTag,depType,y in zip(dev_x, dev_pos,dev_depTag,dev_depType, dev_y)]
        return self._create_examples(dev_data, "dev")

    def get_test_examples(self):
        test = self.corpus.read(self.corpus.blind_test_sents)       #self.corpus.test_sents
        test_x = [[x[0] for x in elem] for elem in test]
        test_pos = [[x[2] for x in elem] for elem in test]
        test_depTag = [[x[3] for x in elem] for elem in test]
        test_depType = [[x[4] for x in elem] for elem in test]
        test_y = [[x[5] for x in elem] for elem in test]
        test_data = [(x,pos, depTag,depType, y) for x, pos, depTag, depType, y in zip(test_x, test_pos, test_depTag, test_depType, test_y)]
        return self._create_examples(test_data, "test")

    def get_pos_dict(self):
        train = self.corpus.read(self.corpus.train_sents)
        train_pos = [[x[2] for x in elem] for elem in train]
        dev = self.corpus.read(self.corpus.dev_sents)
        dev_pos = [[x[2] for x in elem] for elem in dev]
        poses = list(set([elem for sublist in train_pos+dev_pos for elem in sublist])) + ["[CLS]", "[SEP]", "X"]
        pos2idx = {}
        for (i, pos) in enumerate(poses):
            pos2idx[pos] = i
        return pos2idx

    def get_deprels_dict(self):
        train = self.corpus.read(self.corpus.train_sents)
        train_deprels = [[x[4] for x in elem] for elem in train]
        dev = self.corpus.read(self.corpus.dev_sents)
        dev_deprels = [[x[4] for x in elem] for elem in dev]
        deprels = list(set([elem for sublist in train_deprels+dev_deprels for elem in sublist])) + ["[CLS]", "[SEP]", "X"]
        deprel2idx = {}
        for (i, deprel) in enumerate(deprels):
            deprel2idx[deprel] = i
        return deprel2idx

    def get_labels(self):
        train = self.corpus.read(self.corpus.train_sents)
        train_y = [[x[5] for x in elem] for elem in train]
        dev = self.corpus.read(self.corpus.dev_sents)
        dev_y = [[x[5] for x in elem] for elem in dev]
        labels = list(set([elem for sublist in train_y+dev_y for elem in sublist])) + ["[CLS]", "[SEP]", "X"]
        return labels
    #    return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
    #            "[CLS]", "[SEP]", "X"]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for i, (sentence, pos, depTag, depType, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            pos = pos
            depTag = depTag
            depType = depType
            label = label
            examples.append(InputExample(guid=guid, text=text_a, pos=pos, depTag=depTag, depType=depType, label=label))
        return examples

class DataProcessor(object):
    """Processor for parseme formated (.cupt) data."""
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_examples(self, train_name):
        self.train_name = train_name
        t1 = time.time()
        train = readfile(self.data_dir+self.train_name)
        print("time for readfile()", time.time()-t1)
        train_x = [[x[1] for x in elem[0]] for elem in train]
        train_pos = [[x[3] for x in elem[0]] for elem in train]
        train_depTag = [[int(x[4]) for x in elem[0]] for elem in train] 
        train_depType = [[x[5] for x in elem[0]] for elem in train]
        # target/y is actually dependency tag in this project
        train_y = [[x[5] for x in elem[0]] for elem in train]
        train_data = [(x,pos, depTag,depType,y) for x,pos, depTag,depType,y in zip(train_x, train_pos, train_depTag,train_depType, train_y)]
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, dev_name):
        self.dev_name = dev_name
        dev = readfile(self.data_dir+self.dev_name)
        dev_x = [[x[1] for x in elem[0]] for elem in dev]
        dev_pos = [[x[3] for x in elem[0]] for elem in dev]
        dev_depTag = [[int(x[4]) for x in elem[0]] for elem in dev]
        dev_depType = [[x[5] for x in elem[0]] for elem in dev]
        dev_y = [[x[5] for x in elem[0]] for elem in dev]
        dev_data = [(x,pos, depTag,depType,y) for x,pos, depTag,depType,y in zip(dev_x, dev_pos,dev_depTag,dev_depType, dev_y)]
        return self._create_examples(dev_data, "dev")

    def get_test_examples(self, test_name):
        self.test_name = test_name
        test = readfile(self.data_dir+self.test_name)
        test_x = [[x[1] for x in elem[0]] for elem in test]
        test_pos = [[x[3] for x in elem[0]] for elem in test]
        test_depTag = [[int(x[4]) for x in elem[0]] for elem in test]
        test_depType = [[x[5] for x in elem[0]] for elem in test]
        test_y = [[x[5] for x in elem[0]] for elem in test]
        test_data = [(x,pos, depTag,depType, y) for x, pos, depTag, depType, y in zip(test_x, test_pos, test_depTag, test_depType, test_y)]
        return self._create_examples(test_data, "test")

    def get_pos_dict(self):
        train = readfile(self.data_dir+self.train_name) 
        train_pos = [[x[3] for x in elem[0]] for elem in train]
        dev = readfile(self.data_dir+self.dev_name) 
        dev_pos = [[x[3] for x in elem[0]] for elem in dev]
        poses = list(set([elem for sublist in train_pos+dev_pos for elem in sublist])) + ["[CLS]", "[SEP]", "X"]
        pos2idx = {}
        for (i, pos) in enumerate(poses):
            pos2idx[pos] = i
        return pos2idx
        
    def get_deprels_dict(self):
        train = readfile(self.data_dir+self.train_name) 
        train_deprels = [[x[5] for x in elem[0]] for elem in train]
        dev = readfile(self.data_dir+self.dev_name) 
        dev_deprels = [[x[5] for x in elem[0]] for elem in dev]
        deprels = list(set([elem for sublist in train_deprels+dev_deprels for elem in sublist])) + ["[CLS]", "[SEP]", "X"]
        deprel2idx = {}
        for (i, deprel) in enumerate(deprels):
            deprel2idx[deprel] = i
        return deprel2idx

    def get_labels(self):
        train = readfile(self.data_dir+self.train_name) 
        train_y = [[x[5] for x in elem[0]] for elem in train]
        dev = readfile(self.data_dir+self.dev_name) 
        dev_y = [[x[5] for x in elem[0]] for elem in dev]
        labels = list(set([elem for sublist in train_y+dev_y for elem in sublist])) + ["[CLS]", "[SEP]", "X"]
        return labels
    #    return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
    #            "[CLS]", "[SEP]", "X"]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for i, (sentence, pos, depTag, depType, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            pos = pos
            depTag = depTag
            depType = depType
            label = label
            examples.append(InputExample(guid=guid, text=text_a, pos=pos, depTag=depTag, depType=depType, label=label))
        return examples


class NERDataSet(data.Dataset):
    """
    This Dataset prepares the data for BERT-related modles with [CLS] in the 
    begining of the sequence and [SEP] at the end. It also takes care of 
    wordPiece tokenization.
    """
    def __init__(self, data_list, tokenizer, label_map, pos_map=None, deptype_map=None, max_len=100):
        self.max_len = max_len
        self.label_map = label_map
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.pos_map = pos_map
        self.deptype_map = deptype_map

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        input_example = self.data_list[idx]
        text = input_example.text
        pos = input_example.pos
        depTag = input_example.depTag
        depType = input_example.depType
        label = input_example.label
        length = len(label)
        word_tokens = ['[CLS]']
        tag_num = [-1]    # This contains dependency heads of tokens, which will
                            # probably be converted to adjacency matrices. 
        pos_tokens = ['[CLS]']
        deptype_tokens = ['[CLS]']
        label_list = ['[CLS]']
        label_mask = [0]  # value in (0, 1) - 0 signifies invalid token

        input_ids = [self.tokenizer.convert_tokens_to_ids('[CLS]')]
        pos_ids = [self.pos_map['[CLS]']]
        deptype_ids = [self.deptype_map['[CLS]']]
        label_ids = [self.label_map['[CLS]']]

        # iterate over individual tokens and their labels
        w_i = 1   # added the word index of the original words to relate the newly generated tokens to them
        for word, pos, depTag, depType, label in zip(text.split(), pos, depTag, depType, label):
            tokenized_word = self.tokenizer.tokenize(word)

            for token in tokenized_word:
                word_tokens.append(token)
                input_ids.append(self.tokenizer.convert_tokens_to_ids(token))

            if len(tokenized_word) == 0:
                print("WARNING: There is a word that BERT tokenizer returned nothing for:", word)
                print('\t substituted the word with "."')
                token = '.'
                word_tokens.append(token)
                input_ids.append(self.tokenizer.convert_tokens_to_ids(token))
            
            tag_num.append(int(depTag))
            label_list.append(label)
            
            pos_tokens.append(pos)
            deptype_tokens.append(depType)
            
            # some labels in the test data might be unseen and unknown to the training
            if pos in self.pos_map:
                pos_ids.append(self.pos_map[pos])
            else:
                pos_ids.append(self.pos_map['X'])
            if depType in self.deptype_map:
                deptype_ids.append(self.deptype_map[depType])
            else:
                deptype_ids.append(self.deptype_map['X'])
                
            if label in self.label_map:
                label_ids.append(self.label_map[label])
            else:
                label_ids.append(self.label_map['X'])
            label_mask.append(1)
            # len(tokenized_word) > 1 only if it splits word in between, in which case
            # the first token gets assigned MWE(any target) tag and the remaining ones 
            # get assigned X
            for i in range(1, len(tokenized_word)):
                label_list.append('X')
                label_ids.append(self.label_map['X'])
                label_mask.append(0)

                tag_num.append(w_i)
                pos_tokens.append('X')
                pos_ids.append(self.pos_map['X'])
                deptype_tokens.append('X')
                deptype_ids.append(self.deptype_map['X'])
            w_i+=1

        assert len(word_tokens) == len(label_list) == len(input_ids) == len(label_ids) == len(
            label_mask), " ".join[str(len(word_tokens)),str(len(label_list)),str(len(input_ids)),
                    str(len(label_ids)),str(len(label_mask))]

        if len(word_tokens) >= self.max_len:    # Truncating
            word_tokens = word_tokens[:(self.max_len - 1)]
            label_list = label_list[:(self.max_len - 1)]
            tag_num = tag_num[:(self.max_len - 1)]
            input_ids = input_ids[:(self.max_len - 1)]
            label_ids = label_ids[:(self.max_len - 1)]
            label_mask = label_mask[:(self.max_len - 1)]
            pos_tokens = pos_tokens[:(self.max_len - 1)]
            pos_ids = pos_ids[:(self.max_len - 1)]
            deptype_tokens = deptype_tokens[:(self.max_len - 1)]
            deptype_ids = deptype_ids[:(self.max_len - 1)]

        # dpendency heads that refer to a token out of the bound for max_len
        # are replaced with the index for the word itself
        for i in range(len(tag_num)):
            if tag_num[i] > sum(label_mask):
                tag_num[i] = sum(label_mask) #i

        assert len(word_tokens) < self.max_len, len(word_tokens)

        word_tokens.append('[SEP]')
        tag_num.append(-1)
        label_list.append('[SEP]')
        input_ids.append(self.tokenizer.convert_tokens_to_ids('[SEP]'))
        label_ids.append(self.label_map['[SEP]'])
        label_mask.append(0)
        
        pos_tokens.append('[SEP]')
        pos_ids.append(self.pos_map['[SEP]'])
        deptype_tokens.append('[SEP]')
        deptype_ids.append(self.deptype_map['[SEP]'])

        assert len(word_tokens) == len(label_list) == len(input_ids) == len(label_ids) == len(
            label_mask)

        sentence_id = [0 for _ in input_ids]
        attention_mask = [1 for _ in input_ids]

        # Padding
        while len(input_ids) < self.max_len:    # Padding
            input_ids.append(0)
            tag_num.append(-1)
            pos_ids.append(0)
            deptype_ids.append(0)
            label_ids.append(self.label_map['X'])
            attention_mask.append(0)
            sentence_id.append(0)
            label_mask.append(0)

        assert len(word_tokens) == len(label_list)
        assert len(input_ids) == len(label_ids) == len(attention_mask) == len(sentence_id) == len(
            label_mask) == self.max_len, len(input_ids)

        # length is the length of the original sentence
        #length = sum(label_mask)

        # return word_tokens, label_list,
        return torch.LongTensor(input_ids), torch.LongTensor(pos_ids), torch.LongTensor(tag_num), torch.LongTensor(deptype_ids), torch.LongTensor(label_ids), torch.LongTensor(attention_mask), torch.LongTensor(sentence_id), torch.BoolTensor(label_mask), torch.LongTensor([length])


def _is_projective(parse):
    """
    Is the parse tree projective?
    Returns
    --------
    projective : bool
       True if a projective tree.
    """
    for m, h in enumerate(parse):
        for m2, h2 in enumerate(parse):
            #if m2 == m1:
            if h2 == 0 or h==0:
                continue
            if m+1 < h:
                if (
                    m+1 < m2+1 < h < h2
                    or m+1 < h2 < h < m2+1
                    or m2+1 < m+1 < h2 < h
                    or h2 < m+1 < m2+1 < h
                ):
                    #print('*first if', m+1, h, m2+1, h2)
                    return False
            if h < m+1:
                if (
                    h < m2+1 < m+1 < h2
                    or h < h2 < m+1 < m2+1
                    or m2+1 < h < h2 < m+1
                    or h2 < h < m2+1 < m+1
                ):
                    #print('**second if', m+1, h, m2+1, h2)
                    return False
    return True

