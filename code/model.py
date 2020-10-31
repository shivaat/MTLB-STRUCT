import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForTokenClassification, BertModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_adjacency(tag_ids):
    a = np.zeros((tag_ids.size()[0], tag_ids.size()[1], tag_ids.size()[1]), dtype=np.int)
    for i in range(len(tag_ids)):
        for j in range(len(tag_ids[i])):
            if tag_ids[i][j] not in [0, -1]:
                a[i][tag_ids[i][j]-1] = 1
    a = torch.tensor(a).long() 
    return a.to(device)

class CoNLLClassifier(BertForTokenClassification):

    def forward(self, input_ids, tag_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, label_masks=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]  # (b, MAX_LEN, 768)   # token_output

        token_reprs = [embedding[mask] for mask, embedding in zip(label_masks, sequence_output)]
        token_reprs = pad_sequence(sequences=token_reprs, batch_first=True,
                                   padding_value=-1)  # (b, local_max_len, 768)
        tag_rep = [tag[mask] for mask, tag in zip(label_masks,tag_ids)]
        
        tag_rep = pad_sequence(sequences=tag_rep, batch_first=True, padding_value=-1)

        sequence_output = self.dropout(token_reprs)
        
        logits = self.classifier(sequence_output)

        outputs = (logits,)
        if labels is not None:
            labels = [label[mask] for mask, label in zip(label_masks, labels)]
            labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # (b, local_max_len)
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='sum')
            mask = labels != -1
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss /= mask.float().sum()
            outputs = (loss,) + outputs + (labels,)

        return outputs  # (loss), scores, (hidden_states), (attentions)


class DepMultiTaskClassify(nn.Module):    #(BertForTokenClassification):   #(nn.Module):
    def __init__(self, pret_model, num_labels):
        super(DepMultiTaskClassify, self).__init__()
        self.num_labels = num_labels
        self.base_model = BertModel.from_pretrained(pret_model)   #("bert-base-cased")
        H = self.base_model.config.hidden_size
        print('vocab size:', self.base_model.config.vocab_size)
        self.linear = nn.Linear(H, H)
        self.bilinear = nn.Linear(H, H)
        self.root = nn.Parameter(torch.rand(H))
        self.dropout = nn.Dropout(0.2)    #config["dropout"])
        self.classifier = nn.Linear(H, num_labels)

    def forward(self, input_ids, tag_ids, labels=None, label_masks=None):

        sequence_output = self.dropout(self.base_model(input_ids)[0])

        # The following line is performed instead of the mapper operation in torch-struct
        token_reprs = [embedding[mask] for mask, embedding in zip(label_masks, sequence_output)]
        token_reprs = pad_sequence(sequences=token_reprs, batch_first=True,
                                   padding_value=-1)  # (b, local_max_len, 768)
        final2 = torch.einsum("bnh,hg->bng", token_reprs, self.linear.weight)
        final = torch.einsum("bnh,hg,bmg->bnm", token_reprs, self.bilinear.weight, final2)
        root_score = torch.einsum("bnh,h->bn", token_reprs, self.root)
        #final = final[:, 1:-1, 1:-1]
        N = final.shape[1]
        final[:, torch.arange(N), torch.arange(N)] += root_score[:,:] #[:, 1:-1]


        logits = self.classifier(token_reprs)  # (b, local_max_len, num_labels)

        outputs = (logits,)
        if labels is not None:      
            labels = [label[mask] for mask, label in zip(label_masks, labels)]
            labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # (b, local_max_len)
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='sum')
            mask = labels != -1
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss /= mask.float().sum()

            outputs = (loss,) + outputs + (labels,)

        return outputs + (final,)

class DepMultiTaskClassify2(nn.Module):    #(BertForTokenClassification):   #(nn.Module):
    '''
    This is just the same model but smaller (with hidden layer of size H/2) that I have tried.
    '''
    def __init__(self, pret_model, num_labels):
        super(DepMultiTaskClassify2, self).__init__()
        self.num_labels = num_labels
        self.base_model = BertModel.from_pretrained(pret_model)   #("bert-base-cased")
        H = self.base_model.config.hidden_size
        print('vocab size:', self.base_model.config.vocab_size)
        self.linear = nn.Linear(H, int(H/2))
        self.bilinear = nn.Linear(H, int(H/2))
        self.root = nn.Parameter(torch.rand(H))
        self.dropout = nn.Dropout(0.2)    #config["dropout"])
        self.classifier = nn.Linear(H, num_labels)

    def forward(self, input_ids, tag_ids, labels=None, label_masks=None):

        sequence_output = self.dropout(self.base_model(input_ids)[0])

        # The following line is performed instead of the mapper operation in torch-struct
        token_reprs = [embedding[mask] for mask, embedding in zip(label_masks, sequence_output)]
        token_reprs = pad_sequence(sequences=token_reprs, batch_first=True,
                                   padding_value=-1)  # (b, local_max_len, 768)
        linear = torch.einsum("gh->hg", self.linear.weight)
        final2 = torch.einsum("bnh,hg->bng", token_reprs, linear)
        bilinear = torch.einsum("gh->hg", self.bilinear.weight)
        final = torch.einsum("bnh,hg,bmg->bnm", token_reprs, bilinear, final2)
        root_score = torch.einsum("bnh,h->bn", token_reprs, self.root)
        #final = final[:, 1:-1, 1:-1]
        N = final.shape[1]
        final[:, torch.arange(N), torch.arange(N)] += root_score[:,:] #[:, 1:-1]


        logits = self.classifier(token_reprs)  # (b, local_max_len, num_labels)

        outputs = (logits,)
        if labels is not None:      
            labels = [label[mask] for mask, label in zip(label_masks, labels)]
            labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # (b, local_max_len)
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='sum')
            mask = labels != -1
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss /= mask.float().sum()

            outputs = (loss,) + outputs + (labels,)

        return outputs + (final,)
        
class MultiTaskClassifier(torch.nn.Module):
    '''
    This was an experiment with POS tagging as the auxiliary task
    '''
    
    def __init__(self, pre_trained_model, n_labels, n_labels_aux, device, data_parallel=True):
        super(MultiTaskClassifier, self).__init__()
        
        bert = BertModel.from_pretrained(pre_trained_model).to(device=device)
        #if data_parallel:
        #        self.bert = torch.nn.DataParallel(bert)
        #else:
        
        self.bert = bert
        bert_dim = 768
        
        self.dropout = torch.nn.Dropout()
        
        self.n_labels = n_labels
        self.n_labels_aux = n_labels_aux
        
        self.classifier = torch.nn.Linear(bert_dim, n_labels)
        self.classifier_aux = torch.nn.Linear(bert_dim, n_labels_aux)
        
    
    def forward(self, input_ids, tag_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, 
                labels_aux=None, label_masks=None):
                   
        bert_outputs = self.bert(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask)
                            
        sequence_output = bert_outputs[0]  # (b, MAX_LEN, 768)   # token_output
        
        token_reprs = [embedding[mask] for mask, embedding in zip(label_masks, sequence_output)]
        token_reprs = pad_sequence(sequences=token_reprs, batch_first=True,
                                   padding_value=-1)  # (b, local_max_len, 768)
                                   
        sequence_output = self.dropout(token_reprs)
        #import pdb; pdb.set_trace()
        logits = self.classifier(sequence_output)  # (b, local_max_len, num_labels)
        
        logits_aux = self.classifier_aux(sequence_output) 
        
        
        outputs = (logits, logits_aux)
        
        if labels is not None:      
            labels = [label[mask] for mask, label in zip(label_masks, labels)]
            labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # (b, local_max_len)
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='sum')
            mask = labels != -1
            loss = loss_fct(logits.view(-1, self.n_labels), labels.view(-1))
            loss /= mask.float().sum()
            
            labels_aux = [label[mask] for mask, label in zip(label_masks, labels_aux)]
            labels_aux = pad_sequence(labels_aux, batch_first=True, padding_value=-1)  # (b, local_max_len)
            loss_fct_aux = CrossEntropyLoss(ignore_index=-1, reduction='sum')
            mask = labels_aux != -1
            loss_aux = loss_fct_aux(logits_aux.view(-1, self.n_labels_aux), labels_aux.view(-1))
            loss_aux /= mask.float().sum()
            
            outputs = (loss, loss_aux) + outputs + (labels, labels_aux)

        return outputs  # (loss), scores, (hidden_states), (attentions)




