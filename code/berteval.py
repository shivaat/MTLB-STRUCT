import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange, tqdm
from seqeval.metrics import accuracy_score, f1_score, classification_report
#from sklearn.metrics import precision_recall_fscore_support

from torch_struct import DependencyCRF

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def eval(iter_data, model, tags2idx, device_name, mt=False):
    device = device_name
    logger.info("starting to evaluate")
    model = model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps = 0
    predictions, true_labels, data_instances, probs = [], [], [], []
    dep_predictions, dep_gold_labels = [], []
    total_edges = 0
    incorrect_edges = 0
    for batch in tqdm(iter_data):
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_pos_ids, b_tag_ids, b_deptype_ids, b_labels, b_input_mask,\
                b_token_type_ids, b_label_masks, lengths = batch
        #print('b_input_ids size:', b_input_ids.size())    # batch_size*max_len
        lengths = torch.flatten(lengths)
        batch_size, _ = b_tag_ids.shape
        
        with torch.no_grad():
            if not mt:
                tmp_eval_loss, logits, reduced_labels = model(b_input_ids, b_tag_ids,
                                                          token_type_ids=b_token_type_ids,
                                                          attention_mask=b_input_mask,
                                                          labels=b_labels,
                                                          label_masks=b_label_masks)

            else:

                tmp_eval_loss, logits, reduced_labels, final = model(b_input_ids, b_tag_ids, 
                                                        labels=b_labels, label_masks=b_label_masks)
                
                if not lengths.max() <= final.shape[1]: #+ 1:
                    #print("fail to evaluate for dependency:", "max length", lengths.max(), "final shape", final.shape[1])
                    #continue
                    out = torch.zeros(b_tags.size())   # not sure about the size!
                                                        # I cannot think what the size should be
                else:
                    dist = DependencyCRF(final, lengths=lengths)

                    out = dist.argmax
                    dep_predictions.append(out)

                    b_tags = [tag[mask] for mask, tag in zip(b_label_masks, b_tag_ids)]
                    b_tags = pad_sequence(b_tags, batch_first=True, padding_value=0)
                    dep_gold = dist.struct.to_parts(b_tags, lengths=lengths).type_as(out)
                    dep_gold_labels.append(dep_gold)

                    incorrect_edges += (out[:, :].cpu() - dep_gold[:, :].cpu()).abs().sum() / 2.0
                    total_edges += dep_gold.sum()
                    #log_prob = dist.log_prob(dep_labels)
                    #dep_loss = log_prob.sum()
            
        tags_idx = [tags2idx[t] for t in tags2idx]
        logits_probs = F.softmax(logits, dim=2)[:,:, tags_idx]
        preds = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        #print('***',logits_probs)
        #print('logits size:',logits.size())     # batch_size*sentence_len(before padding)
        logits_probs = logits_probs.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        reduced_labels = reduced_labels.to('cpu').numpy()

        labels_to_append = []
        predictions_to_append = []
        logits_to_append = []

        for prediction, r_label, logit in zip(preds, reduced_labels, logits_probs):
            preds = []
            labels = []
            logs = []
            for pred, lab, log in zip(prediction, r_label, logit):
                if lab.item() == -1:  # masked label; -1 means do not collect this label
                    continue
                preds.append(pred)
                labels.append(lab)
                logs.append(log)
            predictions_to_append.append(preds)
            labels_to_append.append(labels)
            logits_to_append.append(logs)

        predictions.extend(predictions_to_append)
        true_labels.extend(labels_to_append)
        data_instances.extend(b_input_ids)
        probs.extend(logits_to_append)

        eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1

    if mt:
        print('num of edges', total_edges, 'incorrect_edges:', incorrect_edges)
        print('aacuracy', (total_edges-incorrect_edges)/total_edges)

    eval_loss = eval_loss / nb_eval_steps
    logger.info("eval loss (only main): {}".format(eval_loss))
    #idx2tags = {tags2idx[t]: t for t in tags2idx}
    # Changing the o-X labels to O, make the labels compatible with seqeval formatting
    idx2tags = {}
    for t in tags2idx:
        if not t.startswith('o-'):
            idx2tags[tags2idx[t]]= t
        else:
            idx2tags[tags2idx[t]]='O'
    pred_tags = [[idx2tags[p_i] for p_i in p] for p in predictions]
    valid_tags = [[idx2tags[l_i] for l_i in l] for l in true_labels]
    logger.info("Seqeval accuracy: {}".format(accuracy_score(valid_tags, pred_tags)))
    fscore = f1_score(valid_tags, pred_tags, zero_division=0)
    logger.info("Seqeval F1-Score: {}".format(fscore))
    logger.info("Seqeval Classification report: -- ")
    logger.info(classification_report(valid_tags, pred_tags, zero_division=0))

    final_labels = [[idx2tags[p_i] for p_i in p] for p in predictions]
    return final_labels, probs, fscore

def eval_blind(iter_data, model, tags2idx, device_name, mt=False):
    device = device_name
    logger.info("starting to evaluate")
    model = model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps = 0
    predictions, true_labels, data_instances, probs = [], [], [], []
    dep_predictions, dep_gold_labels = [], []
    total_edges = 0
    incorrect_edges = 0
    for batch in tqdm(iter_data):
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_pos_ids, b_tag_ids, b_deptype_ids, b_labels, b_input_mask,\
                b_token_type_ids, b_label_masks, lengths = batch
        #print('b_input_ids size:', b_input_ids.size())    # batch_size*max_len
        lengths = torch.flatten(lengths)
        batch_size, _ = b_tag_ids.shape

        with torch.no_grad():
            if not mt:
                tmp_eval_loss, logits, reduced_labels = model(b_input_ids, b_tag_ids,
                                                          token_type_ids=b_token_type_ids,
                                                          attention_mask=b_input_mask,
                                                          labels=b_labels,
                                                          label_masks=b_label_masks)
            else:

                tmp_eval_loss, logits, reduced_labels, final = model(b_input_ids, b_tag_ids,
                                                        labels=b_labels, label_masks=b_label_masks)

                if not lengths.max() <= final.shape[1]: #+ 1:
                    #print("fail to evaluate for dependency:", "max length", lengths.max(), "final shape", final.shape[1])
                    #continue
                    out = torch.zeros(b_tags.size())   # Not sure about the size!!!
                                                        
                else:
                    dist = DependencyCRF(final, lengths=lengths)

                    out = dist.argmax
                    dep_predictions.append(out)

                    b_tags = [tag[mask] for mask, tag in zip(b_label_masks, b_tag_ids)]
                    b_tags = pad_sequence(b_tags, batch_first=True, padding_value=0)
                    dep_gold = dist.struct.to_parts(b_tags, lengths=lengths).type_as(out)
                    dep_gold_labels.append(dep_gold)

                    incorrect_edges += (out[:, :].cpu() - dep_gold[:, :].cpu()).abs().sum() / 2.0
                    total_edges += dep_gold.sum()
                    #log_prob = dist.log_prob(dep_labels)
                    #dep_loss = log_prob.sum()

        tags_idx = [tags2idx[t] for t in tags2idx]
        logits_probs = F.softmax(logits, dim=2)[:,:, tags_idx]
        preds = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        #print('***',logits_probs)
        #print('logits size:',logits.size())     # batch_size*sentence_len(before padding)
        logits_probs = logits_probs.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        reduced_labels = reduced_labels.to('cpu').numpy()

        predictions_to_append = []
        logits_to_append = []

        for prediction, r_label, logit in zip(preds, reduced_labels, logits_probs):
            preds = []
            logs = []
            for pred, lab, log in zip(prediction, r_label, logit):
                if lab.item() == -1:  # masked label; -1 means do not collect this label
                    continue
                preds.append(pred)
                logs.append(log)
            predictions_to_append.append(preds)
            logits_to_append.append(logs)

        predictions.extend(predictions_to_append)
        data_instances.extend(b_input_ids)
        probs.extend(logits_to_append)

        nb_eval_steps += 1

    idx2tags = {tags2idx[t]: t for t in tags2idx}
    #pred_tags = [idx2tags[p_i] for p in predictions for p_i in p]
    final_labels = [[idx2tags[p_i] for p_i in p] for p in predictions]

    return final_labels, probs



