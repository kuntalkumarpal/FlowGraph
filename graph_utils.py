import numpy as np
import pandas as pd
import pickle
import csv
import os
import torch
import sys
import jsonlines
import argparse
import random
random.seed(42)
import itertools
import logging
from collections import defaultdict


### Feature processing
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from filelock import FileLock

from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

### Data Loader and mini-batch preperation
import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import Data


seed =42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def read_examples_from_file(data, connection_type, mode):
    
        guid_index = 1
        examples = []
        example_size = []
        edges = []
        conn_edges = []
        examples_added = 0
        for eachfile in data:
            filetext = eachfile['data']
            edge_index = []
            example = []
            for idx, eachsentattr in enumerate(filetext):
                
                words = eachsentattr['sentence'].split()
                example.append(words)
                guid_index += 1
                relations = eachsentattr['relation_id']
                
                for each_rel in relations:
                    edge_index.append( (idx, each_rel) )
            examples_added += len(filetext)
        
            example_size.append(len(filetext))
            conn_edges.append( edge_index )
            edges.append(list(itertools.combinations(range(0,len(filetext)),2)))
            examples.append(example)
        
        labels = []
        for ee, ec in zip(edges, conn_edges):
            each_label = []
            for e in ee:
                if e in ec:
                    each_label.append(1)
                else:
                    each_label.append(0)
            labels.append(each_label)
        print("Examples to Graph Structure Stats :",len(edges),len(conn_edges),len(example_size),len(examples), len(labels))
        
        if connection_type == "complete":
            conn_edges = edges
        elif connection_type == "linear":
            linear_edges = []
            for each in edges:
                eachgraph_linear_edges = []
                for each_edge in each:
                    # First node in edge is 1 diff from other node
                    if each_edge[1]-each_edge[0] == 1:
                        eachgraph_linear_edges.append(each_edge)
                linear_edges.append(eachgraph_linear_edges)
            conn_edges = linear_edges
        return examples, example_size, conn_edges, edges, labels
    
def convert_examples_to_features(examples : List, 
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        model_type,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=0, #cls_token_segment_id=1 to check why it was 1
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,):
    
    all_features = []
    for each_process in examples:
        features = []
    #     tokens_orignal = []
        for (ex_index, example) in enumerate(each_process):
            tokens = []
            input_ids = []
            seqment_ids = []
            if "roberta" not in model_type:
                for word in example:
                    word_tokens = tokenizer.tokenize(word)
                    if len(word_tokens) > 0:
                        tokens.extend(word_tokens)

                    # Account for [CLS] and [SEP] with "- 2"
                    special_tokens_count = tokenizer.num_special_tokens_to_add()
                    if len(tokens) > max_seq_length - special_tokens_count:
                        tokens = tokens[: (max_seq_length - special_tokens_count)]
        #                 label_ids = label_ids[: (max_seq_length - special_tokens_count)]

                tokens += [sep_token]
                if sep_token_extra:
                    # roberta uses an extra separator b/w pairs of sentences
                    tokens += [sep_token]
                segment_ids = [sequence_a_segment_id] * len(tokens)
                if cls_token_at_end:
                    tokens += [cls_token]
                    segment_ids += [cls_token_segment_id]
                else:
                    tokens = [cls_token] + tokens
                    segment_ids = [cls_token_segment_id] + segment_ids

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
            else:
                ### For RoBERTa ###
                example_sent = " ".join(example)
                input_ids = tokenizer.encode(example_sent, add_special_tokens=True,  max_length=min(max_seq_length, tokenizer.max_len))
                tokens = tokenizer.decode(input_ids).split() # This is just to show not used so <s> , . will get attached to some other tokens on splits
                segment_ids = [sequence_a_segment_id] * len(input_ids)
                
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)


            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        #         label_ids = ([pad_token_label_id] * padding_length) + label_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
        #         label_ids += [pad_token_label_id] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            
            features.append([tokens, input_ids, input_mask, segment_ids])
        all_features.append(features)
    return all_features
    
def prepare_data_graph(features, conn_edges, edges, labels):
    
    data_list = []
    for eachgraph_features, eachgraph_connedges, eachgraph_alledges, eachgraph_labels in zip(features, conn_edges, edges, labels):
        
        token_ids = [e[1] for e in eachgraph_features]
        input_masks = [e[2] for e in eachgraph_features]
        segment_ids = [e[3] for e in eachgraph_features]
        each_feature = torch.tensor(token_ids, dtype=torch.long)
        each_edge_index = torch.tensor(eachgraph_connedges, dtype=torch.long).T
        data = Data(x=each_feature, 
                    edge_index=each_edge_index, 
                    all_edges=eachgraph_alledges, 
                    y=eachgraph_labels, 
                    input_masks=input_masks, 
                    segment_ids=segment_ids)
        data_list.append(data)
        
    return data_list
        
def balance_connected_unconnected_edges(edges, labels):
    
    bal_edges = []
    bal_labels = []
    for eedges, elabels in zip(edges, labels):
        distr = defaultdict(list)
        each_bal_edges = []
        each_bal_labels = []
        for e,l in zip(eedges,elabels):
            distr[l].append(e)
        
        random.shuffle(distr[0])    #select same number of no_connections as connections
        to_consider = len(distr[1]) #Since less
        temp_list = []
        for each in distr[1]:
            temp_list.append((each,1))
        for each in distr[0][:to_consider]:
            temp_list.append((each,0))
        random.shuffle(temp_list)
        for each in temp_list:
            each_bal_edges.append(each[0])
            each_bal_labels.append(each[1])
        
        assert len(each_bal_edges)==len(each_bal_labels)
        
        bal_edges.append(each_bal_edges)
        bal_labels.append(each_bal_labels)

    assert len(edges) == len(bal_edges)
    assert len(labels) == len(bal_labels)
    
    return bal_edges, bal_labels
    
    
def windowed_pairs_selection(edges, conn_edges, labels, window):
    
    win_edges, win_labels = [], []
    for eedges, elabels in zip(edges, labels):
#         print(eedges, elabels)
        sel_edges, sel_labels = [], []
        for e, l in zip(eedges, elabels):
#             print(e,l, e[1]-e[0])
            # Remove considering those edges between nodes which are more than window_size distance apart and are all 0s
            if e[1]-e[0] > window and l==0:
                continue
            sel_edges.append(e)
            sel_labels.append(l)
        win_edges.append(sel_edges)
        win_labels.append(sel_labels)
#         print(sel_edges, sel_labels)
#         print(len(eedges),len(elabels))
#         print(len(sel_edges),len(sel_labels))
                
#         input()
    
    # selecting graph connections which are within the window
    win_conn_edges = []
    for each_graph_conn in conn_edges:
        sel_conn_edges = []
        for e in each_graph_conn:
            if e[1]-e[0] > window:
                continue
            sel_conn_edges.append(e)
        win_conn_edges.append(sel_conn_edges)
        
                
    
    return win_edges, win_conn_edges, win_labels

def find_label_stats(labels):
    
    l = defaultdict(int)
    for each in labels:
        for e in each:
            l[e]+=1
            
    
    return l[0], l[1], round(l[1]/(l[0]+l[1]),2) 
        
def oversample_positive_data(edges, labels):
    
    new_edges, new_labels = [], []
    for eedges, elabels in zip(edges, labels):
        sel_edges, sel_labels = [], []
        for e, l in zip(eedges, elabels):
            sel_edges.append(e)
            sel_labels.append(l)
            if l==1: #If there is an edge n1n2 oversample that with n2n1 since procedural text no back direction
                sel_edges.append((e[1],e[0]))
                sel_labels.append(l)
                
        new_edges.append(sel_edges)
        new_labels.append(sel_labels)
        
    return new_edges, new_labels
    


def prepare_data(filepath, domain, max_seq_len, tokenizer, window, is_oversample, graph_connection, model_type):
    
    Xtrain = []
    Xval = []
    Xtest = []
    with jsonlines.open(os.path.join(filepath,'train.jsonl'),'r') as f:
        for each in f:
            Xtrain.append(each)
    with jsonlines.open(os.path.join(filepath,'val.jsonl'),'r') as f:
        for each in f:
            Xval.append(each)
    with jsonlines.open(os.path.join(filepath,'test.jsonl'),'r') as f:
        for each in f:
            Xtest.append(each)
        
    
    ### Gather Train Features
    ### sentences    : each sentence of each graph
    ### example_size : size of each graph
    ### conn_edges   : graph connections [adj matrix to learn features]
    ### edges        : all edges [complete graph - excluding the self loop]
    ### labels       : ylabels [whether node exists between every pairs of nodes]
    sentences, example_size, conn_edges, edges, labels = read_examples_from_file(Xtrain, graph_connection, "train")
#     print(len(sentences), len(example_size), len(conn_edges), len(edges), len(labels))
#     print(example_size)
        
    features = convert_examples_to_features(examples=sentences, 
        max_seq_length=max_seq_len,
        tokenizer=tokenizer,
        model_type=model_type)
    logger.info("Feature length Train: %d",len(features))
#     logger.info("Features: %s", " ".join([str(x) for x in features[0][:3]]))
    
    if is_oversample:
        edges, labels = oversample_positive_data(edges, labels)
    if window:
        edges,conn_edges, labels = windowed_pairs_selection(edges, conn_edges, labels, window)
    print("Pre-Balanced Train",find_label_stats(labels))
#     edges, labels = balance_connected_unconnected_edges(edges, labels)

    d_train = prepare_data_graph(features, conn_edges, edges, labels)
    print("Post-Balanced Train",find_label_stats(labels))
    _,_,edge_percent = find_label_stats(labels)
#     print(edge_percent)
#     input()
    
    
    
    ### Gather Val Features
    sentences, example_size, conn_edges, edges, labels = read_examples_from_file(Xval, graph_connection, "val")
#     print(len(sentences), len(example_size), len(conn_edges), len(edges), len(labels))
    
    features = convert_examples_to_features(examples=sentences, 
        max_seq_length=max_seq_len,
        tokenizer=tokenizer,
        model_type=model_type)
    logger.info("Feature length Val: %d",len(features))
#     logger.info("Features: %s", " ".join([str(x) for x in features[0][:3]]))
    
    if window:
        edges, conn_edges, labels = windowed_pairs_selection(edges, conn_edges, labels, window)
        
        
    d_val = prepare_data_graph(features, conn_edges, edges, labels)
    print("Val Samples",find_label_stats(labels))
    
    
    
    
    ### Gather Test Features
    sentences, example_size, conn_edges, edges, labels = read_examples_from_file(Xtest, graph_connection, "test")
#     print(len(sentences), len(example_size), len(conn_edges), len(edges), len(labels))
    
    features = convert_examples_to_features(examples=sentences, 
        max_seq_length=max_seq_len,
        tokenizer=tokenizer,
        model_type=model_type)
    logger.info("Feature length Val: %d",len(features))

    if window:
        edges, conn_edges, labels = windowed_pairs_selection(edges, conn_edges, labels, window)
        
        
    d_test = prepare_data_graph(features, conn_edges, edges, labels)
    print("Test Samples",find_label_stats(labels))
    
   
    
    return d_train, d_val, d_test, edge_percent



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--datafile", default="", type=str)
    parser.add_argument("--domain", default="cooking", type=str)
    args = parser.parse_args()
    
    data = prepare_data(args.datafile, args.domain)
#     print(data[0])
    loader = DataLoader(data, batch_size=2, shuffle=False)
    
    for batch in loader:
        print("Batch:",batch) 
        logger.info("Graphs: %d, Nodes: %d, Edges: %d",batch.num_graphs, batch.num_nodes, batch.num_edges)