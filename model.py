from collections  import defaultdict
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv

from transformers import BertModel, RobertaModel
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm1d
from transformers import AutoConfig, AutoModel

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Sequential as Seq, Linear, ReLU
    
import random
import numpy as np

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='max') #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)
    
class BertBiLayerGCN(torch.nn.Module):
    def __init__(self, config, args ):
        super().__init__()
        self.conv_hidden_1 = config.hidden_size
        self.conv_hidden_2 = 128
        self.conv_hidden_out = 64
        self.num_classes = args.num_classes
        self.device = args.device
        self.window_size = args.window_size
        
    
    
        self.bert = BertModel(config)
        self.conv1 = GCNConv(self.conv_hidden_1, self.conv_hidden_2 )
        self.bn1 = BatchNorm1d(self.conv_hidden_2)
        self.conv2 = GCNConv(self.conv_hidden_2, self.conv_hidden_out)
        self.bn2 = BatchNorm1d(self.conv_hidden_out)
        self.linear1 = torch.nn.Linear(2*self.conv_hidden_out,self.num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    
    def forward(self, data):
        node_token_ids, edge_index, all_edges, labels, input_masks, segment_ids = data.x, data.edge_index, data.all_edges, data.y, data.input_masks, data.segment_ids
        num_graphs = data.num_graphs
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        batch = data.batch.reshape(num_nodes,1).to(self.device)
        
        unfolded_input_mask = [eim for im in input_masks for eim in im]
        input_masks = torch.tensor(unfolded_input_mask, dtype=torch.long).to(self.device)          # previous(num_graphs[[11,50][7,50]]) -> [18,50]
        unfolded_segment_ids = [esi for si in segment_ids for esi in si]
        segment_ids = torch.tensor(unfolded_segment_ids, dtype=torch.long).to(self.device)
        ylabels = torch.tensor([e for each in labels for e in each],dtype=torch.long).to(self.device)  #[76]
        num_possible_edges_in_batch = len(ylabels)
        
        assert node_token_ids.size() == segment_ids.size()
        assert segment_ids.size() == input_masks.size()

        out = self.bert(node_token_ids, input_masks, segment_ids )
        cls_tokens = out[1]                         # (num_nodes_in_batch, Bert_hidden_size) [18,768]
        
        x = self.conv1(cls_tokens, edge_index) # (num_nodes_in_batch, 16)         
        x = F.gelu(self.bn1(x))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)          # (num_nodes_in_batch, conv2_out) [18,8]
        x = F.gelu(self.bn2(x))
        x = F.dropout(x, training=self.training)

        # Create the input for every pair of nodes
        edge_repr = torch.empty(num_possible_edges_in_batch,2*self.conv_hidden_out)
        last_node = 0
        index = 0
        for i, each_graph_all_edges in enumerate(all_edges):
            max_node_id = 0
            for each_edge in each_graph_all_edges:
                edge_repr[index] = torch.cat((x[each_edge[0]+last_node],x[each_edge[1]+last_node]),0) #(all_num_possible_edges_in_batch, self.conv_hidden_out*2 [76,2*64]
                index += 1
                max_node_id = max(max_node_id, each_edge[0]+last_node,each_edge[1]+last_node)
            last_node = max_node_id+1
        edge_repr = edge_repr.to(self.device)
        assert last_node == num_nodes  
        
        x = self.linear1(edge_repr)     #(all_num_possible_edges_in_batch, num_classes) [76, 2]
        
        
        preds = x.argmax(dim=1)
        loss = self.loss_fn(x, ylabels)
                                         
        
        return loss, preds, ylabels
    
    
    
    
class BertUniLayerGCN(torch.nn.Module):
    def __init__(self, config, args ):
        super().__init__()
        self.conv_hidden_1 = config.hidden_size
        #self.conv_hidden_2 = 128
        self.conv_hidden_out = 128
        self.num_classes = args.num_classes
        self.device = args.device
        
    
    
        set_seed(args)
        self.bert = BertModel(config)
        self.conv1 = GCNConv(self.conv_hidden_1, self.conv_hidden_out)
        self.bn1 = BatchNorm1d(self.conv_hidden_out)
        self.linear1 = torch.nn.Linear(2*self.conv_hidden_out,self.num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    
    def forward(self, data):
        node_token_ids, edge_index, all_edges, labels, input_masks, segment_ids = data.x, data.edge_index, data.all_edges, data.y, data.input_masks, data.segment_ids
        num_graphs = data.num_graphs
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        batch = data.batch.reshape(num_nodes,1).to(self.device)
        
        unfolded_input_mask = [eim for im in input_masks for eim in im]
        input_masks = torch.tensor(unfolded_input_mask, dtype=torch.long).to(self.device)          # previous(num_graphs[[11,50][7,50]]) -> [18,50]
        unfolded_segment_ids = [esi for si in segment_ids for esi in si]
        segment_ids = torch.tensor(unfolded_segment_ids, dtype=torch.long).to(self.device)
        ylabels = torch.tensor([e for each in labels for e in each],dtype=torch.long).to(self.device)  #[76]
        num_possible_edges_in_batch = len(ylabels)
        
        assert node_token_ids.size() == segment_ids.size()
        assert segment_ids.size() == input_masks.size()

        out = self.bert(node_token_ids, input_masks, segment_ids )
        cls_tokens = out[1]                         # (num_nodes_in_batch, Bert_hidden_size) [18,768]
        
        x = self.conv1(cls_tokens, edge_index) # (num_nodes_in_batch, 16)         
        x = F.gelu(self.bn1(x))
        x = F.dropout(x, training=self.training)

        # Create the input for every pair of nodes
        edge_repr = torch.empty(num_possible_edges_in_batch,2*self.conv_hidden_out)
        last_node = 0
        index = 0
        for i, each_graph_all_edges in enumerate(all_edges):
            max_node_id = 0
            for each_edge in each_graph_all_edges:
                edge_repr[index] = torch.cat((x[each_edge[0]+last_node],x[each_edge[1]+last_node]),0) #(all_num_possible_edges_in_batch, self.conv_hidden_out*2 [76,2*64]
                index += 1
                max_node_id = max(max_node_id, each_edge[0]+last_node,each_edge[1]+last_node)
            last_node = max_node_id+1
        edge_repr = edge_repr.to(self.device)
        assert last_node == num_nodes
        
        x = self.linear1(edge_repr)   
        
        
        preds = x.argmax(dim=1)
        loss = self.loss_fn(x, ylabels) 
        
        return loss, preds, ylabels
    
    
    

class LMProjection(torch.nn.Module):
    def __init__(self, config, args ):
        super().__init__()
        self.conv_hidden_out = config.hidden_size
        self.conv_hidden_out2 = args.gnn_layer1_out_dim
        self.num_classes = args.num_classes
        self.device = args.device
        self.window_size = args.window_size
        self.edge_percent = args.edge_percent
        self.dropout = args.dropout
    
        weights = [self.edge_percent, 1-self.edge_percent]
        self.weights = torch.tensor(weights,dtype=torch.float).to(self.device)
    
        set_seed(args)
        self.bert = AutoModel.from_config(config)
        if args.use_pretrained_weights:
            self.bert = AutoModel.from_pretrained(args.model,config=config)
        self.linear1 = torch.nn.Linear(2*self.conv_hidden_out,self.conv_hidden_out2)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.weights)
        self.linear2 = torch.nn.Linear(self.conv_hidden_out2,self.num_classes)
    
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
    
    
    def forward(self, data):
        node_token_ids, edge_index, all_edges, labels, input_masks, segment_ids = data.x, data.edge_index, data.all_edges, data.y, data.input_masks, data.segment_ids
        num_graphs = data.num_graphs
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        batch = data.batch.reshape(num_nodes,1).to(self.device)
        
        unfolded_input_mask = [eim for im in input_masks for eim in im]
        input_masks = torch.tensor(unfolded_input_mask, dtype=torch.long).to(self.device)          # previous(num_graphs[[11,50][7,50]]) -> [18,50]
        unfolded_segment_ids = [esi for si in segment_ids for esi in si]
        segment_ids = torch.tensor(unfolded_segment_ids, dtype=torch.long).to(self.device)
        ylabels = torch.tensor([e for each in labels for e in each],dtype=torch.long).to(self.device)  #[76]
        num_possible_edges_in_batch = len(ylabels)
        
        assert node_token_ids.size() == segment_ids.size()
        assert segment_ids.size() == input_masks.size()

        out = self.bert(node_token_ids, input_masks, segment_ids )
        cls_tokens = out[1]                         # (num_nodes_in_batch, Bert_hidden_size) [18,768]
        
        x = cls_tokens
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Create the input for every pair of nodes
        edge_repr = torch.empty(num_possible_edges_in_batch,2*self.conv_hidden_out)
        last_node = 0
        index = 0
        for i, each_graph_all_edges in enumerate(all_edges):
            max_node_id = 0
            for each_edge in each_graph_all_edges:
                edge_repr[index] = torch.cat((x[each_edge[0]+last_node],x[each_edge[1]+last_node]),0) #(all_num_possible_edges_in_batch, self.conv_hidden_out*2 [76,2*64]
                index += 1
                max_node_id = max(max_node_id, each_edge[0]+last_node,each_edge[1]+last_node)
            last_node = max_node_id+1
        edge_repr = edge_repr.to(self.device)
        assert last_node == num_nodes
        
        x = self.linear1(edge_repr)   
        x = self.linear2(F.gelu(x))
        
        
        probs = torch.nn.functional.softmax(x, dim=-1)
        loss = self.loss_fn(x, ylabels) 
        
        return loss, probs, ylabels
    
    
    
    
class GCNUniLayerBiProj(torch.nn.Module):
    def __init__(self, config, args ):
        ''' Best one '''
        super().__init__()
        self.conv_hidden_1 = config.hidden_size
        self.conv_hidden_out = args.gnn_layer1_out_dim
        self.num_classes = args.num_classes
        self.device = args.device
        self.window_size = args.window_size
        self.edge_percent = args.edge_percent
        self.dropout = args.dropout
    
        weights = [self.edge_percent, 1-self.edge_percent]
        self.weights = torch.tensor(weights,dtype=torch.float).to(self.device)
    
        set_seed(args)
        self.bert = AutoModel.from_config(config)
        if args.use_pretrained_weights:
            self.bert = AutoModel.from_pretrained(args.model,config=config)
        self.conv1 = GCNConv(self.conv_hidden_1, self.conv_hidden_out)

        self.bn1 = BatchNorm1d(self.conv_hidden_out)
        self.linear1 = torch.nn.Linear(2*self.conv_hidden_out,self.conv_hidden_out)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.weights)
        self.linear2 = torch.nn.Linear(self.conv_hidden_out,self.num_classes)
    
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
    
    
    def forward(self, data):
        node_token_ids, edge_index, all_edges, labels, input_masks, segment_ids = data.x, data.edge_index, data.all_edges, data.y, data.input_masks, data.segment_ids
        num_graphs = data.num_graphs
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        batch = data.batch.reshape(num_nodes,1).to(self.device)
        unfolded_input_mask = [eim for im in input_masks for eim in im]
        input_masks = torch.tensor(unfolded_input_mask, dtype=torch.long).to(self.device)          # previous(num_graphs[[11,50][7,50]]) -> [18,50]
        unfolded_segment_ids = [esi for si in segment_ids for esi in si]
        segment_ids = torch.tensor(unfolded_segment_ids, dtype=torch.long).to(self.device)
        ylabels = torch.tensor([e for each in labels for e in each],dtype=torch.long).to(self.device)  #[76]
        num_possible_edges_in_batch = len(ylabels)
        
        assert node_token_ids.size() == segment_ids.size()
        assert segment_ids.size() == input_masks.size()

        out = self.bert(node_token_ids, input_masks, segment_ids )
        cls_tokens = out[1]                         # (num_nodes_in_batch, Bert_hidden_size) [18,768]
        
        x = self.conv1(cls_tokens, edge_index) # (num_nodes_in_batch, 16)         
        x = self.bn1(F.gelu(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Create the input for every pair of nodes
        edge_repr = torch.empty(num_possible_edges_in_batch,2*self.conv_hidden_out)
        last_node = 0
        index = 0
        for i, each_graph_all_edges in enumerate(all_edges):
            max_node_id = 0
            for each_edge in each_graph_all_edges:
                edge_repr[index] = torch.cat((x[each_edge[0]+last_node],x[each_edge[1]+last_node]),0) #(all_num_possible_edges_in_batch, self.conv_hidden_out*2 [76,2*64]
                index += 1
                max_node_id = max(max_node_id, each_edge[0]+last_node,each_edge[1]+last_node)
            last_node = max_node_id+1
        edge_repr = edge_repr.to(self.device)
    
        assert last_node == num_nodes
        
        x = self.linear1(edge_repr)   
        x = self.linear2(F.gelu(x))
        
        probs = torch.nn.functional.softmax(x, dim=-1)
        loss = self.loss_fn(x, ylabels)  # This loss is from each of multiple GPUs in parallel
    
        return loss, probs, ylabels
    
    
    

class GCNBiLayerBiProj(torch.nn.Module):
    def __init__(self, config, args ):
        super().__init__()
        self.conv_hidden_1 = config.hidden_size
        self.conv_hidden_2 = args.gnn_layer1_out_dim
        self.conv_hidden_out = args.gnn_layer2_out_dim
        self.num_classes = args.num_classes
        self.device = args.device
        self.window_size = args.window_size
        self.edge_percent = args.edge_percent
        self.dropout = args.dropout
    
        weights = [self.edge_percent, 1-self.edge_percent]
        self.weights = torch.tensor(weights,dtype=torch.float).to(self.device)
    
        set_seed(args)
        self.bert = AutoModel.from_config(config)
        if args.use_pretrained_weights:
            self.bert = AutoModel.from_pretrained(args.model,config=config)
        self.conv1 = GCNConv(self.conv_hidden_1, self.conv_hidden_2 )
        self.bn1 = BatchNorm1d(self.conv_hidden_2)
        self.conv2 = GCNConv(self.conv_hidden_2, self.conv_hidden_out)
        self.bn2 = BatchNorm1d(self.conv_hidden_out)
        self.linear1 = torch.nn.Linear(2*self.conv_hidden_out,self.conv_hidden_out)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.weights)
        self.linear2 = torch.nn.Linear(self.conv_hidden_out,self.num_classes)
        
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
    
    
    def forward(self, data):
        node_token_ids, edge_index, all_edges, labels, input_masks, segment_ids = data.x, data.edge_index, data.all_edges, data.y, data.input_masks, data.segment_ids
        num_graphs = data.num_graphs
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        batch = data.batch.reshape(num_nodes,1).to(self.device)
        
        unfolded_input_mask = [eim for im in input_masks for eim in im]
        input_masks = torch.tensor(unfolded_input_mask, dtype=torch.long).to(self.device)          # previous(num_graphs[[11,50][7,50]]) -> [18,50]
        unfolded_segment_ids = [esi for si in segment_ids for esi in si]
        segment_ids = torch.tensor(unfolded_segment_ids, dtype=torch.long).to(self.device)
        ylabels = torch.tensor([e for each in labels for e in each],dtype=torch.long).to(self.device)  #[76]
        num_possible_edges_in_batch = len(ylabels)
        
        assert node_token_ids.size() == segment_ids.size()
        assert segment_ids.size() == input_masks.size()

        out = self.bert(node_token_ids, input_masks, segment_ids )
        cls_tokens = out[1]                         # (num_nodes_in_batch, Bert_hidden_size) [18,768]
        
        x = self.conv1(cls_tokens, edge_index) # (num_nodes_in_batch, 16)         
        x = self.bn1(F.gelu(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)          # (num_nodes_in_batch, conv2_out) [18,8]
        x = self.bn2(F.gelu(x))
        x = F.dropout(x, p=self.dropout,training=self.training)

        # Create the input for every pair of nodes
        edge_repr = torch.empty(num_possible_edges_in_batch,2*self.conv_hidden_out)
        last_node = 0
        index = 0
        for i, each_graph_all_edges in enumerate(all_edges):
            max_node_id = 0
            for each_edge in each_graph_all_edges:
                edge_repr[index] = torch.cat((x[each_edge[0]+last_node],x[each_edge[1]+last_node]),0) #(all_num_possible_edges_in_batch, self.conv_hidden_out*2 [76,2*64]
                index += 1
                max_node_id = max(max_node_id, each_edge[0]+last_node,each_edge[1]+last_node)
            last_node = max_node_id+1
        edge_repr = edge_repr.to(self.device)
        assert last_node == num_nodes  
        
        x = self.linear1(edge_repr)     #(all_num_possible_edges_in_batch, num_classes) [76, 2]
        x = self.linear2(F.gelu(x))
        
        probs = torch.nn.functional.softmax(x, dim=-1)
        loss = self.loss_fn(x, ylabels)
                                         
        
        return loss, probs, ylabels
    
    
    
    
    
    
class GATUniLayerBiProj(torch.nn.Module):
    ''' For Both BERT and RoBERTa'''
    ''' Working '''
    def __init__(self, config, args ):
        super().__init__()
        self.conv_hidden_1 = config.hidden_size
        self.conv_hidden_out = args.gnn_layer1_out_dim
        self.num_classes = args.num_classes
        self.device = args.device
        self.window_size = args.window_size
        self.edge_percent = args.edge_percent
        self.dropout = args.dropout
    
        weights = [self.edge_percent, 1-self.edge_percent]
        self.weights = torch.tensor(weights,dtype=torch.float).to(self.device)
    
        set_seed(args)
        self.bert = AutoModel.from_config(config)
        if args.use_pretrained_weights:
            self.bert = AutoModel.from_pretrained(args.model,config=config)
            
        self.conv1 = GATConv(self.conv_hidden_1, self.conv_hidden_out, heads=4, dropout=self.dropout, concat=False)

        self.bn1 = BatchNorm1d(self.conv_hidden_out)
        self.linear1 = torch.nn.Linear(2*self.conv_hidden_out,self.conv_hidden_out)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.weights)
        self.linear2 = torch.nn.Linear(self.conv_hidden_out,self.num_classes)
        
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
    
    
    def forward(self, data):
        node_token_ids, edge_index, all_edges, labels, input_masks, segment_ids = data.x, data.edge_index, data.all_edges, data.y, data.input_masks, data.segment_ids
        num_graphs = data.num_graphs
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        batch = data.batch.reshape(num_nodes,1).to(self.device)
        
        unfolded_input_mask = [eim for im in input_masks for eim in im]
        input_masks = torch.tensor(unfolded_input_mask, dtype=torch.long).to(self.device)          # previous(num_graphs[[11,50][7,50]]) -> [18,50]
        unfolded_segment_ids = [esi for si in segment_ids for esi in si]
        segment_ids = torch.tensor(unfolded_segment_ids, dtype=torch.long).to(self.device)
        ylabels = torch.tensor([e for each in labels for e in each],dtype=torch.long).to(self.device)  #[76]
        num_possible_edges_in_batch = len(ylabels)
        
        assert node_token_ids.size() == segment_ids.size()
        assert segment_ids.size() == input_masks.size()

        out = self.bert(node_token_ids, input_masks, segment_ids )
        cls_tokens = out[1]                         # (num_nodes_in_batch, Bert_hidden_size) [18,768]
        
        x = self.conv1(cls_tokens, edge_index) # (num_nodes_in_batch, 16)         
        x = self.bn1(F.gelu(x))
        x = F.dropout(x, training=self.training)

        # Create the input for every pair of nodes
        edge_repr = torch.empty(num_possible_edges_in_batch,2*self.conv_hidden_out)
        last_node = 0
        index = 0
        for i, each_graph_all_edges in enumerate(all_edges):
            max_node_id = 0
            for each_edge in each_graph_all_edges:
                edge_repr[index] = torch.cat((x[each_edge[0]+last_node],x[each_edge[1]+last_node]),0) #(all_num_possible_edges_in_batch, self.conv_hidden_out*2 [76,2*64]
                index += 1
                max_node_id = max(max_node_id, each_edge[0]+last_node,each_edge[1]+last_node)
            last_node = max_node_id+1
        edge_repr = edge_repr.to(self.device)
        assert last_node == num_nodes
        
        x = self.linear1(edge_repr)   
        x = self.linear2(F.gelu(x))
        
        
        probs = torch.nn.functional.softmax(x, dim=-1)
        loss = self.loss_fn(x, ylabels) 
        
        return loss, probs, ylabels
    
    

class GATBiLayerBiProj(torch.nn.Module):
    ''' For Both BERT and RoBERTa'''
    def __init__(self, config, args ):
        super().__init__()
        self.conv_hidden_1 = config.hidden_size
        self.conv_hidden_2 = args.gnn_layer1_out_dim
        self.conv_hidden_out = args.gnn_layer2_out_dim
        self.dropout = args.dropout
        self.num_classes = args.num_classes
        self.device = args.device
        self.window_size = args.window_size
        self.edge_percent = args.edge_percent
    
        weights = [self.edge_percent, 1-self.edge_percent]
        self.weights = torch.tensor(weights,dtype=torch.float).to(self.device)
    
    
        set_seed(args)
        self.bert = AutoModel.from_config(config)
        if args.use_pretrained_weights:
            self.bert = AutoModel.from_pretrained(args.model,config=config)
    
        self.conv1 = GATConv(self.conv_hidden_1, self.conv_hidden_2, heads=4, dropout=self.dropout)
        self.conv2 = GATConv(self.conv_hidden_2*4, self.conv_hidden_out, concat=False)

        self.bn1 = BatchNorm1d(self.conv_hidden_2*4)
        self.bn2 = BatchNorm1d(self.conv_hidden_out)
        self.linear1 = torch.nn.Linear(2*self.conv_hidden_out,self.conv_hidden_out)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.weights)
        self.linear2 = torch.nn.Linear(self.conv_hidden_out,self.num_classes)
        
#         torch.nn.init.xavier_uniform_(self.linear1.weight)
#         torch.nn.init.xavier_uniform_(self.linear2.weight)
    
    
    def forward(self, data):
        node_token_ids, edge_index, all_edges, labels, input_masks, segment_ids = data.x, data.edge_index, data.all_edges, data.y, data.input_masks, data.segment_ids
        num_graphs = data.num_graphs
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        batch = data.batch.reshape(num_nodes,1).to(self.device)
        
        unfolded_input_mask = [eim for im in input_masks for eim in im]
        input_masks = torch.tensor(unfolded_input_mask, dtype=torch.long).to(self.device)          # previous(num_graphs[[11,50][7,50]]) -> [18,50]
        unfolded_segment_ids = [esi for si in segment_ids for esi in si]
        segment_ids = torch.tensor(unfolded_segment_ids, dtype=torch.long).to(self.device)
        ylabels = torch.tensor([e for each in labels for e in each],dtype=torch.long).to(self.device)  #[76]
        num_possible_edges_in_batch = len(ylabels)
        
        assert node_token_ids.size() == segment_ids.size()
        assert segment_ids.size() == input_masks.size()

        out = self.bert(node_token_ids, input_masks, segment_ids )
        cls_tokens = out[1]                         # (num_nodes_in_batch, Bert_hidden_size) [18,768]
        
        x = self.conv1(cls_tokens, edge_index) # (num_nodes_in_batch, 16)         
        x = self.bn1(F.gelu(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)          
        x = self.bn2(F.gelu(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Create the input for every pair of nodes
        edge_repr = torch.empty(num_possible_edges_in_batch,2*self.conv_hidden_out)
        last_node = 0
        index = 0
        for i, each_graph_all_edges in enumerate(all_edges):
            max_node_id = 0
            for each_edge in each_graph_all_edges:
                edge_repr[index] = torch.cat((x[each_edge[0]+last_node],x[each_edge[1]+last_node]),0) #(all_num_possible_edges_in_batch, self.conv_hidden_out*2 [76,2*64]
                index += 1
                max_node_id = max(max_node_id, each_edge[0]+last_node,each_edge[1]+last_node)
            last_node = max_node_id+1
        edge_repr = edge_repr.to(self.device)
        assert last_node == num_nodes
        
        x = self.linear1(edge_repr)   
        x = self.linear2(F.gelu(x))
        
        
        probs = torch.nn.functional.softmax(x, dim=-1)
        loss = self.loss_fn(x, ylabels) 
        
        return loss, probs, ylabels
    
    
    
class RandomBaseline:
    ''' Randomly select any of 0 or 1 for each true labels
    '''
    def get_baseline(self, test_data):
        
        ylabels = [eachlabel for each in test_data for eachlabel in each.y]
        preds = [random.choice([0,1]) for _ in range(len(ylabels))]
        return np.array(preds), np.array(ylabels)
    

class WtdRandomBaseline:
    ''' Randomly select any of 0 or 1 for each true labels
    '''
    def __init__(self, args):
        self.edge_percent = args.edge_percent
        
    def get_baseline(self, test_data):
        
        ylabels = [eachlabel for each in test_data for eachlabel in each.y]
        num_ones = round(len(ylabels)*self.edge_percent)
        num_zeros = len(ylabels) - num_ones
        pos_ones = random.sample(range(len(ylabels)), num_ones)
        preds = np.array([0]*len(ylabels))
        preds[pos_ones] = 1
        return preds, np.array(ylabels)
    
    
    

class GCNUniLayerBiProjFullSeq(torch.nn.Module):
    def __init__(self, config, args ):
        ''' Uses the Full sequence repr instead of CLS only '''
        
        super().__init__()
        self.conv_hidden_1 = config.hidden_size
        self.conv_hidden_out = args.gnn_layer1_out_dim
        self.num_classes = args.num_classes
        self.device = args.device
        self.window_size = args.window_size
        self.edge_percent = args.edge_percent
        self.dropout = args.dropout
        self.max_seq_len = args.max_seq_len
    
        weights = [self.edge_percent, 1-self.edge_percent]
        self.weights = torch.tensor(weights,dtype=torch.float).to(self.device)
    
        set_seed(args)
        self.bert = AutoModel.from_config(config)
        if args.use_pretrained_weights:
            self.bert = AutoModel.from_pretrained(args.model,config=config)
        self.conv1 = GCNConv(self.conv_hidden_1*self.max_seq_len, self.conv_hidden_out)

        self.bn1 = BatchNorm1d(self.conv_hidden_out)
        self.linear1 = torch.nn.Linear(2*self.conv_hidden_out,self.conv_hidden_out)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.weights)
        self.linear2 = torch.nn.Linear(self.conv_hidden_out,self.num_classes)
    
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
    
    
    def forward(self, data):
        node_token_ids, edge_index, all_edges, labels, input_masks, segment_ids = data.x, data.edge_index, data.all_edges, data.y, data.input_masks, data.segment_ids
        num_graphs = data.num_graphs
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        batch = data.batch.reshape(num_nodes,1).to(self.device)
        
        unfolded_input_mask = [eim for im in input_masks for eim in im]
        input_masks = torch.tensor(unfolded_input_mask, dtype=torch.long).to(self.device)          # previous(num_graphs[[11,50][7,50]]) -> [18,50]
        unfolded_segment_ids = [esi for si in segment_ids for esi in si]
        segment_ids = torch.tensor(unfolded_segment_ids, dtype=torch.long).to(self.device)
        ylabels = torch.tensor([e for each in labels for e in each],dtype=torch.long).to(self.device)  #[76]
        num_possible_edges_in_batch = len(ylabels)
        
        assert node_token_ids.size() == segment_ids.size()
        assert segment_ids.size() == input_masks.size()

        out = self.bert(node_token_ids, input_masks, segment_ids ) #((43,128,768),(43,768))
        cls_tokens = out[0].reshape(out[0].size()[0],-1) # (41, 98304)
        x = self.conv1(cls_tokens, edge_index) # (num_nodes_in_batch, 16)         
        x = self.bn1(F.gelu(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Create the input for every pair of nodes
        edge_repr = torch.empty(num_possible_edges_in_batch,2*self.conv_hidden_out)
        last_node = 0
        index = 0
        for i, each_graph_all_edges in enumerate(all_edges):
            max_node_id = 0
            for each_edge in each_graph_all_edges:
                edge_repr[index] = torch.cat((x[each_edge[0]+last_node],x[each_edge[1]+last_node]),0) #(all_num_possible_edges_in_batch, self.conv_hidden_out*2 [76,2*64]
                index += 1
                max_node_id = max(max_node_id, each_edge[0]+last_node,each_edge[1]+last_node)
            last_node = max_node_id+1
        edge_repr = edge_repr.to(self.device)
    
        assert last_node == num_nodes
        
        x = self.linear1(edge_repr)   
        x = self.linear2(F.gelu(x))
        
        probs = torch.nn.functional.softmax(x, dim=-1)
        loss = self.loss_fn(x, ylabels)  # This loss is from each of multiple GPUs in parallel
    
        return loss, probs, ylabels
    
    