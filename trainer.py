import argparse
import logging
import numpy as np
from tqdm import tqdm
import random
import os

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve



import torch
import torch.nn.functional as F

from torch_geometric.data import DataLoader
from torch_geometric.nn import DataParallel
from torch_geometric.data import DataListLoader

from graph_utils import prepare_data
from model import BertUniLayerGCN, BertBiLayerGCN, LMProjection, GCNUniLayerBiProj, GCNBiLayerBiProj, GATUniLayerBiProj, GATBiLayerBiProj, GCNUniLayerBiProjFullSeq
from model import RandomBaseline, WtdRandomBaseline

from transformers import AutoConfig

from transformers.data.metrics import simple_accuracy


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


from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    get_linear_schedule_with_warmup,
)

def accuracy_metric(preds, labels, pos_prob, isbaseline=False):
    
    assert len(preds) == len(labels)
    lr_precision, lr_recall = 0, 0
    if not isbaseline:
        lr_precision, lr_recall, _ = precision_recall_curve(labels, pos_prob)
        
    out = {'accuracy': accuracy_score(preds, labels),
           'precision': precision_score(labels, preds),
           'recall': recall_score(labels, preds),
           'f1': f1_score(labels, preds),
           'classification_report':classification_report(labels, preds),
           'roc_auc_score':roc_auc_score(labels,pos_prob)  if not isbaseline else 0,
           "cohen":cohen_kappa_score(labels,preds),
           "pr_auc":auc(lr_recall, lr_precision) if not isbaseline else 0
          }
    return out
            

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

        
logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def evaluate(model, data_loader, args, mode):
    
    total_loss = 0
    total_accuracy = 0
    total_preds = []
    total_ylabel = []
    total_pos_prob = []
    total_steps = 0
    
    logger.info("***** Running evaluation {} ***** %s", mode)
    logger.info("  Num examples = %d", len(data_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    for step, batch in enumerate(data_loader):
            
        loss, preds, ylabels, pos_prob = 0, [], [], []
        model.eval()
        batch_data = batch       # No need for sending data to device
        with torch.no_grad():
            loss, probs, ylabels = model(batch_data)
            if args.n_gpu > 1:
                loss = loss.mean()
            preds = probs.argmax(dim=1)
            pos_prob = probs[:,1]
            total_loss += loss.item()
        total_steps += 1

        # Calculate predictions
        total_preds += preds.detach().cpu().numpy().tolist()
        total_ylabel += ylabels.detach().cpu().numpy().tolist()
        total_pos_prob += pos_prob.detach().cpu().numpy().tolist()

    out_metric = accuracy_metric(total_preds, total_ylabel,total_pos_prob)
    out_metric['loss'] = total_loss/total_steps
    
    return out_metric
        
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--datafile", default="cooking/", type=str)
    parser.add_argument("--domain", default="cooking", type=str)
    parser.add_argument("--num_epochs",default=100,type=int)
    parser.add_argument("--model",default="bert-large-uncased",type=str)
    parser.add_argument("--batch_size",default=4, type=int)
    parser.add_argument("--eval_batch_size",default=2, type=int)
    parser.add_argument("--learning_rate",default=1e-5, type=float)
    parser.add_argument("--max_seq_len",default=128, type=int)
    parser.add_argument("--num_classes",default=2, type=int)
    parser.add_argument("--seed",default=42, type=int)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--output_dir", default="", type=str, help="Path where model needs to be saved.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--custom_model",default="gcn-1lyr-2proj",type=str)
    parser.add_argument("--window_size",default=5,type=int,help="0-Any pairs of nodes, n-edges between nodes no more than n distances apart")
    parser.add_argument("--do_oversample_data",action="store_true",help="Inflate Training positive labels")
    parser.add_argument("--use_pretrained_weights",action="store_true",help="Whether to load pre-trained weights")
    parser.add_argument("--gnn_layer1_out_dim", default=128, type=int, help="The dimension of Graph Neural Network Layer 1 output dimension")
    parser.add_argument("--gnn_layer2_out_dim", default=64, type=int, help="The dimension of Graph Neural Network Layer 2 output dimension")
    parser.add_argument("--dropout", default=0.4, type=float, help="Dropout for each layers.")
    parser.add_argument("--graph_connection_type", default="complete", type=str, help="Type of graph connection you need to learn features.")
    
    
        
    args = parser.parse_args()
    
    args.n_gpu = torch.cuda.device_count()
    
    set_seed(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    
    
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model, num_labels=args.num_classes)
    
    
    data_train, data_val, data_test, edge_percent = prepare_data(args.datafile, args.domain, args.max_seq_len, tokenizer, args.window_size, args.do_oversample_data, args.graph_connection_type, args.model)
    args.edge_percent = edge_percent
    
    
    model = ""
    if args.custom_model == "bertgcn-2lyr":
        #BertModel -> Bert_0-11 layers -> BertPooler -> GCNConv1 -> GCNConv2 ->Projection -> CrossEntropy
        model = BertBiLayerGCN(config, args).to(device)
    elif args.custom_model == "bertgcn-1lyr":
        #BertModel -> Bert_0-11 layers -> BertPooler -> GCNConv1 -> Projection -> CrossEntropy
        model = BertUniLayerGCN(config, args).to(device)
    elif args.custom_model == "lm-projection":
        #BertModel -> Bert_0-11 layers -> BertPooler -> Projection -> CrossEntropy
        model = LMProjection(config, args).to(device)
    elif args.custom_model == "gcn-1lyr-2proj":
        model = GCNUniLayerBiProj(config, args).to(device)
    elif args.custom_model == "gcn-2lyr-2proj":
        model = GCNBiLayerBiProj(config, args).to(device)
    elif args.custom_model == "gat-1lyr-2proj":
        model = GATUniLayerBiProj(config, args).to(device)
    elif args.custom_model == "gat-2lyr-2proj":
        model = GATBiLayerBiProj(config, args).to(device)
    elif args.custom_model == "gcn-1lyr-2proj-fullseq":
        model = GCNUniLayerBiProjFullSeq(config, args).to(device)
    elif args.custom_model == "random-baseline":
        model = RandomBaseline()
        preds, ylabels = model.get_baseline(data_test)
        out_metric_test = accuracy_metric(preds, ylabels, None, True)
        logger.info("Random Baseline :\nTest Performance:  Accuracy: %f, F1: %f, P: %f, R: %f, ROCAUC: %f, PRAUC: %f, Report: %s", 
                    out_metric_test['accuracy'],
                    out_metric_test['f1'], 
                    out_metric_test['precision'],
                    out_metric_test['recall'],
                    out_metric_test['roc_auc_score'],
                    out_metric_test['pr_auc'],
                    out_metric_test['classification_report']
                   )
        exit(0)
    elif args.custom_model == "wtd-random-baseline":
        model = WtdRandomBaseline(args)
        preds, ylabels = model.get_baseline(data_test)
        out_metric_test = accuracy_metric(preds, ylabels, None, True)
        logger.info("Wtd Random Baseline :\nTest Performance:  Accuracy: %f, F1: %f, P: %f, R: %f, ROCAUC: %f, PRAUC: %f, Report: %s", 
                    out_metric_test['accuracy'],
                    out_metric_test['f1'], 
                    out_metric_test['precision'],
                    out_metric_test['recall'],
                    out_metric_test['roc_auc_score'],
                    out_metric_test['pr_auc'],
                    out_metric_test['classification_report']
                   )
        exit(0)
        
        
    else:
        logger.info("Invalid Custom Model name")
        exit(0)
    
    model = DataParallel(model)
    # DO NOT USE SHUFFLE (not handling the batching of edges and gold labels when shuffled)
    # Commented out for multi-gpu support
#     train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=False) 
#     val_loader = DataLoader(data_val, batch_size=args.eval_batch_size, shuffle=False)
#     test_loader = DataLoader(data_test, batch_size=args.eval_batch_size, shuffle=False)

    train_loader = DataListLoader(data_train, batch_size=args.batch_size, shuffle=False) 
    val_loader = DataListLoader(data_val, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataListLoader(data_test, batch_size=args.eval_batch_size, shuffle=False)

    
    # Prepare optimizer and schedule (decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [ { "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                                      "weight_decay": args.weight_decay,},
                                    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
                                    ]
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    print("num_training_steps:",len(train_loader)*args.num_epochs)
    # Scheduler causing the loss to hover around same value
    scheduler = get_linear_schedule_with_warmup( optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=len(train_loader)*args.num_epochs)
    
    
            
    
    # Start Training 
    
    if args.do_train:

        best_loss = 1000
        best_epoch = -1
        best_accuracy = 0
        best_f1 = 0
        best_prauc = 0

        model.zero_grad()
        for epoch in tqdm(range(args.num_epochs)):
            global_loss = 0
            epoch_preds = []
            epoch_ylabels = []
            epoch_pos_prob = []
            global_step = 0
            for step, batch in enumerate(train_loader):

                model.train()
                batch_data = batch
                optimizer.zero_grad()

                loss, probs, ylabels = model(batch_data) # This values is a aggregated n losses,probs,ylabels from n gpus aggregated in GPU0
    
                if args.n_gpu > 1:
                    loss = loss.mean()
                
                preds = probs.argmax(dim=1)
                
                pos_prob = probs[:,1]
                global_loss += loss.item()
                global_step += 1

                # Calculate predictions
                if epoch_preds is None:
                    epoch_preds = preds.detach().cpu().numpy()
                    epoch_ylabels = ylabels.detach().cpu().numpy()
                    epoch_pos_prob = pos_prob.detach().cpu().numpy()
                else:
                    epoch_preds = np.append(epoch_preds, preds.detach().cpu().numpy(), axis=0)
                    epoch_ylabels = np.append(epoch_ylabels, ylabels.detach().cpu().numpy(), axis=0)
                    epoch_pos_prob = np.append(epoch_pos_prob, pos_prob.detach().cpu().numpy(), axis=0)

                # Back Propagate
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()


            out_metric_train = accuracy_metric(epoch_preds, epoch_ylabels, epoch_pos_prob)
            out_metric_train['loss']= global_loss/global_step

            ### Evaluate on val data
            out_metric_val = evaluate(model, val_loader, args, "val")

            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
                
            logger.info("Training Epoch: %d, Loss: %f, Accuracy: %f, F1: %f, PRAUC: %f, LR: %f",epoch, out_metric_train['loss'], out_metric_train['accuracy'], out_metric_train['f1'], out_metric_train['pr_auc'], current_lr)
            logger.info("Validation Epoch: %d, Loss: %f, Accuracy: %f, F1: %f, PRAUC: %f,",epoch, out_metric_val['loss'], out_metric_val['accuracy'],out_metric_val['f1'], out_metric_val['pr_auc'])

            if out_metric_val['pr_auc'] > best_prauc:
                best_prauc = out_metric_val['pr_auc']
                val_loss = out_metric_val['loss']

                if not os.path.exists(os.path.join(args.output_dir,"PRAUC")):
                    os.makedirs(os.path.join(args.output_dir,"PRAUC"))


                logger.info("Best Val Performance(P): %d, Loss: %f, PRAUC: %f, F1: %f, P: %f, R: %f, ROCAUC: %f, PRAUC: %f",
                            epoch, 
                            val_loss, 
                            best_prauc,
                            out_metric_val['f1'], 
                            out_metric_val['precision'], 
                            out_metric_val['recall'],
                            out_metric_val['roc_auc_score'],
                            out_metric_val['pr_auc'])
                model_to_save = os.path.join(args.output_dir,"PRAUC", "best_model.ckpt")
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': out_metric_val['accuracy'],
                'f1':out_metric_val['f1'],
                'precision':out_metric_val['precision'],
                'recall':out_metric_val['recall'],
                'roc_auc_score':out_metric_val['roc_auc_score'],
                'pr_auc':out_metric_val['pr_auc']
                }, model_to_save)
                logger.info("Model Saved to : %s",os.path.join(args.output_dir,"PRAUC"))
                
                
                
            if out_metric_val['f1'] > best_f1:
                best_f1 = out_metric_val['f1']
                val_loss = out_metric_val['loss']
                

                if not os.path.exists(os.path.join(args.output_dir,"F1")):
                    os.makedirs(os.path.join(args.output_dir,"F1"))


                logger.info("Best Val Performance(F1): %d, Loss: %f, Accuracy: %f, F1: %f, P: %f, R: %f, ROCAUC: %f, PRAUC: %f",
                            epoch, 
                            val_loss, 
                            out_metric_val['accuracy'],
                            out_metric_val['f1'], 
                            out_metric_val['precision'], 
                            out_metric_val['recall'],
                            out_metric_val['roc_auc_score'],
                            out_metric_val['pr_auc'])
                model_to_save = os.path.join(args.output_dir,"F1", "best_model.ckpt")
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': out_metric_val['accuracy'],
                'f1':out_metric_val['f1'],
                'precision':out_metric_val['precision'],
                'recall':out_metric_val['recall'],
                'roc_auc_score':out_metric_val['roc_auc_score'],
                'pr_auc':out_metric_val['pr_auc']
                }, model_to_save)
                logger.info("Model Saved to : %s",os.path.join(args.output_dir,"F1"))

                
            if out_metric_val['loss'] < best_loss:
                best_loss = out_metric_val['loss']
                
                if not os.path.exists(os.path.join(args.output_dir,"LOSS")):
                    os.makedirs(os.path.join(args.output_dir,"LOSS"))


                logger.info("Best Val Performance(LOSS): %d, Loss: %f, Accuracy: %f, F1: %f, P: %f, R: %f, ROCAUC: %f, PRAUC: %f",
                            epoch, 
                            out_metric_val["loss"], 
                            out_metric_val['accuracy'],
                            out_metric_val['f1'], 
                            out_metric_val['precision'], 
                            out_metric_val['recall'],
                            out_metric_val['roc_auc_score'],
                            out_metric_val['pr_auc'])
                model_to_save = os.path.join(args.output_dir,"LOSS", "best_model.ckpt")
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': out_metric_val['loss'],
                'accuracy': out_metric_val['accuracy'],
                'f1':out_metric_val['f1'],
                'precision':out_metric_val['precision'],
                'recall':out_metric_val['recall'],
                'roc_auc_score':out_metric_val['roc_auc_score'],
                'pr_auc':out_metric_val['pr_auc']
                }, model_to_save)
                logger.info("Model Saved to : %s",os.path.join(args.output_dir,"LOSS"))


    if args.do_eval:
        

        ## Load a BEST trained model on ACCURACY

        model_to_load = os.path.join(args.output_dir,"PRAUC", "best_model.ckpt")
        checkpoint = torch.load(model_to_load)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        accuracy = checkpoint['accuracy']
        f1 = checkpoint['f1']
        p=checkpoint['precision']
        r=checkpoint['recall']
        rocauc = checkpoint['roc_auc_score']
        prauc = checkpoint['pr_auc']
        logger.info("Model loaded with Val performance(P) : epoch: %d, loss: %f, accuracy: %f, f1: %f, P: %f, R: %f, ROCAUC: %f, PRAUC: %f", epoch, loss, accuracy, f1, p, r, rocauc, prauc)
        
        ### Evaluate on val data
        out_metric_test = evaluate(model, test_loader, args, "test")


        logger.info("Test Performance(P) Loss: %f, Accuracy: %f, F1: %f, P: %f, R: %f, ROCAUC: %f, PRAUC: %f, Report: %s", 
                    out_metric_test['loss'],
                    out_metric_test['accuracy'],
                    out_metric_test['f1'], 
                    out_metric_test['precision'],
                    out_metric_test['recall'],
                    out_metric_test['roc_auc_score'],
                    out_metric_test['pr_auc'],
                    out_metric_test['classification_report']
                   )

        

        ## Load a BEST trained model on F1

        model_to_load = os.path.join(args.output_dir,"F1", "best_model.ckpt")
        checkpoint = torch.load(model_to_load)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        accuracy = checkpoint['accuracy']
        f1 = checkpoint['f1']
        p=checkpoint['precision']
        r=checkpoint['recall']
        rocauc = checkpoint['roc_auc_score']
        prauc = checkpoint['pr_auc']
        logger.info("Model loaded with Val performance(F1) : epoch: %d, loss: %f, accuracy: %f, f1: %f, P: %f, R: %f, ROCAUC: %f, PRAUC: %f", epoch, loss, accuracy, f1, p, r, rocauc, prauc)
        
        ### Evaluate on val data
        out_metric_test = evaluate(model, test_loader, args, "test")


        logger.info("Test Performance(F1) Loss: %f, Accuracy: %f, F1: %f, P: %f, R: %f, ROCAUC: %f, PRAUC: %f, Report: %s", 
                    out_metric_test['loss'],
                    out_metric_test['accuracy'],
                    out_metric_test['f1'], 
                    out_metric_test['precision'],
                    out_metric_test['recall'],
                    out_metric_test['roc_auc_score'],
                    out_metric_test['pr_auc'],
                    out_metric_test['classification_report']
                   )

        

        ## Load a BEST trained model on LOSS

        model_to_load = os.path.join(args.output_dir,"LOSS", "best_model.ckpt")
        checkpoint = torch.load(model_to_load)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        accuracy = checkpoint['accuracy']
        f1 = checkpoint['f1']
        p=checkpoint['precision']
        r=checkpoint['recall']
        rocauc = checkpoint['roc_auc_score']
        prauc = checkpoint['pr_auc']
        logger.info("Model loaded with Val performance(LOSS) : epoch: %d, loss: %f, accuracy: %f, f1: %f, P: %f, R: %f, ROCAUC: %f, PRAUC: %f", epoch, loss, accuracy, f1, p, r, rocauc, prauc)
        
        ### Evaluate on val data
        out_metric_test = evaluate(model, test_loader, args, "test")


        logger.info("Test Performance(LOSS) Loss: %f, Accuracy: %f, F1: %f, P: %f, R: %f, ROCAUC: %f, PRAUC: %f, Report: %s", 
                    out_metric_test['loss'],
                    out_metric_test['accuracy'],
                    out_metric_test['f1'], 
                    out_metric_test['precision'],
                    out_metric_test['recall'],
                    out_metric_test['roc_auc_score'],
                    out_metric_test['pr_auc'],
                    out_metric_test['classification_report']
                    
                   )
