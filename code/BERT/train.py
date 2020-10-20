"""
this model has these functions:
1. cluster the training sentences for better mix
2. mix in arbitury layer
3. semi supervised learning 
"""

import argparse
import glob
import logging as log
import os
import random
import time
import torch.nn.functional as F

import numpy as np
import torch
from eval_utils import f1_score, precision_score, recall_score, classification_report, macro_score
from utils import gen_knn_mix_batch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import pickle
from transformers import *

from read_data import *

from tensorboardX import SummaryWriter

from bert_models import BertModel4Mix

logger = log.getLogger(__name__)

MODEL_CLASSES = {"bert": (BertConfig, BertForTokenClassification, BertTokenizer)}

parser = argparse.ArgumentParser(description='PyTorch BaseNER')
parser.add_argument("--data-dir", default = './data', type = str, required = True)
parser.add_argument("--model-type", default = 'bert', type = str)
parser.add_argument("--model-name", default = 'bert-base-multilingual-cased', type = str)
parser.add_argument("--output-dir", default = './german_eval', type = str)

parser.add_argument('--gpu', default='0,1,2,3', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--train-examples', default = -1, type = int)

parser.add_argument("--labels", default = "", type = str)
parser.add_argument('--config-name', default = '', type = str)
parser.add_argument("--tokenizer-name", default = '', type = str)
parser.add_argument("--max-seq-length", default = 128, type = int)

parser.add_argument("--do-train", action="store_true", help="Whether to run training.")
parser.add_argument("--do-eval", action="store_true", help="Whether to run eval on the dev set.")
parser.add_argument("--do-predict", action="store_true", help="Whether to run predictions on the test set.")
parser.add_argument("--evaluate-during-training", action="store_true", help="Whether to run evaluation during training at each logging step.")
parser.add_argument("--do-lower-case", action="store_true", help="Set this flag if you are using an uncased model.")

parser.add_argument("--batch-size", default = 16, type = int)
parser.add_argument('--eval-batch-size', default = 128, type = int)

parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")

parser.add_argument("--learning-rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight-decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam-epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

parser.add_argument("--num-train-epochs", default=20, type=float, help="Total number of training epochs to perform.")
parser.add_argument("--max-steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument('--warmup-steps', default = 0, type = int,  help="Linear warmup over warmup_steps.")

parser.add_argument('--logging-steps', default = 150, type = int, help="Log every X updates steps.")
parser.add_argument("--save-steps", type=int, default=0, help="Save checkpoint every X updates steps.")
parser.add_argument("--eval-all-checkpoints", action="store_true", help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
parser.add_argument("--overwrite-output-dir", action="store_true", help="Overwrite the content of the output directory")

parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

parser.add_argument("--pad-subtoken-with-real-label", action="store_true", help="give real label to the padded token instead of `-100` ")
parser.add_argument("--subtoken-label-type",default='real', type=str,help="[real|repeat|O] three ways to do pad subtoken with real label. [real] give the subtoken a real label e.g., B -> B I. [repeat] simply repeat the label e.g., B -> B B. [O] give it a O label. B -> B O")


parser.add_argument("--eval-pad-subtoken-with-first-subtoken-only", action="store_true", help="only works when --pad-subtoken-with-real-label is true, in this mode, we only test the prediction of the first subtoken of each word (if the word could be tokenized into multiple subtoken)")
parser.add_argument("--label-sep-cls", action="store_true", help="label [SEP] [CLS] with special labels, but not [PAD]") 

# inter mix 
parser.add_argument('--mix-option', action='store_true',help='mix option')
parser.add_argument('--mix-layers-set', nargs='+', default = [6,9,12], type=int)  
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--beta', default=-1, type=float)
# knn
parser.add_argument("--use-knn-train-data", action="store_true", help="indicates we will use the knn training data")
parser.add_argument('--num-knn-k', default=30, type=int, help='top k we use in the knn')
parser.add_argument('--knn-mix-ratio', default=1, type=float, help='the ratio of the mix sample and the main sample are from the knn, otherwise we randomly sample one to mix')


# semi
parser.add_argument("--u-batch-size", default = 64, type = int)
parser.add_argument('--semi', action='store_true')
parser.add_argument('--T', type = float, default = 1.0,help='sharpen temperature')
parser.add_argument('--sharp', action='store_true')
parser.add_argument('--weight', type = float, default = 1.0)
parser.add_argument("--semi-pkl-file", default = "de_2.pkl", type = str,help='the paired back translated sentences pkl file') 
parser.add_argument("--semi-num", default = 10000, type = int,help='the number of unlabeled examples we use for semi-supervised learning')
parser.add_argument("--ignore-last-n-label", default = 4, type = int, help='the number of last labels we ignore for semi loss') # 4 means cls, pad, sep, O
parser.add_argument("--semi-loss", default = "mse", type = str,help="mse or kld")
parser.add_argument("--num-semi-iter", default = 1, type = int,help="number of semi train numbers in one loop")
parser.add_argument('--warmup-semi', action='store_true',help="warmup semi training")
parser.add_argument("--log-file", default = "results.csv", type = str,help="the file to store resutls")

# [semi loss change] we can also change the loss in current semi train. just force the augmented to approximate original. Instead of mixing them together. 
parser.add_argument("--semi-loss-method", default = "origin", type = str,help="[mix|origin] `mix` is the default one we are using. which set the mean of origin and augmented as the target distribution. `origin` is the one I used before in la-dtl. which means we set the original unlabeled data's prediction as the target and force the augmented data's label distribution to follow the target")
parser.add_argument("--optimizer", default = "adam", type = str,help='optimizer')
parser.add_argument('--special-label-weight', default=0, type=float, help='the special_label_weight in training . default 0')



# intra-mix

parser.add_argument('--intra-mix-ratio', default=-1, type=float, help='the ratio that we do intra mix')
parser.add_argument('--intra-mix-subtoken', action="store_true", help='we mix subtoken ')





args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
args.n_gpu = torch.cuda.device_count()
print("gpu num: ", args.n_gpu)

best_f1 = 0

print('perform mix: ', args.mix_option)
print("mix layers sets: ", args.mix_layers_set)


def set_seed(args):
    logger.info("random seed %s", args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if len(args.gpu) > 0:
        torch.cuda.manual_seed_all(args.seed)

def linear_rampup(current, rampup_length=args.num_train_epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

def train(args, train_dataset, eval_dataset, test_dataset, model, tokenizer, labels, pad_token_label_id,unlabeled_dataset=None):
    global best_f1
    tb_writer = SummaryWriter()
    print('tb_writer.logdir',tb_writer.logdir)
    
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
    labeled_dataloader = train_dataloader
    if args.semi:
        unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size = args.u_batch_size, shuffle = True)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]


    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.optimizer=='adam':
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    elif args.optimizer=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )    
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Total train batch size (w. parallel, accumulation) = %d",
        args.batch_size
        * args.gradient_accumulation_steps),
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    tr_semi_loss, logging_semi_loss = 0.0, 0.0
    
    #eval_f1 = []
    test_f1 = []
    model.zero_grad()

    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc='Epoch')
    set_seed(args)

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")        
        for step, batch in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            if args.semi:
                unlabeled_train_iter = iter(unlabeled_dataloader)
                for semi_iter in range(args.num_semi_iter):
                    try:
                        unlabeled_batch = unlabeled_train_iter.next()
                    except:
                        unlabeled_train_iter = iter(unlabeled_dataloader)
                        unlabeled_batch = unlabeled_train_iter.next()            
                    unlabeled_batch = tuple(t.to(args.device) for t in unlabeled_batch)   
                    with torch.no_grad():

                        inputs1 = {"input_ids": unlabeled_batch[0], "attention_mask": unlabeled_batch[1], "target_a": unlabeled_batch[3]}
                        inputs1["token_type_ids"] = unlabeled_batch[2]

                        # pass the unlabeled data through to get the logits
                        logits_u = model(**inputs1,special_label_weight=args.special_label_weight)[0]
                        # mask out cls,sep,pad,(subtoken) 
                        logits_u = torch.softmax(logits_u, dim = -1)
                        if args.sharp:
                            pt = logits_u**(1/args.T)
                            logits_u = pt / pt.sum(dim=-1, keepdim=True)                            
                        mask_u = torch.zeros(logits_u.shape).cuda()
                        idx = torch.where(unlabeled_batch[3] != 1)
                        mask_u[idx] = torch.tensor([1.0] *  logits_u.shape[-1]).cuda() 
                        logits_u = ((logits_u * mask_u).sum(dim = 1))[:,0:-args.ignore_last_n_label]
                        


                        if args.semi_loss_method=='mix':
                            inputs2 = {"input_ids": unlabeled_batch[4], "attention_mask": unlabeled_batch[5], "target_a": unlabeled_batch[7]}
                            inputs2["token_type_ids"] = unlabeled_batch[6]                            
                            logits_u2 = model(**inputs2,special_label_weight=args.special_label_weight)[0]
                            logits_u2 = torch.softmax(logits_u2, dim = -1)
                            if args.sharp: 
                                pt = logits_u2**(1/args.T)
                                logits_u2 = pt / pt.sum(dim=-1, keepdim=True)

                            mask_u2 = torch.zeros(logits_u2.shape).cuda()
                            idx = torch.where(unlabeled_batch[7] != 1)
                            mask_u2[idx] = torch.tensor([1.0] * logits_u2.shape[-1]).cuda()
                            logits_u2 = ((logits_u2 * mask_u2).sum(dim = 1))[:,0:-args.ignore_last_n_label] 
                            inputs2 = {"input_ids": unlabeled_batch[4], "attention_mask": unlabeled_batch[5], "target_a": unlabeled_batch[7]}
                            inputs2["token_type_ids"] = unlabeled_batch[6]


                            p = (logits_u+logits_u2)/2
                        elif args.semi_loss_method=='origin':
                            p = logits_u
                        targets_u = p.detach()                

                    # unlabeled
                    inputs2 = {"input_ids": unlabeled_batch[4], "attention_mask": unlabeled_batch[5], "target_a": unlabeled_batch[7]}
                    inputs2["token_type_ids"] = unlabeled_batch[6]
                    logits_u2 = model(**inputs2,special_label_weight=args.special_label_weight)[0]
                    logits_u2 = torch.softmax(logits_u2, dim = -1)
                    mask_u2 = torch.zeros(logits_u2.shape).cuda()
                    idx = torch.where(unlabeled_batch[7] != 1)
                    mask_u2[idx] = torch.tensor([1.0] *  logits_u2.shape[-1]).cuda()
                    logits_u2 = ((logits_u2 * mask_u2).sum(dim = 1))[:,0:-args.ignore_last_n_label] 

                    if args.semi_loss_method in ['mix','origin']:                        
                        inputs1 = {"input_ids": unlabeled_batch[0], "attention_mask": unlabeled_batch[1], "target_a": unlabeled_batch[3]}
                        inputs1["token_type_ids"] = unlabeled_batch[2]
                        logits_u = model(**inputs1,special_label_weight=args.special_label_weight)[0]
                        logits_u = torch.softmax(logits_u, dim = -1)
                        mask_u = torch.zeros(logits_u.shape).cuda()
                        idx = torch.where(unlabeled_batch[3] != 1)
                        mask_u[idx] = torch.tensor([1.0] *  logits_u.shape[-1]).cuda()
                        logits_u = ((logits_u * mask_u).sum(dim = 1))[:,0:-args.ignore_last_n_label] 
                        logits = torch.cat([logits_u, logits_u2],dim = 0)
                        targets = torch.cat([targets_u, targets_u], dim = 0)
                        
                        
                    if args.semi_loss=='mse':
                        Lu = F.mse_loss(logits, targets)
                    elif args.semi_loss=='kld':
                        Lu = F.kl_div(logits, targets, None, None, 'batchmean')
                    if args.n_gpu >= 1:
                        Lu = Lu.mean()                            
                    if args.warmup_semi:
                        loss_u = args.weight * linear_rampup(epoch + step/len(epoch_iterator)) * Lu                        
                    else:
                        loss_u = args.weight * Lu                        
                    if args.gradient_accumulation_steps > 1:
                        loss_u = loss_u / args.gradient_accumulation_steps

                tr_semi_loss+= loss_u.item()
                
            batch = tuple(t.to(args.device) for t in batch)
            inputs_a = {"input_ids": batch[0],"attention_mask": batch[1],'subtoken_ids':batch[4]}
            target_a=batch[3]
            # set inputs A and inputs B

            if args.use_knn_train_data:
                batch_b = gen_knn_mix_batch(batch=batch,train_dataset=train_dataset,sent_id_knn_array=args.sent_id_knn_array,
                                            knn_mix_ratio=args.knn_mix_ratio,
                                            train_size=args.train_examples)
                assert len(batch_b)==len(batch)
                inputs_b = {"input_ids": batch_b[0],"attention_mask": batch_b[1]}
                target_b=batch_b[3] 
            else:
                idx=torch.randperm(batch[0].size(0))
                inputs_b = {"input_ids": batch[0][idx],"attention_mask": batch[1][idx]}
                target_b=batch[3][idx]

            if (args.alpha-0)<1e-6:
                l=1
            else:
                if args.beta==-1:
                    l = np.random.beta(args.alpha, args.alpha)
                else:
                    l = np.random.beta(args.alpha, args.beta)
                l = max(l, 1-l)
            mix_layer = np.random.choice(args.mix_layers_set, 1)[0]
            mix_layer = mix_layer -1 
                        
        
            
            if args.mix_option:

                # two kinds of mix. the primary one is the inter mix, the secondary one is the intra mix
                if random.uniform(0, 1) <= args.intra_mix_ratio:
                    # do intra-mix
                    # mix the attention mask to be the longer one. 
                    outputs,loss = model(inputs_a['input_ids'],target_a,l=l,
                                        attention_mask = inputs_a["attention_mask"],
                                        special_label_weight=args.special_label_weight,
                                        subtoken_ids=inputs_a['subtoken_ids'],
                                        do_intra_mix=True,
                                        intra_mix_subtoken=args.intra_mix_subtoken)
                                                         
                else:
                    # do inter-mix
                    inputs_b['input_ids'] = inputs_b['input_ids'].to(args.device)
                    inputs_b["attention_mask"] = inputs_b["attention_mask"].to(args.device)
                    target_b = target_b.to(args.device)  
                    # mix the attention mask to be the longer one. 
                    attention_mask = ((inputs_a["attention_mask"]+inputs_b["attention_mask"])>0).type(torch.long)
                    attention_mask = attention_mask.to(args.device)
                    outputs,loss = model(inputs_a['input_ids'],target_a,inputs_b['input_ids'],
                                        target_b,l, mix_layer,
                                        attention_mask = attention_mask,
                                        special_label_weight=args.special_label_weight,
                                        subtoken_ids=None,
                                        do_intra_mix=False)
                                        
            else:
                # no mix
                outputs,loss = model(inputs_a['input_ids'],target_a,
                                     attention_mask = inputs_a["attention_mask"],
                                     special_label_weight=args.special_label_weight,
                                     subtoken_ids=None,
                                     do_intra_mix=False)
                
            if args.n_gpu >= 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.semi:
                loss=loss+loss_u 
            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (args.evaluate_during_training):
                        
                        results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, eval_dataset, parallel = False, mode="dev", prefix = str(global_step))
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        
                        if results['f1'] >= best_f1:
                            best_f1 = results['f1']
                            results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, test_dataset, parallel = False, mode="test", prefix = str(global_step))
                            test_f1.append(results['f1'])
                            

                            output_dir = os.path.join(args.output_dir, "best")
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            logger.info("Saving best model to %s", output_dir)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model)  
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
                    tb_writer.add_scalar("semi_loss", (tr_semi_loss - logging_semi_loss) / args.logging_steps, global_step)
                    logger.info("logging train info!!!")
                    logger.info("*")
                    logging_semi_loss = tr_semi_loss


                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (model.module if hasattr(model, "module") else model)  
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
            
        # eval and save the best model based on dev set after each epoch
        if (args.evaluate_during_training):
            
            results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, eval_dataset, parallel = False, mode="dev", prefix = str(global_step))
            for key, value in results.items():
                tb_writer.add_scalar("eval_{}".format(key), value, global_step)

            
            if results['f1'] >= best_f1:
                best_f1 = results['f1']
                results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, test_dataset, parallel = False, mode="test", prefix = str(global_step))
                test_f1.append(results['f1'])
                
                output_dir = os.path.join(args.output_dir, "best")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                logger.info("Saving best model to %s", output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model)  
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    args.tb_writer_logdir=tb_writer.logdir
    tb_writer.close()
    return global_step, tr_loss / global_step, test_f1 ,tr_semi_loss/global_step

def output_eval_results(out_label_list,preds_list,input_id_list,file_name):
    with open(file_name,'w') as fout:
        for i in range(len(out_label_list)):
            label=out_label_list[i]
            pred=preds_list[i]
            tokens=input_id_list[i]
            for j in range(len(label)):
                if tokens[j]=='[PAD]':
                    continue
                fout.write('{}\t{}\t{}\n'.format(tokens[j] ,label[j],pred[j]))
            fout.write('\n')


def evaluate(args, model, tokenizer, labels, pad_token_label_id,  eval_dataset = None, parallel = True, mode = 'dev', prefix = ''):
    if eval_dataset is None:
        eval_dataset = read_data(args, tokenizer, labels, pad_token_label_id, mode = mode,
                                 pad_subtoken_with_real_label=args.pad_subtoken_with_real_label)
    eval_dataloader = DataLoader(eval_dataset, batch_size = args.eval_batch_size, shuffle = False)

    if parallel:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****", mode + '-' + prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    all_subtoken_ids=None
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3],'sent_id' : batch[5]}
            inputs["token_type_ids"] = batch[2]
            target=inputs['labels']
            
            logits,tmp_eval_loss  = model(inputs['input_ids'],target,attention_mask = inputs["attention_mask"],
                                     special_label_weight=args.special_label_weight)
            
            

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            all_subtoken_ids=batch[4].detach().cpu().numpy()
            sent_id=inputs['sent_id'].detach().cpu().numpy()
            input_ids=inputs['input_ids'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            all_subtoken_ids = np.append(all_subtoken_ids, batch[4].detach().cpu().numpy(), axis=0)
            sent_id = np.append(sent_id, inputs['sent_id'].detach().cpu().numpy(), axis=0)
            input_ids= np.append(input_ids, inputs['input_ids'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    input_id_list = [[] for _ in range(input_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if args.pad_subtoken_with_real_label  or args.label_sep_cls:

                if args.eval_pad_subtoken_with_first_subtoken_only:
                    if all_subtoken_ids[i,j] ==1: 
                        out_label_list[i].append(label_map[out_label_ids[i][j]])
                        preds_list[i].append(label_map[preds[i][j]])
                        tid=input_ids[i][j]
                        input_id_list[i].append(tokenizer.convert_ids_to_tokens([tid])[0])


                else:
                    if all_subtoken_ids[i,j] in [0,1] and out_label_ids[i, j] != pad_token_label_id:# in this case, we consider all the tokens.
                        out_label_list[i].append(label_map[out_label_ids[i][j]])
                        preds_list[i].append(label_map[preds[i][j]])            
                        input_id_list[i].append(tokenizer.convert_ids_to_tokens([input_ids[i][j]]))    
            else:
                if all_subtoken_ids[i,j] in [0,1] and out_label_ids[i, j] != pad_token_label_id:# in this case, we consider all the tokens.
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])                
                    input_id_list[i].append(tokenizer.convert_ids_to_tokens([input_ids[i][j]]))
    file_name=os.path.join(args.output_dir,'{}_pred_results.tsv'.format(mode))
    output_eval_results(out_label_list,preds_list,input_id_list,file_name)

    macro_scores=macro_score(out_label_list, preds_list)
    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
        'macro_f1':macro_scores['macro_f1'],
        'macro_precision':macro_scores['macro_precision'],
        'macro_recall':macro_scores['macro_recall']
    }

    logger.info("***** Eval results %s *****", mode + '-' + prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list

def main():
    global best_f1
    if (os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir):
        raise ValueError( "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    
    logger.setLevel(log.INFO)
    formatter = log.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    
    if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
            
    fh = log.FileHandler(args.output_dir  +'/' + str(args.train_examples)+'-' + 'log.txt')
    fh.setLevel(log.INFO)
    fh.setFormatter(formatter)

    ch = log.StreamHandler()
    ch.setLevel(log.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    
    logger.info("------NEW RUN-----")

    logger.info("device: %s, n_gpu: %s", args.device, args.n_gpu)

    set_seed(args)

    labels = get_labels(args.labels)
    num_labels = len(labels)
    args.num_labels=num_labels

    pad_token_label_id = CrossEntropyLoss().ignore_index

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name,
        num_labels=num_labels,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name,
        do_lower_case=args.do_lower_case,
    )
        
    model_class = BertModel4Mix(config)

    model = model_class.from_pretrained(
        args.model_name,
        from_tf=bool(".ckpt" in args.model_name),
        config=config,
    )
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    
    

    if args.do_train:
        if args.semi:
            train_dataset, unlabeled_dataset = read_data(args, tokenizer, labels, 
                                                           pad_token_label_id, mode = 'train', train_examples = args.train_examples, 
                                                          pad_subtoken_with_real_label=args.pad_subtoken_with_real_label,
                                                           semi = args.semi)    
            

            
        ###################################################################################################################
        else:
            train_dataset = read_data(args, tokenizer, labels, pad_token_label_id, mode = 'train', train_examples = args.train_examples,
                                      pad_subtoken_with_real_label=args.pad_subtoken_with_real_label)
        if args.use_knn_train_data:
            file_path = os.path.join(args.data_dir, 'sent_id_knn_{}.pkl'.format(args.train_examples))
            I_array = pickle.load(open(file_path, 'rb'))
            args.sent_id_knn_array = I_array[:,1:args.num_knn_k+1]
            assert args.sent_id_knn_array.shape[1]==args.num_knn_k


        
        if args.evaluate_during_training:
            eval_dataset = read_data(args, tokenizer, labels, pad_token_label_id, mode = 'dev',
                                     pad_subtoken_with_real_label=args.pad_subtoken_with_real_label)

            

            test_dataset = read_data(args, tokenizer, labels, pad_token_label_id, mode = 'test',
                                     pad_subtoken_with_real_label=args.pad_subtoken_with_real_label)
            
        else:
            eval_dataset = None
            test_dataset = None
        if args.semi:
            global_step, tr_loss, test_f1, tr_semi_loss = train(args, train_dataset, eval_dataset, test_dataset, model, tokenizer, labels, pad_token_label_id,unlabeled_dataset)
        else:
            global_step, tr_loss, test_f1, tr_semi_loss = train(args, train_dataset, eval_dataset, test_dataset, model, tokenizer, labels, pad_token_label_id)
        print(test_f1)
        logger.info("test_f1", test_f1)
        
        logger.info(" global_step = %s, average loss = %s, average semi_loss = %s, best eval f1 = %s", global_step, tr_loss, tr_semi_loss,best_f1)
        
        
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    
    if args.do_eval:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        output_dir = os.path.join(args.output_dir, "best")
        model = model_class.from_pretrained(output_dir)
        model.to(args.device)
        result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", prefix = 'final')
        
        
    if args.do_predict:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        output_dir = os.path.join(args.output_dir, "best")
        model = model_class.from_pretrained(output_dir)
        model.to(args.device)
        result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test", prefix = 'final')



if __name__ == "__main__":
    main()

    