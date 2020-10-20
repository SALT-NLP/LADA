import logging as log
import os

import numpy as np
from tqdm import tqdm, trange
import pickle
import torch
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler, TensorDataset)
from transformers import *

logger = log.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels,data_dir = None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.augmented_words = augment(guid, data_dir)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids,subtoken_ids,sent_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.subtoken_ids = subtoken_ids
        self.sent_id = sent_id

class UnlabeledFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, mask_ids, input_ids2, input_mask2, segment_ids2, mask_ids2 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.mask_ids = mask_ids

        self.input_ids2 = input_ids2
        self.input_mask2 = input_mask2
        self.segment_ids2 = segment_ids2
        self.mask_ids2 = mask_ids2
        
        
def augment(guid, para): 
    if para is not None:
        try:
            return para[guid].split(' ')
        except:
            return []
    else:
        return []

def read_examples_from_file_knn(data_dir, mode, train_examples):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 0
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(
                        guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(
                guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
    if mode == 'train':
        if train_examples < 0 or train_examples > len(examples):
            return examples
        else:
            return examples[:train_examples]
    return examples

def read_examples_from_file(data_dir, mode, train_examples, args):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 0
    examples = []
    
    if args.semi:
        file_path2 = os.path.join(data_dir, args.semi_pkl_file)
        with open(file_path2, 'rb') as f:
            para = pickle.load(f)
                
    para = para if args.semi else None
        
    
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(
                        guid="{}-{}".format(mode, guid_index), words=words, labels=labels, data_dir = para))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(
                guid="{}-{}".format(mode, guid_index), words=words, labels=labels, data_dir = para))
    if mode == 'train':
        if train_examples < 0 or train_examples > len(examples): 
            if args.semi:
                return (examples, examples[-args.semi_num:])
            else:
                return examples
        else:
            if args.semi:
                return (examples[:train_examples], examples[-args.semi_num:])
            else:
                return examples[:train_examples]
    return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, 
                cls_token='[CLS]', sep_token='[SEP]', pad_token=0, 
                cls_token_segment_id=0, sequence_a_segment_id=0, pad_token_segment_id=0,
                pad_token_label_id=-100, mask_padding_with_zero=True,
                                 omit_sep_cls_token=False,
                                 pad_subtoken_with_real_label=False,
                                subtoken_label_type='real',label_sep_cls=False):
    print('process training data',len(examples))
    

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    max_len=0
    

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))


        tokens = []
        label_ids = []
        subtoken_ids=[]
        # this subtoken_ids array is used to mark whether the token is a subtoken of a word or not
        for word, label in zip(example.words, example.labels):

            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            """
            'real': the defaul one, same as above:
                    O word: Oxx->OOO
                    I word: Ixx->III
                    B word: Bxx->BII
            'repeat': repeat the label:
                    O word: Oxx->OOO
                    I word: Ixx->III
                    B word: Bxx->BBB
            'O': change to O
                    O word: Oxx->OOO
                    I word: Ixx->IOO
                    B word: Bxx->BOO            
            """
            
            if len(word_tokens) > 0:
                if pad_subtoken_with_real_label:
                    if subtoken_label_type=='real':
                        if label[0]=='B':
                            pad_label='I'+label[1:]
                            label_ids.extend([label_map[label]] + [label_map[pad_label]] * (len(word_tokens) - 1))
                            subtoken_ids.extend([1] + [0] * (len(word_tokens) - 1))
                        else: # 'I' and 'O'
                            label_ids.extend([label_map[label]]*len(word_tokens))
                            subtoken_ids.extend([1] + [0] * (len(word_tokens) - 1))
                    elif subtoken_label_type=='repeat':
                        label_ids.extend([label_map[label]]*len(word_tokens))
                        subtoken_ids.extend([1] + [0] * (len(word_tokens) - 1))
                    elif subtoken_label_type=='O':
                        label_ids.extend([label_map[label]] + [label_map['O']] * (len(word_tokens) - 1))
                        subtoken_ids.extend([1] + [0] * (len(word_tokens) - 1))                        
                else:                    
                    label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                    subtoken_ids.extend([1] + [0] * (len(word_tokens) - 1))


        if len(tokens) > max_len:
            max_len=len(tokens)
            
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length -2)]
            label_ids = label_ids[: (max_seq_length -2)]
            subtoken_ids=subtoken_ids[:(max_seq_length -2)]
        
        if omit_sep_cls_token:
            segment_ids = [sequence_a_segment_id] * len(tokens)
            
        elif label_sep_cls:
            tokens += [sep_token]
            label_ids += [label_map['SEP']]
            segment_ids = [sequence_a_segment_id] * len(tokens)
            subtoken_ids+=[-1]

            tokens = [cls_token] + tokens
            label_ids = [label_map['CLS']] + label_ids
            segment_ids = [sequence_a_segment_id] + segment_ids
            subtoken_ids=[-1]+subtoken_ids

            
        else:
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)
            subtoken_ids+=[-1]

            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids
            subtoken_ids=[-1]+subtoken_ids
        
            
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        if label_sep_cls:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [label_map['PAD']] * padding_length  
            subtoken_ids+=[-1]*padding_length
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
            subtoken_ids+=[-1]*padding_length
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(subtoken_ids)==max_seq_length
        

        if ex_index < 2:
            print("******"*10)
            print("*** Example ***")
            print("guid: %s", example.guid)
            print("tokens: %s", " ".join([str(x) for x in tokens]))
            print("input_ids: %s", " ".join([str(x) for x in input_ids]))
            print("input_mask: %s", " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            print("label_ids: %s", " ".join([str(x) for x in label_ids]))
            print("subtoken_ids: %s", " ".join([str(x) for x in subtoken_ids]))
        try:
            sent_id = int(example.guid.split('-')[1])
            assert sent_id==ex_index,('sent_id',sent_id,'ex_index',ex_index,'example.guid',example.guid)
        except:
            print(example.guid)
            print(example.words)
            print(example.labels)
            
            

        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids, subtoken_ids=subtoken_ids,sent_id = sent_id)
        )
    print('=*'*40)
    print('max_len',max_len)
    return features

def convert_unlabeled_examples_to_features(examples,labels ,max_seq_length, tokenizer, 
                cls_token='[CLS]', sep_token='[SEP]', pad_token=0, 
                cls_token_segment_id=0, sequence_a_segment_id=0, pad_token_segment_id=0,
                pad_token_label_id=-100, mask_padding_with_zero=True,
                omit_sep_cls_token=False,
                pad_subtoken_with_real_label=False,label_sep_cls=False):

    features = []
    label_map = {label: i for i, label in enumerate(labels)}

    for (ex_index, example) in tqdm(enumerate(examples)):
        if ex_index % 5000 == 0:
            print("Writing example %d of %d", ex_index, len(examples))
            logger.info("Writing example %d of %d", ex_index, len(examples))


        tokens = []
        mask_ids = []
        
        for word in example.words:

            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            if len(word_tokens) > 0:
                mask_ids.extend([0] + [1] * (len(word_tokens) - 1)) 
            
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length -2)]
            mask_ids = mask_ids[: (max_seq_length -2)]
        
        
        tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        mask_ids += [1]
        

        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids
        
        mask_ids = [1] + mask_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)


        padding_length = max_seq_length - len(input_ids)
        if label_sep_cls:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length  
            segment_ids += [pad_token_segment_id] * padding_length
            mask_ids += [1] * padding_length
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            mask_ids += [1] * padding_length



        tokens2 = []
        mask_ids2 = []
        for word in example.augmented_words:

            word_tokens = tokenizer.tokenize(word)
            tokens2.extend(word_tokens)
            if len(word_tokens) > 0:
                mask_ids2.extend([0] + [1] * (len(word_tokens) - 1)) 

            
        if len(tokens) > max_seq_length - 2:
            tokens2 = tokens2[:(max_seq_length -2)]
            mask_ids2 = mask_ids2[: (max_seq_length -2)]
        
        tokens2 += [sep_token]
        mask_ids2 += [1]
        segment_ids2 = [sequence_a_segment_id] * len(tokens2)

        tokens2 = [cls_token] + tokens2
        mask_ids2 = [1] + mask_ids2
        segment_ids2 = [cls_token_segment_id] + segment_ids2

        input_ids2 = tokenizer.convert_tokens_to_ids(tokens2)

        input_mask2 = [1 if mask_padding_with_zero else 0] * len(input_ids2)

        padding_length = max_seq_length - len(input_ids2)

        
        input_ids2 += [pad_token] * padding_length
        input_mask2 += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids2 += [pad_token_segment_id] * padding_length
        mask_ids2 += [1] * padding_length

        
        if ex_index < 2:
            print("*** semi dataset ***")
            print("*** Example ***")
            print("guid: %s", example.guid)
            print("tokens: %s", " ".join([str(x) for x in tokens]))
            print("input_ids: %s", " ".join([str(x) for x in input_ids]))
            print("input_mask: %s", " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            print("mask_ids: %s", " ".join([str(x) for x in mask_ids]))        
            print("tokens2: %s", " ".join([str(x) for x in tokens2]))
            print("input_ids2: %s", " ".join([str(x) for x in input_ids2]))
            print("input_mask2: %s", " ".join([str(x) for x in input_mask2]))
            print("segment_ids2: %s", " ".join([str(x) for x in segment_ids2]))
            print("mask_ids2: %s", " ".join([str(x) for x in mask_ids2]))        

        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(mask_ids) == max_seq_length
        assert len(input_ids2) == max_seq_length,(len(input_ids2) , max_seq_length)
        assert len(input_mask2) == max_seq_length
        assert len(segment_ids2) == max_seq_length
        assert len(mask_ids2) == max_seq_length
        features.append(
            UnlabeledFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, mask_ids=mask_ids,
            input_ids2=input_ids2, input_mask2=input_mask2, segment_ids2=segment_ids2, mask_ids2=mask_ids2)
        )
    return features



def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def read_data(args, tokenizer, labels, pad_token_label_id, mode, train_examples = -1, 
              omit_sep_cls_token=False,
              pad_subtoken_with_real_label=False,
              semi = False):
    logger.info("Creating features from dataset file at %s", args.data_dir)
    examples = read_examples_from_file(args.data_dir, mode, train_examples,args)
    if not semi or mode is not 'train':
        #examples = examples[0]
        print(mode)
        print('data num: {}'.format(len(examples)))

        features = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer, 
                                    cls_token = tokenizer.cls_token, sep_token =  tokenizer.sep_token, pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                    cls_token_segment_id = 2 if args.model_type in ["xlnet"] else 0, 
                                    sequence_a_segment_id = 0, pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0, 
                                    pad_token_label_id = pad_token_label_id,
                                    omit_sep_cls_token=omit_sep_cls_token,
                                    pad_subtoken_with_real_label=pad_subtoken_with_real_label,
                                    subtoken_label_type=args.subtoken_label_type,
                                    label_sep_cls=args.label_sep_cls)
        
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_subtoken_ids = torch.tensor([f.subtoken_ids for f in features], dtype=torch.long)
        all_sent_id = torch.tensor([f.sent_id for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_subtoken_ids,all_sent_id)
        
        return dataset
    elif semi and mode == 'train':
        (labeled, unlabeled) = examples

        print(mode)
        print('labeled data num: {}'.format(len(labeled)))
        print('unlabeled data num: {}'.format(len(unlabeled)))
        
        
        
        features = convert_examples_to_features(labeled, labels, args.max_seq_length, tokenizer, 
                                    cls_token = tokenizer.cls_token, sep_token =  tokenizer.sep_token, pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                    cls_token_segment_id = 2 if args.model_type in ["xlnet"] else 0, 
                                    sequence_a_segment_id = 0, pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0, 
                                    pad_token_label_id = pad_token_label_id,
                                    omit_sep_cls_token=omit_sep_cls_token,
                                    pad_subtoken_with_real_label=pad_subtoken_with_real_label,
                                    subtoken_label_type=args.subtoken_label_type,
                                    label_sep_cls=args.label_sep_cls)

        unlabeled_features = convert_unlabeled_examples_to_features(unlabeled, labels, args.max_seq_length, tokenizer, 
                                    cls_token = tokenizer.cls_token, sep_token =  tokenizer.sep_token, pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                    cls_token_segment_id = 2 if args.model_type in ["xlnet"] else 0, 
                                    sequence_a_segment_id = 0, pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0, 
                                    pad_token_label_id = pad_token_label_id,
                                    omit_sep_cls_token=omit_sep_cls_token,
                                    pad_subtoken_with_real_label=pad_subtoken_with_real_label,
                                    label_sep_cls=args.label_sep_cls)

        
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_subtoken_ids = torch.tensor([f.subtoken_ids for f in features], dtype=torch.long)
        all_sent_id = torch.tensor([f.sent_id for f in features], dtype=torch.long)
        labeled_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_subtoken_ids,all_sent_id)
        all_input_ids = torch.tensor([f.input_ids for f in unlabeled_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in unlabeled_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in unlabeled_features], dtype=torch.long)
        all_mask_ids = torch.tensor([f.mask_ids for f in unlabeled_features], dtype=torch.long)
        all_input_ids2 = torch.tensor([f.input_ids2 for f in unlabeled_features], dtype=torch.long)
        all_input_mask2 = torch.tensor([f.input_mask2 for f in unlabeled_features], dtype=torch.long)
        all_segment_ids2 = torch.tensor([f.segment_ids2 for f in unlabeled_features], dtype=torch.long)
        all_mask_ids2 = torch.tensor([f.mask_ids2 for f in unlabeled_features], dtype=torch.long)

        unlabeled_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_mask_ids, all_input_ids2, all_input_mask2, all_segment_ids2, all_mask_ids2)
        return labeled_dataset, unlabeled_dataset