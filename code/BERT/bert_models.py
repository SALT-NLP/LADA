"""
implement the mixBert model for NER here
"""
import torch
import torch.nn as nn
from pytorch_transformers import *
from transformers.modeling_bert import *
import numpy as np

import torch.nn.functional as F
def sample_id(index):
    
    i = np.random.choice(index, 1)
    
    return i
class BertModel4Mix(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel4Mix, self).__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder4Mix(config)

        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # todo
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()
    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings
    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    def forward(self, input_ids, 
                       target_a, 
                       input_ids2 = None,
                       target_b=None, 
                       l = None, 
                       mix_layer = 1000, 
                       attention_mask=None, 
                       token_type_ids=None, 
                       position_ids=None, 
                       head_mask=None,
                       special_label_weight=1,
                       subtoken_ids=None,
                       do_intra_mix=False,
                       intra_mix_subtoken=True):

        if attention_mask is None:
            if input_ids2 is not None:
                attention_mask2 = torch.ones_like(input_ids2)
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask2=attention_mask


        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            if input_ids2 is not None:
                token_type_ids2 = torch.zeros_like(input_ids2)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if input_ids2 is not None:

            extended_attention_mask2 = attention_mask2.unsqueeze(1).unsqueeze(2)

            extended_attention_mask2 = extended_attention_mask2.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_attention_mask2 = (1.0 - extended_attention_mask2) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) 
        else:
            head_mask = [None] * self.config.num_hidden_layers



        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        if input_ids2 is not None:
            embedding_output2 = self.embeddings(input_ids2, position_ids=position_ids, token_type_ids=token_type_ids2)

        if input_ids2 is not None:
            encoder_outputs = self.encoder(embedding_output, embedding_output2, l, mix_layer, extended_attention_mask, extended_attention_mask2 , head_mask=head_mask)
        else:
            encoder_outputs = self.encoder(embedding_output, attention_mask = extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]
    
        if do_intra_mix:
            subtoken_invalid_ids=[-1] # we mix first token and sub-token
            # compute the loss
            labels4train_a = torch.zeros([target_a.shape[0], target_a.shape[1], self.num_labels]).cuda()
            labels4train_b = torch.zeros([target_a.shape[0], target_a.shape[1], self.num_labels]).cuda()


            # logits
            for i in range(0, target_a.shape[0]):

                effective_ids=[]
                for k in range(len(subtoken_ids[i])):
                    if subtoken_ids[i][k] not in subtoken_invalid_ids:
                        effective_ids.append(k)


                for j in range(0, target_a.shape[1]):
                    if subtoken_ids[i][j] not in subtoken_invalid_ids: 

                        mix_index = sample_id(effective_ids)

                        
                        labels4train_a[i][j][target_a[i][j]] = 1
                        labels4train_b[i][j][target_a[i][mix_index]] = 1

                        labels4train_a[i][j] = l * labels4train_a[i][j] + (1-l) * labels4train_b[i][j]
                        sequence_output[i][j] = l * sequence_output[i][j] + (1-l) * sequence_output[i][mix_index]

            # loss
            sequence_output = self.dropout(sequence_output) 
            logits = self.classifier(sequence_output)

            loss_mask=attention_mask.view(-1, 1).repeat(1, self.num_labels).view(target_a.shape[0], target_a.shape[1],self.num_labels).type(torch.float).cuda()
            
            loss_mask[:,:,-3:]=loss_mask[:,:,-3:]*special_label_weight            
            loss = - torch.sum(F.log_softmax(logits, dim=2) * labels4train_a*loss_mask, dim=2)

            return logits,loss


        else:
            sequence_output = self.dropout(sequence_output) # dropout here
            # compute the loss
            labels4train_a = torch.zeros([target_a.shape[0], target_a.shape[1], self.num_labels]).cuda()
            labels4train_a.scatter_(2,target_a.unsqueeze(2),1)
                
                
            if input_ids2 is not None:
                labels4train_b = torch.zeros([target_a.shape[0], target_a.shape[1], self.num_labels]).cuda()
                labels4train_b.scatter_(2,target_b.unsqueeze(2),1)
                mixed_target = l * labels4train_a + (1 - l) * labels4train_b
                
            else:
                mixed_target =  labels4train_a 
            logits=self.classifier(sequence_output)
            
            loss_mask=attention_mask.view(-1, 1).repeat(1, self.num_labels).view(target_a.shape[0], target_a.shape[1],self.num_labels).type(torch.float).cuda()
            
            
            loss_mask[:,:,-3:]=loss_mask[:,:,-3:]*special_label_weight

            
            

            loss = - torch.sum(F.log_softmax(logits, dim=2) * mixed_target*loss_mask, dim=2)

            return logits,loss 



class BertEncoder4Mix(nn.Module):
    def __init__(self, config):
        super(BertEncoder4Mix, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self,
                hidden_states,
                hidden_states2 = None,
                l = None,
                mix_layer = 1000,
                attention_mask=None,
                attention_mask2=None,
                head_mask=None):

        all_hidden_states = ()
        all_attentions = ()

        if mix_layer == -1: # mix from the bottom
            if hidden_states2 is not None:
                hidden_states = l * hidden_states + (1-l)*hidden_states2

        for i, layer_module in enumerate(self.layer):
            if i <= mix_layer:

                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

                if hidden_states2 is not None:
                    layer_outputs2 = layer_module(hidden_states2, attention_mask2, head_mask[i])
                    hidden_states2 = layer_outputs2[0]
            
            if i == mix_layer:
                if hidden_states2 is not None:
                    hidden_states = l * hidden_states + (1-l)*hidden_states2

            if i > mix_layer:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)
        
        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)



class ClassificationBert(nn.Module):
    def __init__(self, 
                 hidden_size,
                 model_name,
                 num_labels=-1,
                 mix_option = False):
        super(ClassificationBert, self).__init__()
        

        if mix_option:
            self.bert = BertModel4Mix.from_pretrained(model_name)
        else:
            self.bert = BertModel.from_pretrained(model_name)    
        self.classifier = nn.Linear(hidden_size, num_labels)
        

    def forward(self, x, x2=None, l=None, mix_layer = 0):

        if x2 is not None:
            all_hidden,Lx = self.bert(x,x2,l, mix_layer)

        else:
            all_hidden,Lx = self.bert(x)
        logits=self.classifier(all_hidden[0])

        return logits

