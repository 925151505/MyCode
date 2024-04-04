import logging
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel
from transformers.activations import gelu, gelu_new
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from Fusion_module import MAG
from loss import SupConLoss
from augmentation import *
from config import *
logger = logging.getLogger(__name__)

def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))
ACT2FN = {
    "gelu": gelu,
    "relu": torch.nn.functional.relu,
    "gelu_new": gelu_new,
    "mish": mish,
}


class MAG_BertModel(BertPreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.MAG = MAG(
            config.hidden_size,
            multimodal_config.beta_shift,
            multimodal_config.dropout_prob,
        )

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        mode,
        cat_feature=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        #Early fusion with
        fused_embedding = self.MAG(embedding_output, visual, acoustic)
        encoder_outputs = self.encoder(
            fused_embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        if mode == 'test':
            cls_feat = encoder_outputs[0][:, 0, :]
            pdist = nn.PairwiseDistance(p=1)#Calculate similarity function (p=2 Euclidean distance and p=1 Manhattan distance)
            for i in range(encoder_outputs[0].shape[0]):
                #Calculate category similarity results：
                #score = list(map(lambda x: x @ cls_feat[i], cat_feature))#Dot product(max)
                score = [pdist(torch.unsqueeze(x, 0), torch.unsqueeze(cls_feat[i], 0)) for x in cat_feature]#European and Manhattan(min)
                #score = [torch.cosine_similarity(x, cls_feat[i], dim=0) for x in cat_feature]#Cosine distance [-1, 1] 1 is most similar
                encoder_outputs[0][i, 0, :] = 0.9 * encoder_outputs[0][i, 0, :] + 0.1 * cat_feature[score.index(min(score))]


        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = sequence_output, pooled_output
        return outputs


class PMEE_BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = MAG_BertModel(config, multimodal_config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        self.contrastive = SupConLoss()

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        mode,
        cat_feature=None,
        label_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        outputs = self.bert(
            input_ids,
            visual,
            acoustic,
            mode,
            cat_feature,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        #Obtain the final output vector, then perform category comparison learning and induce prototype vectors
        ##################################
        pooled_output = outputs[1]
        if mode == 'train':
            new_cat_feature = aug(outputs[0], label_ids, mode='aug')  #There are 6 elements in the list, each with 768 dimensions
            feature = outputs[0]
            label_ids = label_ids.squeeze()
            cls_tokens, labelss = aug(feature, label_ids, mode='cons')
            cls_tokens = torch.unsqueeze(cls_tokens, 1)
            contrastive_loss = self.contrastive(cls_tokens, labelss)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)


        if mode == 'train':
            return logits, contrastive_loss, new_cat_feature
        else:
            return logits