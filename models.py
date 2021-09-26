import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

from neural import TransformerInterEncoder, SSingleMatchNet, FuseNet
from utils import CHINSESE_MODEL_PATH


class SwagIntrospector(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = logits
        if labels is not None:
            labels = labels.type_as(logits)
            loss_fct = torch.nn.BCEWithLogitsLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1), labels.view(-1))
            outputs = (loss, logits)

        return outputs  # (loss), scores, (hidden_states), (attentions)


class BertSumIntrospector(BertPreTrainedModel):
    def __init__(self, bert_config, **kwargs):
        super(BertSumIntrospector, self).__init__(bert_config)
        self.bert_config = bert_config
        self.intro_config = kwargs["intro_config"]
        self.bert = BertModel(bert_config)
        self.encoder = TransformerInterEncoder(
            bert_config.hidden_size,
            kwargs["intro_config"].ff_size,
            kwargs["intro_config"].heads,
            kwargs["intro_config"].dropout,
            kwargs["intro_config"].inter_layers
        )
        self.criterion = torch.nn.BCELoss()
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            clses=None,
            mask_cls=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None
    ):
        # bert的输入为bs*seqlen
        # sequence_output:[bs,512,768]
        sequence_outputs = self.bert(input_ids, attention_mask, token_type_ids)[0]

        sents_vec = sequence_outputs[torch.arange(sequence_outputs.size(0)).unsqueeze(1), clses]
        print(sents_vec[:, :, 0:20])
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        # sent_scores:[bs, n_sent]
        logits = self.encoder(sents_vec, mask_cls)

        outputs = logits
        if labels is not None:
            if mask_cls is not None:
                active_loss = mask_cls.view(-1) == 1
                active_logits = logits.view(-1)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = self.criterion(active_logits, active_labels)
            else:
                loss = self.criterion(logits.view(-1), labels.view(-1))
            outputs = (loss, logits)
        return outputs

    def seperate_seq(self, sequence_outputs, question_lens, choice_lens, doc_lens):
        doc_seq_output = sequence_outputs.new(sequence_outputs.size()).zero_()
        ques_seq_output = sequence_outputs.new(sequence_outputs.size()).zero_()
        choice_seq_output = sequence_outputs.new(sequence_outputs.size()).zero_()
        for i in range(doc_lens.size(0)):
            ques_seq_output[i, :question_lens[i]] = sequence_outputs[i, :question_lens[i]]
            choice_seq_output[i, :choice_lens[i]] = \
                sequence_outputs[i, question_lens[i]:question_lens[i] + choice_lens[i]]
            doc_seq_output[i, :doc_lens[i]] = \
                sequence_outputs[i, question_lens[i] + choice_lens[i]:question_lens[i] + choice_lens[i] + doc_lens[i]]
        return ques_seq_output, choice_seq_output, doc_seq_output


class AttenSumIntrospector(BertPreTrainedModel):
    def __init__(self, bert_config, **kwargs):
        super(AttenSumIntrospector, self).__init__(bert_config)
        self.bert_config = bert_config
        self.intro_config = kwargs["intro_config"]
        self.bert = BertModel(bert_config)
        self.ssmatch = SSingleMatchNet(bert_config)
        self.fuse = FuseNet(bert_config)
        self.classifier2 = nn.Linear(2 * bert_config.hidden_size, bert_config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.encoder = TransformerInterEncoder(
            bert_config.hidden_size,
            kwargs["intro_config"].ff_size,
            kwargs["intro_config"].heads,
            kwargs["intro_config"].dropout,
            kwargs["intro_config"].inter_layers
        )
        self.criterion = torch.nn.BCELoss()
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            question_lens=None,
            choice_lens=None,
            doc_lens=None,
            clses=None,
            mask_cls=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None
    ):
        # bert的输入为bs*seqlen
        # (bs, 4, 512) -> (bs*4, 512)
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        question_lens = question_lens.view(-1, question_lens.size(0) * question_lens.size(1)).squeeze(0)
        choice_lens = choice_lens.view(-1, choice_lens.size(0) * choice_lens.size(1)).squeeze(0)
        doc_lens = doc_lens.view(-1, doc_lens.size(0) * doc_lens.size(1)).squeeze(0)
        expand_clses = clses.unsqueeze(1).repeat(1, 4, 1).view(-1, clses.size(-1))
        expand_mask_cls = mask_cls.unsqueeze(1).repeat(1, 4, 1).view(-1, mask_cls.size(-1))

        # sequence_output:[bs*4,512,768]
        sequence_outputs = self.bert(input_ids, attention_mask, token_type_ids)[0]

        # 获取question，choice，doc分别对应的向量:[bs*4,512,768]
        ques_seq_output, choice_seq_output, doc_seq_output = \
            self.seperate_seq(sequence_outputs, question_lens, choice_lens, doc_lens)

        # 注意力机制
        # output:[bs*4, 512, 768]
        pc_output = self.ssmatch([doc_seq_output, choice_seq_output, choice_lens])
        cp_output = self.ssmatch([choice_seq_output, doc_seq_output, doc_lens])
        pq_output = self.ssmatch([doc_seq_output, ques_seq_output, question_lens])
        qp_output = self.ssmatch([ques_seq_output, doc_seq_output, doc_lens])

        pa_fuse = self.fuse([pc_output, cp_output])
        pq_fuse = self.fuse([pq_output, qp_output])

        cat_output = torch.cat([pa_fuse, pq_fuse], -1)
        classifier_output = self.classifier2(self.dropout(cat_output))
        # sents_vec:[bs*4, n_sent, 768]
        sents_vec = doc_seq_output[torch.arange(classifier_output.size(0)).unsqueeze(1), expand_clses]
        sents_vec = sents_vec * expand_mask_cls[:, :, None].float()
        # sent_scores:[bs*4, n_sent]
        sent_scores = self.encoder(sents_vec, expand_mask_cls)
        # reshape_sent_scores:[bs, 4, n_sent]
        reshape_sent_scores = sent_scores.view(-1, 4, sent_scores.size(-1))
        # logits:[bs, n_sent]
        logits = torch.mean(reshape_sent_scores, 1)

        outputs = logits
        if labels is not None:
            labels = labels.type_as(logits)
            # Only keep active parts of the loss
            if mask_cls is not None:
                active_loss = mask_cls.reshape(-1) == 1
                active_logits = logits.reshape(-1)[active_loss]
                active_labels = labels.reshape(-1)[active_loss]
                loss = self.criterion(active_logits, active_labels)
            else:
                loss = self.criterion(logits.view(-1), labels.view(-1))
            outputs = (loss, logits)
        return outputs

    def seperate_seq(self, sequence_outputs, question_lens, choice_lens, doc_lens):
        doc_seq_output = sequence_outputs.new(sequence_outputs.size()).zero_()
        ques_seq_output = sequence_outputs.new(sequence_outputs.size()).zero_()
        choice_seq_output = sequence_outputs.new(sequence_outputs.size()).zero_()
        for i in range(doc_lens.size(0)):
            ques_seq_output[i, :question_lens[i]] = sequence_outputs[i, :question_lens[i]]
            choice_seq_output[i, :choice_lens[i]] = \
                sequence_outputs[i, question_lens[i]:question_lens[i] + choice_lens[i]]
            doc_seq_output[i, :doc_lens[i]] = \
                sequence_outputs[i, question_lens[i] + choice_lens[i]:question_lens[i] + choice_lens[i] + doc_lens[i]]
        return ques_seq_output, choice_seq_output, doc_seq_output
