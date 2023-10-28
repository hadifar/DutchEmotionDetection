import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torchcrf import CRF
from transformers import PreTrainedModel, AutoModel
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead


# class MyModel(nn.Module):
#
#     def __int__(self, encoder):
#         self.encoder = encoder
#
#     def __call__(self, x):
#         return self.encoder(x)


# class RobertaSimple(BertPreTrainedModel):
#     _keys_to_ignore_on_load_unexpected = [r"pooler"]
#
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#
#         self.bert = RobertaModel.from_pretrained(config, add_pooling_layer=False)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#
#     def forward(
#             self,
#             input_ids=None,
#             attention_mask=None,
#             token_type_ids=None,
#             position_ids=None,
#             head_mask=None,
#             inputs_embeds=None,
#             labels=None,
#             output_attentions=None,
#             output_hidden_states=None,
#             return_dict=None,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
#             Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
#             1]``.
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         bs, ss, hs = input_ids.shape
#
#         # input_ids.view()
#
#         outputs = self.bert(
#             input_ids.view(bs * ss, hs),
#             attention_mask=attention_mask.view(bs * ss, hs),
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#
#         sequence_output = outputs[0][:, 0, :]
#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)
#         logits = logits.view(bs, ss, -1)
#
#         loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output
#
#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
#
#


class Encoder(PreTrainedModel):
    def __init__(self, config=None):
        super().__init__(config)

        self.transformer = AutoModel.from_config(config)
        self.classifier = RobertaClassificationHead(config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                ):
        encoder_output = self.transformer(input_ids, attention_mask)
        return self.classifier(encoder_output[0])


class RobertaCRF(nn.Module):
    # _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, encoder, crf):
        # super().__init__(config)
        super().__init__()
        self.encoder = encoder
        self.config = encoder.config
        self.num_labels = self.config.num_labels
        self.crf = crf
        if self.crf == 1:
            self.crf_tagger = CRF(num_tags=self.num_labels, batch_first=True)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        bs, ss, hs = input_ids.shape

        # input_ids.view()

        outputs = self.encoder(input_ids.view(bs * ss, hs), attention_mask.view(bs * ss, hs))

        logits = outputs.logits
        logits = logits.view(bs, ss, -1)

        loss = None
        if self.crf == 1:
            if labels is not None:
                log_likelihood, tags = self.crf_tagger(logits, labels.squeeze(-1)), self.crf_tagger.decode(logits)
                loss = 0 - log_likelihood
            else:
                tags = self.crf_tagger.decode(logits)

            tags = torch.Tensor(tags)

            if not return_dict:
                output = (tags,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return loss, tags
        else:
            if labels is not None:
                if not hasattr(self.config, 'problem_type') or self.config.problem_type is None:
                    if self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)

            if self.config.problem_type == "single_label_classification":
                return loss, logits.argmax(-1)
            elif self.config.problem_type == 'multi_label_classification':
                return loss, torch.gt(logits, 0).int()
