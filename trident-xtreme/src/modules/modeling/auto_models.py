from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.utils.generic import ModelOutput
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.auto.modeling_auto import AutoModel
from trident.utils.logging import get_logger

from .heads import ClassificationHead

log = get_logger(__name__)


@dataclass
class TokenClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    labels: Optional[torch.Tensor] = None


class AutoModelForCLSClassification(LightningModule):
    def __init__(self, hidden_dropout: float = 0.1, **kwargs):
        super().__init__()
        self.model = AutoModel.from_pretrained(**kwargs)
        self.base_model = self.model
        # self.encoder = self.model  # alias
        self.dropout = nn.Dropout(hidden_dropout)
        self.num_labels = kwargs.get("num_labels", 2)
        self.classifier = ClassificationHead(
            hidden_size=self.model.config.hidden_size,
            num_labels=self.num_labels,
        )

    def forward(self, **kwargs) -> SequenceClassifierOutput:
        outputs = self.model(
            input_ids=kwargs.get("input_ids"),
            attention_mask=kwargs.get("attention_mask"),
            token_type_ids=kwargs.get("token_type_ids"),
            position_ids=kwargs.get("position_ids"),
            # head_mask=kwargs.get("head_mask"),
            # inputs_embeds=kwargs.get("inputs_embeds"),
            # output_attentions=kwargs.get("output_attentions"),
            # output_hidden_states=kwargs.get("output_hidden_states"),
            # return_dict=kwargs.get("return_dict"),
        )
        sequence_output = outputs[0][:, 0, :]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        labels = kwargs.get("labels")
        if labels is not None:
            loss = cross_entropy(
                logits.view(-1, self.classifier.num_labels), labels.view(-1)
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AutoModelForTokenClassification(LightningModule):
    def __init__(self, hidden_dropout: float = 0.1, **kwargs):
        super().__init__()
        self.num_labels = kwargs.get("num_labels", 1)
        self.config = AutoConfig.from_pretrained(
            kwargs.get("pretrained_model_name_or_path")
        )
        self.config.update({"hidden_dropout_prob": hidden_dropout})
        self.roberta = AutoModel.from_pretrained(
            kwargs.get("pretrained_model_name_or_path"), config=self.config
        )
        self.hidden_size = self.roberta.config.hidden_size
        # self.encoder = self.model  # alias
        self.dropout = nn.Dropout(hidden_dropout)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def embed(self, *args):
        return self.roberta(input_ids=args[0], attention_mask=args[1])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> TokenClassifierOutput:
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            labels=labels,
        )
