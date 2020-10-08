import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config, GPT2PreTrainedModel


class GPT2LMHeadModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

    def forward(self, input_ids):
        transformer_outputs = self.transformer(input_ids)

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        return lm_logits
