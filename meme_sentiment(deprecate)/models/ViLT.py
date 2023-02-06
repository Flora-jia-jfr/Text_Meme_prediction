import torch
from torch import nn
from transformers import ViltModel, ViltConfig

class ViLTForMemeSentimentClassification(nn.Module):

    def __init__(self, n_sentiment):
        super(ViLTForMemeSentimentClassification, self).__init__()
        self.configuration = ViltConfig()
        self.vilt_model = ViltModel(self.configuration)
        # Freeze the pretrained_model
        # for param in self.vilt_model.parameters():
        #     param.requires_grad = False
        self.fc = nn.Linear(in_features=self.configuration.hidden_size, out_features=n_sentiment)

    @torch.no_grad()
    def get_hidden(self, inputs):
        return self.vilt_model(**inputs)['pooler_output']

    def forward(self, inputs):
        hidden_out = self.get_hidden(inputs)
        fc_out = self.fc(hidden_out)
        return fc_out