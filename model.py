import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, BertForTokenClassification
from torchcrf import CRF 

#Based on tutorial provided by hugging face transformers
class NER(torch.nn.Module):

  def __init__(self, hidden_size = 256):
    super().__init__()

    self.embeddings = BertModel.from_pretrained('bert-base-uncased')
    self.GRU = torch.nn.GRU(768, hidden_size, batch_first = True, bidirectional = True)
    self.intermediate_layer = torch.nn.Linear(2 * hidden_size, hidden_size)
    self.dropout = torch.nn.Dropout(.1)
    self.final_layer = torch.nn.Linear(hidden_size, 3)
    

  def forward(self, tokens, masks):
    embedd = self.embeddings(tokens, attention_mask = masks)
    output, _ = self.GRU(embedd.last_hidden_state)
    output = output.reshape(-1, output.shape[2])
    output = self.intermediate_layer(output)
    output = self.dropout(output)
    
    return F.log_softmax(self.final_layer(output), dim=1)

#Based on tutorial provided by hugging face transformers and
#https://pytorch-crf.readthedocs.io/en/stable/#torchcrf.CRF.forward
class NERCRF(torch.nn.Module):
  def __init__(self, hidden_size = 256):
    super().__init__()

    self.embeddings = BertModel.from_pretrained('bert-base-uncased')
   
    self.GRU = torch.nn.GRU(768, hidden_size, batch_first = True, bidirectional = True)
    self.intermediate_layer = torch.nn.Linear(2 * hidden_size, hidden_size)
    self.dropout = torch.nn.Dropout(.1)
    self.final_layer = torch.nn.Linear(hidden_size, 4)
    self.crf = CRF(4, batch_first=True)
    

  def forward(self, tokens, masks, labels):
    embedd = self.embeddings(tokens, attention_mask = masks)
    output, _ = self.GRU(embedd.last_hidden_state)
    output = self.intermediate_layer(output)
    output = self.dropout(output)
    output = self.final_layer(output)
    masks = masks.type(torch.uint8)
    if labels is not None:
      loss = -1 * self.crf(F.log_softmax(output, dim = 2), labels, mask=masks, reduction='mean')
      return loss
    else:
      prediction = self.crf.decode(output, mask=masks)
      return prediction

#https://pytorch.org/tutorials/beginner/saving_loading_models.html
#Tutorial to save and load pytorch models
def save_model(model, name):
    from torch import save
    from os import path
    
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % name))
    raise ValueError("model type '%s' doesn't exist" % str(type(model)))


def load_model(model, isCRF = False):
    from torch import load
    from os import path
    if isCRF:
      r = NERCRF()
    else:
      r = NER()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r