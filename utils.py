import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class NERDataset(Dataset):

  def __init__(self, filename, maxlen, is_CRF = False):
    f = open(filename)
    
    exs = []
    words = []
    labels = []
    for line in f:
      
      if len(line.strip()) > 0:
        fields = line.split("\t")
        words.append(fields[0])
        labels.append(fields[1].strip("\n"))
      else:
        exs.append(NERExample(words, labels))
        words = []
        labels = []
    
    f.close()
    
    self.exs = exs
    #Initialize the BERT tokenizer
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    self.maxlen = maxlen
    self.is_CRF = is_CRF

  def __len__(self):
    return len(self.exs)
  
  def __getitem__(self, index):
    sentence = self.exs[index].words
    labels = self.exs[index].labels
    tokens = []
    
    ex_labels = []
    for i in range(0, len(sentence)):
      word = sentence[i]
      label = labels[i]
      tokenized_word = self.tokenizer.tokenize(word)
      to_len = len(tokenized_word)
      
      tokens.extend(tokenized_word)
      ex_labels.extend([label] * to_len)

    if self.is_CRF:
      labels = [3] + ex_labels + [3]
    else:
      labels = [-1] + ex_labels + [-1]
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    
    if len(tokens) < self.maxlen:
      tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]
      if self.is_CRF:
        labels = labels + [ 3 for _ in range(self.maxlen - len(labels))]
      else:
        labels = labels + [ -1 for _ in range(self.maxlen - len(labels))]
    
    for i in range(0, len(labels)):
      if labels[i] == 'O':
        labels[i] = 0
      elif labels[i] == 'B':
        labels[i] = 1
      elif labels[i] == 'I':
        labels[i] = 2
    
    
    
    tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) 
    tokens_ids_tensor = torch.tensor(tokens_ids) 
    labels_tensor = torch.tensor(labels)
    attn_mask = ((tokens_ids_tensor != 0)).long()

    return tokens_ids_tensor, attn_mask, labels_tensor

class NERTestDataset(Dataset):

  def __init__(self, filename, maxlen):
    f = open(filename)
    
    exs = []
    words = []
    labels = []
    for line in f:
      
      if len(line.strip()) > 0:
        fields = line.split("\t")
        words.append(fields[0].strip("\n"))
        labels.append(True)
      else:
        exs.append(NERExample(words, labels))
        words = []
        labels = []
    
    f.close()
    
    self.exs = exs
    #Initialize the BERT tokenizer
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    self.maxlen = maxlen

  def __len__(self):
    return len(self.exs)
  
  def __getitem__(self, index):
    sentence = self.exs[index].words
    labels = self.exs[index].labels
    tokens = []
    ex_labels = []
    subwords = []
    for i in range(0, len(sentence)):
      word = sentence[i]
      label = labels[i]
      tokenized_word = self.tokenizer.tokenize(word)
      subwords.append(tokenized_word)
      to_len = len(tokenized_word)
      tokens.extend(tokenized_word)
      ex_labels.extend([label] * to_len)


    labels = [False] + ex_labels + [False]
    tokens = ['[CLS]'] + tokens + ['[SEP]']

    tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) 
    tokens_ids_tensor = torch.tensor(tokens_ids) 
    mask_tensor = torch.tensor(labels)
    attn_mask = ((tokens_ids_tensor != 0) ).long()

    return tokens_ids_tensor, attn_mask, mask_tensor, subwords

class NERExample:

  def __init__(self, words, labels):
        self.words = words
        self.labels = labels

  def __repr__(self):
      return repr(self.words) + "; label=" + repr(self.labels)

  def __str__(self):
      return self.__repr__()

def load_data(dataset_path, maxlen, num_workers=2, batch_size=16, is_CRF = False):
    dataset = NERDataset(dataset_path, maxlen, is_CRF)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)

def load_test_data(dataset_path, maxlen, num_workers=2, batch_size=1):
    dataset = NERTestDataset(dataset_path, maxlen)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)


