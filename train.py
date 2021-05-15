import utils
import model
import torch
from transformers import BertTokenizer
from sklearn.metrics import confusion_matrix
import torch.utils.tensorboard as tb
import os


#Train BERT model
def train(args):
  train_data = utils.load_data('data/train/train.txt', 100)
  dev_data = utils.load_data('data/dev/dev.txt', 100)
  train_logger, valid_logger = None, None

  #Intialize tensorboard logger
  log_name = '%s' % (args.name)
  if args.log_dir is not None:
      train_logger = tb.SummaryWriter(os.path.join(args.log_dir, 'train', log_name))
      valid_logger = tb.SummaryWriter(os.path.join(args.log_dir, 'valid', log_name))

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  global_step = 0
  global_step2 = 0
  nerModel = model.NER()
  nerModel.to(device)
  optim = torch.optim.Adam(nerModel.parameters(), lr=2e-5, weight_decay = 1e-6)
  best_loss = 1000

  #Train for 4 epochs as suggested by BERT paper
  for epoch in range(4):
    lossList = []
    nerModel.train()
    count = 0
    for x, y, z in train_data:
      x = x.to(device)
      y = y.to(device)
      z = z.to(device)

      output = nerModel(x, y)
      loss = loss_func(output, z)
      loss.backward()
      optim.step()
      optim.zero_grad()
      lossList.append(loss.item())
      if train_logger:
        train_logger.add_scalar('loss', loss, global_step=global_step)
      global_step += 1


    lossList2 = []
    nerModel.eval()

    for x, y, z in dev_data:
      x = x.to(device)
      y = y.to(device)
      z = z.to(device)
      output = nerModel(x, y)
      loss = loss_func(output, z)
      lossList2.append(loss.item())
      if valid_logger:
        valid_logger.add_scalar('loss', loss, global_step=global_step2)
      global_step2 += 1

    #Save model
    model.save_model(nerModel, 'nerModel')
    print(sum(lossList)/len(train_data), sum(lossList2)/len(dev_data))

#Train BERTCRF model
def trainBERTCRF(args):
  train_data = utils.load_data('data/train/train.txt', 100, is_CRF = True)
  dev_data = utils.load_data('data/dev/dev.txt', 100, is_CRF = True)
  #test_data = utils.load_test_data('data/test/test.nolabels.txt', 60)
  train_logger, valid_logger = None, None

   #Intialize tensorboard logger
  log_name = '%s' % (args.name)
  if args.log_dir is not None:
      train_logger = tb.SummaryWriter(os.path.join(args.log_dir, 'train', log_name))
      valid_logger = tb.SummaryWriter(os.path.join(args.log_dir, 'valid', log_name))

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  global_step = 0
  global_step2 = 0

  nerModel = model.NERCRF()
  nerModel.to(device)
  optim = torch.optim.Adam(nerModel.parameters(), lr=2e-5, weight_decay = 1e-6)
  best_loss = 1000


  for epoch in range(4):
    lossList = []
    nerModel.train()
    for x, y, z in train_data:
      x = x.to(device)
      y = y.to(device)
      z = z.to(device)
      loss = nerModel(x, y, z)
      loss.backward()
      optim.step()
      optim.zero_grad()
      lossList.append(loss.item())
      if train_logger:
        train_logger.add_scalar('loss', loss, global_step=global_step)
      global_step += 1

    lossList2 = []
    nerModel.eval()

    for x, y, z in dev_data:
      x = x.to(device)
      y = y.to(device)
      z = z.to(device)
      loss = nerModel(x, y, z)
      # loss = loss_func(output, z)
      lossList2.append(loss.item())
      if valid_logger:
        valid_logger.add_scalar('loss', loss, global_step=global_step2)
      global_step2 += 1


    model.save_model(nerModel, 'nerCRFModel')
    print(sum(lossList)/len(train_data), sum(lossList2)/len(dev_data))

#Test BERTCRF
def testCRF(model_name):
  test_data = utils.load_test_data('data/dev/dev.nolabels.txt', 60)


  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  nerModel = model.load_model(model_name, isCRF = True)
  nerModel.to(device)
  nerModel.eval()
  count = 0
  f = open("devCRF.out", "w")
  li = ['O', 'B', 'I', 'O']
  for x, y, z, sub in test_data:
    x = x.to(device)
    y = y.to(device)
    z = z.to(device)
    #Tags are already given using the decode function from CRF
    output = nerModel(x, y, None)
    output = output[0]

    count = 1

    #Go through the output and get the corresponding tag from index
    for sub_words in sub:
      tags = []
      for ind_sub in sub_words:
        tags.append(li[output[count]])
        count+=1
      #Order matters here. We need to check B then I and last O
      if 'B' in tags:
        f.write('B')
        f.write('\n')
      elif 'I' in tags:
        f.write('I')
        f.write('\n')
      else:
        f.write('O')
        f.write('\n')


    f.write("\n")
    print(x.shape)

def test(model_name):
  test_data = utils.load_test_data('data/dev/dev.nolabels.txt', 60)


  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  nerModel = model.load_model(model_name)
  nerModel.to(device)
  nerModel.eval()
  count = 0
  f = open("dev.out", "w")
  li = ['O', 'B', 'I']

  for x, y, z, sub in test_data:
    x = x.to(device)
    y = y.to(device)
    z = z.to(device)
    output = nerModel(x, y)
    #Turn log softmax to values between 0 and 1
    prob = torch.exp(output)
    max_vals, max_ind = torch.max(prob, dim = -1)
    #Get the indicies with the highest probability
    masked_ind = max_ind[z.squeeze(0)]
    count = 0
    #Go through all the sub-tokens (nam, ##it)
    for sub_words in sub:
      tags = []
      for ind_sub in sub_words:
        tags.append(li[masked_ind[count].item()])
        count+=1
      #Order matters here. We need to check B then I and last O
      if 'B' in tags:
        f.write('B')
        f.write('\n')
      elif 'I' in tags:
        f.write('I')
        f.write('\n')
      else:
        f.write('O')
        f.write('\n')



    f.write("\n")
    print(output.shape)


  f.close()

    #print(max_ind.sum())

#Custom loss function to ignore all the padded values
def loss_func(output, labels):
  labels = labels.reshape(-1)
  mask = (labels >= 0).float()
  temp = torch.tensor([x for x in range(output.shape[0])])
  num_tokens = int(torch.sum(mask).item())
  outputs = output[temp, labels] * mask
  return -1 * torch.sum(outputs)/num_tokens

#Create a confusion matrix
def create_confusion_matrix():
  dataset = utils.NERDataset("data/dev/dev.txt", 60)
  dataset2 = utils.NERTestDataset("dev.out", 60)
  pred = []
  label = []
  label_names = ['O', 'B', 'I']
  for i in range(0, len(dataset)):
    example1 = dataset.exs[i]
    example2 = dataset2.exs[i]
    pred.extend(example1.labels)
    label.extend(example2.words)

  for i in range(1, len(pred)):
    if pred[i] == 'I' and pred[i-1] == 'O':
      pred[i] = 'B'

  print(confusion_matrix(label, pred, label_names))

#Calculate the data statistics
def datasetStats():
  trainDataset = utils.NERDataset("data/dev/dev.txt", 60)
  devDataset = utils.NERDataset("data/dev/dev.txt", 60)
  testDataset = utils.NERTestDataset("data/test/test.nolabels.txt", 60)
  vocabSet = set()
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  vocabSet2 = set()
  unknowns = []
  print(len(trainDataset), len(devDataset), len(testDataset))
  for ex in trainDataset.exs:
    for word in ex.words:
      vocabSet.add(word)
      tw = tokenizer.tokenize(word)
      for w in tw:
        vocabSet2.add(w)
        if w == '[UNK]':
          unknowns.append(word)


  print(len(vocabSet))
  print(len(vocabSet2))




if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()

  parser.add_argument('-t', '--test')
  parser.add_argument('-m', '--model')
  parser.add_argument('-s', '--stat')
  parser.add_argument('-log_dir', '--log_dir')
  parser.add_argument('-n', '--name')
  # Put custom arguments here

  args = parser.parse_args()

  if args.test == 'n' and args.model == 'BERT':
    train(args)
  elif args.test == 'n' and args.model == 'BERTCRF':
    trainBERTCRF(args)
  elif args.model == 'BERT':
    test(args.test)
  elif args.model == 'BERTCRF':
    testCRF(args.test)
  elif args.stat == 'confusion':
    create_confusion_matrix()
  elif args.stat == 'stat':
    datasetStats()
