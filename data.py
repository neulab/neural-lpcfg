#!/usr/bin/env python3
import numpy as np
import torch
import pickle

class Dataset(object):
  def __init__(self, data_file, load_dep=False):
    data = pickle.load(open(data_file, 'rb')) #get text data
    self.sents = self._convert(data['source']).long()
    self.other_data = data['other_data']
    self.sent_lengths = self._convert(data['source_l']).long()
    self.batch_size = self._convert(data['batch_l']).long()
    self.batch_idx = self._convert(data['batch_idx']).long()
    self.vocab_size = data['vocab_size'][0]
    self.num_batches = self.batch_idx.size(0)
    self.word2idx = data['word2idx']
    self.idx2word = data['idx2word']
    self.load_dep = load_dep

  def _convert(self, x):
    return torch.from_numpy(np.asarray(x))

  def __len__(self):
    return self.num_batches

  def __getitem__(self, idx):
    assert(idx < self.num_batches and idx >= 0)
    start_idx = self.batch_idx[idx]
    end_idx = start_idx + self.batch_size[idx]
    length = self.sent_lengths[idx].item()
    sents = self.sents[start_idx:end_idx]
    other_data = self.other_data[start_idx:end_idx]
    sent_str = [d[0] for d in other_data]
    tags = [d[1] for d in other_data]
    actions = [d[2] for d in other_data]
    binary_tree = [d[3] for d in other_data]
    spans = [d[5] for d in other_data]
    if(self.load_dep):
      heads = [d[7] for d in other_data]
    batch_size = self.batch_size[idx].item()
    # original data includes </s>, which we don't need
    data_batch = [sents[:, 1:length-1], length-2, batch_size, tags, actions, 
                  spans, binary_tree, other_data]
    if(self.load_dep):
      data_batch.append(heads)
    return data_batch
