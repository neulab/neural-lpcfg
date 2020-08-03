#!/usr/bin/env python3
import sys
import os

import argparse
import json
import random
import shutil
import copy

from collections import defaultdict

import torch
from torch import cuda
import numpy as np
import time
import logging
from data import Dataset
from utils import *
from models import CompPCFG, LexicalizedCompPCFG
from torch.nn.init import xavier_uniform_
from torch.utils.tensorboard import SummaryWriter

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

import pdb

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

# Program options
parser.add_argument('--mode', default='train', help='train/test')
parser.add_argument('--test_file', default='data/preprocessed/ptb-test.pkl')
# Data path options
parser.add_argument('--train_file', default='data/preprocessed/ptb-train.pkl')
parser.add_argument('--val_file', default='data/preprocessed/ptb-val.pkl')
parser.add_argument('--save_path', default='compound-pcfg.pt', help='where to save the model')
parser.add_argument('--pretrained_word_emb', default="", help="word emb file")
# Model options
parser.add_argument('--model', default='LexicalizedCompPCFG', type=str, help='model name')
parser.add_argument('--load_model', default='', type=str, help='checkpoint file of stored model')
parser.add_argument('--init_gain', default=1., type=float, help='gain of xaviar initialization')
parser.add_argument('--init_model', default='', help='initial lexicalized pcfg with compound pcfg')
# Generative model parameters
parser.add_argument('--z_dim', default=64, type=int, help='latent dimension')
parser.add_argument('--t_states', default=60, type=int, help='number of preterminal states')
parser.add_argument('--nt_states', default=30, type=int, help='number of nonterminal states')
parser.add_argument('--state_dim', default=256, type=int, help='symbol embedding dimension')
parser.add_argument('--nt_emission', action="store_true", help='allow a single word span with a non-terminal')
parser.add_argument('--scalar_dir_scores', action="store_true", help='using scalar dir scores instead neural ones')
parser.add_argument('--seperate_nt_emb_for_emission', action="store_true", help='seperate nt embeddings for emission probability')
parser.add_argument('--head_first', action="store_true", help="first generate head and direction")
parser.add_argument('--tie_word_emb', action="store_true", help="tie the word embeddings")
parser.add_argument('--flow_word_emb', action="store_true", help="emit words via invertible flow")
parser.add_argument('--freeze_word_emb', action="store_true", help="freeze word embeddings")
# Inference network parameters
parser.add_argument('--h_dim', default=512, type=int, help='hidden dim for variational LSTM')
parser.add_argument('--w_dim', default=512, type=int, help='embedding dim for variational LSTM')
# Optimization options
parser.add_argument('--num_epochs', default=10, type=int, help='number of training epochs')
parser.add_argument('--lr', default=0.001, type=float, help='starting learning rate')
parser.add_argument('--delay_step', default=1, type=int, help='number of backprop before step')
parser.add_argument('--max_grad_norm', default=3, type=float, help='gradient clipping parameter')
parser.add_argument('--max_length', default=30, type=float, help='max sentence length cutoff start')
parser.add_argument('--len_incr', default=1, type=int, help='increment max length each epoch')
parser.add_argument('--final_max_length', default=40, type=int, help='final max length cutoff')
parser.add_argument('--eval_max_length', default=None, type=int, help='max length in evaluation. set to the same as final_max_length by default')
parser.add_argument('--beta1', default=0.75, type=float, help='beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for adam')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--seed', default=3435, type=int, help='random seed')
parser.add_argument('--print_every', type=int, default=1000, help='print stats after N batches')
parser.add_argument('--supervised_signals', nargs="*", default = [], help="supervised signals to use")
parser.add_argument('--opt_level', type=str, default="O0", help="mixed precision")
parser.add_argument('--t_emb_init', type=str, default="", help="initial value of t_emb")
parser.add_argument('--vocab_mlp_identity_init', action='store_true', help="initialize vocab_mlp as identity function")
# Evaluation optiones
parser.add_argument('--evaluate_dep', action='store_true', help='evaluate dependency parsing results')

parser.add_argument('--log_dir', type=str, default="", help='tensorboard logdir')

args = parser.parse_args()

if(args.eval_max_length is None):
  args.eval_max_length = args.final_max_length

# tensorboard
if(args.log_dir == ""):
  writer = SummaryWriter()
else:
  writer = SummaryWriter(log_dir=args.log_dir)
global_step = 0

def add_scalars(main_tag, tag_scalar_dict, global_step):
  for tag in tag_scalar_dict:
    writer.add_scalar("{}/{}".format(main_tag, tag), tag_scalar_dict[tag], global_step)

def main(args):
  global global_step
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if(args.mode == 'train'):
    train_data = Dataset(args.train_file, load_dep=args.evaluate_dep)
    val_data = Dataset(args.val_file, load_dep=args.evaluate_dep)  
    train_sents = train_data.batch_size.sum()
    vocab_size = int(train_data.vocab_size)    
    max_len = max(val_data.sents.size(1), train_data.sents.size(1))
    print('Train: %d sents / %d batches, Val: %d sents / %d batches' % 
          (train_data.sents.size(0), len(train_data), val_data.sents.size(0), len(val_data)))
    if(not args.pretrained_word_emb == ""):
      pretrained_word_emb_matrix = get_word_emb_matrix(args.pretrained_word_emb, train_data.idx2word)
    else:
      pretrained_word_emb_matrix = None
  else:
    test_data = Dataset(args.test_file, load_dep=args.evaluate_dep)
    vocab_size = int(test_data.vocab_size)
    max_len = test_data.sents.size(1)
    print("Test: %d sents / %d batches" % (test_data.sents.size(0), len(test_data)))
    if(not args.pretrained_word_emb == ""):
      pretrained_word_emb_matrix = get_word_emb_matrix(args.pretrained_word_emb, test_data.idx2word)
    else:
      pretrained_word_emb_matrix = None
  print('Vocab size: %d, Max Sent Len: %d' % (vocab_size, max_len))
  print('Save Path', args.save_path)
  cuda.set_device(args.gpu)
  if(args.model == 'CompPCFG'):
    model = CompPCFG(vocab = vocab_size,
                    state_dim = args.state_dim,
                    t_states = args.t_states,
                    nt_states = args.nt_states,
                    h_dim = args.h_dim,
                    w_dim = args.w_dim,
                    z_dim = args.z_dim)
    init_model = None
  elif(args.model == 'LexicalizedCompPCFG'):
    if args.init_model != '':
      init_model = CompPCFG(vocab = vocab_size,
                       state_dim = args.state_dim,
                       t_states = args.t_states,
                       nt_states = args.nt_states,
                       h_dim = args.h_dim,
                       w_dim = args.w_dim,
                       z_dim = args.z_dim)
      init_model.load_state_dict(torch.load(args.init_model)["model"])
      args.supervised_signals = ["phrase", "tag", "nt"] 
    else:
      init_model = None
    model = LexicalizedCompPCFG(vocab = vocab_size,
                                state_dim = args.state_dim,
                                t_states = args.t_states,
                                nt_states = args.nt_states,
                                h_dim = args.h_dim,
                                w_dim = args.w_dim,
                                z_dim = args.z_dim,
                                nt_emission=args.nt_emission,
                                scalar_dir_scores=args.scalar_dir_scores,
                                seperate_nt_emb_for_emission=args.seperate_nt_emb_for_emission,
                                head_first=args.head_first,
                                tie_word_emb=args.tie_word_emb,
                                flow_word_emb=args.flow_word_emb,
                                freeze_word_emb=args.freeze_word_emb,
                                pretrained_word_emb=pretrained_word_emb_matrix,
                                supervised_signals=args.supervised_signals)
  else:
    raise NotImplementedError
  for name, param in model.named_parameters():    
    if param.dim() > 1:
      xavier_uniform_(param, args.init_gain)
  if(args.t_emb_init != ""):
    t_emb_init = np.loadtxt(args.t_emb_init)
    model.t_emb.data.copy_(torch.from_numpy(t_emb_init))
  if(args.vocab_mlp_identity_init):
    model.vocab_mlp[0].bias.data.copy_(torch.zeros(args.state_dim))
    model.vocab_mlp[0].weight.data.copy_(torch.cat([torch.eye(args.state_dim, args.state_dim), torch.zeros(args.state_dim, args.z_dim)], dim=1))
  if(args.load_model != ''):
    print("Loading model from {}.".format(args.load_model))    
    model.load_state_dict(torch.load(args.load_model)["model"])
    print("Model loaded from {}.".format(args.load_model))    
  print("model architecture")
  print(model)
  model.train()
  model.cuda()
  if init_model:
    init_model.eval()
    init_model.cuda()
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas = (args.beta1, args.beta2))
  if args.opt_level != "O0":
    model.pcfg.huge = 1e4
    model, optimizer = amp.initialize(
      model, optimizer, opt_level=args.opt_level, 
      keep_batchnorm_fp32=True, loss_scale="dynamic"
    )
  if(args.mode == "test"):
    print('--------------------------------')
    print('Checking validation perf...')    
    test_ppl, test_f1 = eval(test_data, model)
    print('--------------------------------')
    return
  best_val_ppl = 1e5
  best_val_f1 = 0
  epoch = 0
  while epoch < args.num_epochs:
    start_time = time.time()
    epoch += 1  
    print('Starting epoch %d' % epoch)
    train_nll = 0.
    train_kl = 0.
    num_sents = 0.
    num_words = 0.
    all_stats = [[0., 0., 0.]]
    if(args.evaluate_dep):
      dep_stats = [[0., 0., 0.]]
    b = b_ = 0
    optimization_delay_count_down = args.delay_step
    for i in np.random.permutation(len(train_data)):
      b += 1
      gold_tree = None
      if(not args.evaluate_dep):
        sents, length, batch_size, _, _, gold_spans, gold_binary_trees, _ = train_data[i]      
      else:
        sents, length, batch_size, gold_tags, gold_actions, gold_spans, gold_binary_trees, _, heads = train_data[i]
        if(len(args.supervised_signals)):
          gold_tree = []
          for j in range(len(heads)):
            gold_tree.append(get_span2head(gold_spans[j], heads[j], gold_actions=gold_actions[j], gold_tags=gold_tags[j]))
            for span, (head, label) in gold_tree[j].items():
              if(span[0] == span[1]):
                gold_tree[j][span] = (head, PT2ID[label])
              else:
                f = lambda x : x[:x.find('-')] if x.find('-') != -1 else x
                g = lambda y : y[:y.find('=')] if y.find('=') != -1 else y
                gold_tree[j][span] = (head, NT2ID[f(g(label))])
      if length > args.max_length or length == 1: #length filter based on curriculum 
        continue
      b_ += 1
      sents = sents.cuda()
      if init_model:
        gold_tree = []
        with torch.no_grad():
          _, _, _, argmax_spans = init_model(sents, argmax=True)
          for j in range(len(argmax_spans)):
            gold_tree.append({})
            for span in argmax_spans[j]:
              if(span[0] == span[1]):
                gold_tree[j][(span[0], span[1])] = (-1, span[2] - args.nt_states)
              else:
                gold_tree[j][(span[0], span[1])] = (-1, span[2])
      nll, kl, binary_matrix, argmax_spans = model(sents, argmax=True, gold_tree=gold_tree)      
      loss = (nll + kl).mean()
      if(args.opt_level != "O0"):
        with amp.scale_loss(loss, optimizer) as scaled_loss:
          scaled_loss.backward()
      else:
        loss.backward()
      train_nll += nll.sum().item()
      train_kl += kl.sum().item()
      if(optimization_delay_count_down == 1):
        if args.max_grad_norm > 0:
            if args.opt_level == "O0":
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(amp.master_params(
                    optimizer), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        optimization_delay_count_down = args.delay_step
      else:
        optimization_delay_count_down -= 1
      num_sents += batch_size
      num_words += batch_size * (length + 1) # we implicitly generate </s> so we explicitly count it
      for bb in range(batch_size):
        span_b = [(a[0], a[1]) for a in argmax_spans[bb] if a[0] != a[1]] #ignore labels
        span_b_set = set(span_b[:-1])
        update_stats(span_b_set, [set(gold_spans[bb][:-1])], all_stats)
        if(args.evaluate_dep):
          update_dep_stats(argmax_spans[bb], heads[bb], dep_stats)
      if b_ % args.print_every == 0:
        all_f1 = get_f1(all_stats)
        dir_acc, undir_acc = get_dep_acc(dep_stats) if args.evaluate_dep else (0., 0.)
        param_norm = sum([p.norm()**2 for p in model.parameters()]).item()**0.5
        gparam_norm = sum([p.grad.norm()**2 for p in model.parameters() 
                           if p.grad is not None]).item()**0.5
        log_str = 'Epoch: %d, Batch: %d/%d, |Param|: %.6f, |GParam|: %.2f,  LR: %.4f, ' + \
                  'ReconPPL: %.2f, NLLloss: %.4f, KL: %.4f, PPLBound: %.2f, ValPPL: %.2f, ValF1: %.2f, ' + \
                  'CorpusF1: %.2f, DirAcc: %.2f, UndirAcc: %.2f, Throughput: %.2f examples/sec'
        print(log_str %
              (epoch, b, len(train_data), param_norm, gparam_norm, args.lr, 
               np.exp(train_nll / num_words), train_nll / num_words, train_kl /num_sents, 
               np.exp((train_nll + train_kl)/num_words), best_val_ppl, best_val_f1, 
               all_f1[0], dir_acc, undir_acc, num_sents / (time.time() - start_time)))
        # print an example parse
        tree = get_tree_from_binary_matrix(binary_matrix[0], length)
        action = get_actions(tree)
        sent_str = [train_data.idx2word[word_idx] for word_idx in list(sents[0].cpu().numpy())]
        if(args.evaluate_dep):
          print("Pred Tree: %s" % get_tagged_parse(get_tree(action, sent_str), argmax_spans[0]))
        else:
          print("Pred Tree: %s" % get_tree(action, sent_str))
        print("Gold Tree: %s" % get_tree(gold_binary_trees[0], sent_str))

        # tensorboard
        global_step += args.print_every
        add_scalars(main_tag="train",
                           tag_scalar_dict={"ParamNorm": param_norm,
                                            "ParamGradNorm": gparam_norm,
                                            "ReconPPL": np.exp(train_nll / num_words),
                                            "KL": train_kl /num_sents, 
                                            "PPLBound": np.exp((train_nll + train_kl)/num_words), 
                                            "CorpusF1": all_f1[0], 
                                            "DirAcc": dir_acc,
                                            "UndirAcc": undir_acc,
                                            "Throughput (examples/sec)": num_sents / (time.time() - start_time),
                                            "GPU memory usage": torch.cuda.memory_allocated()},
                           global_step=global_step)
        if(args.evaluate_dep):
          writer.add_text("Pred Tree", get_tagged_parse(get_tree(action, sent_str), argmax_spans[0]), global_step)                   
        else:
          writer.add_text("Pred Tree", get_tree(action, sent_str), global_step)                   
        writer.add_text("Gold Tree", get_tree(gold_binary_trees[0], sent_str), global_step)

    args.max_length = min(args.final_max_length, args.max_length + args.len_incr)
    print('--------------------------------')
    print('Checking validation perf...')    
    val_ppl, val_f1 = eval(val_data, model)
    print('--------------------------------')
    if val_ppl < best_val_ppl:
      best_val_ppl = val_ppl
      best_val_f1 = val_f1
      checkpoint = {
        'args': args.__dict__,
        'model': model.cpu().state_dict(),
        'word2idx': train_data.word2idx,
        'idx2word': train_data.idx2word
      }
      print('Saving checkpoint to %s' % args.save_path)
      torch.save(checkpoint, args.save_path)
      model.cuda()

def eval(data, model):
  global global_step
  model.eval()
  num_sents = 0
  num_words = 0
  total_nll = 0.
  total_kl = 0.
  corpus_f1 = [0., 0., 0.] 
  corpus_f1_by_cat = [defaultdict(int), defaultdict(int), defaultdict(int)]
  dep_stats = [[0., 0., 0.]]
  sent_f1 = [] 

  # f = open("tmp.txt", "w")

  with torch.no_grad():
    for i in range(len(data)):
      if(not args.evaluate_dep):
        sents, length, batch_size, _, gold_actions, gold_spans, gold_binary_trees, other_data = data[i] 
      else:
        sents, length, batch_size, gold_tags, gold_actions, gold_spans, gold_binary_trees, other_data, heads = data[i] 
      span_dicts = []
      for j in range(batch_size):
        span_dict = {}
        for l, r, nt in get_nonbinary_spans_label(gold_actions[j])[0]:
          span_dict[(l, r)] = nt
        span_dicts.append(span_dict)
      if length == 1 or length > args.eval_max_length:
        continue
      sents = sents.cuda()
      # note that for unsuperised parsing, we should do model(sents, argmax=True, use_mean = True)
      # but we don't for eval since we want a valid upper bound on PPL for early stopping
      # see eval.py for proper MAP inference
      nll, kl, binary_matrix, argmax_spans = model(sents, argmax=True)
      total_nll += nll.sum().item()
      total_kl  += kl.sum().item()
      num_sents += batch_size
      num_words += batch_size*(length +1) # we implicitly generate </s> so we explicitly count it

      gold_tree = []
      for j in range(len(heads)):
        gold_tree.append(get_span2head(gold_spans[j], heads[j], gold_actions=gold_actions[j], gold_tags=gold_tags[j]))
        for span, (head, label) in gold_tree[j].items():
          if(span[0] == span[1]):
            gold_tree[j][span] = (head, PT2ID[label])
          else:
            f = lambda x : x[:x.find('-')] if x.find('-') != -1 else x
            g = lambda y : y[:y.find('=')] if y.find('=') != -1 else y
            gold_tree[j][span] = (head, f(g(label)))

      for b in range(batch_size):
        # for a in argmax_spans[b]:
        #   if((a[0], a[1]) in span_dicts[b]):
        #     f.write("{}\t{}\n".format(a[2], span_dicts[b][(a[0], a[1])]))

        span_b = [(a[0], a[1]) for a in argmax_spans[b] if a[0] != a[1]] #ignore labels
        span_b_set = set(span_b[:-1])        
        gold_b_set = set(gold_spans[b][:-1])
        tp, fp, fn = get_stats(span_b_set, gold_b_set) 
        corpus_f1[0] += tp
        corpus_f1[1] += fp
        corpus_f1[2] += fn
        tp_by_cat, all_by_cat = get_stats_by_cat(span_b_set, gold_b_set, gold_tree[b])
        for j in tp_by_cat:
          corpus_f1_by_cat[0][j] += tp_by_cat[j]
        for j in all_by_cat:
          corpus_f1_by_cat[1][j] += all_by_cat[j]
        # sent-level F1 is based on L83-89 from https://github.com/yikangshen/PRPN/test_phrase_grammar.py

        model_out = span_b_set
        std_out = gold_b_set
        overlap = model_out.intersection(std_out)
        prec = float(len(overlap)) / (len(model_out) + 1e-8)
        reca = float(len(overlap)) / (len(std_out) + 1e-8)
        if len(std_out) == 0:
          reca = 1. 
          if len(model_out) == 0:
            prec = 1.
        f1 = 2 * prec * reca / (prec + reca + 1e-8)
        sent_f1.append(f1)

        if(args.evaluate_dep):
          update_dep_stats(argmax_spans[b], heads[b], dep_stats)
  tp, fp, fn = corpus_f1  
  prec = tp / (tp + fp)
  recall = tp / (tp + fn)
  corpus_f1 = 2*prec*recall/(prec+recall) if prec+recall > 0 else 0.
  for j in corpus_f1_by_cat[1]:
    corpus_f1_by_cat[2][j] = corpus_f1_by_cat[0] / corpus_f1_by_cat[1]
  sent_f1 = np.mean(np.array(sent_f1))
  dir_acc, undir_acc = get_dep_acc(dep_stats) if args.evaluate_dep else (0., 0.)
  recon_ppl = np.exp(total_nll / num_words)
  ppl_elbo = np.exp((total_nll + total_kl)/num_words) 
  kl = total_kl /num_sents
  print('ReconPPL: %.2f, KL: %.4f, NLLloss: %.4f, PPL (Upper Bound): %.2f' %
        (recon_ppl, kl, total_nll / num_words, ppl_elbo))
  print('Corpus F1: %.2f, Sentence F1: %.2f' %
        (corpus_f1*100, sent_f1*100))
  if(args.evaluate_dep):
    print('DirAcc: %.2f, UndirAcc: %.2f'%(dir_acc, undir_acc))
  print('Corpus Recall by Category: {}'.format(corpus_f1_by_cat[2]))
  # tensorboard
  add_scalars(main_tag="validation",
                     tag_scalar_dict={"ReconPPL": recon_ppl,
                                       "KL": kl,
                                       "PPL (Upper Bound)": ppl_elbo,
                                       "Corpus F1": corpus_f1 * 100, 
                                       "Sentence F1": sent_f1*100,
                                       "DirAcc": dir_acc if args.evaluate_dep else 0,
                                       "UndirAcc": undir_acc if args.evaluate_dep else 0},
                     global_step=global_step)
  model.train()
  return ppl_elbo, sent_f1*100

if __name__ == '__main__':
  main(args)
