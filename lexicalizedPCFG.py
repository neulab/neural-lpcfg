#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import random
from torch.cuda import memory_allocated
import pdb

class LexicalizedPCFG(nn.Module):
  # Lexicalized PCFG:
  # S → A[x]            A ∈ N, x ∈ 𝚺
  # A[x] → B[x] C[y]    A, B, C ∈ N ∪ P, x, y ∈ 𝚺
  # A[x] → B[y] C[x]    A, B, C ∈ N ∪ P, x, y ∈ 𝚺
  # T[x] → x            T ∈ P, x ∈ 𝚺

  def __init__(self, nt_states, t_states, nt_emission=False, supervised_signals = []):
    super(LexicalizedPCFG, self).__init__()
    self.nt_states = nt_states
    self.t_states = t_states
    self.states = nt_states + t_states
    self.nt_emission = nt_emission
    self.huge = 1e9

    if(self.nt_emission):
      self.word_span_slice = slice(self.states)
    else:
      self.word_span_slice = slice(self.nt_states,self.states)
    
    self.supervised_signals = supervised_signals

  # def logadd(self, x, y):
  #   d = torch.max(x,y)  
  #   return torch.log(torch.exp(x-d) + torch.exp(y-d)) + d    
  def logadd(self, x, y):
    names = x.names
    assert names == y.names, "Two operants' names are not matched {} and {}.".format(names, y.names)
    return torch.logsumexp(torch.stack([x.rename(None), y.rename(None)]), dim=0).refine_names(*names)

  def logsumexp(self, x, dim=1):
    d = torch.max(x, dim)[0]    
    if x.dim() == 1:
      return torch.log(torch.exp(x - d).sum(dim)) + d
    else:
      return torch.log(torch.exp(x - d.unsqueeze(dim).expand_as(x)).sum(dim)) + d

  def __get_scores(self, unary_scores, rule_scores, root_scores, dir_scores):
    # INPUT
    # unary scores : b x n x (NT + T)
    # rule scores : b x (NT+T) x (NT+T) x (NT+T)
    # root_scores : b x NT
    # dir_scores : 2 x b x NT x (NT + T) x (NT + T) x N
    # OUTPUT
    # rule scores: 2 x B x (NT x T) x (NT x T) x (NT x T) x N
    #              (D, B, T, TL, TR, H)
    # root_scores : b x NT x n
    #              (B, T, H)
    assert unary_scores.names == ('B', 'H', 'T')
    assert rule_scores.names == ('B', 'T', 'TL', 'TR')
    assert root_scores.names == ('B', 'T')
    assert dir_scores.names == ('D', 'B', 'T', 'H', 'TL', 'TR')

    rule_shape = ('D', 'B', 'T', 'H', 'TL', 'TR')
    root_shape = ('B', 'T', 'H')
    rule_scores = rule_scores.align_to(*rule_shape) \
                + dir_scores.align_to(*rule_shape)
    
    if rule_scores.size('H') == 1:
      rule_scores = rule_scores.expand(-1, -1, -1, unary_scores.size('H'), -1, -1)
    return rule_scores, root_scores, unary_scores

  def __get_scores(self, unary_scores, rule_scores, root_scores):
    # INPUT
    # unary scores : b x n x (NT + T)
    # rule scores : b x (NT+T) x (NT+T) x (NT+T)
    # root_scores : b x NT
    # dir_scores : 2 x b x NT x (NT + T) x (NT + T) x N
    # OUTPUT
    # rule scores: 2 x B x (NT x T) x (NT x T) x (NT x T) x N
    #              (D, B, T, TL, TR, H)
    # root_scores : b x NT x n
    #              (B, T, H)
    assert unary_scores.names == ('B', 'H', 'T')
    assert rule_scores.names == ('B', 'T', 'H', 'TL', 'TR', 'D')
    assert root_scores.names == ('B', 'T')

    rule_shape = ('D', 'B', 'T', 'H', 'TL', 'TR')
    root_shape = ('B', 'T', 'H')
    rule_scores = rule_scores.align_to(*rule_shape)
    
    if rule_scores.size('H') == 1:
      rule_scores = rule_scores.expand(-1, -1, -1, unary_scores.size('H'), -1, -1)
    return rule_scores, root_scores, unary_scores
  
  def print_name_size(self, x):
    print(x.size(), x.names)
  
  def print_memory_usage(self, lineno, device="cuda:0"):
    print("Line {}: {}M".format(lineno, int(memory_allocated(device)/1000000)))

  def cross_bracket(self, l, r, gold_brackets):
    for bl, br in gold_brackets:
      if((bl<l<=br and r>br) or (l<bl and bl<=r<br)):
        return True
    return False
  
  def get_mask(self, B, N, T, gold_tree):
    mask = self.beta.new(B, N+1, N+1, T, N).fill_(0)
    for i in range(B):
      gold_brackets = gold_tree[i].keys()
      if "phrase" in self.supervised_signals:
        for l in range(N):
          for r in range(l, N):
            if(self.cross_bracket(l, r, gold_brackets)):
              mask[i][l, r+1].fill_(-self.huge)
      for l, r in gold_brackets:
        mask[i][l, r+1].fill_(-self.huge)
        acceptable_heads = slice(gold_tree[i][(l, r)][0], gold_tree[i][(l, r)][0] + 1)\
                           if "head" in self.supervised_signals else slice(l, r+1)
        if(l == r):
          if(gold_tree[i][(l, r)][1] < self.t_states and "tag" in self.supervised_signals):
            mask[i][l, r+1, gold_tree[i][(l, r)][1] + self.nt_states, acceptable_heads] = 0
          else:
            mask[i][l, r+1, :, acceptable_heads] = 0
        else:
          if(gold_tree[i][(l, r)][1] < self.nt_states and "nt" in self.supervised_signals):
            mask[i][l, r+1, gold_tree[i][(l, r)][1], acceptable_heads] = 0
          else:
            mask[i][l, r+1, :, acceptable_heads] = 0
    return mask
  
  def _inside(self, gold_tree=None, **kwargs):
    #inside step
    
    rule_scores, root_scores, unary_scores = self.__get_scores(**kwargs)
    
    # statistics
    B = rule_scores.size('B')
    N = unary_scores.size('H')
    T = self.states

    # uses conventional python numbering scheme: [s, t] represents span [s, t)
    # this scheme facilitates fast computation
    # f[s, t] = logsumexp(f[s, :] * f[:, t])
    self.beta = rule_scores.new(B, N + 1, N + 1, T, N).fill_(-self.huge).refine_names('B', 'L', 'R', 'T', 'H')
    self.beta_ = rule_scores.new(B, N + 1, N + 1, T).fill_(-self.huge).refine_names('B', 'L', 'R', 'T')
    
    if(not gold_tree is None):
      mask = self.get_mask(B, N, T, gold_tree)
    else:
      mask = self.beta.new(B, N+1, N+1, T, N).fill_(0)

    # initialization: f[k, k+1]
    for k in range(N):
      for state in range(self.states):
        if(not self.nt_emission and state < self.nt_states):
          continue
        self.beta[:, k, k+1, state, k] = mask[:, k, k+1, state, k]
        self.beta_[:, k, k+1, state] = unary_scores[:, k, state].rename(None) + mask[:, k, k+1, state, k].rename(None)

    # span length w, at least 2
    for W in np.arange(2, N+1):
      
      # start point s
      for l in range(N-W+1):
        r = l + W
        
        f = lambda x:torch.logsumexp(x.align_to('B', 'T', 'H', ...).rename(None).reshape(B, self.nt_states, W, -1), dim=3).refine_names('B', 'T', 'H')
        left = lambda x, y, z: x.rename(T='TL').align_as(z) + y.rename(T='TR').align_as(z) + z
        right = lambda x, y, z: x.rename(T='TL').align_as(z) + y.rename(T='TR').align_as(z) + z
        g = lambda x, y, x_, y_, z: torch.cat((left(x, y_, z[0]).align_as(z), 
                                           right(x_, y, z[1]).align_as(z)), dim='D')

        if W == 2:
          tmp = g(self.beta[:, l, l+1, self.word_span_slice, l:r],
                  self.beta[:, l+1, r, self.word_span_slice, l:r],
                  self.beta_[:, l, l+1, self.word_span_slice],
                  self.beta_[:, l+1, r, self.word_span_slice],
                  rule_scores[:, :, :, l:r, self.word_span_slice, self.word_span_slice])
          tmp = f(tmp)

        elif W == 3:
          tmp1 = g(self.beta[:, l, l+1, self.word_span_slice, l:r],
                   self.beta[:, l+1, r, :self.nt_states, l:r],
                   self.beta_[:, l, l+1, self.word_span_slice],
                   self.beta_[:, l+1, r, :self.nt_states],
                   rule_scores[:, :, :, l:r, self.word_span_slice, :self.nt_states])
          tmp2 = g(self.beta[:, l, r-1, :self.nt_states, l:r],
                   self.beta[:, r-1, r, self.word_span_slice, l:r],
                   self.beta_[:, l, r-1, :self.nt_states],
                   self.beta_[:, r-1, r, self.word_span_slice],
                   rule_scores[:, :, :, l:r, :self.nt_states, self.word_span_slice])
          tmp = self.logadd(f(tmp1), f(tmp2))
        
        elif W >= 4:
          tmp1 = g(self.beta[:, l, l+1, self.word_span_slice, l:r],
                   self.beta[:, l+1, r, :self.nt_states, l:r],
                   self.beta_[:, l, l+1, self.word_span_slice],
                   self.beta_[:, l+1, r, :self.nt_states],
                   rule_scores[:, :, :, l:r, self.word_span_slice, :self.nt_states])
          
          tmp2 = g(self.beta[:, l, r-1, :self.nt_states, l:r],
                   self.beta[:, r-1, r, self.word_span_slice, l:r],
                   self.beta_[:, l, r-1, :self.nt_states],
                   self.beta_[:, r-1, r, self.word_span_slice],
                   rule_scores[:, :, :, l:r, :self.nt_states, self.word_span_slice])
          
          tmp3 = g(self.beta[:, l, l+2:r-1, :self.nt_states, l:r].rename(R='U'),
                   self.beta[:, l+2:r-1, r, :self.nt_states, l:r].rename(L='U'),
                   self.beta_[:, l, l+2:r-1, :self.nt_states].rename(R='U'),
                   self.beta_[:, l+2:r-1, r, :self.nt_states].rename(L='U'),
                   rule_scores[:, :, :, l:r, :self.nt_states, :self.nt_states].align_to('D', 'B', 'T', 'H', 'U', ...))
          tmp = self.logadd(self.logadd(f(tmp1), f(tmp2)), f(tmp3))

        tmp = tmp + mask[:, l, r, :self.nt_states, l:r]
        self.beta[:, l, r, :self.nt_states, l:r] = tmp.rename(None)
        tmp_ = torch.logsumexp(tmp + unary_scores[:, l:r, :self.nt_states].align_as(tmp), dim='H')
        self.beta_[:, l, r, :self.nt_states] = tmp_.rename(None)

    
    log_Z = self.beta_[:, 0, N, :self.nt_states] + root_scores
    log_Z = torch.logsumexp(log_Z, dim='T')
    return log_Z

  def _viterbi(self, **kwargs):
    #unary scores : b x n x T
    #rule scores : b x NT x (NT+T) x (NT+T)
    
    rule_scores, root_scores, unary_scores = self.__get_scores(**kwargs)
    
    # statistics
    B = rule_scores.size('B')
    N = unary_scores.size('H')
    T = self.states
    
    # # dummy rules
    # rule_scores = torch.cat([rule_scores, \
    #                          rule_scores.new(B, self.t_states, T, T) \
    #                          .fill_(-self.huge)], dim=1)
    
    self.scores = rule_scores.new(B, N+1, N+1, T, N).fill_(-self.huge).refine_names('B', 'L', 'R', 'T', 'H')
    self.scores_ = rule_scores.new(B, N+1, N+1, T).fill_(-self.huge).refine_names('B', 'L', 'R', 'T')
    self.bp = rule_scores.new(B, N+1, N+1, T, N).long().fill_(-1).refine_names('B', 'L', 'R', 'T', 'H')
    self.left_bp = rule_scores.new(B, N+1, N+1, T, N).long().fill_(-1).refine_names('B', 'L', 'R', 'T', 'H')
    self.right_bp = rule_scores.new(B, N+1, N+1, T, N).long().fill_(-1).refine_names('B', 'L', 'R', 'T', 'H')
    self.dir_bp = rule_scores.new(B, N+1, N+1, T, N).long().fill_(-1).refine_names('B', 'L', 'R', 'T', 'H')
    self.new_head_bp = rule_scores.new(B, N+1, N+1, T).long().fill_(-1).refine_names('B', 'L', 'R', 'T')
    self.argmax = rule_scores.new(B, N, N).long().fill_(-1)
    self.argmax_tags = rule_scores.new(B, N).long().fill_(-1)
    self.spans = [[] for _ in range(B)]
    
    # initialization: f[k, k+1]
    for k in range(N):
      for state in range(self.states):
        if(not self.nt_emission and state < self.nt_states):
          continue
        self.scores[:, k, k+1, state, k] = 0  
        self.scores_[:, k, k+1, state] = unary_scores[:, k, state].rename(None)
        self.new_head_bp[:, k, k+1, state] = k

    for W in np.arange(2, N+1):
      for l in range(N-W+1):
        r = l + W
        
        left = lambda x, y, z: x.rename(T='TL').align_as(z) + y.rename(T='TR').align_as(z) + z
        right = lambda x, y, z: x.rename(T='TL').align_as(z) + y.rename(T='TR').align_as(z) + z
        g = lambda x, y, x_, y_, z: torch.cat((left(x, y_, z[0]).align_as(z), 
                                           right(x_, y, z[1]).align_as(z)), dim='D')
        
        # self.print_name_size(self.scores[:, l, l+1:r, :, l:r])
        # self.print_name_size(rule_scores[:, :, :, l:r, :self.nt_states, :self.nt_states, l:r].align_to('D', 'B', 'T', 'H', 'U', ...))
        tmp = g(self.scores[:, l, l+1:r, :, l:r].rename(R='U'),
                self.scores[:, l+1:r, r, :, l:r].rename(L='U'),
                self.scores_[:, l, l+1:r, :].rename(R='U'),
                self.scores_[:, l+1:r, r, :].rename(L='U'),
                rule_scores[:, :, :, l:r, :, :].align_to('D', 'B', 'T', 'H', 'U', ...))
        
        tmp = tmp.align_to('B', 'T', 'H', 'D', 'U', 'TL', 'TR').flatten(['D', 'U', 'TL', 'TR'], 'position')

        assert(tmp.size('position') == self.states * self.states * (W-1) * 2), "{}".format(tmp.size('position'))
        # view once and marginalize    
        tmp, max_pos = torch.max(tmp, dim=3)

        max_pos = max_pos.rename(None)
        right_child = max_pos % self.states
        max_pos /= self.states
        left_child = max_pos % self.states
        max_pos /= self.states
        max_idx = max_pos % (W-1) + l + 1
        max_pos = max_pos / int(W - 1)
        max_dir = max_pos
        
        self.scores[:, l, r, :self.nt_states, l:r] = tmp.rename(None)
        tmp_ = tmp + unary_scores[:, l:r, :self.nt_states].align_as(tmp)
        tmp_, new_head = torch.max(tmp_, dim='H')
        self.scores_[:, l, r, :self.nt_states] = tmp_.rename(None)
        
        self.bp[:, l, r, :self.nt_states, l:r] = max_idx        
        self.left_bp[:, l, r, :self.nt_states, l:r] = left_child
        self.right_bp[:, l, r, :self.nt_states, l:r] = right_child
        self.dir_bp[:, l, r, :self.nt_states, l:r] = max_dir
        self.new_head_bp[:, l, r, :self.nt_states] = new_head.rename(None) + l

    max_score = self.scores_[:, 0, N, :self.nt_states] + root_scores
    max_score, max_idx = torch.max(max_score, dim='T')
    for b in range(B):
      self._backtrack(b, 0, N, max_idx[b].item())      
    return self.scores, self.argmax, self.spans

  def _backtrack(self, b, s, t, state, head=-1):
    if(head == -1):
      head = int(self.new_head_bp[b][s][t][state])
    u = int(self.bp[b][s][t][state][head])
    assert(s < t), "s: %d, t %d"%(s, t)
    left_state = int(self.left_bp[b][s][t][state][head])
    right_state = int(self.right_bp[b][s][t][state][head])
    direction = int(self.dir_bp[b][s][t][state][head])
    self.argmax[b][s][t-1] = 1
    if s == t-1:
      self.spans[b].insert(0, (s, t-1, state, head))
      self.argmax_tags[b][s] = state
      return None      
    else:
      self.spans[b].insert(0, (s, t-1, state, head))
      if(direction == 0):
        assert head < u, "head: {} < u: {}".format(head, u)
        self._backtrack(b, s, u, left_state, head)
        self._backtrack(b, u, t, right_state)
      else:
        assert head >= u, "head: {} >= u: {}".format(head, u)
        self._backtrack(b, s, u, left_state)
        self._backtrack(b, u, t, right_state, head)
        
    return None