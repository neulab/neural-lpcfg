#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import random
  
class PCFG(nn.Module):
  def __init__(self, nt_states, t_states):
    super(PCFG, self).__init__()
    self.nt_states = nt_states
    self.t_states = t_states
    self.states = nt_states + t_states
    self.huge = 1e9

  def logadd(self, x, y):
    d = torch.max(x,y)  
    return torch.log(torch.exp(x-d) + torch.exp(y-d)) + d    

  def logsumexp(self, x, dim=1):
    d = torch.max(x, dim)[0]    
    if x.dim() == 1:
      return torch.log(torch.exp(x - d).sum(dim)) + d
    else:
      return torch.log(torch.exp(x - d.unsqueeze(dim).expand_as(x)).sum(dim)) + d

  def _inside(self, unary_scores, rule_scores, root_scores):
    #inside step
    #unary scores : b x n x T
    #rule scores : b x NT  x (NT+T) x (NT+T)
    #root : b x NT
    
    # statistics
    batch_size = unary_scores.size(0)
    n = unary_scores.size(1)

    # uses conventional python numbering scheme: [s, t] represents span [s, t)
    # this scheme facilitates fast computation
    # f[s, t] = logsumexp(f[s, :] * f[:, t])
    self.beta = unary_scores.new(batch_size, n + 1, n + 1, self.states).fill_(-self.huge)

    # initialization: f[k, k+1]
    for k in range(n):
      for state in range(self.t_states):
        self.beta[:, k, k+1, self.nt_states + state] = unary_scores[:, k, state]        

    # span length w, at least 2
    for w in np.arange(2, n+1):
      
      # start point s
      for s in range(n-w+1):
        t = s + w
        
        f = lambda x:torch.logsumexp(x.view(batch_size, self.nt_states, -1), dim=2)

        if w == 2:
          tmp = self.beta[:, s, s+1, self.nt_states:].unsqueeze(2).unsqueeze(1) \
              + self.beta[:, s+1, t, self.nt_states:].unsqueeze(1).unsqueeze(2) \
              + rule_scores[:, :, self.nt_states:, self.nt_states:]
          tmp = f(tmp)
        elif w == 3:
          tmp1 = self.beta[:, s, s+1, self.nt_states:].unsqueeze(2).unsqueeze(1) \
               + self.beta[:, s+1, t, :self.nt_states].unsqueeze(1).unsqueeze(2) \
               + rule_scores[:, :, self.nt_states:, :self.nt_states]
          tmp2 = self.beta[:, s, t-1, :self.nt_states].unsqueeze(2).unsqueeze(1) \
               + self.beta[:, t-1, t, self.nt_states:].unsqueeze(1).unsqueeze(2) \
               + rule_scores[:, :, :self.nt_states, self.nt_states:]
          tmp = self.logadd(f(tmp1), f(tmp2))
        elif w >= 4:
          tmp1 = self.beta[:, s, s+1, self.nt_states:].unsqueeze(2).unsqueeze(1) \
               + self.beta[:, s+1, t, :self.nt_states].unsqueeze(1).unsqueeze(2) \
               + rule_scores[:, :, self.nt_states:, :self.nt_states]
          tmp2 = self.beta[:, s, t-1, :self.nt_states].unsqueeze(2).unsqueeze(1) \
               + self.beta[:, t-1, t, self.nt_states:].unsqueeze(1).unsqueeze(2) \
               + rule_scores[:, :, :self.nt_states, self.nt_states:]
          tmp3 = self.beta[:, s, s+2:t-1, :self.nt_states].unsqueeze(3).unsqueeze(1) \
               + self.beta[:, s+2:t-1, t, :self.nt_states].unsqueeze(1).unsqueeze(3) \
               + rule_scores[:, :, :self.nt_states, :self.nt_states].unsqueeze(2)
          tmp = self.logadd(self.logadd(f(tmp1), f(tmp2)), f(tmp3))

        self.beta[:, s, t, :self.nt_states] = tmp
      log_Z = self.beta[:, 0, n, :self.nt_states] + root_scores
      log_Z = self.logsumexp(log_Z, 1)
    return log_Z

  def _viterbi(self, unary_scores, rule_scores, root_scores):
    #unary scores : b x n x T
    #rule scores : b x NT x (NT+T) x (NT+T)
    
    batch_size = unary_scores.size(0)
    n = unary_scores.size(1)
    
    # dummy rules
    rule_scores = torch.cat([rule_scores, \
                             rule_scores.new(batch_size, self.t_states, self.states, self.states) \
                             .fill_(-self.huge)], dim=1)
    
    self.scores = unary_scores.new(batch_size, n+1, n+1, self.states).fill_(-self.huge)
    self.bp = unary_scores.new(batch_size, n+1, n+1, self.states).fill_(-1)
    self.left_bp = unary_scores.new(batch_size, n+1, n+1, self.states).fill_(-1)
    self.right_bp = unary_scores.new(batch_size, n+1, n+1, self.states).fill_(-1)
    self.argmax = unary_scores.new(batch_size, n, n).fill_(-1)
    self.argmax_tags = unary_scores.new(batch_size, n).fill_(-1)
    self.spans = [[] for _ in range(batch_size)]
    
    for k in range(n):
      for state in range(self.t_states):
        self.scores[:, k, k + 1, self.nt_states + state] = unary_scores[:, k, state]   

    for w in np.arange(2, n+1):
      for s in range(n-w+1):
        t = s + w
        
        tmp = self.scores[:, s, s+1:t, :].unsqueeze(3).unsqueeze(1) \
            + self.scores[:, s+1:t, t, :].unsqueeze(1).unsqueeze(3) \
            + rule_scores.unsqueeze(2)

        # view once and marginalize    
        tmp, max_pos = torch.max(tmp.view(batch_size, self.states, -1), dim=2)

        # step by step marginalization
        # tmp = self.logsumexp(tmp, dim=4)
        # tmp = self.logsumexp(tmp, dim=3)
        # tmp = self.logsumexp(tmp, dim=2)

        max_idx = max_pos / (self.states * self.states) + s + 1
        left_child = (max_pos % (self.states * self.states)) / self.states
        right_child = max_pos % self.states
        
        self.scores[:, s, t, :self.nt_states] = tmp[:, :self.nt_states]
        
        self.bp[:, s, t, :self.nt_states] = max_idx[:, :self.nt_states]         
        self.left_bp[:, s, t, :self.nt_states] = left_child[:, :self.nt_states]
        self.right_bp[:, s, t, :self.nt_states] = right_child[:, :self.nt_states]
    max_score = self.scores[:, 0, n, :self.nt_states] + root_scores
    max_score, max_idx = torch.max(max_score, 1)
    for b in range(batch_size):
      self._backtrack(b, 0, n, max_idx[b].item())      
    return self.scores, self.argmax, self.spans

  def _backtrack(self, b, s, t, state):
    u = int(self.bp[b][s][t][state])
    assert(s < t), "s: %d, t %d"%(s, t)
    left_state = int(self.left_bp[b][s][t][state])
    right_state = int(self.right_bp[b][s][t][state])
    self.argmax[b][s][t-1] = 1
    if s == t-1:
      self.spans[b].insert(0, (s, t-1, state))
      self.argmax_tags[b][s] = state - self.nt_states
      return None      
    else:
      self.spans[b].insert(0, (s, t-1, state))
      self._backtrack(b, s, u, left_state)
      self._backtrack(b, u, t, right_state)
    return None  
