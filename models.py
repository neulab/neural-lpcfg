import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PCFG import PCFG
from lexicalizedPCFG import LexicalizedPCFG
from random import shuffle
from flow import FlowWordEmission
import pdb

class ResidualLayer(nn.Module):
  def __init__(self, in_dim = 100,
               out_dim = 100):
    super(ResidualLayer, self).__init__()
    self.lin1 = nn.Linear(in_dim, out_dim)
    self.lin2 = nn.Linear(out_dim, out_dim)

  def forward(self, x):
    return F.relu(self.lin2(F.relu(self.lin1(x)))) + x

class CompPCFG(nn.Module):
  def __init__(self, vocab = 100,
               h_dim = 512, 
               w_dim = 512,
               z_dim = 64,
               state_dim = 256, 
               t_states = 10,
               nt_states = 10,
               **kwargs):
    super(CompPCFG, self).__init__()
    self.state_dim = state_dim
    self.t_emb = nn.Parameter(torch.randn(t_states, state_dim))
    self.nt_emb = nn.Parameter(torch.randn(nt_states, state_dim))
    self.root_emb = nn.Parameter(torch.randn(1, state_dim))
    self.pcfg = PCFG(nt_states, t_states)
    self.nt_states = nt_states
    self.t_states = t_states
    self.all_states = nt_states + t_states
    self.dim = state_dim
    self.register_parameter('t_emb', self.t_emb)
    self.register_parameter('nt_emb', self.nt_emb)
    self.register_parameter('root_emb', self.root_emb)
    self.rule_mlp = nn.Linear(state_dim+z_dim, self.all_states**2)
    self.root_mlp = nn.Sequential(nn.Linear(z_dim + state_dim, state_dim),
                                  ResidualLayer(state_dim, state_dim),
                                  ResidualLayer(state_dim, state_dim),                         
                                  nn.Linear(state_dim, self.nt_states))
    if z_dim > 0:
      self.enc_emb = nn.Embedding(vocab, w_dim)
      self.enc_rnn = nn.LSTM(w_dim, h_dim, bidirectional=True, num_layers = 1, batch_first = True)
      self.enc_params = nn.Linear(h_dim*2, z_dim*2)
    self.z_dim = z_dim
    self.vocab_mlp = nn.Sequential(nn.Linear(z_dim + state_dim, state_dim),
                                   ResidualLayer(state_dim, state_dim),
                                   ResidualLayer(state_dim, state_dim),
                                   nn.Linear(state_dim, vocab))
      
  def enc(self, x):
    emb = self.enc_emb(x)
    h, _ = self.enc_rnn(emb)    
    params = self.enc_params(h.max(1)[0])
    mean = params[:, :self.z_dim]
    logvar = params[:, self.z_dim:]    
    return mean, logvar

  def kl(self, mean, logvar):
    result =  -0.5 * (logvar - torch.pow(mean, 2)- torch.exp(logvar) + 1)
    return result

  def forward(self, x, argmax=False, use_mean=False, **kwargs):
    #x : batch x n
    n = x.size(1)
    batch_size = x.size(0)
    if self.z_dim > 0:
      mean, logvar = self.enc(x)
      kl = self.kl(mean, logvar).sum(1) 
      z = mean.new(batch_size, mean.size(1)).normal_(0, 1)
      z = (0.5*logvar).exp()*z + mean    
      kl = self.kl(mean, logvar).sum(1) 
      if use_mean:
        z = mean
      self.z = z
    else:
      self.z = torch.zeros(batch_size, 1).cuda()

    t_emb = self.t_emb
    nt_emb = self.nt_emb
    root_emb = self.root_emb

    root_emb = root_emb.expand(batch_size, self.state_dim)
    t_emb = t_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, n, self.t_states, self.state_dim)
    nt_emb = nt_emb.unsqueeze(0).expand(batch_size, self.nt_states, self.state_dim)

    if self.z_dim > 0:
      root_emb = torch.cat([root_emb, z], 1)
      z_expand = z.unsqueeze(1).expand(batch_size, n, self.z_dim)
      z_expand = z_expand.unsqueeze(2).expand(batch_size, n, self.t_states, self.z_dim)
      t_emb = torch.cat([t_emb, z_expand], 3)
      nt_emb = torch.cat([nt_emb, z.unsqueeze(1).expand(batch_size, self.nt_states, 
                                                         self.z_dim)], 2)
    root_scores = F.log_softmax(self.root_mlp(root_emb), 1)
    unary_scores = F.log_softmax(self.vocab_mlp(t_emb), 3)
    x_expand = x.unsqueeze(2).expand(batch_size, x.size(1), self.t_states).unsqueeze(3)
    unary = torch.gather(unary_scores, 3, x_expand).squeeze(3)
    rule_score = F.log_softmax(self.rule_mlp(nt_emb), 2) # nt x t**2
    rule_scores = rule_score.view(batch_size, self.nt_states, self.all_states, self.all_states)
    log_Z = self.pcfg._inside(unary, rule_scores, root_scores)
    if self.z_dim == 0:
      kl = torch.zeros_like(log_Z)
    if argmax:
      with torch.no_grad():
        max_score, binary_matrix, spans = self.pcfg._viterbi(unary, rule_scores, root_scores)
        self.tags = self.pcfg.argmax_tags
      return -log_Z, kl, binary_matrix, spans
    else:
      return -log_Z, kl

class LexicalizedCompPCFG(nn.Module):
  def __init__(self, vocab = 100,
               h_dim = 512, 
               w_dim = 512,
               z_dim = 64,
               state_dim = 256, 
               t_states = 10,
               nt_states = 10,
               scalar_dir_scores = False,
               seperate_nt_emb_for_emission = False,
               head_first = False,
               tie_word_emb = False,
               variant='IV',
               flow_word_emb=False,
               couple_layers=4,
               cell_layers=1,
               pretrained_word_emb=None,
               freeze_word_emb=False,
               nt_emission = False,
               supervised_signals=[]):
    super(LexicalizedCompPCFG, self).__init__()
    self.state_dim = state_dim
    self.t_emb = nn.Parameter(torch.randn(t_states, state_dim))
    self.nt_emb = nn.Parameter(torch.randn(nt_states, state_dim))
    self.root_emb = nn.Parameter(torch.randn(1, state_dim))
    self.pcfg = LexicalizedPCFG(nt_states, t_states, nt_emission=nt_emission, supervised_signals=supervised_signals)
    self.nt_states = nt_states
    self.t_states = t_states
    self.all_states = nt_states + t_states
    self.dim = state_dim
    self.register_parameter('t_emb', self.t_emb)
    self.register_parameter('nt_emb', self.nt_emb)
    if seperate_nt_emb_for_emission:
      self.nt_emb_emission = nn.Parameter(torch.randn(nt_states, state_dim))
      self.register_parameter('nt_emb_emission', self.nt_emb_emission)
    else:
      self.nt_emb_emission = None
    self.register_parameter('root_emb', self.root_emb)
    self.head_first = head_first
    self.variant = variant
    if(not head_first):
      self.rule_mlp = nn.Linear(state_dim+state_dim+z_dim, 2 * self.all_states**2)
    else:
      if self.variant == 'I':
        self.head_mlp = nn.Linear(state_dim+state_dim+z_dim, 2 * self.all_states)
        self.rule_mlp = nn.Linear(state_dim+state_dim+z_dim, self.all_states**2)
      elif self.variant == 'II':
        self.head_mlp = nn.Sequential(nn.Linear(state_dim+state_dim+z_dim, state_dim),
                                      ResidualLayer(state_dim, state_dim),
                                      nn.Linear(state_dim, 2 * self.all_states))
        self.rule_mlp = nn.Linear(state_dim+state_dim+z_dim, self.all_states**2)                              
      elif self.variant == 'III':
        self.head_mlp = nn.Sequential(nn.Linear(state_dim+state_dim+z_dim, state_dim),
                                      ResidualLayer(state_dim, state_dim),
                                      nn.Linear(state_dim, 2 * self.all_states))
        self.rule_mlp = nn.Linear(state_dim+z_dim, self.all_states**2)
      elif self.variant == 'IV':
        self.head_mlp = nn.Sequential(nn.Linear(state_dim+state_dim+z_dim, state_dim),
                                      ResidualLayer(state_dim, state_dim),
                                      nn.Linear(state_dim, 2 * self.all_states))
        self.left_rule_mlp = nn.Linear(state_dim+state_dim+z_dim, self.all_states**2) 
        self.right_rule_mlp = nn.Linear(state_dim+state_dim+z_dim, self.all_states**2) 
      else:
        raise NotImplementedError
    self.word_emb = nn.Embedding(vocab, state_dim)
    if not pretrained_word_emb is None:
      self.word_emb.load_state_dict({'weight':torch.from_numpy(pretrained_word_emb)})
    if freeze_word_emb:
      self.word_emb.weight.requires_grad = False
    self.root_mlp = nn.Sequential(nn.Linear(z_dim + state_dim, state_dim),
                                  ResidualLayer(state_dim, state_dim),
                                  ResidualLayer(state_dim, state_dim),                         
                                  nn.Linear(state_dim, self.nt_states))
    if z_dim > 0:
      self.enc_emb = nn.Embedding(vocab, w_dim)
      self.enc_rnn = nn.LSTM(w_dim, h_dim, bidirectional=True, num_layers = 1, batch_first = True)
      self.enc_params = nn.Linear(h_dim*2, z_dim*2)
    self.z_dim = z_dim
    self.flow_word_emb = flow_word_emb
    if self.flow_word_emb:
      self.vocab_mlp = nn.Sequential(nn.Linear(z_dim + state_dim, state_dim),
                                   ResidualLayer(state_dim, state_dim),
                                   ResidualLayer(state_dim, state_dim))
      self.emit_prob = FlowWordEmission(state_dim, vocab, couple_layers, cell_layers, state_dim)
      self.word_emb.weight.requires_grad = False
      if tie_word_emb:
        self.emit_prob.word_emb.weight = self.word_emb.weight   
    else:
      self.vocab_mlp = nn.Sequential(nn.Linear(z_dim + state_dim, state_dim),
                                   ResidualLayer(state_dim, state_dim),
                                   ResidualLayer(state_dim, state_dim))
      
      self.emit_prob = nn.Linear(state_dim, vocab)
      if tie_word_emb:
        self.emit_prob.weight = self.word_emb.weight   
    
    # if(scalar_dir_scores):
    #   self.scalar_dir_scores = nn.Parameter(torch.randn(nt_states, self.all_states, self.all_states))
    #   self.register_parameter('scalar_dir_scores', self.scalar_dir_scores)
    # else:
    #   self.dir_mlp = nn.Linear(state_dim+z_dim, self.all_states**2)
    #   self.scalar_dir_scores = None
      
  def enc(self, x):
    emb = self.enc_emb(x)
    h, _ = self.enc_rnn(emb)    
    params = self.enc_params(h.max(1)[0])
    mean = params[:, :self.z_dim]
    logvar = params[:, self.z_dim:]    
    return mean, logvar

  def kl(self, mean, logvar):
    result =  -0.5 * (logvar - torch.pow(mean, 2)- torch.exp(logvar) + 1)
    return result

  def forward(self, x, argmax=False, use_mean=False, gold_tree=None):
    #x : batch x n
    n = x.size(1)
    batch_size = x.size(0)
    if self.z_dim > 0:
      mean, logvar = self.enc(x)
      kl = self.kl(mean, logvar).sum(1) 
      z = mean.new(batch_size, mean.size(1)).normal_(0, 1)
      z = (0.5*logvar).exp()*z + mean    
      kl = self.kl(mean, logvar).sum(1) 
      if use_mean:
        z = mean
      self.z = z
    else:
      self.z = torch.zeros(batch_size, 1).cuda()

    t_emb = self.t_emb
    nt_emb = self.nt_emb
    nt_emb_emission = self.nt_emb_emission
    root_emb = self.root_emb

    root_emb = root_emb.expand(batch_size, self.state_dim)
    t_emb = t_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, n, self.t_states, self.state_dim)
    nt_emb = nt_emb.unsqueeze(0).expand(batch_size, self.nt_states, self.state_dim)
    nt_emb_emission = nt_emb_emission.unsqueeze(0).expand(batch_size, self.nt_states, self.state_dim) \
                                                        if not nt_emb_emission is None else None

    if self.z_dim > 0:
      root_emb = torch.cat([root_emb, z], 1)
      z_expand = z.unsqueeze(1).expand(batch_size, n, self.z_dim)
      z_expand = z_expand.unsqueeze(2).expand(batch_size, n, self.t_states, self.z_dim)
      t_emb = torch.cat([t_emb, z_expand], 3)
      nt_emb = torch.cat([nt_emb, z.unsqueeze(1).expand(batch_size, self.nt_states, 
                                                         self.z_dim)], 2)
      nt_emb_emission = torch.cat([nt_emb_emission, z.unsqueeze(1).expand(batch_size, self.nt_states, 
                                                         self.z_dim)], 2) \
                                                         if not nt_emb_emission is None else None  
    root_scores = F.log_softmax(self.root_mlp(root_emb), 1)
    if nt_emb_emission is None:
      T_emb = torch.cat([nt_emb.unsqueeze(1).expand(-1, n, -1, -1),
                        t_emb], dim=2)
    else:
      T_emb = torch.cat([nt_emb_emission.unsqueeze(1).expand(-1, n, -1, -1),
                        t_emb], dim=2)
    if(self.flow_word_emb):
      unary = self.emit_prob(self.vocab_mlp(T_emb), x)
    else:  
      unary_scores = F.log_softmax(self.emit_prob(self.vocab_mlp(T_emb)), 3)
      x_expand = x.unsqueeze(2).expand(batch_size, x.size(1), self.all_states).unsqueeze(3)
      unary = torch.gather(unary_scores, 3, x_expand).squeeze(3)
    unary = unary.refine_names('B', 'H', 'T')
    
    x_emb = self.word_emb(x)
    nt_x_emb = torch.cat([x_emb.unsqueeze(1).expand(-1, self.nt_states, -1, -1), 
                          nt_emb.unsqueeze(2).expand(-1, -1, n, -1)], dim=3)
    if(not self.head_first):
      rule_score = F.log_softmax(self.rule_mlp(nt_x_emb), 3) # nt x t**2
      rule_scores = rule_score.view(batch_size, self.nt_states, n, self.all_states, self.all_states, 2)
    else:
      if self.variant in ['I', 'II']:
        rule_score = self.rule_mlp(nt_x_emb) # nt x t**2
        rule_scores = rule_score.view(batch_size, self.nt_states, n, self.all_states, self.all_states)
        head_score = F.log_softmax(self.head_mlp(nt_x_emb), 3) # nt x t**2
        head_scores = head_score.view(batch_size, self.nt_states, n, self.all_states, 2)
        left_scores = F.log_softmax(rule_scores, dim=4).unsqueeze(-1)
        right_scores = F.log_softmax(rule_scores, dim=3).unsqueeze(-1)
        rule_scores = torch.cat([head_scores[:, :, :, :, 0:1].unsqueeze(4) + left_scores,
                                head_scores[:, :, :, :, 1:2].unsqueeze(3) + right_scores], dim=-1)
      elif self.variant == 'III':
        rule_score = self.rule_mlp(nt_emb.unsqueeze(2).expand(-1, -1, n, -1)) # nt x t**2
        rule_scores = rule_score.view(batch_size, self.nt_states, n, self.all_states, self.all_states)
        head_score = F.log_softmax(self.head_mlp(nt_x_emb), 3) # nt x t**2
        head_scores = head_score.view(batch_size, self.nt_states, n, self.all_states, 2)
        left_scores = F.log_softmax(rule_scores, dim=4).unsqueeze(-1)
        right_scores = F.log_softmax(rule_scores, dim=3).unsqueeze(-1)
        rule_scores = torch.cat([head_scores[:, :, :, :, 0:1].unsqueeze(4) + left_scores,
                                head_scores[:, :, :, :, 1:2].unsqueeze(3) + right_scores], dim=-1)
      elif self.variant == 'IV':
        left_rule_score = self.left_rule_mlp(nt_x_emb) # nt x t**2
        right_rule_score = self.right_rule_mlp(nt_x_emb) # nt x t**2
        left_rule_scores = left_rule_score.view(batch_size, self.nt_states, n, self.all_states, self.all_states)
        right_rule_scores = right_rule_score.view(batch_size, self.nt_states, n, self.all_states, self.all_states)
        head_score = F.log_softmax(self.head_mlp(nt_x_emb), 3) # nt x t**2
        head_scores = head_score.view(batch_size, self.nt_states, n, self.all_states, 2)
        left_scores = F.log_softmax(left_rule_scores, dim=4).unsqueeze(-1)
        right_scores = F.log_softmax(right_rule_scores, dim=3).unsqueeze(-1)
        rule_scores = torch.cat([head_scores[:, :, :, :, 0:1].unsqueeze(4) + left_scores,
                                head_scores[:, :, :, :, 1:2].unsqueeze(3) + right_scores], dim=-1)
      else:
        raise NotImplementedError

    # if self.scalar_dir_scores is None:
    #   dir_score = self.dir_mlp(nt_emb).view(batch_size, self.nt_states, self.all_states, self.all_states)
    # else:
    #   dir_score = self.scalar_dir_scores.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    # dir_scores = F.logsigmoid(torch.stack([dir_score, -dir_score]))
    
    rule_scores = rule_scores.refine_names('B', 'T', 'H', 'TL', 'TR', 'D')
    root_scores = root_scores.refine_names('B', 'T')
    # dir_scores = dir_scores.refine_names('D', 'B', 'T', 'TL', 'TR').align_to('D', 'B', 'T', 'H', 'TL', 'TR')
    log_Z = self.pcfg._inside(unary_scores = unary, 
                              rule_scores = rule_scores, 
                              root_scores = root_scores,
                              gold_tree=gold_tree)
    if self.z_dim == 0:
      kl = torch.zeros_like(log_Z)
    
    if(log_Z.sum().item() > -0.1):
      pdb.set_trace()

    if argmax:
      with torch.no_grad():
        max_score, binary_matrix, spans = self.pcfg._viterbi(unary_scores = unary, 
                                                             rule_scores = rule_scores, 
                                                             root_scores = root_scores)
        self.tags = self.pcfg.argmax_tags
      return -log_Z, kl, binary_matrix, spans
    else:
      return -log_Z, kl
