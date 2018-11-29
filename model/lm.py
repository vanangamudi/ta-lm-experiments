import os
import re
import sys
import json
import time
import random
from pprint import pprint, pformat

sys.path.append('..')
import config
from anikattu.logger import CMDFilter
import logging
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(name)s.%(funcName)s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from functools import partial


import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable

from anikattu.utilz import Var, LongVar, init_hidden

class Base(nn.Module):
    def __init__(self, config, name):
        super(Base, self).__init__()
        self._name = name
        self.log = logging.getLogger(self._name)
        size_log_name = '{}.{}'.format(self._name, 'size')
        self.log.info('constructing logger: {}'.format(size_log_name))
        self.size_log = logging.getLogger(size_log_name)
        self.size_log.info('size_log')
        self.log.setLevel(config.CONFIG.LOG.MODEL.level)
        self.size_log.setLevel(config.CONFIG.LOG.MODEL.level)
        self.print_instance = 0
        
    def __(self, tensor, name='', print_instance=False):
        if isinstance(tensor, list) or isinstance(tensor, tuple):
            for i in range(len(tensor)):
                self.__(tensor[i], '{}[{}]'.format(name, i))
        else:
            self.size_log.debug('{} -> {}'.format(name, tensor.size()))
            if self.print_instance or print_instance:
                self.size_log.debug(tensor)

            
        return tensor

    def name(self, n):
        return '{}.{}'.format(self._name, n)

class LM(Base):
    def __init__(self, config, name, vocab_size, loss_function):
        super().__init__(config, name)
        self.config = config
        self.embed_size = config.HPCONFIG.embed_size
        self.hidden_size = config.HPCONFIG.hidden_size
        self.vocab_size = vocab_size
        self.loss_function = loss_function
        
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        
        self.lm  = nn.GRUCell(self.embed.embedding_dim, self.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.answer = nn.Linear(self.hidden_size, self.vocab_size)

        if config.CONFIG.cuda:
             self.cuda()

    def initial_hidden(self, batch_size):
        ret = Variable(torch.zeros( batch_size, self.lm.hidden_size))
        ret = ret.cuda() if self.config.CONFIG.cuda else ret
        return ret
    
    def forward(self, input_, state):
        input_emb  = self.__( self.embed(input_),  'input_emb')
        state = self.lm(input_emb, state)
        return F.log_softmax(self.answer(state), dim=-1), state
        
        
