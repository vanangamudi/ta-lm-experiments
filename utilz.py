import os
import re
import sys
import glob
from pprint import pprint, pformat

import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np

from functools import partial
from collections import namedtuple, defaultdict, Counter


from anikattu.tokenizer import word_tokenize
from anikattu.tokenstring import TokenString
from anikattu.trainer.lm import Trainer , Tester, Predictor
from anikattu.datafeed import DataFeed, MultiplexedDataFeed
from anikattu.dataset import NLPDataset as Dataset, NLPDatasetList as DatasetList
from anikattu.utilz import tqdm, ListTable
from anikattu.vocab import Vocab
from anikattu.utilz import Var, LongVar, init_hidden, pad_seq
from nltk.tokenize import WordPunctTokenizer
word_punct_tokenizer = WordPunctTokenizer()
word_tokenize = word_punct_tokenizer.tokenize

from tace16.tace16 import tace16_to_utf8, utf8_to_tace16

VOCAB =  ['PAD', 'UNK', 'GO', 'EOS']
PAD = VOCAB.index('PAD')

"""
    Local Utilities, Helper Functions

"""
Sample   =  namedtuple('Sample', ['id', 'sequence'])

def __repr__(self):
    return '<{}:{}, {}>'.format(
        self.__class__.__name__,
        self.id,
        tace16_to_utf8([int(i) for i in self.sequence])
    )
     

Sample.__repr__ = __repr__

def unicodeToAscii(s):
    import unicodedata
    return ''.join(
        c for c in unicodedata.normalize('NFKC', s)
        if unicodedata.category(c) != 'Mn'
    )

def load_tawiki_data(config, dataset_name='tawiki', max_sample_size=None):
    samples = []
    skipped = 0

    vocab = Counter()
    
    try:
        filename = glob.glob('../dataset/tawiki_lines.txt')[0]
              
        log.info('processing file: {}'.format(filename))
        dataset = open(filename).readlines()
        for i, line in enumerate(tqdm(dataset, desc='processing {}'.format(filename))):
            import string

            #print(line)
            try:
                line = line.strip()
                
                if len(line) > 20:

                    for j, segment in enumerate(line.split('. ')):
                        if len(segment) < 20:
                            continue
                        
                        samples.append(
                            Sample(
                                id = '{}.{}.{}'.format(dataset_name, i ,j),
                                sequence = [str(i) for i in utf8_to_tace16(segment)]
                            )
                        )
                    """
                    samples.append(
                            Sample(
                                id = '{}.{}'.format(dataset_name, i),
                                sequence = [str(i) for i in utf8_to_tace16(line)]
                            )
                        )
                    """
            except:
                log.exception('{}.{}.{} -  {}'.format(dataset_name, i, j, word))
    except:
        skipped += 1
        log.exception('{}.{} -  {}'.format(dataset_name, i, line))

    print('skipped {} samples'.format(skipped))
    
    samples = sorted(samples, key=lambda x: len(x.sequence), reverse=True)
    if max_sample_size:
        samples = samples[:max_sample_size]

    log.info('building vocab...')
    for sample in samples:
        vocab.update(sample.sequence)

    return os.path.basename(filename), samples, vocab



def load_data(config, max_sample_size=None):
    dataset = {}
    filename, train_samples, vocab = load_tawiki_data(config)
    vocab = Vocab(vocab, special_tokens=VOCAB)

    pivot = int(config.CONFIG.split_ratio * len(train_samples))
    
    dataset[filename] = Dataset(filename, (train_samples[:pivot], train_samples[pivot:]), vocab, vocab)

    return DatasetList('ta-lm', dataset.values())
        

# ## Loss and accuracy function
def loss(ti, output, batch, loss_function, *args, **kwargs):
    indices, (sequence, ), _ = batch
    output, state = output
    return loss_function(output, sequence[:, ti+1]) 


def accuracy(ti, output, batch, *args, **kwargs):
    indices, (sequence, ), _ = batch
    output, state = output
    return (output.max(dim=1)[1] == sequence[:, ti+1]).sum().float()/float(answer.size(0))


def repr_function(output, batch, VOCAB, dataset):
    indices, (sequence, ), _ = batch
    results = []
    output = output.max(1)[1]
    results = []
    for idx, o in zip(indices, output):
        op = ' '.join([VOCAB[i] for i in o])
        results.append(op)
    return results


def batchop(datapoints, VOCAB, *args, **kwargs):
    indices = [d.id for d in datapoints]
    sequence = []
    for d in datapoints:
        s = []
        sequence.append([VOCAB[w] for w in d.sequence])

    sequence    = LongVar(pad_seq(sequence))
    batch = indices, (sequence, ), ()
    return batch


def portion(dataset, percent):
    return dataset[ : int(len(dataset) * percent) ]


def train(config, argv, name, ROOT_DIR,  model, dataset):
    _batchop = partial(batchop, VOCAB=dataset.input_vocab)
    predictor_feed = DataFeed(name, dataset.testset, batchop=_batchop, batch_size=1)
    train_feed     = DataFeed(name,
                              portion(dataset.trainset, config.HPCONFIG.trainset_size),
                              batchop=_batchop,
                              batch_size=config.CONFIG.batch_size)
    
    predictor = Predictor(name,
                          model=model,
                          directory=ROOT_DIR,
                          feed=predictor_feed,
                          repr_function=partial(repr_function
                                                , VOCAB=dataset.input_vocab
                                                , dataset=dataset.testset_dict))

    loss_ = partial(loss, loss_function=nn.NLLLoss())
    test_feed, tester = {}, {}
    
    def acc(*args, **kwargs):
        return -1 * loss_(*args, **kwargs)
    
    for subset in dataset.datasets:
        test_feed[subset.name]      = DataFeed(subset.name,
                                               subset.testset,
                                               batchop=_batchop,
                                               batch_size=config.CONFIG.batch_size)

        tester[subset.name] = Tester(name     = subset.name,
                                     config   = config,
                                     model    = model,
                                     directory = ROOT_DIR,
                                     loss_function = loss_,
                                     accuracy_function = acc,
                                     feed = test_feed[subset.name],
                                     save_model_weights=False)

    test_feed[name]      = DataFeed(name,
                                    dataset.testset,
                                    batchop=_batchop,
                                    batch_size=config.CONFIG.batch_size)

    tester[name] = Tester(name  = name,
                                  config   = config,
                                  model    = model,
                                  directory = ROOT_DIR,
                                  loss_function = loss_,
                                  accuracy_function = loss_,
                                  feed = test_feed[name],
                                  predictor=predictor)


    def do_every_checkpoint(epoch):
        if config.CONFIG.plot_metrics:
            from matplotlib import pyplot as plt
            fig = plt.figure(figsize=(10, 5))
            
        for t in tester.values():
            t.do_every_checkpoint(epoch)

            if config.CONFIG.plot_metrics:
                plt.plot(list(t.loss), label=t.name)

        if config.CONFIG.plot_metrics:
            plt.savefig('loss.png')
            plt.close()
        


    trainer = Trainer(name=name,
                      config = config,
                      model=model,
                      directory=ROOT_DIR,
                      optimizer  = optim.Adam(model.parameters()),
                      loss_function = loss_,
                      checkpoint = config.CONFIG.CHECKPOINT,
                      do_every_checkpoint = do_every_checkpoint,
                      epochs = config.CONFIG.EPOCHS,
                      feed = train_feed,
    )



    for e in range(config.CONFIG.EONS):

        if not trainer.train():
            raise Exception

        dump = open('{}/results/eon_{}.csv'.format(ROOT_DIR, e), 'w')
        log.info('on {}th eon'.format(e))
        results = ListTable()
        for ri in tqdm(range(predictor_feed.num_batch), desc='running prediction on eon: {}'.format(e)):
            output, _results = predictor.predict(ri)
            results.extend(_results)
        dump.write(repr(results))
        dump.close()

        

