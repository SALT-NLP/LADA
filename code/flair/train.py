from flair.data import Corpus
from flair.datasets import CONLL_03
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings
from typing import List
import argparse
parser = argparse.ArgumentParser(description='PyTorch mixNER')

parser.add_argument("--use-knn-train-data", action="store_true", help="indicates we will use the knn training data")
parser.add_argument('--num-knn-k', default=5, type=int, help='top k we use in the knn')
parser.add_argument('--knn-mix-ratio', default=0.5, type=float, help='the ratio of the mix sample and the main sample are from the knn, otherwise we randomly sample one to mix')
parser.add_argument('--train-examples', default = -1, type = int)
parser.add_argument('--mix-layer', default = 0, type=int, help='0: mix at layer 0, before lstm, 1: mix at layer 1, after lstm, 2: mix at both layer 1/0, randomly')  
parser.add_argument('--mini-batch-size', default = 2, type=int)  

parser.add_argument('--mix-option', action='store_true',help='mix option')
parser.add_argument('--alpha', default=60, type=float)
parser.add_argument('--beta', default=1.5, type=float)

parser.add_argument("--exp-save-name", default = 'example-ner', type = str, required = True)
parser.add_argument('--train-epochs', default = 40, type = int)

parser.add_argument('--patience', default = 3, type = int)

parser.add_argument("--use-crf", action="store_true", help="use crf layer")

parser.add_argument("--debug", action="store_true", help="debug")

parser.add_argument("--log-file", default = "./code/flair/resources/tasks/conll_03/test_results.csv", type = str,help="the file to store resutls")

args = parser.parse_args()

# 1. get the corpus
corpus: Corpus = CONLL_03(base_path='./code/flair/resources/tasks',tag_to_bioes='ner')

# subsampling the corpus
corpus.train.sentences=corpus.train.sentences[0:args.train_examples]
corpus.train.total_sentence_count=args.train_examples

# knn file
args.knn_idx_file='./code/flair/resources/tasks/conll_03/sent_id_knn{}.pkl'.format(args.train_examples)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    # GloVe embeddings
    WordEmbeddings('glove'),

    # contextual string embeddings, forward
    PooledFlairEmbeddings('news-forward', pooling='min'),

    # contextual string embeddings, backward
    PooledFlairEmbeddings('news-backward', pooling='min'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        use_crf=args.use_crf,
                                        tag_type=tag_type,
                                        debug=args.debug)

# initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)
folder='./code/flair/resources/taggers/{}'.format(args.exp_save_name)
results = trainer.train(folder,
              train_with_dev=False,
              max_epochs=args.train_epochs,
              mini_batch_size=args.mini_batch_size,
              patience=args.patience,
              args=args)

