import numpy as np

from kerasTraining import train_model
from data import BidirectionalLMDataset, Vocabulary, UnicodeCharsVocabulary


def main():
    # load the vocab
    vocab = load_vocab(vocab_file, 50)
    
    # print("from vocab", vocab)

    # define the options
    batch_size = 128  # batch size for each GPU
    n_gpus = 3

    # number of tokens in training data (this for 1B Word Benchmark)
    n_train_tokens = 768648884

    options = {
     'bidirectional': True,

     'char_cnn': {'activation': 'relu',
      'embedding': {'dim': 16},
      'filters': [[1, 32],
       [2, 32],
       [3, 64],
       [4, 128],
       [5, 256],
       [6, 512],
       [7, 1024]],
      'max_characters_per_token': 50,
      'n_characters': 261,
      'n_highway': 2},
    
     'dropout': 0.1,
    
     'lstm': {
      'cell_clip': 3,
      'dim': 4096,
      'n_layers': 2,
      'proj_clip': 3,
      'projection_dim': 512,
      'use_skip_connections': True},
    
     'all_clip_norm_val': 10.0,
    
     'n_epochs': 10,
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': 20,
     'n_negative_samples_batch': 8192,
    }

    prefix = train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                      shuffle_on_load=True)

    tf_save_dir = outfile
    tf_log_dir = outfile
    # train(options, data, n_gpus, tf_save_dir, tf_log_dir)
    train_model(options,data, tf_save_dir)


def load_vocab(vocab_file, max_word_length=None):
    if max_word_length:
        return UnicodeCharsVocabulary(vocab_file, max_word_length,
                                      validate_file=True)
    else:
        return Vocabulary(vocab_file, validate_file=True)




if __name__ == '__main__':
    vocab_file = './fixtures/vocab.txt'
    outfile = 'weights.h5'
    train_prefix = './fixtures/data.txt'
    

    main()

