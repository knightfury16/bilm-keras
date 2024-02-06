from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Concatenate
from keras.optimizers import Adagrad
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
import h5py
import re

from data import Vocabulary, UnicodeCharsVocabulary, InvalidNumberOfCharacters



def build_model(options):
# Define word input
    word_input = Input(shape=(options['batch_size'], options['unroll_steps']))

    # Define word embedding
    word_embedding = Embedding(input_dim=options['n_tokens_vocab'], output_dim=options['lstm']['projection_dim'])(word_input)

    # Define char input
    char_input = Input(shape=(options['batch_size'], options['unroll_steps'], options['char_cnn']['max_characters_per_token']))

    # Define char embedding
    char_embedding = Embedding(input_dim=options['char_cnn']['n_characters'], output_dim=options['char_cnn']['embedding']['dim'])(char_input)

    # Flatten char embedding
    char_embedding_flattened = K.reshape((options['batch_size'], options['unroll_steps'], -1))(char_embedding)

    # Concatenate word and char embeddings
    combined_embedding = Concatenate(axis=-1)([word_embedding, char_embedding_flattened])

    # LSTM layer
    lstm = LSTM(options['lstm']['dim'], return_sequences=True)(combined_embedding)

    # Output layer
    output = Dense(options['n_tokens_vocab'], activation='softmax')(lstm)

    # Define model
    model = Model(inputs=[word_input, char_input], outputs=output)
    return model


# Compile the Keras model
def compile_model(model, options):
    optimizer = Adagrad(learning_rate=options.get('learning_rate', 0.2))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

# Define Keras callbacks
checkpoint_callback = ModelCheckpoint(filepath='model_checkpoint.h5', save_best_only=True)
tensorboard_callback = TensorBoard(log_dir='logs')




# Main training loop
def train_model(options, data, outfile):
    #Build model
    model = build_model(options)
    
    #Compile Model
    compile_model(model, options)
    
    #Train model
    model.fit(data,
              epochs=options['n_epochs'],
              steps_per_epoch=options['steps_per_epoch'],
              callbacks=[checkpoint_callback, tensorboard_callback])
    
    #After training dump weights
    # dump_weights(model, outfile)

def dump_weights(model, outfile):
    '''
    Dump the trained weights from a Keras model to a HDF5 file.
    '''
    def _get_outname(tf_name):
        outname = re.sub(':0$', '', tf_name)
        outname = outname.lstrip('lm/')
        outname = re.sub('/rnn/', '/RNN/', outname)
        outname = re.sub('/multi_rnn_cell/', '/MultiRNNCell/', outname)
        outname = re.sub('/cell_', '/Cell', outname)
        outname = re.sub('/lstm_cell/', '/LSTMCell/', outname)
        if '/RNN/' in outname:
            if 'projection' in outname:
                outname = re.sub('projection/kernel', 'W_P_0', outname)
            else:
                outname = re.sub('/kernel', '/W_0', outname)
                outname = re.sub('/bias', '/B', outname)
        return outname

    with h5py.File(outfile, 'w') as fout:
        for layer in model.layers:
            for v in layer.trainable_weights:
                if 'softmax' in v.name:
                    # don't dump these
                    continue
                outname = _get_outname(v.name)
                print("Saving variable {0} with name {1}".format(
                    v.name, outname))
                shape = v.shape
                dset = fout.create_dataset(outname, shape, dtype='float32')
                values = v.numpy()
                dset[...] = values


