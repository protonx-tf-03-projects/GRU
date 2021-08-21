import os
from argparse import ArgumentParser
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import rmsprop
from model.gru_rnn import GRU_RNN

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    home_dir = os.getcwd()

    # FIXME
    # Arguments users used when running command lines
    parser.add_argument(
        "--data_folder", default='{}/data'.format(home_dir), type=str)
    parser.add_argument(
        "--model-folder", default='{}/model/'.format(home_dir), type=str)
    
    parser.add_argument("--model", default='gru', type=str)
    parser.add_argument("--units", default=128, type=int)
    parser.add_argument("--embedding-size", default=1000, type=int)
    parser.add_argument("--vocab-size", default=1000, type=int)
    parser.add_argument("--input-length", default=1000, type=int)
    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--optimizers", default='rmsprop', type=str)
    parser.add_argument("--loss-function",
                        default='binary_crossentropy', type=str)

    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=1000, type=int)

    args = parser.parse_args()

    # FIXME
    # Project Description

    print('---------------------Welcome to ${name}-------------------')
    print('Github: ${accout}')
    print('Email: ${email}')
    print('---------------------------------------------------------------------')
    print('Training ${name} model with hyper-params:') # FIXME
    print('===========================')
    
    # print arguments
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')
    
    # FIXME

    # Prepair dataset
    data_folder = args.data_folder

    # Split data
    train_ds = None # <-- fixme
    val_ds = None  # <-- fixme

    # Initializing model
    if args.model == 'lstm':
      # LSTM model
      # model = LSTM()
      pass
    elif args.model == 'tanh':
      # Tanh model
      # model = Tanh()
      pass 
    else:
      model = GRU_RNN(args.units, args.embedding_size,
                        args.vocab_size, args.input_length)

    # Set up loss function
    if args.loss_function == 'binary_crossentropy':
      loss_object = tf.keras.losses.BinaryCrossentropy(
          from_logits=False, label_smoothing=0, axis=-1, reduction="auto", name="binary_crossentropy")
    else:
      # Adam or whatever
      pass

    # Optimizer Definition
    if args.loss_function == 'rmsprop':
      optimizers = tf.keras.optimizers.RMSprop(
          learning_rate=args.learning_rate, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False, name=args.optimizers)
    else: 
      pass

    # Compile optimizer and loss function into model
    model.compile(optimizer=optimizers, loss=loss_object, metrics=['acc'])

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)

    # Training model
    model.fit(train_ds, epochs=args.epochs, batch_size=args.batch_size,
                validation_data=val_ds, callbacks=[early_stopping])

    # Saving model
    model.save(args.model_folder)

    # Do Prediction


