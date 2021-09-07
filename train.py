import os
from argparse import ArgumentParser
import tensorflow as tf
from model.gru_rnn import GRU_RNN
from data import Dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == "__main__":
    tf.keras.backend.clear_session()
    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    home_dir = os.getcwd()

    # Arguments users used when running command lines
    parser.add_argument(
        "--model-folder", default='{}/tmp/model/'.format(home_dir), type=str)
    parser.add_argument(
        "--checkpoint-folder", default='{}/tmp/checkpoints/'.format(home_dir), type=str)

    parser.add_argument("--data-path", default='data/IMDB_Dataset.csv', type=str)
    parser.add_argument("--data-name", default='review', type=str)
    parser.add_argument("--label-name", default='sentiment', type=str)
    parser.add_argument(
        "--data-classes", default={'negative': 0, 'positive': 1}, type=set)
    parser.add_argument("--num-class", default=2, type=int)

    parser.add_argument("--model", default='gru', type=str)
    parser.add_argument("--units", default=128, type=int)
    parser.add_argument("--embedding-size", default=128, type=int)
    parser.add_argument("--vocab-size", default=10000, type=int)
    parser.add_argument("--max-length", default=256, type=int)
    
    parser.add_argument("--learning-rate", default=0.0008, type=float)
    parser.add_argument("--optimizer", default='rmsprop', type=str)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--buffer-size", default=128, type=int)
    parser.add_argument("--epochs", default=12, type=int)

    args = parser.parse_args()

    # Project Description
    print(' ')
    print('---------------------Welcome to GRU Team | TF03 | ProtonX-------------------')
    print('Github: joeeislovely | anhdungpro97 | ttduongtran')
    print('Email: nguyenminh.sangatpa@gmail.com | anhdung1951997@gmail.com | ttduongtran@gmail.com')
    print('---------------------------------------------------------------------')
    print(f'Training {args.model.upper()} model with hyper-params:')
    print('===========================')
    
    # print arguments
    for i, arg in enumerate(vars(args)):
      print('{}. {}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')
    
    # Prepair dataset
    dataset = Dataset(args.data_path, args.vocab_size, data_classes=args.data_classes)
    
    train_ds, val_ds = dataset.build_dataset(
        args.max_length, args.test_size, args.buffer_size, args.batch_size, args.data_name, args.label_name)

    sentences_tokenizer = dataset.sentences_tokenizer
    sentences_tokenizer_size = len(sentences_tokenizer.word_counts) + 1
    
    # Initializing variables
    input_length = args.max_length

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
                      sentences_tokenizer_size, input_length, num_class=args.num_class)
    
    
    # Set up loss function
    losses = tf.keras.losses.CategoricalCrossentropy(
        name="categorical_crossentropy")
        
    # Optimizer Definition
    if args.optimizer == 'rmsprop':
      optimizer = tf.keras.optimizers.RMSprop(
          learning_rate=args.learning_rate, name='rmsprop')
    else: 
      optimizer = tf.keras.optimizers.Adam(
          learning_rate=args.learning_rate, name='adam')

    # Compile optimizer and loss function into model
    metrics = ['accuracy', 'mse']
    model.compile(optimizer=optimizer,
                  loss=losses, metrics=metrics)
    
    # model.summary()

    # Callbacks: Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)

    # Callbacks: checkpoint training
    # include the epoch in the file name
    checkpoint_path = "tmp/checkpoints/cp-{epoch:04d}.ckpt/"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Save weights, every 4-epochs.
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True, save_freq=5)

    # Training model
    model.fit(train_ds, epochs=args.epochs,
                        batch_size=args.batch_size, 
                        validation_data=val_ds,
                        verbose=1, 
                        callbacks=[checkpoint])

    # Saving model
    model.save(f"{args.model_folder}/{args.model}.h5py")

    # Do Prediction


