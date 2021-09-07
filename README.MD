# GRU

Library Logo <--- **FIXME**

<p align="center">
    <img src='https://storage.googleapis.com/protonx-cloud-storage/transformer/protonx-transf.png' width=200 class="center">
</p>

Compare and evaluate recurrent neural networks (RNN) with different types of recurrent units: 
- Gated recurrent unit (GRU)
- Long short-term memory (LSTM)
- Tanh

Slide about your project (if it's available) <--- **FIXME**

Architecture Image  <--- **FIXME**

Advisors:
- Github: [bangoc123](https://github.com/bangoc123)
- Email: protonxai@gmail.com

Authors:
| # | Name | Github | Email |
|---|------|--------|-------|
|1|Sang Nguyen|[joeeislovely](https://github.com/joeeislovely)|nguyenminh.sangatpa@gmail.com|
|2|Dung Nguyen|[anhdungpro97](https://github.com/anhdungpro97)|anhdung1951997@gmail.com|
|3|Duong Tran|[ttduongtran](https://github.com/ttduongtran)|ttduongtran@gmail.com|

## I.  Set up environment

- Step 1: Create the environment (Make sure you have installed [Miniconda](https://docs.conda.io/en/latest/miniconda.html))

```python
conda env create -f environment.yml
```

- Step 2: Active environment
```
conda activate gru
```

## II.  Set up your dataset

- Guide user how to download your data and set the data pipeline <--- **FIXME**
- References: [NLP](https://github.com/bangoc123/transformer) and [CV](https://github.com/bangoc123/mlp-mixer)

## III. Training Process

Training script:


```python

!python train.py --epochs ${epochs} 
                --model-folder ${model_folder}
                --checkpoint-folder ${checkpoint_folder}
                --data-path ${data_path}
                --data-name ${data_name}
                --label-name ${label_name}
                --data-classes ${data_classes}
                --num-class ${num_classes}
                --model ${model} 
                --units ${units}
                --embedding-size ${embedding_size}
                --vocab-size ${vocab_size}
                --max-length ${max_length}
                --learning-rate ${learning_rate}
                --optimizer ${optimizer}
                --test-size ${test_size}
                --batch-size ${batch_size}
                --buffer-size ${buffer_size}

```

Example for IMDB dataset:

```python
!python train.py --epochs 20 --model gru --optimizer rmsprop --units 128 --embedding-size 128 --vocab-size=10000 --max-length 256 --learning-rate 0.0008  --test-size 0.2 --batch-size 32 --buffer-size 128
```

Example others dataset:

```python
!python train.py --epochs 20 --model gru --learning-rate 0.0008 --optimizer rmsprop --model-folder /tmp/model/ --checkpoint-folder /tmp/checkpoints/ --data-path data/IMDB_Dataset.csv --data-name review  --label-name sentiment --data-classes {'negative': 0, 'positive': 1} --num-class 2 --units 128 --embedding-size 128 --vocab-size=10000 --max-length 256  --test-size 0.2 --batch-size 32 --buffer-size 128
```


There are some `important` arguments for the script you should consider when running it:
- `model-folder`: Directory model. (E.g. `tmp/model`)
- `checkpoint-folder`: Directory checkpoints. (E.g. `tmp/checkpoints`)
- `data-path`: The path of dataset (Must be csv format. E.g. `data/IMDB_Dataset.csv`)
- `data-name`: The folder of validation data
- `data-name`: Name of data column that having sentences will be train. (e.g. `review`)
- `label-name`:  Name of label column that having labels will be train. (e.g. `sentiment` )
- `data-classes`: Set of labels that you need to convert to categorical. (E.g. `{'negative': 0, 'positive': 1}`)
- `num-class`: Number of labels in your dataset. (e.g. `2` labels)
- `model`: Choose one model that you want to test including: `gru`, `lstm`, `tanh`. Default is `gru`
- `units`: Hidden dimension. Default is `128`
- `embedding-size`: Embedding dimension. Default is `128`
- `vocab-size`: Vocabulary size. Default is `10000`
- `max-length`: The maximum length of a sentence you want to keep when preprocessing. Default value `256`
- `learning-rate`: Learning rate. Default is `0.005`
- `optimizer`: Choose one optimizer that you want to apply, including: `rmsprop` and `adam`. Default is `rmsprop`
- `test-size`: Split to train (1-x) and test (x) dataset with ratio. Default is `0.2`
- `batch-size`: The batch size of the dataset. Default value `64`

## IV. Predict Process

```bash
python predict.py --test-data ${link_to_test_data}
```

## V. Result and Comparision

### 1. GRU
```
Epoch 7/10
625/625 [==============================] - 175s 280ms/step - loss: 0.3127 - accuracy: 0.8690 - mse: 0.0954 - val_loss: 0.2497 - val_accuracy: 0.8968 - val_mse: 0.0748
Epoch 8/10
625/625 [==============================] - 175s 281ms/step - loss: 0.2106 - accuracy: 0.9200 - mse: 0.0608 - val_loss: 0.2646 - val_accuracy: 0.8956 - val_mse: 0.0772
Epoch 9/10
625/625 [==============================] - 175s 280ms/step - loss: 0.1551 - accuracy: 0.9430 - mse: 0.0435 - val_loss: 0.3356 - val_accuracy: 0.8761 - val_mse: 0.0949
Epoch 10/10
625/625 [==============================] - 175s 280ms/step - loss: 0.1029 - accuracy: 0.9641 - mse: 0.0279 - val_loss: 0.3228 - val_accuracy: 0.8862 - val_mse: 0.0880
```

### 2. LSTM
```
Epoch 10/10 ... 
```
### 2. TANH
```
Epoch 10/10 ... 
```

Your comments about these results <--- **FIXME**


## VI. Running Test

When you want to modify the model, you need to run the test to make sure your change does not affect the whole system.

In the `./folder-name` **(FIXME)** folder please run:

```bash
pytest
```


