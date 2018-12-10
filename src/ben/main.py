import torch
import pandas as pd
import torch
from torchtext import data
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Convolution1D
from keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.layers import SpatialDropout1D
from keras.layers.recurrent import GRU

from src.loader.loader import load_data

print(torch.cuda.is_available())
d = load_data()
print(d)
