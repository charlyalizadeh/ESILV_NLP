import torch
import torch.nn as nn
import numpy as np
from ffnn import predict_ffnn


model = torch.load('./models/ffnn')
X_test = np.load('data/processed/test_fr_core_news_lg_fr_core_news_lg.npy')
y_test_pred = predict_ffnn(X_test, model)
y_test_pred = (y_test_pred.numpy() + 1).astype(np.int64)
print(y_test_pred)
np.savetxt('result_test.txt', y_test_pred, fmt='%u')
