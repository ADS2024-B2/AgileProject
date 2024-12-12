from pymongo import MongoClient
import numpy as np
import pandas as pd
import torch
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import rmse_score

# Replace with the correct MongoDB URI
mongo_uri = "mongodb://localhost:27017"  # For host-based MongoDB
# If MongoDB is in a Docker container, use "mongodb" as the host, or use the Docker network name

client = MongoClient(mongo_uri)
db = client['your_database']  # Replace with your database name
collection = db['your_collection']  # Replace with your collection name

# Find all documents in the collection
documents = collection.find()

# Get user ratings matrix
dataset = documents['????']

model = ExplicitFactorizationModel(loss='regression',
                                   embedding_dim=128,  # latent dimensionality
                                   n_iter=10,  # number of epochs of training
                                   batch_size=1024,  # minibatch size
                                   l2=1e-9,  # strength of L2 regularization
                                   learning_rate=1e-3
                                #    ,use_cuda=torch.cuda.is_available()
                                   )

train, test = random_train_test_split(dataset, random_state=np.random.RandomState(42))

model.fit(train, verbose=False)

# If we were to log each model retraining metrics:
# train_rmse = rmse_score(model, train)
# test_rmse = rmse_score(model, test)
# print('Train RMSE {:.3f}, test RMSE {:.3f}'.format(train_rmse, test_rmse))

torch.save(model, 'spotlight_explicit_model.pth')