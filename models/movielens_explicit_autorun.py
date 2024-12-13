from pymongo import MongoClient
import numpy as np
import pandas as pd
import torch
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import rmse_score

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://ghitaoudrhiri02:ghitaoudrhiri02@cluster0.fllnb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
   client.admin.command('ping')
   # print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
   print(e)

db =  client["AGILE"]
collection = db["ratings"]

# Retrieve all documents from the collection
data = list(collection.find({}))

# Convert the data to a pandas DataFrame
dataset = pd.DataFrame(data)

# Drop the MongoDB-specific _id field (optional)
if '_id' in dataset.columns:
    dataset = dataset.drop('_id', axis=1)

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