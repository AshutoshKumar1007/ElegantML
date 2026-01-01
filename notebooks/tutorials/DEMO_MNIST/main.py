#main.py
import numpy as np
from ...__init__ import *
from config import *
from trainer import Trainer
from evaluator import Evaluator
from utils import load_images,load_labels

#Load the MNIST DATA
train_images,Mean,Std = load_images(os.path.join(CURR_DIR , r'\train-images-idx3-ubyte\train-images-idx3-ubyte'))
train_labels = load_labels(os.path.join(CURR_DIR  ,r'\train-labels-idx1-ubyte\train-labels-idx1-ubyte'))
xs = train_images
ys = train_labels
print(Mean.shape,Std.shape)

# Train-Test-Split
ratio = 1 - VALIDATION_SPLIT
n = xs.shape[0]
split_index = int(n*ratio)
xtrain,xval = xs[:split_index], xs[split_index:]
ytrain, yval = ys[:split_index], ys[split_index:]

# Model definition
model = Sequential([
    Linear(784, 512,False),
    BatchNorm(512),
    Relu(),
    Dropout(keep_prop=0.85),

    Linear(512, 256,False),
    BatchNorm(256),
    Relu(),
    Dropout(keep_prop=0.7),
    
    Linear(256, 128,False),
    BatchNorm(128),
    Relu(),
    Dropout(keep_prop=0.85),
    
    Linear(128, 10,False),
    BatchNorm(10),
    Softmax() 
])

total_params = sum(np.prod(param.value.shape)  for param in model.parameters())
print("Total number of parameters:", total_params)

#Train
trainer = Trainer(
    model = model,
    criterion= CrossEntropyLoss(),
    optim=Adam(model.parameters()),
    regularizer= L2regularizer(model),
    config= __import__('config')
)
losses = trainer.train(xtrain=xs,ytrain=ys)

#Evaluate
evaluator = Evaluator(model)
val_acc = evaluator.evaluate(xval,yval)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")