#evaluator.py
import numpy as np

class Evaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, data, labels):
        correct = 0
        for i in range(len(data)):
            self.model.eval()
            x = data[i].reshape(1, -1)
            y = labels[i]
            y_pred = self.model(x)
            if y_pred.argmax() == y.argmax():
                correct += 1
        return correct / len(data)