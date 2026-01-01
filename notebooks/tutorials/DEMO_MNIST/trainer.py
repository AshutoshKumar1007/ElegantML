#trainer.py
import numpy as np
from tqdm import tqdm

class Trainer:
    def __init__(self,model,criterion,optim,regularizer,config):
        self.model = model
        self.criterion = criterion
        self.optim = optim
        self.regularizer = regularizer
        self.batch_size = config.BATCH_SIZE
        self.epoches = config.EPOCHES
        self.lr = config.LEARLING_RATE
        self.flag = True
    def train(self,xtrain,ytrain):
        num_samples = xtrain.shape[0]
        train_losses = []
        for epoch in tqdm(range(1, self.epochs + 1), desc="Training"):
            self.model.train()
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            xtrain_shuffled = xtrain[indices]
            ytrain_shuffled = ytrain[indices]
            epoch_loss = 0

            for i in range(0, num_samples, self.batch_size):
                x_batch = xtrain_shuffled[i:i + self.batch_size]
                y_batch = ytrain_shuffled[i:i + self.batch_size]

                y_pred = self.model(x_batch)
                loss = self.loss_fn(y_pred, y_batch) + self.model.regulizer()
                dAl = self.loss_fn.backward()
                self.model.backprop(dAl)

                self.optimizer.lr = self.lr if self.flag else self.lr * 0.1
                self.optimizer.step()
                epoch_loss += loss

            avg_loss = epoch_loss / (num_samples // self.batch_size)
            train_losses.append(avg_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

            if avg_loss < 0.006 and self.flag:
                self.model.save('model_checkpoint0.plk')
                print("Model checkpoint saved.")
                self.flag = False
        
        return train_losses
