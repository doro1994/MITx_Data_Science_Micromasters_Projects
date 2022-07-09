import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions
#
#input = torch.randn(32, 1, 42, 28)
#model = nn.Conv2d(1, 32, (3, 3))
#output = model(input)
#print(output.shape)
#model = nn.Conv2d(32, 32, (3, 3))
#output = model(output)
#print(output.shape)
#model = nn.ReLU()
#output = model(output)
#print(output.shape)
#model = nn.MaxPool2d((2, 2))
#output = model(output)
#print(output.shape)
#model = nn.Conv2d(32, 64, (3, 3))
#output = model(output)
#print(output.shape)
#model = nn.Conv2d(64, 64, (3, 3))
#output = model(output)
#print(output.shape)
#model = nn.ReLU()
#output = model(output)
#print(output.shape)
#model = nn.MaxPool2d((2, 2))
#output = model(output)
#print(output.shape)
#model = Flatten()
#output = model(output)
#print(output.shape)
#
#model = nn.Linear(2560, 256)
#output = model(output)
#print(output.shape)
#model = nn.Dropout(0.5)
#output = model(output)
#print(output.shape)
#model = nn.Linear(256, 10)
#output = model(output)
#print(output.shape)


class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        self.img_rows, self.img_cols = img_rows, img_cols
        self.fc1 = nn.Conv2d(1, 32, (3, 3))
        self.fc2 = nn.Conv2d(32, 32, (3, 3))
        self.fc3 = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(0.25)
        self.fc4 = nn.Conv2d(32, 64, (3, 3))
        self.fc5 = nn.Conv2d(64, 64, (3, 3))
        self.fc6 = nn.MaxPool2d((2, 2))
        self.flatten = Flatten()        
        self.hidden = nn.Linear(1792, 128)  
        self.output = nn.Linear(128, 20)
        

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.relu(self.hidden(x))
        x = self.dropout(x)
        x = self.output(x)
        
        out_first_digit = x[:, :10]
        out_second_digit = x[:, -10:]

        return out_first_digit, out_second_digit

def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = CNN(input_dimension) # TODO add proper layers to CNN class above

    # Train
    #train_model(train_batches, dev_batches, model)
    train_model(train_batches, dev_batches, model,  momentum=0.9, n_epochs=30)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
