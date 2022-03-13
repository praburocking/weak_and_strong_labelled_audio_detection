from torch.utils.data import DataLoader
import os
import pandas as pd
import torch
from getting_and_init_the_data import get_dataset, get_data_loader
import numpy as np
from models.crnn_model import CRNNModel
from copy import deepcopy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

root_dir = '../data/audioset_strong/audioset_strong/'
train_dir = os.path.normpath(root_dir + "bal_train/")
test_dir = os.path.normpath(root_dir + "test/")
val_dir = os.path.normpath(root_dir + "eval/")


def data_loading(batch=10, dataset_type="Strong"):
    root_dir = '../data/audioset_weak/audioset_weak/' if dataset_type == "Weak" else '../data/audioset_strong/audioset_strong/'
    train_dir = os.path.normpath(root_dir + "bal_train/")
    test_dir = os.path.normpath(root_dir + "test/")
    val_dir = os.path.normpath(root_dir + "eval/")
    train_loader = get_data_loader(get_dataset(train_dir, key_class='labels', dataset_type=dataset_type), batch, True, True)
    test_loader = get_data_loader(get_dataset(test_dir, key_class='labels', dataset_type=dataset_type), batch, True, True)
    val_loader = get_data_loader(get_dataset(val_dir, key_class='labels', dataset_type=dataset_type), batch, True, True)
    return {"train": train_loader, "test": test_loader, "val": val_loader}


def training(model, epoch, train_loader, device, loss_function, optimizer, val_loader, patience,dataset_type="Strong"):
    best_model = None
    lowest_val_loss = np.inf
    best_val_epoch = 0
    patience_counter = 0
    for j in range(epoch):
        train_loss = []
        val_loss = []
        model.train()
        for i, (features, label) in enumerate(train_loader):
            features = features.to(device=device, dtype=torch.double)
            label = label.to(device=device, dtype=torch.double)

            score = model(features)
            # print(score.shape)
            # print(label.shape)
            # print(label)
            # score=torch.reshape(score,(-1,))
            loss = loss_function(score, label)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            train_loss.append(loss)

        model.eval()
        with torch.no_grad():
            for i, (features, label) in enumerate(val_loader):
                features = features.to(device=device, dtype=torch.double)
                label = label.to(device=device, dtype=torch.double)

                score = model(features)
                loss = loss_function(score, label)
                val_loss.append(loss)

        train_mean = torch.mean(torch.tensor(train_loss))
        val_mean = torch.mean(torch.tensor(val_loss))
        print("epoch ------ (" + str(j) + "/" + str(epoch - 1) + ") training loss---- " + str(
            train_mean) + "validation loss---- " + str(val_mean))
        if lowest_val_loss > val_mean:

            lowest_val_loss = val_mean
            print("new lowest val " + str(lowest_val_loss))
            best_val_epoch = j
            best_model = deepcopy(model.state_dict())
            # torch.save(best_model, save_model_file)
            patience_counter = 0
        else:
            if patience < patience_counter:
                print("breaking the epoch loop as there is no improvement in the loss")
                break
            else:
                patience_counter = patience_counter + 1
    print("best epoch--- " + str(best_val_epoch) + "lowest loss function " + str(lowest_val_loss))
    return best_model


def testing(mod, model_best_stats, test_loader, device, loss_function, dataset_type="Strong"):
    cur_label = np.zeros((0, 10)) if dataset_type == "Weak" else np.zeros((0, 10, 10))
    cur_output = np.zeros((0, 10)) if dataset_type == "Weak" else np.zeros((0, 10, 10))
    with torch.no_grad():
        test_loss = []
        correct = None
        # mod = model().to(device=device, dtype=torch.double)
        mod.load_state_dict(model_best_stats)
        temp_correct = 0
        total_iter_counter = 0
        for i, (features, label) in enumerate(test_loader):
            features = features.to(device=device, dtype=torch.double)
            label = label.to(device=device, dtype=torch.double)
            score = mod(features)
           # print(label.shape)
            #print(score.shape)
            loss = loss_function(score, label)
            test_loss.append(loss)
            output = (score > 0.5).float()
            cur_label = np.append(cur_label, label.cpu().numpy(), axis=0)
            cur_output = np.append(cur_output, output.cpu().numpy(), axis=0)
            #print("output shape " + str(output.shape))
            temp_correct = temp_correct + (output == label).double().sum() / len(label)
            total_iter_counter = i
        # cur_label = torch.tensor(cur_label)
        # cur_output = torch.tensor(cur_output)
        correct = (cur_output == cur_label).sum() / len(cur_label)
        test_loss = torch.mean(torch.tensor(test_loss))
        print("total  test loss ------ " + str(test_loss) + " Test accuracy ---" + str(correct))
        print("accuracy from temp_correct " + str(temp_correct / total_iter_counter))
        print("cur_output shape " + str(cur_output.shape))
        return cur_output,correct


def __main__():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')
    labels = ["Music", "Speech", "Vehicle", "Car", "Animal", "Engine", "Boat,Water Vehicle", "Train", "Siren", "Dog"]
    batch = 20
    channel = 1
    seconds = 10
    feature_len = 128
    dataset_types = ["Weak","Strong"]
    accuracies={}
    for dataset_type in dataset_types:
        in_dim = (batch, channel, seconds, feature_len)
        out_dim = (30, 1, 10, 10)
        cnn_kernels_size = ((1, 3))
        pool_kernels_size = ((1, 5))
        train_loader, test_loader, val_loader = data_loading(batch=batch, dataset_type=dataset_type).values()

        crnn_model = CRNNModel(dataset_type=dataset_type).to(device=device, dtype=torch.double)
        optimizer = torch.optim.Adam(params=crnn_model.parameters(), lr=0.001)
        loss_function = torch.nn.BCEWithLogitsLoss()

        epoch = 200
        patience = 30
        model_best_stats = training(crnn_model, epoch, train_loader, device, loss_function, optimizer, val_loader,
                                    patience,dataset_type=dataset_type)
        torch.save(model_best_stats, dataset_type+'_best_model.mdl')
        #model_best_stats=torch.load(dataset_type+'_best_model.mdl')
        crnn_model = CRNNModel(dataset_type=dataset_type).to(device=device, dtype=torch.double)
        if model_best_stats is not None:
            _,accuracy=testing(crnn_model, model_best_stats, test_loader, device, loss_function, dataset_type=dataset_type)
            accuracies[dataset_type]=accuracy
    for i in accuracies.items():
        print("accuracy for the dataset "+str(i[0])+" --->"+str(i[1]))


if __name__ == "__main__":
    __main__()
