import torch
import sys
import pickle as pkl
import numpy as np
import torch.nn as nn
import time
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import IntervalsDataset, pad_collate
from lstm_loupe import LSTMModel

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
        print("Cuda is available.")
        sys.stdout.flush()
    else:
        device = "cpu"

    (
        all_accelerations,
        all_mics,
        all_data_availabilities,
        all_subject_ids,
        all_neighbor_counts,
        labels_eff,
        labels_frust,
        labels_sts,
    ) = pkl.load(open("time_series_1min_padding_ready.pkl", "rb"))

    all_accelerations = np.asarray(all_accelerations)
    all_mics = np.asarray(all_mics)
    all_subject_ids = np.asarray(all_subject_ids)
    all_neighbor_counts = np.asarray(all_neighbor_counts)
    labels_eff = np.asarray(labels_eff)

    labels_eff[labels_eff < 5] = 0
    labels_eff[labels_eff == 5] = 1

    (
        X_train_acc,
        X_test_acc,
        X_train_mic,
        X_test_mic,
        y_train,
        y_test,
    ) = train_test_split(
        all_accelerations,
        all_mics,
        labels_eff,
        test_size=0.2,
        random_state=42,
        stratify=labels_eff,
    )

    X_train_acc, X_val_acc, X_train_mic, X_val_mic, y_train, y_val = train_test_split(
        X_train_acc,
        X_train_mic,
        y_train,
        test_size=0.25,
        random_state=42,
        stratify=y_train,
    )

    train_dataset = IntervalsDataset(X_train_acc, X_train_mic, y_train)
    val_dataset = IntervalsDataset(X_val_acc, X_val_mic, y_val)
    test_dataset = IntervalsDataset(X_test_acc, X_test_mic, y_test)

    dataloader_train = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate
    )

    dataloader_val = DataLoader(
        val_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate
    )
    dataloader_test = DataLoader(
        test_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate
    )

    inp_dim = 5
    hid_dim = 32
    n_layers = 2
    out_dim = 1
    batch_size = 32
    num_epochs = 500
    model = LSTMModel(
        inp_dim,
        hid_dim,
        n_layers,
        batch_size,
        cluster_size,
        cluster_output_dim,
        out_dim,
        device,
    )
    model = model.to(device)

    cl_weights = compute_class_weight("balanced", np.unique(labels_eff), labels_eff)
    cl_weights = torch.from_numpy(cl_weights).float()
    cl_weights = cl_weights.to(device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=cl_weights[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
    # nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.8, patience=100)

    train_losses = []
    val_losses = []
    train_bacs = []
    test_bacs = []
    val_bacs = []

    for i in np.arange(num_epochs):

        model.train()
        running_loss = 0.0
        start = time.time()
        for i_batch, cur in enumerate(dataloader_train):
            cur_acc = cur[0]
            cur_mic = cur[1]
            cur_y = cur[2]
            cur_acc_len = cur[3]
            cur_mic_len = cur[4]
            optimizer.zero_grad()
            outputs = model(cur_acc, cur_mic, cur_acc_len, cur_mic_len)
            target = torch.tensor(cur_y).float()
            target = target.to(device)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        mean_train_loss = running_loss / i_batch
        train_losses.append(mean_train_loss)

        model.eval()
        y_tr_batches = []
        y_tr_out = []
        with torch.no_grad():
            for i_batch, cur in enumerate(dataloader_train):
                cur_acc = cur[0]
                cur_mic = cur[1]
                cur_y = cur[2]
                cur_acc_len = cur[3]
                cur_mic_len = cur[4]
                outputs = model(cur_acc, cur_mic, cur_acc_len, cur_mic_len)
                y_out = np.round(torch.sigmoid(outputs.cpu())).detach().numpy()
                y_tr_batches.extend(cur_y)
                y_tr_out.extend(y_out)
        cur_train_bac = balanced_accuracy_score(y_tr_batches, y_tr_out)
        train_bacs.append(cur_train_bac)

        y_val_batches = []
        y_val_out = []
        running_val_loss = 0
        with torch.no_grad():
            for i_batch, cur in enumerate(dataloader_val):
                cur_acc = cur[0]
                cur_mic = cur[1]
                cur_y = cur[2]
                cur_acc_len = cur[3]
                cur_mic_len = cur[4]
                outputs = model(cur_acc, cur_mic, cur_acc_len, cur_mic_len)
                target = torch.tensor(cur_y).float()
                target = target.to(device)
                val_loss = criterion(outputs, target)
                running_val_loss += val_loss.item()
                y_out = np.round(torch.sigmoid(outputs.cpu())).detach().numpy()
                y_val_batches.extend(cur_y)
                y_val_out.extend(y_out)
        mean_val_loss = running_val_loss / i_batch
        scheduler.step(mean_val_loss)
        val_losses.append(mean_val_loss)
        cur_val_bac = balanced_accuracy_score(y_val_batches, y_val_out)
        val_bacs.append(cur_val_bac)

        y_test_batches = []
        y_test_out = []
        with torch.no_grad():
            for i_batch, cur in enumerate(dataloader_test):
                cur_acc = cur[0]
                cur_mic = cur[1]
                cur_y = cur[2]
                cur_acc_len = cur[3]
                cur_mic_len = cur[4]
                outputs = model(cur_acc, cur_mic, cur_acc_len, cur_mic_len)
                target = torch.tensor(cur_y).float()
                target = target.to(device)
                y_out = np.round(torch.sigmoid(outputs.cpu())).detach().numpy()
                y_test_batches.extend(cur_y)
                y_test_out.extend(y_out)
        cur_test_bac = balanced_accuracy_score(y_test_batches, y_test_out)
        test_bacs.append(cur_test_bac)
        end = time.time()
        print(f"Epoch took {end-start} seconds")
        print(f"Current LR:{optimizer.param_groups[0]['lr']}")
        print(f"Epoch {i} training loss:{mean_train_loss}")
        print(f"Epoch {i} validation loss:{mean_val_loss}")
        print(f"Train BAC:{cur_train_bac}")
        print(f"Val BAC:{cur_val_bac}")
        print(f"Test BAC:{cur_test_bac}")
        sys.stdout.flush()
        torch.save(
            model.state_dict(),
            f"./Results-Loupe/"
            f"LSTM_Vlad_8Clusters_Adam_withoutbalanced_weights_epoch_{i}.pt",
        )
        pkl.dump(
            (train_losses, val_losses, train_bacs, val_bacs, test_bacs),
            open(
                f"./Results-Loupe/LSTM_Vlad_8Clusters_Adam_withoutbalanced_"
                f"losses_bacs_dim{hid_dim}",
                "wb",
            ),
        )
