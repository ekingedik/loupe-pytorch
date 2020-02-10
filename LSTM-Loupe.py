import pickle as pkl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import time
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import itertools
from loupe_pytorch import NetVLAD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys

class LSTMModel(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, layer_dim, batch_size, output_dim, device
    ):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.batch_size = batch_size
        self.lstm_acc_subject = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True
        )
        self.lstm_acc_neighbors = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True
        )
        self.lstm_mic_subject = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True
        )
        self.lstm_mic_neighbors = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True
        )

        self.vlad_acc = NetVLAD(
            feature_size=self.hidden_dim,
            max_samples=5,
            cluster_size=4,
            output_dim=16,
            gating=False,
        )

        self.vlad_mic = NetVLAD(
            feature_size=self.hidden_dim,
            max_samples=5,
            cluster_size=4,
            output_dim=16,
            gating=False,
        )

        self.fc1 = nn.Linear(32, 8)
        self.fcout = nn.Linear(8, output_dim)
        self.device = device

    def forward(self, x_acc, x_mic, acc_lens, mic_lens):
        h_s_acc = torch.zeros(
            self.layer_dim, self.batch_size, self.hidden_dim
        ).requires_grad_()
        c_s_acc = torch.zeros(
            self.layer_dim, self.batch_size, self.hidden_dim
        ).requires_grad_()
        h_s_mic = torch.zeros(
            self.layer_dim, self.batch_size, self.hidden_dim
        ).requires_grad_()
        c_s_mic = torch.zeros(
            self.layer_dim, self.batch_size, self.hidden_dim
        ).requires_grad_()

        acc_s_batch = x_acc[0 : np.shape(x_acc)[0] : 5, :, :]
        mic_s_batch = x_mic[0 : np.shape(x_mic)[0] : 5, :, :]
        acc_s_lengths = torch.tensor(acc_lens[0 : np.shape(x_acc)[0] : 5])
        mic_s_lengths = torch.tensor(mic_lens[0 : np.shape(x_acc)[0] : 5])

        non_valid_acc_all = np.where(np.asarray(acc_lens) == 0)[0]
        non_valid_mic_all = np.where(np.asarray(mic_lens) == 0)[0]

        non_valid_acc_ids = torch.tensor(np.where(acc_s_lengths == 0)[0])
        non_valid_mic_ids = torch.tensor(np.where(mic_s_lengths == 0)[0])

        modified_acc_s_lengths = torch.clone(acc_s_lengths)
        modified_acc_s_lengths[acc_s_lengths == 0] = 1
        modified_mic_s_lengths = torch.clone(mic_s_lengths)
        modified_mic_s_lengths[mic_s_lengths == 0] = 1

        acc_s_batch = acc_s_batch.to(self.device)
        mic_s_batch = mic_s_batch.to(self.device)
        acc_s_lengths = acc_s_lengths.to(self.device)
        mic_s_lengths = mic_s_lengths.to(self.device)
        non_valid_acc_ids = non_valid_acc_ids.to(self.device)
        non_valid_mic_ids = non_valid_mic_ids.to(self.device)
        h_s_acc = h_s_acc.to(self.device)
        c_s_acc = c_s_acc.to(self.device)
        h_s_mic = h_s_mic.to(self.device)
        c_s_mic = c_s_mic.to(self.device)

        modified_acc_s_lengths = modified_acc_s_lengths.to(self.device)
        modified_mic_s_lengths = modified_mic_s_lengths.to(self.device)

        acc_s_packed = pack_padded_sequence(
            acc_s_batch, modified_acc_s_lengths, batch_first=True, enforce_sorted=False
        )
        mic_s_packed = pack_padded_sequence(
            mic_s_batch, modified_mic_s_lengths, batch_first=True, enforce_sorted=False
        )

        acc_s_packed = acc_s_packed.to(self.device)
        mic_s_packed = mic_s_packed.to(self.device)

        packed_out_acc_s, (h_s_acc, c_s_acc) = self.lstm_acc_subject(
            acc_s_packed.float(), (h_s_acc, c_s_acc)
        )
        packed_out_mic_s, (h_s_mic, c_s_mic) = self.lstm_mic_subject(
            mic_s_packed.float(), (h_s_mic, c_s_mic)
        )

        out_acc_s, acc_s_sizes = pad_packed_sequence(packed_out_acc_s, batch_first=True)
        out_mic_s, mic_s_sizes = pad_packed_sequence(packed_out_mic_s, batch_first=True)

        ending_acc_s_outputs = torch.zeros(
            (int(np.shape(x_acc)[0] / 5), self.hidden_dim)
        )
        ending_mic_s_outputs = torch.zeros(
            (int(np.shape(x_acc)[0] / 5), self.hidden_dim)
        )
        for b_num, cur_size in zip(np.arange(int(np.shape(x_acc)[0] / 5)), acc_s_sizes):
            ending_acc_s_outputs[b_num, :] = out_acc_s[b_num, cur_size - 1, :]
        for b_num, cur_size in zip(np.arange(int(np.shape(x_acc)[0] / 5)), mic_s_sizes):
            ending_mic_s_outputs[b_num, :] = out_mic_s[b_num, cur_size - 1, :]

        # Get last time step
        out_acc_s_last_timestep = ending_acc_s_outputs.to(self.device)
        out_mic_s_last_timestep = ending_mic_s_outputs.to(self.device)

        out_neighbors_acc = []
        out_neighbors_mic = []
        # Neighbor's data
        for i in np.arange(1, 5):
            h_n_acc = torch.zeros(
                self.layer_dim, self.batch_size, self.hidden_dim
            ).requires_grad_()
            c_n_acc = torch.zeros(
                self.layer_dim, self.batch_size, self.hidden_dim
            ).requires_grad_()
            h_n_mic = torch.zeros(
                self.layer_dim, self.batch_size, self.hidden_dim
            ).requires_grad_()
            c_n_mic = torch.zeros(
                self.layer_dim, self.batch_size, self.hidden_dim
            ).requires_grad_()
            cur_acc_n_batch = x_acc[i : np.shape(x_acc)[0] : 5, :, :]
            cur_mic_n_batch = x_mic[i : np.shape(x_mic)[0] : 5, :, :]
            cur_acc_n_lengths = torch.tensor(acc_lens[i : np.shape(x_acc)[0] : 5])
            cur_mic_n_lengths = torch.tensor(mic_lens[i : np.shape(x_mic)[0] : 5])

            non_valid_acc_ids = torch.tensor(np.where(cur_acc_n_lengths == 0)[0])
            non_valid_mic_ids = torch.tensor(np.where(cur_mic_n_lengths == 0)[0])

            modified_cur_acc_n_lengths = torch.clone(cur_acc_n_lengths)
            modified_cur_acc_n_lengths[cur_acc_n_lengths == 0] = 1
            modified_cur_mic_n_lengths = torch.clone(cur_mic_n_lengths)
            modified_cur_mic_n_lengths[cur_mic_n_lengths == 0] = 1

            cur_acc_n_batch = cur_acc_n_batch.to(self.device)
            cur_mic_n_batch = cur_mic_n_batch.to(self.device)
            cur_acc_n_lengths = cur_acc_n_lengths.to(self.device)
            cur_mic_n_lengths = cur_mic_n_lengths.to(self.device)
            modified_cur_acc_n_lengths = modified_cur_acc_n_lengths.to(self.device)
            modified_cur_mic_n_lengths = modified_cur_mic_n_lengths.to(self.device)
            h_n_acc = h_n_acc.to(self.device)
            c_n_acc = c_n_acc.to(self.device)
            h_n_mic = h_n_mic.to(self.device)
            c_n_mic = c_n_mic.to(self.device)

            cur_acc_n_packed = pack_padded_sequence(
                cur_acc_n_batch,
                modified_cur_acc_n_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            cur_mic_n_packed = pack_padded_sequence(
                cur_mic_n_batch,
                modified_cur_mic_n_lengths,
                batch_first=True,
                enforce_sorted=False,
            )

            cur_acc_n_packed = cur_acc_n_packed.to(self.device)
            cur_mic_n_packed = cur_mic_n_packed.to(self.device)

            packed_out_neighbor_acc, (h_n_acc, c_n_acc) = self.lstm_acc_neighbors(
                cur_acc_n_packed.float(), (h_n_acc, c_n_acc)
            )
            packed_out_neighbor_mic, (h_n_mic, c_n_mic) = self.lstm_mic_neighbors(
                cur_mic_n_packed.float(), (h_n_mic, c_n_mic)
            )

            out_neighbor_acc, acc_n_sizes = pad_packed_sequence(
                packed_out_neighbor_acc, batch_first=True
            )
            out_neighbor_mic, mic_n_sizes = pad_packed_sequence(
                packed_out_neighbor_mic, batch_first=True
            )

            ending_acc_n_outputs = torch.zeros(
                (int(np.shape(x_acc)[0] / 5), self.hidden_dim)
            )
            ending_mic_n_outputs = torch.zeros(
                (int(np.shape(x_mic)[0] / 5), self.hidden_dim)
            )
            for b_num, cur_size in zip(
                np.arange(int(np.shape(x_acc)[0] / 5)), acc_n_sizes
            ):
                ending_acc_n_outputs[b_num, :] = out_neighbor_acc[
                    b_num, cur_size - 1, :
                ]
            for b_num, cur_size in zip(
                np.arange(int(np.shape(x_acc)[0] / 5)), mic_n_sizes
            ):
                ending_mic_n_outputs[b_num, :] = out_neighbor_mic[
                    b_num, cur_size - 1, :
                ]
            ending_acc_n_outputs = ending_acc_n_outputs.to(self.device)
            ending_mic_n_outputs = ending_mic_n_outputs.to(self.device)
            out_neighbors_acc.append(ending_acc_n_outputs)
            out_neighbors_mic.append(ending_mic_n_outputs)

        out_neighbors_acc.insert(0, out_acc_s_last_timestep)
        out_neighbors_mic.insert(0, out_mic_s_last_timestep)

        all_out_acc = torch.zeros((int(np.shape(x_acc)[0] / 5), 5, self.hidden_dim)).to(
            self.device
        )

        all_out_mic = torch.zeros((int(np.shape(x_acc)[0] / 5), 5, self.hidden_dim)).to(
            self.device
        )

        for bc in np.arange(5):
            all_out_acc[:, bc, :] = out_neighbors_acc[bc]
            all_out_mic[:, bc, :] = out_neighbors_mic[bc]

        for non_valid_id in non_valid_acc_all:
            batch_num = int(non_valid_id / 5)
            rem = non_valid_id % 5
            all_out_acc[batch_num, rem, :] = 0

        for non_valid_id in non_valid_mic_all:
            batch_num = int(non_valid_id / 5)
            rem = non_valid_id % 5
            all_out_mic[batch_num, rem, :] = 0

        all_out_acc = self.vlad_acc(all_out_acc)
        all_out_mic = self.vlad_mic(all_out_mic)

        all_out = torch.cat((all_out_acc, all_out_mic), 1)
        all_out = F.relu(self.fc1(all_out))
        out = self.fcout(all_out)
        return out.view(-1)


class IntervalsDataset(Dataset):
    def __init__(self, acc_list, mic_list, y):
        self.accelerations = acc_list
        self.mics = mic_list
        self.labels = y

    def __len__(self):
        return len(self.accelerations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        accs_arr = self.accelerations
        mics_arr = self.mics
        labels_arr = self.labels
        selected_accs = accs_arr[idx]
        selected_mics = mics_arr[idx]
        selected_label = labels_arr[idx]
        return selected_accs, selected_mics, selected_label


def pad_collate(batch):
    all_accs, all_mics, all_labels = zip(*batch)
    all_to_pad_accs = []
    all_to_pad_mics = []
    actual_acc_lengths = []
    actual_mic_lengths = []
    for i in np.arange(len(all_accs)):
        selected_accs = all_accs[i]
        selected_mics = all_mics[i]
        for j in np.arange(5):
            cur_acc = selected_accs[j]
            cur_mic = selected_mics[j]
            cur_acc = np.asarray(cur_acc)
            cur_mic = np.asarray(cur_mic)
            # TODO: Fix for the bug in extraction code, can be deleted after new data
            # TODO: is obtained
            if np.shape(cur_acc) == (0, 0):
                cur_acc = np.zeros((0, 5))
            elif np.shape(cur_acc) == (0, 6):
                cur_acc = np.zeros((0, 5))
            actual_acc_lengths.append(np.shape(cur_acc)[0])
            actual_mic_lengths.append(np.shape(cur_mic)[0])
            all_to_pad_accs.append(torch.from_numpy(cur_acc))
            all_to_pad_mics.append(torch.from_numpy(cur_mic))
    # TODO: Check for erroneous entries, where all data is 0
    # TODO: and make their lengths 0 too, so they will be padded
    padded_acc = pad_sequence(all_to_pad_accs, batch_first=True, padding_value=-10)
    padded_mic = pad_sequence(all_to_pad_mics, batch_first=True, padding_value=-10)
    return padded_acc, padded_mic, all_labels, actual_acc_lengths, actual_mic_lengths


def get_labels(labels_eff, labels_frust, labels_sts, mc=False):
    if mc is True:
        y_eff_mc = np.zeros((labels_eff.shape))
        y_eff_mc[labels_eff < 4] = -1
        y_eff_mc[labels_eff > 4] = 1

        y_sts_mc = np.zeros((labels_sts.shape))
        y_sts_mc[labels_sts < 4] = -1
        y_sts_mc[labels_sts > 5] = 1

        y_frs_mc = np.zeros((labels_frust.shape))
        y_frs_mc[labels_frust < 3] = -1
        y_frs_mc[labels_frust > 4] = 1

        return y_eff_mc, y_sts_mc, y_frs_mc
    else:
        y_eff_bin = np.zeros((labels_eff.shape))
        y_eff_bin[labels_eff > 3] = 1

        y_sts_bin = np.zeros((labels_sts.shape))
        y_sts_bin[labels_sts > 4] = 1

        y_frs_bin = np.zeros((labels_frust.shape))
        y_frs_bin[labels_frust > 2] = 1

        return y_eff_bin, y_sts_bin, y_frs_bin


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if torch.cuda.is_available():
    device = "cuda"
    print ("Cuda is available.")
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

X_train_acc, X_test_acc, X_train_mic, X_test_mic, y_train, y_test = train_test_split(
    all_accelerations,
    all_mics,
    labels_eff,
    test_size=0.2,
    random_state=42,
    stratify=labels_eff,
)

X_train_acc, X_val_acc, X_train_mic, X_val_mic, y_train, y_val = train_test_split(
    X_train_acc, X_train_mic, y_train, test_size=0.25, random_state=42, stratify=y_train
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
num_epochs = 2000
model = LSTMModel(inp_dim, hid_dim, n_layers, batch_size, out_dim, device)
model = model.to(device)

cl_weights = compute_class_weight("balanced", np.unique(labels_eff), labels_eff)
cl_weights = torch.from_numpy(cl_weights).float()
cl_weights = cl_weights.to(device)
#criterion = nn.BCEWithLogitsLoss(pos_weight=cl_weights[1])
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
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
        f"./Results-Loupe/LSTM_Vlad_8Clusters_Adam_withoutbalanced_weights_epoch_{i}.pt",
    )
    pkl.dump(
        (train_losses, val_losses, train_bacs, val_bacs, test_bacs),
        open(f"./Results-Loupe/LSTM_Vlad_8Clusters_Adam_withoutbalanced_losses_bacs_dim{hid_dim}", "wb"),
    )
