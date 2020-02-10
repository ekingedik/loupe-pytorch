import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from loupe_pytorch import NetVLAD


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer_dim,
        batch_size,
        cluster_size,
        cluster_output_dim,
        output_dim,
        device,
    ):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.batch_size = batch_size
        self.cluster_size = cluster_size
        self.cluster_output_dim = cluster_output_dim
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
            cluster_size=self.cluster_size,
            output_dim=self.cluster_output_dim,
            gating=False,
        )

        self.vlad_mic = NetVLAD(
            feature_size=self.hidden_dim,
            max_samples=5,
            cluster_size=self.cluster_size,
            output_dim=self.cluster_output_dim,
            gating=False,
        )

        self.fc1 = nn.Linear(self.cluster_output_dim * 2, 8)
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
