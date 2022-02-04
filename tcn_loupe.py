import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loupe_pytorch import NetVLAD
from TCN import TemporalConvNet


class TCNModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        levels,
        batch_size,
        output_dim,
        kernel_size,
        dropout,
        cluster_size,
        cluster_output_dim,
        gating,
        bnorm,
        device,
    ):
        super(TCNModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.levels = levels
        self.batch_size = batch_size
        self.num_channels = [self.hidden_dim] * levels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.cluster_size = cluster_size
        self.cluster_output_dim = cluster_output_dim
        self.gating = gating
        self.bnorm = bnorm
        self.tcn_subject_acc = TemporalConvNet(
            input_dim,
            self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )
        self.tcn_neighbors_acc = TemporalConvNet(
            input_dim,
            self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )
        self.tcn_subject_mic = TemporalConvNet(
            input_dim,
            self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )
        self.tcn_neighbors_mic = TemporalConvNet(
            input_dim,
            self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )

        self.vlad_acc = NetVLAD(
            feature_size=self.hidden_dim,
            max_samples=5,
            cluster_size=self.cluster_size,
            output_dim=self.cluster_output_dim,
            gating=self.gating,
            add_batch_norm=self.bnorm,
        )

        self.vlad_mic = NetVLAD(
            feature_size=self.hidden_dim,
            max_samples=5,
            cluster_size=self.cluster_size,
            output_dim=self.cluster_output_dim,
            gating=self.gating,
            add_batch_norm=self.bnorm,
        )

        self.fc1 = nn.Linear(int(self.cluster_output_dim * 2), self.cluster_output_dim)
        self.dp = nn.Dropout(p=self.dropout)
        self.fcout = nn.Linear(self.cluster_output_dim, output_dim)
        self.device = device

    def forward(self, x_acc, x_mic, acc_lens, mic_lens):

        # (int(np.shape(x_acc)[0] / 5) is the current batch size
        acc_s_batch = x_acc[0 : np.shape(x_acc)[0] : 5, :, :]
        mic_s_batch = x_mic[0 : np.shape(x_mic)[0] : 5, :, :]
        acc_s_lengths = torch.tensor(acc_lens[0 : np.shape(x_acc)[0] : 5])
        mic_s_lengths = torch.tensor(mic_lens[0 : np.shape(x_acc)[0] : 5])

        acc_s_batch = acc_s_batch.permute(0, 2, 1)
        mic_s_batch = mic_s_batch.permute(0, 2, 1)

        acc_s_batch = acc_s_batch.to(self.device)
        mic_s_batch = mic_s_batch.to(self.device)

        out_acc_s = self.tcn_subject_acc(acc_s_batch.float())
        out_mic_s = self.tcn_subject_mic(mic_s_batch.float())

        ending_acc_s_outputs = torch.zeros(
            (int(np.shape(x_acc)[0] / 5), self.hidden_dim)
        )
        ending_mic_s_outputs = torch.zeros(
            (int(np.shape(x_acc)[0] / 5), self.hidden_dim)
        )

        for b_num, cur_size in zip(
            np.arange(int(np.shape(x_acc)[0] / 5)), acc_s_lengths
        ):
            if cur_size == 0:
                continue
            else:
                ending_acc_s_outputs[b_num, :] = out_acc_s[b_num, :, cur_size - 1]
        for b_num, cur_size in zip(
            np.arange(int(np.shape(x_acc)[0] / 5)), mic_s_lengths
        ):
            if cur_size == 0:
                continue
            else:
                ending_mic_s_outputs[b_num, :] = out_mic_s[b_num, :, cur_size - 1]

        # Get last time step
        out_acc_s_last_timestep = ending_acc_s_outputs.to(self.device)
        out_mic_s_last_timestep = ending_mic_s_outputs.to(self.device)

        out_neighbors_acc = []
        out_neighbors_mic = []
        # Neighbor's data
        for i in np.arange(1, 5):

            cur_acc_n_batch = x_acc[i : np.shape(x_acc)[0] : 5, :, :]
            cur_mic_n_batch = x_mic[i : np.shape(x_mic)[0] : 5, :, :]
            cur_acc_n_lengths = torch.tensor(acc_lens[i : np.shape(x_acc)[0] : 5])
            cur_mic_n_lengths = torch.tensor(mic_lens[i : np.shape(x_mic)[0] : 5])

            cur_acc_n_batch = cur_acc_n_batch.permute(0, 2, 1)
            cur_mic_n_batch = cur_mic_n_batch.permute(0, 2, 1)

            cur_acc_n_batch = cur_acc_n_batch.to(self.device)
            cur_mic_n_batch = cur_mic_n_batch.to(self.device)

            out_neighbor_acc = self.tcn_neighbors_acc(cur_acc_n_batch.float())
            out_neighbor_mic = self.tcn_neighbors_mic(cur_mic_n_batch.float())

            ending_acc_n_outputs = torch.zeros(
                (int(np.shape(x_acc)[0] / 5), self.hidden_dim)
            )
            ending_mic_n_outputs = torch.zeros(
                (int(np.shape(x_mic)[0] / 5), self.hidden_dim)
            )
            for b_num, cur_size in zip(
                np.arange(int(np.shape(x_acc)[0] / 5)), cur_acc_n_lengths
            ):
                if cur_size == 0:
                    continue
                else:
                    ending_acc_n_outputs[b_num, :] = out_neighbor_acc[
                        b_num, :, cur_size - 1
                    ]
            for b_num, cur_size in zip(
                np.arange(int(np.shape(x_acc)[0] / 5)), cur_mic_n_lengths
            ):
                if cur_size == 0:
                    continue
                else:
                    ending_mic_n_outputs[b_num, :] = out_neighbor_mic[
                        b_num, :, cur_size - 1
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

        all_out_acc = self.vlad_acc(all_out_acc)
        all_out_mic = self.vlad_mic(all_out_mic)

        all_out = torch.cat((all_out_acc, all_out_mic), 1)
        all_out = self.dp(F.leaky_relu(self.fc1(all_out)))
        out = self.fcout(all_out)
        return out
