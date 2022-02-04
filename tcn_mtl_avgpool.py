import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        device
    ):
        super(TCNModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.levels = levels
        self.batch_size = batch_size
        self.num_channels = [self.hidden_dim] * levels
        self.kernel_size = kernel_size
        self.dropout = dropout
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

        self.fc_eff = nn.Linear(
            int(self.hidden_dim * 2), self.hidden_dim
        )
        self.fc_fst = nn.Linear(
            int(self.hidden_dim * 2), self.hidden_dim
        )
        self.fc_sts = nn.Linear(
            int(self.hidden_dim * 2), self.hidden_dim
        )

        self.drp_eff = nn.Dropout(p=self.dropout)
        self.drp_fst = nn.Dropout(p=self.dropout)
        self.drp_sts = nn.Dropout(p=self.dropout)

        self.out_eff = nn.Linear(self.hidden_dim , output_dim)
        self.out_fst = nn.Linear(self.hidden_dim , output_dim)
        self.out_sts = nn.Linear(self.hidden_dim , output_dim)
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

        for b_num, cur_size in zip(np.arange(int(np.shape(x_acc)[0] / 5)), acc_s_lengths):
            if cur_size == 0:
                continue
            else:
                ending_acc_s_outputs[b_num, :] = out_acc_s[b_num, :, cur_size - 1]
        for b_num, cur_size in zip(np.arange(int(np.shape(x_acc)[0] / 5)), mic_s_lengths):
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

        pooled_outputs_acc = []
        pooled_outputs_mic = []
        for bn in np.arange(int(np.shape(x_acc)[0] / 5)):
            # Get the data of the subject entered the ESM
            sbj_acc = out_neighbors_acc[0][bn, :].view(1, self.hidden_dim)
            sbj_mic = out_neighbors_mic[0][bn, :].view(1, self.hidden_dim)
            # Get the neighbours
            n1_acc = out_neighbors_acc[1][bn, :].view(1, self.hidden_dim)
            n2_acc = out_neighbors_acc[2][bn, :].view(1, self.hidden_dim)
            n3_acc = out_neighbors_acc[3][bn, :].view(1, self.hidden_dim)
            n4_acc = out_neighbors_acc[4][bn, :].view(1, self.hidden_dim)
            n1_mic = out_neighbors_mic[1][bn, :].view(1, self.hidden_dim)
            n2_mic = out_neighbors_mic[2][bn, :].view(1, self.hidden_dim)
            n3_mic = out_neighbors_mic[3][bn, :].view(1, self.hidden_dim)
            n4_mic = out_neighbors_mic[4][bn, :].view(1, self.hidden_dim)
            if len(torch.nonzero(n1_acc)) != 0:
                sbj_acc = torch.cat((sbj_acc, n1_acc), dim=0)
            if len(torch.nonzero(n2_acc)) != 0:
                sbj_acc = torch.cat((sbj_acc, n2_acc), dim=0)
            if len(torch.nonzero(n3_acc)) != 0:
                sbj_acc = torch.cat((sbj_acc, n3_acc), dim=0)
            if len(torch.nonzero(n4_acc)) != 0:
                sbj_acc = torch.cat((sbj_acc, n4_acc), dim=0)
            if len(torch.nonzero(n1_mic)) != 0:
                sbj_mic = torch.cat((sbj_mic, n1_mic), dim=0)
            if len(torch.nonzero(n2_mic)) != 0:
                sbj_mic = torch.cat((sbj_mic, n2_mic), dim=0)
            if len(torch.nonzero(n3_mic)) != 0:
                sbj_mic = torch.cat((sbj_mic, n3_mic), dim=0)
            if len(torch.nonzero(n4_mic)) != 0:
                sbj_mic = torch.cat((sbj_mic, n4_mic), dim=0)
            avg_pooled_acc = torch.mean(sbj_acc, dim=0)
            pooled_outputs_acc.append(avg_pooled_acc)
            avg_pooled_mic = torch.mean(sbj_mic, dim=0)
            pooled_outputs_mic.append(avg_pooled_mic)
        pooled_outputs_acc = torch.stack(pooled_outputs_acc)
        pooled_outputs_mic = torch.stack(pooled_outputs_mic)
        all_out = torch.cat((pooled_outputs_acc, pooled_outputs_mic), 1)
        all_out_eff = self.drp_eff(F.leaky_relu(self.fc_eff(all_out)))
        all_out_fst = self.drp_fst(F.leaky_relu(self.fc_fst(all_out)))
        all_out_sts = self.drp_sts(F.leaky_relu(self.fc_sts(all_out)))
        out_eff = self.out_eff(all_out_eff)
        out_fst = self.out_fst(all_out_fst)
        out_sts = self.out_sts(all_out_sts)
        return out_eff, out_fst, out_sts
