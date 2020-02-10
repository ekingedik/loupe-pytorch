from torch.utils.data import Dataset
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


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
