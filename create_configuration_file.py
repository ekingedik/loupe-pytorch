import yaml
import itertools
from pathlib import Path
import numpy as np

# Parameters
hid_dim = [16, 32, 64, 128]
n_layers = [1, 2, 3]
batch_size = [16, 32, 64]
cluster_size = [2, 4, 8, 16]
cluster_output_dim = [8, 16, 32, 64]
lrs = [1e-4, 1e-3, 1e-5]
gating = [True, False]
bnorm = [True, False]
# loupe = ["NetVLAD","NetRVLAD","SoftDBoW"]

all_perms = list(
    itertools.product(
        *[
            hid_dim,
            n_layers,
            batch_size,
            cluster_size,
            cluster_output_dim,
            lrs,
            gating,
            bnorm,
        ]
    )
)

for i in np.arange(500, 2500):
    cur_perm = all_perms[i]
    cur_dict = {}
    cur_dict["hid_dim"] = cur_perm[0]
    cur_dict["n_layers"] = cur_perm[1]
    cur_dict["batch_size"] = cur_perm[2]
    cur_dict["cluster_size"] = cur_perm[3]
    cur_dict["cluster_output_dim"] = cur_perm[4]
    cur_dict["lrs"] = cur_perm[5]
    cur_dict["gating"] = cur_perm[6]
    cur_dict["bnorm"] = cur_perm[7]
    cur_dict["experiment_id"] = int(i)
    outf = Path().cwd() / "config_files" / f"{i}.yaml"
    with open(outf, "w") as file:
        _ = yaml.dump(cur_dict, file)
