
import os
from tqdm import tqdm
import h5py

def merge_files(data_path, new_path=None, filter=None):
    if new_path is None:
        new_path = str(data_path).rstrip("/") + ".hdf5"
    fnames = [name for name in os.listdir(data_path) if name.endswith(".hdf5")]
    fnames = sorted(fnames, key=lambda x: int(''.join(x.split(".")[0].split(':'))))
    with h5py.File(os.path.join(data_path, fnames[0]), "r") as tmp_f:
        keys = list(tmp_f.keys())
    new_f = h5py.File(new_path, "w")
    for k in keys:
        new_f.create_group(k)
    for name in tqdm(fnames):
        if filter is not None and not filter(name):
            continue
        print("==> merging: ", name)
        with h5py.File(os.path.join(data_path, name), "r") as f:
            for k in f.keys():
                new_f[k].create_dataset("{}".format(name.split(".")[0]), data=f[k])
    print("==> saving to: ", new_path)
    new_f.close()

data_path = '/mnt/bn/magic/phyre/phyre_combination_data/4_in_8'
new_path = '/mnt/bn/magic/phyre/phyre_combination_data/4_in_8_10templates.hdf5'
filter = lambda x: int(x.split(":")[0]) <= 10009

merge_files(data_path, new_path, filter)