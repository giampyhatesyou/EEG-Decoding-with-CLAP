import os
from .preprocessing_eegmusic_dataset import Preprocessing_EEGMusic_dataset, Preprocessing_EEGMusic_Test_dataset


# CHANGED(baseline): added cv_mode / cv_held_out_id params (default "within"/-1) and forward them
#                    to both datasets; defaults reproduce the original within-subject split.
def get_dataset(dataset, dataset_dir, subset, download=True,
                cv_mode="within", cv_held_out_id=-1):

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if dataset == "preprocessing_eegmusic":
        d = Preprocessing_EEGMusic_dataset(
            root=dataset_dir, base_dir=dataset_dir, download=download, subset=subset,
            cv_mode=cv_mode, cv_held_out_id=cv_held_out_id)  # CHANGED(baseline): added base_dir + cv args
    elif dataset == "preprocessing_eegmusic_test":
        d = Preprocessing_EEGMusic_Test_dataset(
            root=dataset_dir, base_dir=dataset_dir, download=download, subset=subset,
            cv_mode=cv_mode, cv_held_out_id=cv_held_out_id)  # CHANGED(baseline): added cv args
    else:
        raise NotImplementedError("Dataset not implemented")
    return d
