# AudioSet_Subset_Classification

This code is a modified version from [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn).

## Pretraining
We first pretrain our model on 20% data of AudioSet to get a classifier which can handle 527 classes.
For the 20% data, we use unbalanced part 00-07 as our training set. Other parts are same to [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn). Please refer PANNs for details.


## Finetune

Please copy the project data (9 target classes of AudioSet) to ./datasets/audioset_project.
In ./datasets/audioset_project, we should have:
1) copy_maps.scp, which is a mapping dict to map the orignial audio names to the project audio names
2) audio  folder including audio files
3) data   folder including data list of Kaldi format
4) metadata  folder of data list copied from the full AudioSet. (Seems to be useless in our fine-tune code.)
5) split  folder of data list of audio files spliting (designed for k-fold validation)
Then, remove the overlap data samples (the overlap between AudioSet and our project test set)
```sh
for ID in {00..07}; do
    python utils/dataset_project.py  filter_wavs --csv_dir=$DATASET_DIR_PROJECT"/metadata/unbalanced_partial_csvs" --split_dir=$DATASET_DIR_PROJECT"/split" --map_path=$DATASET_DIR_PROJECT"/copy_maps.scp"
done
```

We then Pack Project Audio Data to .h5 files for training and testing. Please note, in our code, we are only interested in 9 classes instead of all 527 classes in the whole dataset.
```sh
for ID in {0..4}; do
    python utils/dataset_project.py pack_waveforms_to_hdf5 --csv_path=$DATASET_DIR_PROJECT"/metadata/eval_segments.csv" --audios_dir=$DATASET_DIR_PROJECT"/audio" --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/project_eval_data_$ID.h5"  --split_dir=$DATASET_DIR_PROJECT"/split" --split_to_test_id=$ID
done
```
We have 5 splits in total. We use 0 1 2 as training set, 3 as validation set, 4 as our test set. This system use index to sample data as input to train, so we can simply combine the index of 0 1 2 splits as our training set.
```sh
python utils/create_indexes.py combine_full_indexes_project  --indexes_hdf5s_dir=$WORKSPACE"/hdf5s/indexes" --full_indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/project_train.h5" --split_id_list_to_train="0,1,2
```

Finetune(Further) train our CNN14 on the project data (with only 9 classes). We do not fix any trainable parameters and tune the whole model. Last layer for classifiction is trained from scratch.
```sh
python pytorch/finetune_project.py train --workspace=$WORKSPACE --sample_rate=32000  --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=14000 --model_type='Transfer_Cnn14'  --pretrained_checkpoint_path='checkpoints_downloaded/'$CHECKPOINT_FILE_NAME  --cuda  --early_stop=20000  --loss_type='clip_bce' --learning_rate=1e-4
```

We can observe the mAP during training. Also, last step, we can check the F1 scores of our model on by the code I wrote. This script works well on classification using cross entropy loss. However, maybe it has some problems on binary cross entropy loss. Anyway, actually,  the mAP reported in the script above is more than enough to show the performance.
Note: before this step, pls select one model in the workspace/checkpoints/ and copy it into CHECKPOINT_SOURCE_DIR as the model you want to test.
```sh
CHECKPOINT_SOURCE_DIR='checkpoints_downloaded'
CHECKPOINT_FILE_NAME='20_percent_Cnn14.pth'

python pytorch/test_model_project.py test --workspace=$WORKSPACE  --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=14000 --model_type='Cnn14' --loss_type='clip_bce' --augmentation='mixup' --batch_size=32 --learning_rate=1e-3 --cuda --split_id=4 --checkpoint_source_dir=$CHECKPOINT_SOURCE_DIR --checkpoint_file_name=$CHECKPOINT_FILE_NAME
```

## Dataset & Model

- Pretrain Model Link:
https://entuedu-my.sharepoint.com/:u:/r/personal/pang0208_e_ntu_edu_sg/Documents/20_percent_Cnn14._no_overlap.pth?csf=1&web=1&e=h4FiLm

- Pretrain Dataset Link: 
Please proceed to lab cluster. The file is in /home3/pang0208/AudioSet_Subset_Classification/datasets/audioset201906.tar.gz

- Finetune Dataset Link: 
https://entuedu-my.sharepoint.com/:u:/g/personal/pang0208_e_ntu_edu_sg/EY1HvvVQ3MFPmlyh0pAc-ecBWoRZTJ77N5pWynZog4UuLQ?e=cOFvcj

- FInetune Dataset Other Link: https://entuedu-my.sharepoint.com/:u:/g/personal/pang0208_e_ntu_edu_sg/EUICl8EyH31DtbON_NMv1roBCUCE0L3Ju6Kmw1Ccw-JrSQ?e=U2B0kl

## Usage
Users can train the model from scratch by executing the commands in runme.sh. The runme.sh consists of three parts. 1. Pack downloaded wavs to hdf5 file to speed up loading. 2. Pretrain. 3. Finetune.

### Command Options

| Name                    | Type     | Default                              | Description                                                            |
| ---------------------   | -------- | ------------------------------------ | ----------------------------------------------------------------       |
| pretrain_dataset        | `string` | `./datasets/audioset201906`          | Pretrain dataset path                                                  |
| workspace\*             | `string` | `./workspaces/audioset_tagging`      | Workspaces path for hdf waveforms and indexes as well as results       |
| finetune_dataset        | `string` | `./datasets/audioset_project`        | Finetune dataset path                                                  |
| finetune_dataset_others | `string` | `./datasets/audioset_project_others` | Finetune dataset path for others class                                 |
| checkpoint_path         | `string` | `./checkpoints_downloaded`           | Path for pretrain model and finetune model(for testing)                |
| checkpoint_file         | `string` | `20_percent_Cnn14.pth`               | Pretrain model file name                                               |
| pretrain                | `bool`   | `false`                              | False: Run fintune, True: Run pretrain                                 |
| pretrain_pack_data      | `bool`   | `true`                               | (Pretrain) Whether to create hdf5. Can disable if hdf5 data is created |
| finetune_pack_data      | `bool`   | `true`                               | (Finetune) Whether to create hdf5. Can disable if hdf5 data is created |

\* required options

### Examples

#### Pretrain
```sh
bash runme.sh --pretrain_dataset ./datasets/audioset_pretraining --finetune_dataset ./datasets/audioset_project --finetune_dataset_others ./datasets/audioset_project_others --workspace ./workspaces/audioset_tagging --finetune_pack_data false --pretrain true --pretrain_pack_data false
```

#### Finetune
```sh
bash runme.sh --pretrain_dataset ./datasets/audioset_pretraining --finetune_dataset ./datasets/audioset_project --finetune_dataset_others ./datasets/audioset_project_others --workspace ./workspaces/audioset_tagging --finetune_pack_data false --pretrain false --pretrain_pack_data false
```

## Results

| Model  | Pretrain Data | mAP |
| ------------- | ------------- | ------------- |
| CNN14  | 20%  | 87.5 |
