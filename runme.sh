#!/bin/bash

# ====================================
# ====================================
# ====================================
# ============= Pretrain =============

pretrain()
{
  DATASET_DIR=$1
  WORKSPACE=$2
  DATA_PROCESS=$3

  if $DATA_PROCESS; then
    echo "------ Start preparing dataset ------"

    # ============ Download dataset ============
    #echo "------ Download metadata ------"
    #mkdir -p $DATASET_DIR"/metadata"

    # Video list csv
    #wget -O $DATASET_DIR"/metadata/eval_segments.csv" http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv
    #wget -O $DATASET_DIR"/metadata/balanced_train_segments.csv" http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv
    #wget -O $DATASET_DIR"/metadata/unbalanced_train_segments.csv" http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv

    # Class labels indices
    #wget -O $DATASET_DIR"/metadata/class_labels_indices.csv" http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv

    # Quality of counts
    #wget -O $DATASET_DIR"/metadata/qa_true_counts.csv" http://storage.googleapis.com/us_audioset/youtube_corpus/v1/qa/qa_true_counts.csv

    #echo "Download metadata to $DATASET_DIR/metadata"

    # Split large unbalanced csv file (2,041,789) to 41 partial csv files.
    # Each csv file contains at most 50,000 audio info.
    #echo "------ Split unbalanced csv to csvs ------"
    #python3 utils/dataset.py split_unbalanced_csv_to_partial_csvs --unbalanced_csv=$DATASET_DIR/metadata/unbalanced_train_segments.csv --unbalanced_partial_csvs_dir=$DATASET_DIR"/metadata/unbalanced_partial_csvs"

    #echo "------ Download wavs ------"
    # Download evaluation wavs
    #python3 utils/dataset.py download_wavs --csv_path=$DATASET_DIR"/metadata/eval_segments.csv" --audios_dir=$DATASET_DIR"/audios/eval_segments"

    # Download balanced train wavs
    #python3 utils/dataset.py download_wavs --csv_path=$DATASET_DIR"/metadata/balanced_train_segments.csv" --audios_dir=$DATASET_DIR"/audios/balanced_train_segments"

    # Download unbalanced train wavs. Users may consider executing the following
    # commands in parallel. One simple way is to open 41 terminals and execute
    # one command in one terminal.
    #for IDX in {00..07}; do
      #echo $IDX
      #python utils/dataset.py download_wavs --csv_path=$DATASET_DIR"/metadata/unbalanced_partial_csvs/unbalanced_train_segments_part$IDX.csv" --audios_dir=$DATASET_DIR"/audios/unbalanced_train_segments/unbalanced_train_segments_part$IDX"
    #done

    # ============ Pack waveform and target to hdf5 ============
    # Pack evaluation waveforms to a single hdf5 file
    echo "------ Pack evaluation waveforms into hdf5 ------"
    python utils/dataset.py pack_waveforms_to_hdf5 --csv_path=$DATASET_DIR"/metadata/eval_segments.csv" --audios_dir=$DATASET_DIR"/audios/eval_segments" --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/eval.h5"

    # Pack balanced training waveforms to a single hdf5 file
    echo "------ Pack balanced training waveforms into hdf5 ------"
    python utils/dataset.py pack_waveforms_to_hdf5 --csv_path=$DATASET_DIR"/metadata/balanced_train_segments.csv" --audios_dir=$DATASET_DIR"/audios/balanced_train_segments" --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/balanced_train.h5"

    # Pack unbalanced training waveforms to hdf5 files. Users may consider 
    # executing the following commands in parallel to speed up. One simple 
    # way is to open 41 terminals and execute one command in one terminal.
    echo "------ Pack unbalanced training waveforms into hdf5 ------"
    for IDX in {00..40}; do
        echo "Packing waveform with id $IDX"
        python utils/dataset.py pack_waveforms_to_hdf5 --csv_path=$DATASET_DIR"/metadata/unbalanced_partial_csvs/unbalanced_train_segments_part$IDX.csv" --audios_dir=$DATASET_DIR"/audios/unbalanced_train_segments/unbalanced_train_segments_part$IDX" --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/unbalanced_train/unbalanced_train_part$IDX.h5"
    done

    # ============ Prepare training indexes ============
    # Balanced training indexes
    echo "------ Prepare balanced training indexes ------"
    python utils/create_indexes.py create_indexes --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/balanced_train.h5" --indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/balanced_train.h5"

    echo "------ Prepare evaluation indexes ------"
    python utils/create_indexes.py create_indexes --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/eval.h5" --indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/eval.h5"

    echo "------ Prepare unbalanced training indexes ------"
    # Unbalanced training indexes
    for IDX in {00..07}; do
        echo "Creating index with id $IDX"
        python utils/create_indexes.py create_indexes --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/unbalanced_train/unbalanced_train_part$IDX.h5" --indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/unbalanced_train/unbalanced_train_part$IDX.h5"
    done

    # Combine balanced and unbalanced training indexes to a full training indexes hdf5
    echo "------ Combine training indexes ------"
    python utils/create_indexes.py combine_full_indexes --indexes_hdf5s_dir=$WORKSPACE"/hdf5s/indexes" --full_indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/full_train.h5"
  fi

  # ============ Blacklist for training (optional) ============
  # Audios in the balck list will not be used in training

  #python utils/create_black_list.py dcase2017task4 --workspace=$WORKSPACE

  # ============ Train & Inference ============
  echo "------ Training starts ------"
  python pytorch/main.py train --workspace=$WORKSPACE --data_type='full_train' --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=14000 --model_type='Cnn14' --loss_type='clip_bce' --balanced='balanced' --augmentation='mixup' --batch_size=32 --learning_rate=1e-3 --resume_iteration=0 --early_stop=300000 --cuda

  # Plot statistics
  python utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select=1_aug
}

# ====================================
# ====================================
# ====================================
# ============= Finetune =============
finetune()
{
  DATASET_DIR_PROJECT=$1
  WORKSPACE=$2
  DATA_PROCESS=$3
  DATASET_DIR_PROJECT_OTHERS=$6

  if $DATA_PROCESS; then
    # Remove the overlap data samples (the overlap between AudioSet and our project test set)
    for ID in {00..07}; do
        python utils/dataset_project.py  filter_wavs --csv_dir=$DATASET_DIR_PROJECT"/metadata/unbalanced_partial_csvs" --split_dir=$DATASET_DIR_PROJECT"/split" --map_path=$DATASET_DIR_PROJECT"/copy_maps.scp"
    done

    python utils/dataset_project.py  filter_and_rename_others  --csv_dir=$DATASET_DIR_PROJECT_OTHERS"/balanced_train_segments.csv" --data_dir=$DATASET_DIR_PROJECT_OTHERS"/audios/balanced_train_segments/"  --output_dir=$DATASET_DIR_PROJECT"/audios/others/"  --split_dir=$DATASET_DIR_PROJECT"/split"

    # Pack Project Audio Data to h5
    for ID in {0..4}; do
        python utils/dataset_project.py pack_project_waveforms_to_hdf5 --csv_path=$DATASET_DIR_PROJECT"/metadata/eval_segments.csv" --audios_dir=$DATASET_DIR_PROJECT"/audio" --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/project_eval_data_$ID.h5"  --split_dir=$DATASET_DIR_PROJECT"/split" --split_to_test_id=$ID
    done

    for ID in {0..4}; do
        python utils/dataset_project.py pack_project_waveforms_to_hdf5 --csv_path=$DATASET_DIR_PROJECT_OTHERS"/metadata/eval_segments.csv" --audios_dir=$DATASET_DIR_PROJECT_OTHERS"/audios/others" --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/project_eval_data_others_$ID.h5"  --split_dir=$DATASET_DIR_PROJECT_OTHERS"/split" --split_to_test_id=$ID
    done

    for IDX in {0..4}; do
        python utils/create_indexes.py create_indexes --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/project_eval_data_$IDX.h5" --indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/project_eval_data_$IDX.h5"
    done

    for ID in {0..4}; do
        python utils/create_indexes.py create_indexes --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/project_eval_data_others_$ID.h5" --indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/project_eval_data_others_$ID.h5"
    done
    # Our project has 5 splits. Combine split 0 1 2 into training data. Split 3 as development set, Split 4 as test set.
    python utils/create_indexes.py combine_full_indexes_project  --indexes_hdf5s_dir=$WORKSPACE"/hdf5s/indexes" --full_indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/project_train.h5" --split_id_list_to_train="0,1,2"

    python utils/create_indexes.py combine_full_indexes_project  --indexes_hdf5s_dir=$WORKSPACE"/hdf5s/indexes" --full_indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/project_val.h5" --split_id_list_to_train="3"

    python utils/create_indexes.py combine_full_indexes_project  --indexes_hdf5s_dir=$WORKSPACE"/hdf5s/indexes" --full_indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/project_eval.h5" --split_id_list_to_train="4"

  fi
  
  CHECKPOINT_SOURCE_DIR=$4
  CHECKPOINT_FILE_NAME=$5
  CHECKPOINT_FINETUNE="20000_iterations.pth"
  # Finetune on our project data 
  python pytorch/finetune_project.py train --workspace=$WORKSPACE --sample_rate=32000  --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=14000 --model_type='Transfer_Cnn14'  --pretrained_checkpoint_path=$CHECKPOINT_SOURCE_DIR'/'$CHECKPOINT_FILE_NAME  --cuda  --early_stop=20000  --loss_type='clip_bce' --learning_rate=1e-4 

  # !!!!! Note: before this step, pls select one model in the workspace/checkpoints/ and copy it into CHECKPOINT_SOURCE_DIR as the model you want to test.
  CHECKPOINT_COPY=$(find $WORKSPACE/checkpoints/finetune_project/ -name "20000_iterations.pth")
  cp $CHECKPOINT_COPY $CHECKPOINT_SOURCE_DIR
  # Test F1 score of the model for project
  python pytorch/test_model_project.py test --workspace=$WORKSPACE  --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=14000 --model_type='Transfer_Cnn14' --loss_type='clip_bce' --augmentation='mixup' --batch_size=32 --learning_rate=1e-3 --cuda --split_id=4 --checkpoint_source_dir=$CHECKPOINT_SOURCE_DIR --checkpoint_file_name=$CHECKPOINT_FINETUNE 
  #python pytorch/test_model_project.py test --workspace=$WORKSPACE  --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=14000 --model_type='Cnn14' --loss_type='clip_bce' --augmentation='mixup' --batch_size=32 --learning_rate=1e-3 --cuda --split_id=4 --checkpoint_source_dir=$CHECKPOINT_SOURCE_DIR --checkpoint_file_name=$CHECKPOINT_FILE_NAME
}

# Users can modify the following paths
pretrain_dataset=${pretrain_dataset:-"./datasets/audioset201906"}
workspace=${workspace:-"./workspaces/audioset_tagging"}
finetune_dataset=${finetune_dataset:-"./datasets/audioset_project"}
finetune_dataset_others=${finetune_dataset_others:-"./datasets/audioset_project_others"}
checkpoint_path=${checkpoint_path:-"./checkpoints_downloaded"}
checkpoint_file=${checkpoint_file:-"20_percent_Cnn14.pth"}
pretrain=${pretrain:-false}
pretrain_pack_data=${pretrain_pack_data:-true}
finetune_pack_data=${finetune_pack_data:-true}

while [ $# -gt 0 ]; do

  if [[ $1 == *"--"* ]]; then
    param="${1/--/}"
    declare $param="$2"
  fi
  shift
done

#DATASET_DIR="/storage/pang0208/datasets/audioset201906"
#WORKSPACE="/storage/pang0208/workspaces/audioset_tagging"

# ============ Inference with pretrained modela ============
# Inference audio tagging with pretrained model
#MODEL_TYPE="Cnn14"
#CHECKPOINT_PATH="Cnn14_mAP=0.431.pth"
#wget -O $CHECKPOINT_PATH "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
#python3 pytorch/inference.py audio_tagging --model_type=$MODEL_TYPE --checkpoint_path=$CHECKPOINT_PATH --audio_path="resources/R9_ZSCveAHg_7s.wav" --cuda

# Inference sound event detection with pretrained model
#MODEL_TYPE="Cnn14_DecisionLevelMax"
#CHECKPOINT_PATH="Cnn14_DecisionLevelMax_mAP=0.385.pth"
#wget -O $CHECKPOINT_PATH "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"
#python3 pytorch/inference.py sound_event_detection --model_type=$MODEL_TYPE --checkpoint_path=$CHECKPOINT_PATH --audio_path="resources/R9_ZSCveAHg_7s.wav" --cuda

#DATASET_DIR_PROJECT="/storage/pang0208/datasets/audioset_project"
#CHECKPOINT_SOURCE_DIR='checkpoints_downloaded'
#CHECKPOINT_FILE_NAME='20_percent_Cnn14.pth'

if $pretrain; then
  pretrain $pretrain_dataset $workspace $pretrain_pack_data
else
  finetune $finetune_dataset $workspace $finetune_pack_data $checkpoint_path $checkpoint_file $finetune_dataset_others
fi

#finetune $finetune_dataset $workspace $finetune_pack_data $checkpoint_path $checkpoint_file $finetune_dataset_others
#echo "The result can be seen in the $workspace/audioset_tagging/logs"
