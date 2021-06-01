import numpy as np
import argparse
import csv
import os
import glob
import datetime
import time
import logging
import h5py
import librosa
import shutil



from utilities import (create_folder, get_filename, create_logging, 
    float32_to_int16, pad_or_truncate, read_metadata)
import config


def split_unbalanced_csv_to_partial_csvs(args):
    """Split unbalanced csv to part csvs. Each part csv contains up to 50000 ids. 
    """
    
    unbalanced_csv_path = args.unbalanced_csv
    unbalanced_partial_csvs_dir = args.unbalanced_partial_csvs_dir
    
    create_folder(unbalanced_partial_csvs_dir)
    
    with open(unbalanced_csv_path, 'r') as f:
        lines = f.readlines()

    lines = lines[3:]   # Remove head info
    audios_num_per_file = 50000
    
    files_num = int(np.ceil(len(lines) / float(audios_num_per_file)))
    
    for r in range(files_num):
        lines_per_file = lines[r * audios_num_per_file : 
            (r + 1) * audios_num_per_file]
        
        out_csv_path = os.path.join(unbalanced_partial_csvs_dir, 
            'unbalanced_train_segments_part{:02d}.csv'.format(r))

        with open(out_csv_path, 'w') as f:
            f.write('empty\n')
            f.write('empty\n')
            f.write('empty\n')
            for line in lines_per_file:
                f.write(line)
        
        print('Write out csv to {}'.format(out_csv_path))


def download_wavs(args):
    """Download videos and extract audio in wav format.
    """

    # Paths
    csv_path = args.csv_path
    audios_dir = args.audios_dir
    mini_data = args.mini_data
    
    if mini_data:
        logs_dir = '_logs/download_dataset/{}'.format(get_filename(csv_path))
    else:
        logs_dir = '_logs/download_dataset_minidata/{}'.format(get_filename(csv_path))
    
    create_folder(audios_dir)
    create_folder(logs_dir)
    create_logging(logs_dir, filemode='w')
    logging.info('Download log is saved to {}'.format(logs_dir))

    # Read csv
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    lines = lines[3:]   # Remove csv head info

    if mini_data:
        lines = lines[0 : 10]   # Download partial data for debug
    
    download_time = time.time()

    # Download
    for (n, line) in enumerate(lines):
        
        items = line.split(', ')
        audio_id = items[0]
        start_time = float(items[1])
        end_time = float(items[2])
        duration = end_time - start_time
        
        logging.info('{} {} start_time: {:.1f}, end_time: {:.1f}'.format(
            n, audio_id, start_time, end_time))
        
        # Download full video of whatever format
        video_name = os.path.join(audios_dir, '_Y{}.%(ext)s'.format(audio_id))
        os.system("youtube-dl --quiet -o '{}' -x https://www.youtube.com/watch?v={}"\
            .format(video_name, audio_id))

        video_paths = glob.glob(os.path.join(audios_dir, '_Y' + audio_id + '.*'))

        # If download successful
        if len(video_paths) > 0:
            video_path = video_paths[0]     # Choose one video

            # Add 'Y' to the head because some video ids are started with '-'
            # which will cause problem
            audio_path = os.path.join(audios_dir, 'Y' + audio_id + '.wav')

            # Extract audio in wav format
            os.system("ffmpeg -loglevel panic -i {} -ac 1 -ar 32000 -ss {} -t 00:00:{} {} "\
                .format(video_path, 
                str(datetime.timedelta(seconds=start_time)), duration, 
                audio_path))
            
            # Remove downloaded video
            os.system("rm {}".format(video_path))
            
            logging.info("Download and convert to {}".format(audio_path))
                
    logging.info('Download finished! Time spent: {:.3f} s'.format(
        time.time() - download_time))

    logging.info('Logs can be viewed in {}'.format(logs_dir))


def pack_project_waveforms_to_hdf5(args):
    """Pack waveform and target of several audio clips to a single hdf5 file. 
    This can speed up loading and training.
    """

    # Arguments & parameters
    audios_dir = args.audios_dir
    csv_path = args.csv_path
    waveforms_hdf5_path = args.waveforms_hdf5_path
    mini_data = args.mini_data

    split_dir = args.split_dir
    split_to_test_id = args.split_to_test_id

    clip_samples = config.clip_samples
    #classes_num = config.classes_num
    classes_num = 9
    sample_rate = config.sample_rate
    #id_to_ix = config.id_to_ix
    id_to_ix = {'screaming':0,'crying_sobbing':1,'chatter':2,'motor_vehicle_(road)':3,'emergency_vehicle':4,'siren':5,'explosion':6,'gunshot_gunfire':7, 'breaking':8 ,'others': 9}


    id_dict = {'screaming':'/m/03qc9zr','crying_sobbing':'/m/0463cq4','chatter':'/m/07rkbfh','motor_vehicle_(road)':'/m/012f08','emergency_vehicle':'/m/03j1ly','siren':'/m/03kmc9','explosion':'/m/014zdl','gunshot_gunfire':'/m/032s66', 'breaking': '/m/07pc8lb'} 




    # Paths
    if mini_data:
        prefix = 'mini_'
        waveforms_hdf5_path += '.mini'
    else:
        prefix = ''

    create_folder(os.path.dirname(waveforms_hdf5_path))

    logs_dir = '_logs/pack_waveforms_to_hdf5/{}{}'.format(prefix, get_filename(csv_path))
    create_folder(logs_dir)
    create_logging(logs_dir, filemode='w')
    logging.info('Write logs to {}'.format(logs_dir))
    
	# Read csv file
    
    #split_file = open(split_dir+"/fold_filter_"+str(split_to_test_id)+".txt","r")
    split_file = open(split_dir+"/fold_"+str(split_to_test_id)+".txt","r")
    lines = split_file.readlines()
    audio_names = []
    target_names = []
    for line in lines:
        audio_names.append(line.split(' ')[0]+'.wav')
        target_names.append(id_to_ix[line.split(' ')[2][0:-1]])





    #meta_dict = read_metadata(csv_path, classes_num, id_to_ix)

    '''if mini_data:
        mini_num = 10
        for key in meta_dict.keys():
            meta_dict[key] = meta_dict[key][0 : mini_num]'''

    #audios_num = len(meta_dict['audio_name'])
    audios_num = len(lines)



    # Pack waveform to hdf5
    total_time = time.time()

    with h5py.File(waveforms_hdf5_path, 'w') as hf:
        hf.create_dataset('audio_name', shape=((audios_num,)), dtype='S20')
        hf.create_dataset('waveform', shape=((audios_num, clip_samples)), dtype=np.int16)
        hf.create_dataset('target', shape=((audios_num, classes_num)), dtype=np.bool)
        hf.attrs.create('sample_rate', data=sample_rate, dtype=np.int32)

        targets = np.zeros((audios_num, classes_num), dtype=np.bool)

        # Pack waveform & target of several audio clips to a single hdf5 file
        for n in range(audios_num):
            audio_path = os.path.join(audios_dir, audio_names[n])
            if os.path.isfile(audio_path):
                logging.info('{} {}'.format(n, audio_path))
                (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
                audio = pad_or_truncate(audio, clip_samples)

                hf['audio_name'][n] = audio_names[n].encode()
                hf['waveform'][n] = float32_to_int16(audio)
                
                ix = target_names[n]
                if ix < classes_num:
                    targets[n, ix] = 1
                logging.info('{}'.format(targets[n]))
                hf['target'][n] = targets[n]
            else:
                logging.info('{} File does not exist! {}'.format(n, audio_path))

    logging.info('Write to {}'.format(waveforms_hdf5_path))
    logging.info('Pack hdf5 time: {:.3f}'.format(time.time() - total_time))
          

def pack_waveforms_to_hdf5(args):
    """Pack waveform and target of several audio clips to a single hdf5 file.
    This can speed up loading and training.
    """

    # Arguments & parameters
    audios_dir = args.audios_dir
    csv_path = args.csv_path
    waveforms_hdf5_path = args.waveforms_hdf5_path
    mini_data = args.mini_data

    split_dir = args.split_dir
    split_to_test_id = args.split_to_test_id

    clip_samples = config.clip_samples
    classes_num = config.classes_num
    sample_rate = config.sample_rate
    id_to_ix = config.id_to_ix



    #id_dict = {'screaming':'/m/03qc9zr','crying_sobbing':'/m/0463cq4','chatter':'/m/07rkbfh','motor_vehicle_(road)':'/m/012f08','emergency_vehicle':'/m/03j1ly','siren':'/m/03kmc9','explosion':'/m/014zdl','gunshot_gunfire':'/m/032s66', 'breaking': '/m/07pc8lb'}




    sample_rate = config.sample_rate
    id_to_ix = config.id_to_ix



    id_dict = {'screaming':'/m/03qc9zr','crying_sobbing':'/m/0463cq4','chatter':'/m/07rkbfh','motor_vehicle_(road)':'/m/012f08','emergency_vehicle':'/m/03j1ly','siren':'/m/03kmc9','explosion':'/m/014zdl','gunshot_gunfire':'/m/032s66', 'breaking': '/m/07pc8lb'}




    # Paths
    if mini_data:
        prefix = 'mini_'
        waveforms_hdf5_path += '.mini'
    else:
        prefix = ''

    create_folder(os.path.dirname(waveforms_hdf5_path))

    logs_dir = '_logs/pack_waveforms_to_hdf5/{}{}'.format(prefix, get_filename(csv_path))
    create_folder(logs_dir)
    create_logging(logs_dir, filemode='w')
    logging.info('Write logs to {}'.format(logs_dir))

        # Read csv file
    split_file = open(split_dir+"/fold_"+str(split_to_test_id)+".txt","r")
    #split_file = open(split_dir+"/fold_filter_"+str(split_to_test_id)+".txt","r")
    lines = split_file.readlines()
    audio_names = []
    target_names = []
    for line in lines:
        audio_names.append(line.split(' ')[0]+'.wav')
        if line.split(' ')[2][0:-1]=="others":
            target_names.append("others")
        else:
            target_names.append(id_dict[line.split(' ')[2][0:-1]])





    meta_dict = read_metadata(csv_path, classes_num, id_to_ix)

    '''if mini_data:
        mini_num = 10
        for key in meta_dict.keys():
            meta_dict[key] = meta_dict[key][0 : mini_num]'''

    #audios_num = len(meta_dict['audio_name'])
    audios_num = len(lines)



    # Pack waveform to hdf5
    total_time = time.time()

    with h5py.File(waveforms_hdf5_path, 'w') as hf:
        hf.create_dataset('audio_name', shape=((audios_num,)), dtype='S20')
        hf.create_dataset('waveform', shape=((audios_num, clip_samples)), dtype=np.int16)
        hf.create_dataset('target', shape=((audios_num, classes_num)), dtype=np.bool)
        hf.attrs.create('sample_rate', data=sample_rate, dtype=np.int32)

        targets = np.zeros((audios_num, classes_num), dtype=np.bool)

        # Pack waveform & target of several audio clips to a single hdf5 file
        for n in range(audios_num):
            audio_path = os.path.join(audios_dir, audio_names[n])

            if os.path.isfile(audio_path):
                logging.info('{} {}'.format(n, audio_path))
                (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
                audio = pad_or_truncate(audio, clip_samples)

                hf['audio_name'][n] = audio_names[n].encode()
                hf['waveform'][n] = float32_to_int16(audio)
                ix = id_to_ix[target_names[n]]
                targets[n, ix] = 1
                #print(targets[n])
                hf['target'][n] = targets[n]
            else:
                logging.info('{} File does not exist! {}'.format(n, audio_path))

    logging.info('Write to {}'.format(waveforms_hdf5_path))
    logging.info('Pack hdf5 time: {:.3f}'.format(time.time() - total_time))

def filter_and_rename_others(args):
    csv_dir = args.csv_dir
    data_dir = args.data_dir
    output_dir = args.output_dir 
    split_dir = args.split_dir    

    csv_files_name = []
    #for i in range(8):
    csv_files_name.append(csv_dir)
    audio_label_lists = []    
    audio_name_lists = []

    black_label_list = ['/m/03qc9zr','/m/0463cq4','/m/07rkbfh','/m/012f08','/m/03j1ly','/m/03kmc9','/m/014zdl','/m/032s66','/m/07pc8lb']



    for name in csv_files_name:
        with open(name, 'r') as f:
            lines = f.readlines()
        lines = lines[3:]
        
        for (n, line) in enumerate(lines):
            items = line.split(', ')

            audio_id = items[0]           
            audio_name_lists.append('Y'+ audio_id+".wav")
            audio_labels = items[3][1:-2].split(",")
            #print(audio_labels)
            audio_label_lists.append(audio_labels)

            #existing_audio_names_list.append(audio_id)
    #print(audio_label_lists)
    
    f1 = open(split_dir+"/fold_0.txt","w") 
    f2 = open(split_dir+"/fold_1.txt","w")
    f3 = open(split_dir+"/fold_2.txt","w")
    f4 = open(split_dir+"/fold_3.txt","w")
    f5 = open(split_dir+"/fold_4.txt","w")

    file_list = []
    file_list.append(f1)
    file_list.append(f2)
    file_list.append(f3)
    file_list.append(f4)
    file_list.append(f5)
       

    #with open(name, 'r') as f:
    #    lines = f.readlines()    

    k=0
    for i in range(len(audio_name_lists)):
        #print(data_dir + audio_name_lists[i])
        #print(output_dir + "others-"+format(i, '05d')+".wav")
        flag = 1
        #shutil.copyfile(data_dir + audio_name_lists[i], output_dir + "others-"+format(i, '05d')+".wav") 
        fold_id = i%5
        #print(audio_label_lists[i])
         
        for label in audio_label_lists[i]:
            if label in black_label_list:
                flag = 0   
        if flag == 1 and os.path.exists(data_dir + '/'+ audio_name_lists[i]):# and k<=2000:
            k+=1
            shutil.copyfile(data_dir + '/'+ audio_name_lists[i], output_dir + "others-"+format(i, '05d')+".wav")
            file_list[fold_id].write("others-"+format(i, '05d') +" XXX others\n")
        if k%10 == 0:
            print(k)
    #    os.rename(data_dir+audio_names[i],data_dir+"others-"+format(i, '05d')+".wav")

    print(len(audio_name_lists))
    print(k)        
        

    #with open(map_path, 'r') as f:
    #    lines = f.readlines()
    
    



def filter_wavs(args):
    csv_dir = args.csv_dir
    split_dir = args.split_dir
    map_path = args.map_path
    
    csv_files_name = []
    for i in range(8):
        csv_files_name.append(csv_dir+"/unbalanced_train_segments_part0"+str(i)+".csv")
    
    existing_audio_names_list = []

    print("opening csv files")


    for name in csv_files_name:
        with open(name, 'r') as f:
            lines = f.readlines()
        lines = lines[3:]
        for (n, line) in enumerate(lines):
            items = line.split(', ')
            audio_id = items[0]
            existing_audio_names_list.append(audio_id)
    
    with open(map_path, 'r') as f:
        lines = f.readlines()

    maping_dict_id2name = {}
    maping_dict_name2id = {}
    for (n,line) in enumerate(lines):
        items = line.split(' ')
        items_2 = items[0]
        items_3 = items_2.split('/audio_raw/')
        items_4 = items_3[1]
        items_5 = items_4.split('_')
        audio_name_1 = items_5[0]
        
        #items = line.split(', ')

        #print(len(items))
        #print(items)
        items_2 = items[1]
        items_3 = items_2.split('/audio/')
        items_4 = items_3[1]
        items_5 = items_4.split('-')
        audio_name_2_id = items_5[1][0:5]
 
        maping_dict_id2name[audio_name_2_id] = audio_name_1
        maping_dict_name2id[audio_name_1] = audio_name_2_id
    print("opening mapping dict")
    #print(maping_dict)
    

    print("scan the existing names in training")
    existing_audio_names_within_training = []
    for name in existing_audio_names_list:
        if maping_dict_name2id.get(name) is not None:
            existing_audio_names_within_training.append(name)
            


    for i in range(5):
        split_file = open(split_dir+"/fold_"+str(i)+".txt","r")
        lines = split_file.readlines()
        #audio_names = []
        #target_names = []
        output_lines = []
        #for line in lines:

 
        for (n,line) in enumerate(lines):
            audio_name_id= line.split(' ')[0].split('-')[1]
                

            if  maping_dict_id2name.get(audio_name_id) not in existing_audio_names_within_training :
                output_lines.append(line)
        #target_names.append(id_dict[line.split(' ')[2][0:-1]])

              
        #existing_audio_names_list.append(audio_id)   
       
        print("writing into one file")
    
        with open(split_dir+"/fold_filter_"+str(i)+".txt", 'w') as f:
            f.writelines(output_lines)
            f.close()








if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_split = subparsers.add_parser('split_unbalanced_csv_to_partial_csvs')
    parser_split.add_argument('--unbalanced_csv', type=str, required=True, help='Path of unbalanced_csv file to read.')
    parser_split.add_argument('--unbalanced_partial_csvs_dir', type=str, required=True, help='Directory to save out split unbalanced partial csv.')

    parser_download_wavs = subparsers.add_parser('download_wavs')
    parser_download_wavs.add_argument('--csv_path', type=str, required=True, help='Path of csv file containing audio info to be downloaded.')
    parser_download_wavs.add_argument('--audios_dir', type=str, required=True, help='Directory to save out downloaded audio.')
    parser_download_wavs.add_argument('--mini_data', action='store_true', default=True, help='Set true to only download 10 audios for debugging.')

    parser_pack_wavs = subparsers.add_parser('pack_waveforms_to_hdf5')
    parser_pack_wavs.add_argument('--csv_path', type=str, required=True, help='Path of csv file containing audio info to be downloaded.')
    parser_pack_wavs.add_argument('--audios_dir', type=str, required=True, help='Directory to save out downloaded audio.')
    parser_pack_wavs.add_argument('--waveforms_hdf5_path', type=str, required=True, help='Path to save out packed hdf5.')
    parser_pack_wavs.add_argument('--mini_data', action='store_true', default=False, help='Set true to only download 10 audios for debugging.')

    parser_pack_wavs.add_argument('--split_dir', type=str, required=True, help='Path of the split files')
    parser_pack_wavs.add_argument('--split_to_test_id', type=int, required=True,help='Int, ID, to use to test in the split')

    parser_filter_wavs = subparsers.add_parser('filter_wavs')
    parser_filter_wavs.add_argument('--csv_dir', type=str, required=True, help='Path of csv file containing audio info to be downloaded.')
    parser_filter_wavs.add_argument('--split_dir', type=str, required=True, help='Path of the split files')
    parser_filter_wavs.add_argument('--map_path', type=str, required=True, help='Path of the mapping file')


    parser_project_pack_wavs = subparsers.add_parser('pack_project_waveforms_to_hdf5')
    parser_project_pack_wavs.add_argument('--csv_path', type=str, required=True, help='Path of csv file containing audio info to be downloaded.')
    parser_project_pack_wavs.add_argument('--audios_dir', type=str, required=True, help='Directory to save out downloaded audio.')
    parser_project_pack_wavs.add_argument('--waveforms_hdf5_path', type=str, required=True, help='Path to save out packed hdf5.')
    parser_project_pack_wavs.add_argument('--mini_data', action='store_true', default=False, help='Set true to only download 10 audios for debugging.')
    parser_project_pack_wavs.add_argument('--split_dir', type=str, required=True, help='Path of the split files')
    parser_project_pack_wavs.add_argument('--split_to_test_id', type=int, required=True,help='Int, ID, to use to test in the split')

    
  

    parser_filter_and_rename_others = subparsers.add_parser('filter_and_rename_others')
    parser_filter_and_rename_others.add_argument('--csv_dir', type=str, required=True, help='Path of csv file containing audio info to be downloaded.')
    #parser_filter_and_rename_others.add_argument('--split_dir', type=str, required=True, help='Path of the split files')
    #parser_filter_and_rename_others.add_argument('--map_path', type=str, required=True, help='Path of the mapping file')
    parser_filter_and_rename_others.add_argument('--data_dir', type=str, required=True, help='Path of wav file containing audio info to be downloaded.')
    parser_filter_and_rename_others.add_argument('--output_dir', type=str, required=True, help='Path of wav file containing audio info to be downloaded.') 
    parser_filter_and_rename_others.add_argument('--split_dir', type=str, required=True, help='Path of wav file containing audio info to be downloaded.')



 
    args = parser.parse_args()
    
    if args.mode == 'split_unbalanced_csv_to_partial_csvs':
        split_unbalanced_csv_to_partial_csvs(args)
    
    elif args.mode == 'download_wavs':
        download_wavs(args)

    elif args.mode == 'pack_waveforms_to_hdf5':
        pack_waveforms_to_hdf5(args)

    elif args.mode == 'pack_project_waveforms_to_hdf5':
        pack_project_waveforms_to_hdf5(args)

    elif args.mode == 'filter_wavs':
        filter_wavs(args)
    elif args.mode == 'filter_and_rename_others':
        filter_and_rename_others(args)
    else:
        raise Exception('Incorrect arguments!')