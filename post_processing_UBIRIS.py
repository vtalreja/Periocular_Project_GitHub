import numpy as np
import os
import random
import glob
import shutil
import pandas as pd
import csv
import matplotlib.pyplot as plt


def create_class_folders_npy_txt_FRGC(folder, dest_folder):
    '''Function to create class folders from the numpy files and also create a complete text file with file names and a full numpy file with concatenated numpys '''

    li_class_folder = []
    all_subjs_numpy = np.zeros((1, 512))
    file_name_dict = {}
    file_name_dict['file_names'] = []
    file_name_dict['file_location'] = []
    if not (os.path.exists(dest_folder)):
        os.mkdir(dest_folder)
    for ext in ['/*.npy']:
        li_files = glob.glob(folder + ext)
        li_files = sorted(li_files)
        for file_name in li_files:
            base_name = os.path.basename(file_name)
            class_name = base_name.split('d')[0]  # Get the class name from the file name to create the class folder
            subj_folder =  os.path.join(dest_folder, class_name)
            if class_name in li_class_folder:
                shutil.copy(file_name, subj_folder)
            else:
                li_class_folder.append(class_name)
                os.mkdir(os.path.join(dest_folder, class_name))
                shutil.copy(file_name, subj_folder)
            # for sample in selected_nps:
            sample_numpy = np.load(file_name)
            all_subjs_numpy = np.concatenate((all_subjs_numpy, sample_numpy), axis=0)
            file1 = open(os.path.join(dest_folder, 'file_names.txt'), "a")
            file1.write(base_name + ',' + subj_folder + "\n")
            file_name_dict['file_names'].append(base_name)
            file_name_dict['file_location'].append(subj_folder)
    subjs_numpy = all_subjs_numpy[1:, :]
    np.save(os.path.join(dest_folder, 'all_subj_features.npy'), subjs_numpy)
    df = pd.DataFrame(file_name_dict, columns=['file_names', 'file_location'])
    df.to_csv(os.path.join(dest_folder, 'file_names.csv'), index=False,
              header=True)  # Save the csv


def selection_images_UBIRIS_V2(folder_1, folder_2, num_of_subjs, samples_per_subj, dest_folder):
    '''Function to select about 1000 images from the UBIRIS_V2 features from 2 different folders corresponding to num_of_subjs subjects and samples_per_subj '''

    all_subjs_numpy = np.zeros((1, 512))
    file_name_dict = {}
    file_name_dict['file_names'] = []
    file_name_dict['file_location'] = []
    if not (os.path.exists(dest_folder)):
        os.mkdir(dest_folder)
    li_ids = list(range(1, 520, 2))
    selected_ids = []
    while len(selected_ids) != 150:
        selected_ids = random.sample(li_ids, num_of_subjs)
        for val_deleted in [407, 409]:
            if val_deleted in selected_ids:
                selected_ids.remove(val_deleted)
    selected_ids = sorted(selected_ids)
    for id in selected_ids:
        if id <= 260:
            folder = folder_1
        else:
            folder = folder_2
        li_nps = glob.glob(folder + '/C' + str(id) + '_*.npy')
        subj_folder = os.path.join(dest_folder, 'C' + str(id))
        print(id, len(li_nps))
        if len(li_nps) >= samples_per_subj:
            if not (os.path.exists(subj_folder)):
                os.mkdir(subj_folder)
            selected_nps = random.sample(li_nps, samples_per_subj)
            selected_nps = sorted(selected_nps)
            for sample in selected_nps:
                sample_numpy = np.load(sample)
                all_subjs_numpy = np.concatenate((all_subjs_numpy, sample_numpy), axis=0)
                base_name = os.path.basename(sample)
                file1 = open(os.path.join(dest_folder, 'file_names.txt'), "a")
                file1.write(base_name + ',' + subj_folder + "\n")

                file_name_dict['file_names'].append(base_name)
                file_name_dict['file_location'].append(subj_folder)
                shutil.copy(os.path.join(folder, base_name), subj_folder)
    subjs_numpy = all_subjs_numpy[1:, :]
    np.save(os.path.join(dest_folder, 'all_subj_features.npy'), subjs_numpy)
    df = pd.DataFrame(file_name_dict, columns=['file_names', 'file_location'])
    df.to_csv(os.path.join(dest_folder, 'file_names.csv'), index=False,
              header=True)  # Save the csv


def draw_cmc():
    x_axis_list = [1, 5, 10, 15, 20]
    y_axis_list = [.889, .97, .977, .984, .991]
    SCNN = [.824, .945, .965, .973, .982]
    TIP = [.7456, .88, .925, .94, .95]
    TIFS = [.7108, .84, .885, .91, .94]
    # y_axis_list = [.874, .924, .955, .966, .978]
    # SCNN = [.911, .965, .975, .977, .98]
    # TIP = [.854, .91, .945, .965, .975]
    # TIFS = [.783, .865, .905, .92, .93]
    plt.figure(figsize=(10, 5))
    plt.plot(x_axis_list, y_axis_list, label='Ours (Rank1 = 88.9%')
    plt.plot(x_axis_list, SCNN, label='TIFS 17 (Rank1 = 82.4%)')
    plt.plot(x_axis_list, TIP, label='TIP 13 (Rank1 = 74.56%) ')
    plt.plot(x_axis_list, TIFS, label='TIFS 15 (Rank1 = 71.08%)')
    # plt.plot(x_axis_list, y_axis_list, label='Ours (Rank1 = 87.4%')
    # plt.plot(x_axis_list, SCNN, label='TIFS 17 (Rank1 = 91.13%)')
    # plt.plot(x_axis_list, TIP, label='TIP 13 (Rank1 = 85.36%) ')
    # plt.plot(x_axis_list, TIFS, label='TIFS 15 (Rank1 = 78.36%)')
    plt.title("CMC curve for UBIRIS_V2")
    plt.xlabel('Rank')
    plt.ylabel('Recognition Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('/home/n-lab/Documents/Periocular_Project_GitHub/Figs', 'CMC_UBIRIS_V2.png'))


def measure_euclidean_dist(text_file_loc, image_array_loc, dest_folder):
    '''Function to measure euclidean distance between different images and save it into Csvs'''

    # Open the text file to read the file names and unique ids for the images to be compared
    test_images_f = open(text_file_loc, 'r')

    # dest_folder = '/home/n-lab/Documents/Periocular_project/Datasets/UBIris_v2/Selected_Test_Images_Features_Scores_CSVs' # Folder to save the csv files
    if not (os.path.exists(dest_folder)):
        os.mkdir(dest_folder)

    # Read the text file all lines at a time as a list of strings where each string corresponds to a single line from the text file
    test_images_d = test_images_f.readlines()  # We will be having 1050 rows and  two columns where the first column is the file name and the second column is the location of that file.

    # Load the numpy file that is actually the complete image array of all the 1050 images
    image_array = np.load(image_array_loc)

    # Just a toy array
    # image_array = np.array([[1,2,3,4,5],[2,5,3,1,0],[4,5,76,8,1],[3,4,7,8,0],
    #               [1,0,0,1,0],[1,1,1,0,0],[0,0,0,0,1],[0,0,1,0,0]]) # shape 8 x 5

    for i in range(
            image_array.shape[0]):  # Looping through the image array to consider a different gallery image everytime.
        matching_score_dict = {'gallery_filename': [], 'probe_filename': [],
                               'Score': []}  # Dictionary to store the gallery file name and probe file names. This dict is useful to save the result as csv
        Gallery_image = image_array[i, :]  # creation of the gallery image
        Gallery_image = np.expand_dims(Gallery_image,
                                       axis=0)  # Expand the dimension so that the gallery image is of shape 128 x 512 x 1. This will make it easier to duplicate the gallery image along the dimension 2
        Gallery_matrix = np.repeat(Gallery_image, image_array.shape[0] - 1,
                                   0)  # Repeat the gallery image array to create the Gallery matrix for comparuison
        Probe_matrix = np.delete(image_array, i,
                                 0)  # Delete the gallery image from the complete image matrix to create the probe matrix
        dropped_column = np.delete(range(image_array.shape[0]), i,
                                   0)  # He we are trying to drop the column i in order to create the probe file names to be saved in the csv file
        Gallery_file_name = test_images_d[i].strip().split(',')[
            0]  # Get the name of the gallery image file name from the text file
        matching_score_dict['gallery_filename'] = list(np.repeat(Gallery_file_name, image_array.shape[
            0] - 1))  # repeat the gallery file name and create a list to be saved in the dictionary for the csv saving
        matching_score_dict['probe_filename'] = [test_images_d[j].strip().split(',')[0] for j in
                                                 dropped_column]  # Get the file names of all the probe images and assign it to the dict
        matching_score_dict['Score'] = list(np.linalg.norm(Gallery_matrix - Probe_matrix,
                                                           axis=1))  # Calculate the Euclidean distance between the gallery matrix and probe matrix
        df = pd.DataFrame(matching_score_dict, columns=['gallery_filename', 'probe_filename',
                                                        'Score'])  # Create the dataframe from the matching_score_dict to be saved in the csv
        df.to_csv(os.path.join(dest_folder, '{}.csv'.format(os.path.splitext(Gallery_file_name)[0])), index=False,
                  header=True)  # Save the csv



def genuine_impostor_scores(folder,split_str='_'):
    genuine_list = []
    impostor_list = []
    li_csvs = os.listdir(folder)
    for csv_file in li_csvs:
        print(csv_file)
        df = pd.read_csv(os.path.join(folder,csv_file))
        gen_df = df[df['gallery_filename'].str.split(split_str,n=1,expand=True)[0] == df['probe_filename'].str.split(split_str,n=1,expand=True)[0]]
        impostor_df = df[df['gallery_filename'].str.split(split_str,n=1,expand=True)[0] != df['probe_filename'].str.split(split_str,n=1,expand=True)[0]]
        genuine_list.extend(list(gen_df['Score']))
        impostor_list.extend(list(impostor_df['Score']))
    print(len(genuine_list))
    print(len(impostor_list))

    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=genuine_list, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    n, bins, patches = plt.hist(x=impostor_list, bins='auto', color='red',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Genuine vs Impostor')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig('histogram_FRGC_model_FRGC.png')


Image_folder_FRGC = '/home/n-lab/Documents/Periocular_project/Datasets/FRGC_dataset_extracted_periocular_images/Fall2002_Images_copped_periocular_cleaned_Features_FRGC'
Class_folder_FRGC = '/home/n-lab/Documents/Periocular_project/Datasets/FRGC_dataset_extracted_periocular_images/Fall2002_Images_copped_periocular_cleaned_Features_FRGC_Class_Folders'
CSVs_folder_FRGC = '/home/n-lab/Documents/Periocular_project/Datasets/FRGC_dataset_extracted_periocular_images/Fall2002_Images_copped_periocular_cleaned_Features_FRGC_Class_Folders_Features_Scores_CSVs'
create_class_folders_npy_txt_FRGC(Image_folder_FRGC,Class_folder_FRGC)
# selection_images('/home/n-lab/Documents/Periocular_project/Datasets/UBIris_v2/CLASSES_400_300_Part1_Features','/home/n-lab/Documents/Periocular_project/Datasets/UBIris_v2/CLASSES_400_300_Part2_Features',150,7,'/home/n-lab/Documents/Periocular_project/Datasets/UBIris_v2/Selected_Test_Images_Features_Class_Folders_1')
# measure_euclidean_dist(
#     text_file_loc='/home/n-lab/Documents/Periocular_project/Datasets/UBIris_v2/Selected_Test_Images_Features_Class_Folders/file_names.txt',
#     image_array_loc='/home/n-lab/Documents/Periocular_project/Datasets/UBIris_v2/Selected_Test_Images_Features_Class_Folders/all_subj.features.npy',
#     dest_folder='/home/n-lab/Documents/Periocular_project/Datasets/UBIris_v2/Selected_Test_Images_Features_Scores_CSVs')
# genuine_impostor_scores(folder='/home/n-lab/Documents/Periocular_project/Datasets/UBIris_v2/Selected_Test_Images_Features_Scores_CSVs')
measure_euclidean_dist(text_file_loc=os.path.join(Class_folder_FRGC,'file_names.txt'),image_array_loc=os.path.join(Class_folder_FRGC,'all_subj_features.npy'), dest_folder=CSVs_folder_FRGC)
genuine_impostor_scores(CSVs_folder_FRGC,split_str='d')
# draw_cmc()
