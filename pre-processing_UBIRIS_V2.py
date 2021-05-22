import numpy as np
import os
import glob
import shutil
import pandas as pd
from skimage.io import imread, imsave
from utils import *



def create_class_folders(Image_folder, Destination_folder):
    '''This function is used to read an image folder and separate out the images and text files in that
    folder into different class/subject folders based on the image filename   '''
    li_class_folder = []
    male_list = []
    female_list = []
    for ext in ['/*.jpg']:
        li_files = glob.glob(Image_folder + ext)
        li_files = sorted(li_files)
        for file_name in li_files:
            base_name = os.path.basename(file_name)
            class_name = base_name.split('_')[0]  # Get the class name from the file name to create the class folder
            if 'jpg' in (ext):
                if class_name in li_class_folder:
                    shutil.copy(file_name, os.path.join(Destination_folder, class_name))
                else:
                    li_class_folder.append(class_name)
                    os.mkdir(os.path.join(Destination_folder, class_name))
                    shutil.copy(file_name, os.path.join(Destination_folder, class_name))
    #         else:
    #             shutil.copy(file_name, os.path.join(Destination_folder, class_name))
    #             file1 = open(file_name, 'r')
    #             txt_contents = file1.readlines()
    #             if 'Male' in txt_contents[6]:
    #                 if class_name not in male_list:
    #                     male_list.append(class_name)
    #             if 'Female' in txt_contents[6]:
    #                 if class_name not in female_list:
    #                     female_list.append(class_name)
    # print("Number of Males is {}, Females is {}".format(len(male_list), len(female_list)))
    # common_list = [l for l in male_list if l in female_list]
    # # combined_list = male_list + female_list
    # # difference_list = (list(list(set(combined_list)-set(li_class_folder)) + list(set(li_class_folder)-set(combined_list))))
    # print(common_list)


def copy_left_right_folder(Image_dir, i, final_destination_folder, Only_Images=False):
    class_folder = os.path.join(Image_dir, 'C' + str(i))
    if os.path.exists(class_folder):
        if not os.path.exists(final_destination_folder):
            os.mkdir(final_destination_folder)
        if Only_Images:
            li_files = glob.glob(class_folder + '/*.jpg')
        else:
            li_files = os.listdir(class_folder)
        for file_name in li_files:
            shutil.copy(os.path.join(class_folder, file_name),
                        final_destination_folder)


def consolidate_left_right(Image_dir, Destination_folder, Only_Images=False, eye='both'):
    '''Consolidate the left and right image folders into separate folders or consolidate
    them into a single folder depending on the flag 'eye' '''
    joint_class_name = 0
    for i in range(1, 522, 2):
        joint_class_name = joint_class_name + 1
        final_destination_folder = os.path.join(Destination_folder, 'subj_' + str(joint_class_name))
        if eye == 'left':
            copy_left_right_folder(Image_dir, i, final_destination_folder, Only_Images)
        elif eye == 'right':
            copy_left_right_folder(Image_dir, i + 1, final_destination_folder, Only_Images)
        elif eye == 'both':
            copy_left_right_folder(Image_dir, i, final_destination_folder, Only_Images)
            copy_left_right_folder(Image_dir, i + 1, final_destination_folder, Only_Images)
        else:
            raise ValueError('flag eye can take only left, right or both value')


def create_csv_for_attrbiute(Image_dir,csv_file):
    all_data = []
    id_not_present=[]
    class_names = os.listdir(Image_dir)
    df = pd.read_csv(os.path.join(csv_file))
    for class_name in class_names:
        print(class_name)
        class_name_int = int(class_name[1:])
        if class_name_int % 2 == 0:
            match_class_name = 'C'+str(class_name_int-1)
        else:
            match_class_name = 'C' + str(class_name_int)
        gen_df = df[df['id'].str.split('_', n=1, expand=True)[0] ==
                    match_class_name]
        if gen_df.empty:
            id_not_present.append(class_name)
            if class_name in ['C520', 'C521', 'C519', 'C483', 'C484', 'C517', 'C518', 'C522']:
                gender_label = 'Male'
                class_folder = os.path.join(Image_dir, class_name)
                li_files = glob.glob(class_folder + '/*.jpg')
                for file_name in li_files:
                    base_name = os.path.basename(file_name)
                    all_data.append([base_name, class_name, gender_label])
            elif class_name in ['C428', 'C427']:
                gender_label = 'Female'
                class_folder = os.path.join(Image_dir, class_name)
                li_files = glob.glob(class_folder + '/*.jpg')
                for file_name in li_files:
                    base_name = os.path.basename(file_name)
                    all_data.append([base_name, class_name, gender_label])
        else:
            gender_label = gen_df['gender_label'].unique()[0]
            class_folder = os.path.join(Image_dir, class_name)
            li_files = glob.glob(class_folder + '/*.jpg')
            for file_name in li_files:
                base_name = os.path.basename(file_name)
                all_data.append([base_name, class_name, gender_label])
    all_data = np.array(all_data)
    save_csv(data=all_data,path='attribute_data_UBIRIS_V2.csv',fieldnames=['id','class_name_label','gender_label'])
    print(id_not_present,len(id_not_present))


def create_dir(folder):
    if not (os.path.exists(folder)):
        os.mkdir(folder)



if __name__ == '__main__':
    Image_folder = '/home/n-lab/Documents/Periocular_project/Datasets/UBIris_v2/CLASSES_400_300_Part2_JPG'
    Class_folder = '/home/n-lab/Documents/Periocular_project/Datasets/UBIris_v2/Class_Folders'
    Consolidated_folder = '/home/n-lab/Documents/Periocular_project/Datasets/UBIris_v2/Consolidated_Folders'

    # Creating Directories
    # create_dir(Class_folder)
    # create_dir(Consolidated_folder)

    # create_class_folders(Image_folder, Class_folder)
    # consolidate_left_right(Class_folder,Consolidated_folder,True,'both')
    create_csv_for_attrbiute(Class_folder,'attribute_data.csv')
