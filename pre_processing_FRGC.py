import numpy as np
import os
import glob
import shutil
from skimage.io import imread, imsave
from utils import *
from collections import Counter


def rem_files(Image_Folder):
    for ext in ['/*.jpg']:
        li_files = glob.glob(Image_folder + ext)
        li_files = sorted(li_files)
        count = 0
        for file_name in li_files:
            base_name = os.path.basename(file_name)
            image_num = int(base_name.split('_')[1])
            if image_num > 1:
                os.remove(file_name)
                count += 1
    print(count)


def create_class_folders(Image_folder, Destination_folder, eye=None):
    '''This function is used to read an image folder and separate out the images and text files in that
    folder into different class/subject folders based on the image filename   '''
    li_class_folder = []
    Count_list = []
    count = 1
    for ext in ['/*.jpg']:
        li_files = glob.glob(Image_folder + ext)
        li_files = sorted(li_files)
        for file_name in li_files:
            base_name = os.path.basename(file_name)
            class_name = base_name.split('d')[0]  # Get the class name from the file name to create the class folder
            if eye:
                dest = os.path.join(Destination_folder, class_name + eye)
            else:
                dest = os.path.join(Destination_folder, class_name)
            if class_name in li_class_folder:
                shutil.copy(file_name, dest)
                count = count + 1
            else:
                Count_list.append(count)
                count = 1
                li_class_folder.append(class_name)
                os.mkdir(dest)
                shutil.copy(file_name, dest)
    print(Counter(Count_list[1:]))
    print(len(Count_list[1:]))
    print(min(Count_list[1:]))
    print(max(Count_list[1:]))
    print(len(li_class_folder))
    print(li_class_folder)
    print(Count_list)


def create_csv_for_attrbiute(folder):
    fd = open(
        '/home/n-lab/Documents/Periocular_project/Datasets/FRGC/FRGC-2.0-dist/bbaselite_FRGC_2.0/bbaselite_data_frgc2.0.sql',
        'r')
    sqlfile = fd.readlines()
    class_label_list = [sqlfile[i].split('\t')[0].strip().split('S')[1] for i in range(10, 578)]
    gender_label_list = [sqlfile[i].split('\t')[6].strip() for i in range(10, 578)]
    ethnicity_label_list = [sqlfile[i].split('\t')[2].strip() for i in range(10, 578)]
    all_data = []
    no_attrbiute_list = []
    class_names = os.listdir(folder)
    class_names = sorted(class_names)
    for class_name in class_names:
        class_folder = os.path.join(folder, class_name)
        li_files = glob.glob(class_folder + '/*.jpg')
        li_files = sorted(li_files)
        for file_name in li_files:
            base_name = os.path.basename(file_name)
            image_name = base_name.split('.')[0]
            if class_name[0:5] in class_label_list:
                idx = class_label_list.index(class_name[0:5])
                ethnicity_label = ethnicity_label_list[idx]
                if '04367' in class_name:
                    ethnicity_label = 'White'
                elif '04360' in class_name or '04579' in class_name:
                    ethnicity_label = 'Asian'
                elif '04305' in class_name:
                    ethnicity_label = 'Asian'
                elif '04282' in class_name:
                    ethnicity_label = 'Hispanic'
                if 'Unknown' in ethnicity_label:
                    ethnicity_label = 'Hispanic'
                if 'Black' in ethnicity_label:
                    ethnicity_label = 'Hispanic'
                if 'Southern' in ethnicity_label or 'Middle' in ethnicity_label:
                    ethnicity_label = 'Asian'
                all_data.append([base_name, class_name, gender_label_list[idx], ethnicity_label])
            else:
                if class_name not in no_attrbiute_list:
                    no_attrbiute_list.append(class_name)

    all_data = np.array(all_data)
    save_csv(data=all_data, path='attribute_data_FRGC_Spring2004.csv',
             fieldnames=['id', 'class_name_label', 'gender_label', 'ethnicity_label'])
    print(no_attrbiute_list)
    print(len(no_attrbiute_list))


def create_dir(folder):
    if not (os.path.exists(folder)):
        os.mkdir(folder)


def count_images(folder):
    print(len(os.listdir(folder)))
    for subj in os.listdir(folder):
        subject_folder = os.path.join(folder, subj)
        li_files = os.listdir(subject_folder)
        print('subject id : {} , count " {}'.format(subj, len(li_files)))


if __name__ == '__main__':
    Image_folder = '/home/n-lab/Documents/Periocular_project/Datasets/FRGC_dataset_extracted_periocular_images_MTCNN/Spring2004_cropped_Left_Eye'
    Class_folder = '/home/n-lab/Documents/Periocular_project/Datasets/FRGC_dataset_extracted_periocular_images_MTCNN/Spring2004_cropped_Class_Folders'
    # Image_folder = '/home/n-lab/Documents/Periocular_project/Datasets/FRGC_dataset_extracted_periocular_images/Fall2002_Images_copped_periocular_cleaned'
    # Class_folder = '/home/n-lab/Documents/Periocular_project/Datasets/FRGC_dataset_extracted_periocular_images/Class_Folders'
    # Image_folder = '/home/n-lab/Documents/Periocular_project/Datasets/FRGC_dataset_extracted_periocular_images/Fall2002_Images_copped_periocular_cleaned_Features_UBIPr'
    # Class_folder = '/home/n-lab/Documents/Periocular_project/Datasets/FRGC_dataset_extracted_periocular_images/Fall2002_Images_copped_periocular_cleaned_Features_UBIPr_Class_Folders'

    # count_images(Class_folder)
    # rem_files(Image_folder)
    # create_dir(Class_folder)
    # create_class_folders(Image_folder, Class_folder,'L')
    create_csv_for_attrbiute(Class_folder)
    # rem_files(Image_folder)
