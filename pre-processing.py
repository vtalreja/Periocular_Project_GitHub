


import numpy as np
import os
import glob
import shutil
from skimage.io import imread,imsave






def create_class_folders(Image_folder,Destination_folder):
    '''This function is used to read an image folder and separate out the images and text files in that
    folder into different class/subject folders based on the image filename   '''
    li_class_folder = []
    male_list = []
    female_list = []
    for ext in ['/*.jpg','/*.txt']:
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
            else:
                shutil.copy(file_name, os.path.join(Destination_folder, class_name))
                file1 = open(file_name, 'r')
                txt_contents = file1.readlines()
                if 'Male' in txt_contents[6]:
                    if class_name not in male_list:
                        male_list.append(class_name)
                if 'Female' in txt_contents[6]:
                    if class_name not in female_list:
                        female_list.append(class_name)
    print("Number of Males is {}, Females is {}".format(len(male_list),len(female_list)))
    common_list = [l for l in male_list if l in female_list]
    # combined_list = male_list + female_list
    # difference_list = (list(list(set(combined_list)-set(li_class_folder)) + list(set(li_class_folder)-set(combined_list))))
    print(common_list)



def count_left_right_eye (Image_dir):
    '''Function to count the number of left eye folders and right eye folders in the Image_dir and also count if there are any overlaps and mixups'''
    class_names = os.listdir(Image_dir)
    class_names = sorted(class_names)
    left_eye_list =[]
    right_eye_list = []
    for class_name in class_names:
        class_folder = os.path.join(Image_dir,class_name)
        li_files = glob.glob(class_folder + '/*.txt')
        li_files = sorted(li_files)
        for file_name in li_files:
            file1 = open(file_name, 'r')
            txt_contents = file1.readlines()
            if 'LPoseAngle' in txt_contents[0] or 'LGazeAngle' in txt_contents[0]:
                if class_name not in left_eye_list:
                    left_eye_list.append(class_name)
            if 'RPoseAngle' in txt_contents[0] or 'RGazeAngle' in txt_contents[0]:
                if class_name not in right_eye_list:
                    right_eye_list.append(class_name)
    print("Number of left eyes is {}, right eyes is {}".format(len(left_eye_list),len(right_eye_list)))
    print(left_eye_list)
    print(right_eye_list)
    common_list = [l for l in left_eye_list if l in right_eye_list]
    print(common_list)


def consolidate_left_right (Image_dir, Destination_folder,Only_Images=False):
    '''Consolidate the left and right image folders in a single subject folder'''
    # class_names = os.listdir(Image_dir)
    # class_names = sorted(class_names)
    joint_class_name = 0
    for i in range(1,517,2):
        joint_class_name = joint_class_name + 1
        class_folder_left = os.path.join(Image_dir,'C'+str(i))
        class_folder_right = os.path.join(Image_dir,'C'+str(i+1))
        if Only_Images:
            if os.path.exists(class_folder_left):
                os.mkdir(os.path.join(Destination_folder, 'subj_' + str(joint_class_name)))
                li_files_left = glob.glob(class_folder_left + '/*.jpg')
                for file_name in li_files_left:
                    shutil.copy(os.path.join(class_folder_left, file_name),
                                os.path.join(Destination_folder, 'subj_' + str(joint_class_name)))
            if os.path.exists(class_folder_right):
                li_files_right = glob.glob(class_folder_right + '/*.jpg')
                for file_name in li_files_right:
                    shutil.copy(os.path.join(class_folder_right, file_name),
                            os.path.join(Destination_folder, 'subj_' + str(joint_class_name)))
        else:
            if os.path.exists(class_folder_left):
                os.mkdir(os.path.join(Destination_folder, 'subj_' + str(joint_class_name)))
                li_files_left = os.listdir(class_folder_left)
                for file_name in li_files_left:
                    shutil.copy(os.path.join(class_folder_left, file_name),
                                os.path.join(Destination_folder, 'subj_' + str(joint_class_name)))
            if os.path.exists(class_folder_right):
                li_files_right = os.listdir(class_folder_right)
                for file_name in li_files_right:
                    shutil.copy(os.path.join(class_folder_right, file_name),
                                os.path.join(Destination_folder, 'subj_' + str(joint_class_name)))

def count_classes_more_images (Image_dir):
    '''Find the number of subjects with more than 30 images. implies 2 sessions'''
    class_names = os.listdir(Image_dir)
    class_names = sorted(class_names)
    li_classes =[]
    li_empty_folders = []
    for class_name in class_names:
        class_folder = os.path.join(Image_dir,class_name)
        li_files = glob.glob(class_folder + '/*.jpg')
        if len(li_files) > 30:
            li_classes.append(class_name)
        if len(li_files) == 0:
            li_empty_folders.append(class_name)
    print('Total number of subjects with more than 30 images are {} and they are {}'.format(len(li_classes),li_classes))
    print('Total number of subjects with 0 images are {} and they are {}'.format(len(li_empty_folders),li_empty_folders))

def split_male_female (Image_dir,Destination_folder):
    '''Split the Consolidated folders into male and Female folders'''
    class_names = os.listdir(Image_dir)
    class_names = sorted(class_names)
    os.mkdir(os.path.join(Destination_folder, 'Male'))
    os.mkdir(os.path.join(Destination_folder, 'Female'))
    male_folder = os.path.join(Destination_folder,'Male')
    female_folder = os.path.join(Destination_folder,'Female')
    male_list = []
    female_list = []
    for class_name in class_names:
        class_folder = os.path.join(Image_dir,class_name)
        li_files = glob.glob(class_folder + '/*.txt')
        li_files = sorted(li_files)
        for file_name in li_files:
            # base_name = os.path.basename(file_name)
            image_name = file_name.split('.')[0]
            file1 = open(file_name, 'r')
            txt_contents = file1.readlines()
            if 'Male' in txt_contents[6]:
                if class_name not in male_list:
                    os.mkdir(os.path.join(male_folder, class_name))
                    male_class_folder = os.path.join(male_folder,class_name)
                    male_list.append(class_name)
                # shutil.copy(file_name, male_class_folder)
                shutil.copy(image_name+'.jpg', male_class_folder)
            if 'Female' in txt_contents[6]:
                if class_name not in female_list:
                    os.mkdir(os.path.join(female_folder, class_name))
                    female_class_folder = os.path.join(female_folder,class_name)
                    female_list.append(class_name)
                # shutil.copy(file_name, female_class_folder)
                shutil.copy(image_name+'.jpg', female_class_folder)


def copy_images_training (Image_dir,Destination_folder):
    gender_dict = {}
    for gender in ['Male','Female']:
        gender_dict[gender] = []
        gender_folder = os.path.join(Image_dir,gender)
        class_names = os.listdir(gender_folder)
        class_names = sorted(class_names)
        dest_gender_folder = os.path.join(Destination_folder,gender)
        os.mkdir(dest_gender_folder)
        for class_name in class_names:
            class_folder = os.path.join(gender_folder,class_name)
            li_files = glob.glob(class_folder + '/*.jpg')
            li_files = sorted(li_files)
            for file_name in li_files:
                image_name = file_name.split('.')[0]
                image_file = imread(file_name)
                if image_file.shape[1] in [561,651]:
                # file1 = open(file_name, 'r')
                # txt_contents = file1.readlines()
                # if txt_contents[-1] in ['561;','651;']:
                    if class_name not in gender_dict[gender]:
                        dest_class_folder = os.path.join(dest_gender_folder,class_name)
                        os.mkdir(dest_class_folder)
                        gender_dict[gender].append(class_name)
                    shutil.copy(file_name, dest_class_folder)
                    # shutil.copy(image_name+'.txt', dest_class_folder)



















if __name__ == '__main__':
    Image_folder = '/home/n-lab/Documents/Periocular_project/Datasets/ubipr/UBIPeriocular'
    Class_folder = '/home/n-lab/Documents/Periocular_project/Datasets/ubipr/Class_Folders'
    Consolidated_folder = '/home/n-lab/Documents/Periocular_project/Datasets/ubipr/Consolidated_Folders'
    Male_Female_folder = '/home/n-lab/Documents/Periocular_project/Datasets/ubipr/Male_Female_Folders'
    Male_Female_training_folder ='/home/n-lab/Documents/Periocular_project/Datasets/ubipr/Male_Female_Training_Folders_Images'

    if not (os.path.exists(Class_folder)):
        os.mkdir(Class_folder)
    if not (os.path.exists(Consolidated_folder)):
        os.mkdir(Consolidated_folder)
    if not (os.path.exists(Male_Female_folder)):
        os.mkdir(Male_Female_folder)
    if not (os.path.exists(Male_Female_training_folder)):
        os.mkdir(Male_Female_training_folder)
    # create_class_folders(Image_folder, Class_folder)
    # count_left_right_eye(Class_folder)
    # consolidate_left_right(Class_folder,Consolidated_folder,True)
    # count_classes_more_images(Consolidated_folder)
    # split_male_female(Consolidated_folder,Male_Female_folder)
    # copy_images_training(Male_Female_folder,Male_Female_training_folder)













