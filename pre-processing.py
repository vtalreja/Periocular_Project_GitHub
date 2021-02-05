import numpy as np
import os
import glob
import shutil
from skimage.io import imread, imsave


def create_class_folders(Image_folder, Destination_folder):
    '''This function is used to read an image folder and separate out the images and text files in that
    folder into different class/subject folders based on the image filename   '''
    li_class_folder = []
    male_list = []
    female_list = []
    for ext in ['/*.jpg', '/*.txt']:
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
    print("Number of Males is {}, Females is {}".format(len(male_list), len(female_list)))
    common_list = [l for l in male_list if l in female_list]
    # combined_list = male_list + female_list
    # difference_list = (list(list(set(combined_list)-set(li_class_folder)) + list(set(li_class_folder)-set(combined_list))))
    print(common_list)


def count_left_right_eye(Image_dir):
    '''Function to count the number of left eye folders and right eye folders in the Image_dir and also count if there are any overlaps and mixups'''
    class_names = os.listdir(Image_dir)
    class_names = sorted(class_names)
    left_eye_list = []
    right_eye_list = []
    for class_name in class_names:
        class_folder = os.path.join(Image_dir, class_name)
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
    print("Number of left eyes is {}, right eyes is {}".format(len(left_eye_list), len(right_eye_list)))
    print(left_eye_list)
    print(right_eye_list)
    common_list = [l for l in left_eye_list if l in right_eye_list]
    print(common_list)


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
    for i in range(1, 517, 2):
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


def count_classes_more_images(Image_dir,count):
    '''Find the number of subjects who have more than 'count' images'''
    class_names = os.listdir(Image_dir)
    class_names = sorted(class_names)
    li_classes = []
    li_empty_folders = []
    for class_name in class_names:
        class_folder = os.path.join(Image_dir, class_name)
        li_files = glob.glob(class_folder + '/*.jpg')
        if len(li_files) > count:
            li_classes.append(class_name)
        if len(li_files) == 0:
            li_empty_folders.append(class_name)
    print(
        'Total number of subjects with more than {} images are {} and they are {}'.format(count,len(li_classes), li_classes))
    print(
        'Total number of subjects with 0 images are {} and they are {}'.format(len(li_empty_folders), li_empty_folders))

def split_classes_more_images(Image_dir,count,Only_Images):
    class_names = os.listdir(Image_dir)
    class_names = sorted(class_names)
    li_classes = []
    for class_name in class_names:
        class_folder = os.path.join(Image_dir, class_name)
        if Only_Images:
            li_files = glob.glob(class_folder + '/*.jpg')
        else:
            li_files = os.listdir(class_folder)
        if len(li_files) > count:
            li_classes.append(class_name)
            for file_name in li_files:
                if Only_Images:
                    base_name = os.path.basename(file_name)
                else:
                    base_name = file_name
                if 'S2' in base_name:
                    if not os.path.exists (class_folder+'_S2'):
                        os.mkdir(class_folder+'_S2')
                    shutil.move(os.path.join(class_folder,file_name),class_folder+'_S2')





def split_male_female(Image_dir, Destination_folder):
    '''Split the Consolidated folders into male and Female folders'''
    class_names = os.listdir(Image_dir)
    class_names = sorted(class_names)
    os.mkdir(os.path.join(Destination_folder, 'Male'))
    os.mkdir(os.path.join(Destination_folder, 'Female'))
    male_folder = os.path.join(Destination_folder, 'Male')
    female_folder = os.path.join(Destination_folder, 'Female')
    male_list = []
    female_list = []
    for class_name in class_names:
        class_folder = os.path.join(Image_dir, class_name)
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
                    male_class_folder = os.path.join(male_folder, class_name)
                    male_list.append(class_name)
                # shutil.copy(file_name, male_class_folder)
                shutil.copy(image_name + '.jpg', male_class_folder)
            if 'Female' in txt_contents[6]:
                if class_name not in female_list:
                    os.mkdir(os.path.join(female_folder, class_name))
                    female_class_folder = os.path.join(female_folder, class_name)
                    female_list.append(class_name)
                # shutil.copy(file_name, female_class_folder)
                shutil.copy(image_name + '.jpg', female_class_folder)


def copy_images_training(Image_dir, Destination_folder):
    '''Move only the image with height 561 or 651 to the Destination folder'''
    gender_dict = {}
    for gender in ['Male', 'Female']:
        gender_dict[gender] = []
        gender_folder = os.path.join(Image_dir, gender)
        class_names = os.listdir(gender_folder)
        class_names = sorted(class_names)
        dest_gender_folder = os.path.join(Destination_folder, gender)
        os.mkdir(dest_gender_folder)
        for class_name in class_names:
            class_folder = os.path.join(gender_folder, class_name)
            li_files = glob.glob(class_folder + '/*.jpg')
            li_files = sorted(li_files)
            for file_name in li_files:
                image_name = file_name.split('.')[0]
                image_file = imread(file_name)
                if image_file.shape[1] in [561, 651]:
                    # file1 = open(file_name, 'r')
                    # txt_contents = file1.readlines()
                    # if txt_contents[-1] in ['561;','651;']:
                    if class_name not in gender_dict[gender]:
                        dest_class_folder = os.path.join(dest_gender_folder, class_name)
                        os.mkdir(dest_class_folder)
                        gender_dict[gender].append(class_name)
                    shutil.copy(file_name, dest_class_folder)
                    # shutil.copy(image_name+'.txt', dest_class_folder)


def create_dir(folder):
    if not (os.path.exists(folder)):
        os.mkdir(folder)

def diff_two_folders(folder_1, folder_2):
    li_files_1=os.listdir(folder_1)
    li_files_2=os.listdir(folder_2)

    difference_list = (list(list(set(li_files_1)-set(li_files_2)) + list(set(li_files_2)-set(li_files_1))))
    print(difference_list)



if __name__ == '__main__':
    Image_folder = '/home/n-lab/Documents/Periocular_project/Datasets/ubipr/UBIPeriocular'
    Class_folder = '/home/n-lab/Documents/Periocular_project/Datasets/ubipr/Class_Folders'
    Class_folder_single_eye = '/home/n-lab/Documents/Periocular_project/Datasets/ubipr/Class_Folders_Left_Images'
    Class_folder_single_eye_split = '/home/n-lab/Documents/Periocular_project/Datasets/ubipr/Class_Folders_Left_Images_Split'
    Consolidated_folder = '/home/n-lab/Documents/Periocular_project/Datasets/ubipr/Consolidated_Folders'
    Male_Female_folder = '/home/n-lab/Documents/Periocular_project/Datasets/ubipr/Male_Female_Folders'
    Male_Female_training_folder = '/home/n-lab/Documents/Periocular_project/Datasets/ubipr/Male_Female_Training_Folders_Images'

    # Creating Directories
    create_dir(Class_folder)
    create_dir(Consolidated_folder)
    create_dir(Male_Female_folder)
    create_dir(Male_Female_training_folder)
    create_dir(Class_folder_single_eye)
    create_dir(Class_folder_single_eye_split)

    # create_class_folders(Image_folder, Class_folder)
    # count_left_right_eye(Class_folder)
    # consolidate_left_right(Class_folder,Class_folder_single_eye,True,'left')
    # count_classes_more_images(Consolidated_folder,15)
    # count_classes_more_images(Class_folder_single_eye_split,15)
    # split_male_female(Consolidated_folder,Male_Female_folder)
    # copy_images_training(Male_Female_folder,Male_Female_training_folder)
    # split_classes_more_images(Class_folder_single_eye_split,30,False)
