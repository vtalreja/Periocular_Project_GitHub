


import numpy as np
import os
import glob
import shutil





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




if __name__ == '__main__':
    Image_folder = '/home/n-lab/Documents/Periocular_project/Datasets/ubipr/UBIPeriocular'
    Class_folder = '/home/n-lab/Documents/Periocular_project/Datasets/ubipr/Class_Folders'
    if not (os.path.exists(Class_folder)):
        os.mkdir(Class_folder)
    # create_class_folders(Image_folder, Class_folder)
    # count_left_right_eye(Class_folder)












