'''This code is used to read an image folder and separate out the images and text files in that folder into different class/subject folders based on the image filename   '''


import numpy as np
import os
import glob
import shutil


Image_folder = '/home/n-lab/Documents/Periocular_project/Datasets/ubipr/UBIPeriocular'
Destination_folder = '/home/n-lab/Documents/Periocular_project/Datasets/ubipr/Class_Folders'
if not (os.path.exists(Destination_folder)):
    os.mkdir(Destination_folder)
li_class_folder = []
male_list =[]
female_list =[]

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




