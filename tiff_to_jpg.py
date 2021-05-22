from PIL import Image
import glob
import os
import shutil

Image_folder = '/home/n-lab/Documents/Periocular_project/Datasets/UBIris_v2/CLASSES_400_300_Part2'
Dest_folder = '/home/n-lab/Documents/Periocular_project/Datasets/UBIris_v2/CLASSES_400_300_Part2_JPG'

# for name in glob.glob(Image_folder +'/*.tif'):
#     im = Image.open(name)
#     name = str(name).rstrip(".tif")
#     im.save(name + '.jpg', 'JPEG')


li_files = os.listdir(Image_folder)
for name in li_files:
    if name.endswith('tiff'):
        im = Image.open(os.path.join(Image_folder,name))
        name = str(name).rstrip(".tiff")
        im.save(os.path.join(Dest_folder,name + '.jpg'), 'JPEG')





print ("Conversion from tif/tiff to jpg completed!")