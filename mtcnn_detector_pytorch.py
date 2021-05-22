import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
import os
import glob


class FaceDetector(object):
    """
    Face detector class
    """

    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

    def _draw(self, frame, boxes, probs, landmarks,image_name,Dest_Folder):
        """
        Draw landmarks and boxes for each face detected
        """
        # try:
        count = 0
        for box, prob, ld in zip(boxes, probs, landmarks):
            count += 1
            self.crop_eyes(ld,frame,count,image_name,Dest_Folder)

            # Draw rectangle on frame
            # cv2.rectangle(frame,
            #                 (box[0], box[1]),
            #                 (box[2], box[3]),
            #                 (0, 0, 255),
            #                 thickness=2)
            #
            # # Show probability
            # cv2.putText(frame, str(
            #     prob), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            #
            # # Draw landmarks
            # cv2.circle(frame, tuple(ld[0]), 5, (0, 0, 255), -1)
            # cv2.circle(frame, tuple(ld[1]), 5, (0, 0, 255), -1)
            # cv2.circle(frame, tuple(ld[2]), 5, (0, 0, 255), -1)
            # cv2.circle(frame, tuple(ld[3]), 5, (0, 0, 255), -1)
            # cv2.circle(frame, tuple(ld[4]), 5, (0, 0, 255), -1)





        # except:
        #     pass

        return frame

    def crop_eyes(self,ld,frame,count,image_name,Dest_Folder):
        left_eye_x = ld[0][0]
        left_eye_y = ld[0][1]
        right_eye_x = ld[1][0]
        right_eye_y = ld[1][1]
        left_eye_x1 = int(left_eye_x - (ld[2][0] - left_eye_x) + 5)
        left_eye_x2 = int(ld[2][0] - 5)
        left_eye_y1 = int(left_eye_y - 0.75 * (ld[2][1] - left_eye_y))
        left_eye_y2 = int(left_eye_y + 0.75 * (ld[2][1] - left_eye_y))
        right_eye_x1 = int(ld[2][0] + 5)
        right_eye_x2 = int(right_eye_x + (right_eye_x - ld[2][0]) - 5)
        right_eye_y1 = int(right_eye_y - 0.75 * (ld[2][1] - right_eye_y))
        right_eye_y2 = int(right_eye_y + 0.75 * (ld[2][1] - right_eye_y))
        dest_left = os.path.join(Dest_Folder + '_Left_Eye', image_name + ('_{}_'.format(count)))
        dest_right = os.path.join(Dest_Folder + '_Right_Eye', image_name + ('_{}_'.format(count)))

        cropped_left_eye = frame[left_eye_y1:left_eye_y2, left_eye_x1:left_eye_x2, :]
        cropped_right_eye = frame[right_eye_y1:right_eye_y2, right_eye_x1:right_eye_x2, :]
        print(cropped_left_eye.shape)
        print(cropped_right_eye.shape)
        if np.all(cropped_right_eye.shape) and np.all(cropped_left_eye.shape):
        # if cropped_left_eye.shape[0] != 0 and cropped_left_eye.shape[1] != 0 and cropped_right_eye.shape[0] != 0 and cropped_right_eye[1] !=0 :
            cv2.imwrite(dest_left+'left_eye.jpg', frame[left_eye_y1:left_eye_y2, left_eye_x1:left_eye_x2, :])
            cv2.imwrite(dest_right+'right_eye.jpg', frame[right_eye_y1:right_eye_y2, right_eye_x1:right_eye_x2, :])

    def run(self,Image_Folder,Dest_Folder):
        """
            Run the FaceDetector and draw landmarks and boxes around detected faces
        """
        li_folders = os.listdir(Image_Folder)
        li_folders = sorted(li_folders)
        for folder in li_folders:
            image_folder = os.path.join(Image_Folder, folder)
            li_files = glob.glob(image_folder + '/*.JPG')
            li_files = sorted(li_files)
            for file_name in li_files:
                print (file_name)
                base_name = os.path.basename(file_name)
                image_name = base_name.split('.')[0]
                frame = cv2.imread(file_name)
                # frame = cv2.imread('/home/n-lab/Documents/Periocular_project/Datasets/FRGC_dataset_extracted_periocular_images/Fall2002_Images_periocular/2002-269/04494d08.JPG')
                # try:
                # detect face box, probability and landmarks
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                # draw on frame
                if np.all(boxes):
                    self._draw(frame, boxes, probs, landmarks,image_name,Dest_Folder,)

                # except:
                #     pass

                # Show the frame
                # cv2.imwrite('trial_mtcnn.jpg',frame)
                # cv2.imshow('Face Detection', frame)

                # if cv2.waitKey(100):
                #     # break
                #
                # # cap.release()
                #     cv2.destroyAllWindows()

    def run_no_subfolders(self,Image_Folder,Dest_Folder):
        """
            Run the FaceDetector and draw landmarks and boxes around detected faces
        """
        li_files = glob.glob(Image_Folder + '/*.jpg')
        li_files = sorted(li_files)
        for file_name in li_files:
            print (file_name)
            base_name = os.path.basename(file_name)
            image_name = base_name.split('.')[0]
            frame = cv2.imread(file_name)
            # frame = cv2.imread('/home/n-lab/Documents/Periocular_project/Datasets/FRGC_dataset_extracted_periocular_images/Fall2002_Images_periocular/2002-269/04494d08.JPG')
            # try:
            # detect face box, probability and landmarks
            boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
            # draw on frame
            if np.all(boxes):
                self._draw(frame, boxes, probs, landmarks,image_name,Dest_Folder,)

            # except:
            #     pass

            # Show the frame
            # cv2.imwrite('trial_mtcnn.jpg',frame)
            # cv2.imshow('Face Detection', frame)

            # if cv2.waitKey(100):
            #     # break
            #
            # # cap.release()
            #     cv2.destroyAllWindows()
# Run the app
Image_Folder='/home/n-lab/Documents/Periocular_project/Datasets/FRGC/FRGC-2.0-dist/nd1/Spring2003'
Dest_Folder = '/home/n-lab/Documents/Periocular_project/Datasets/FRGC_dataset_extracted_periocular_images_MTCNN/Spring2003_cropped'
mtcnn = MTCNN()
fcd = FaceDetector(mtcnn)
# fcd.run(Image_Folder,Dest_Folder)
fcd.run_no_subfolders(Image_Folder,Dest_Folder)