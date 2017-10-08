import numpy as np
import cv2
import csv
import random

class TeslaVideo:
    def __init__(self, video_path, label_path):
        self.process_labels(label_path)
        self.cap = cv2.VideoCapture(video_path)
        self.epoch = 0  # how many epochs have been trained
        self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # self.label_dict = {}
        self.frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.load_all_images()
        self.current_index = 0

    def r(self):
        return 0.1

    def load_all_images(self):

        print('start to load all images')
        batch,label = self.next_batch(int(self.total_frames))
        random.shuffle(batch,self.r)
        random.shuffle(label,self.r)
        self.data = batch
        self.label = label
        print('all images loaded')


    def next_batch(self,batch_num):
        next_frame_index = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        print('next frame index:',next_frame_index)
        if next_frame_index + batch_num > self.total_frames:
            self.epoch += 1
            self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)  # set the pointer to the beginning

        list = []
        labels = []
        for i in range(0,batch_num):
            if self.cap.isOpened():
                # print('ratio: ',ratio)
                frame_index = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                a,frame = self.cap.read()
                frame = frame / 255.0  # normalize to 0~1

                resized_image = self.crop_resize_image(frame)
                list.append(resized_image)
                tmp  = self.label_dict[str(int(frame_index))]
                labels.append([tmp])
        return [list,labels]
    def next_batch2(self,batch_num):

        if self.current_index + batch_num > len(self.data):
            self.epoch += 1
            self.current_index = 0

        list = self.data[self.current_index:self.current_index+batch_num]
        labels = self.label[self.current_index:self.current_index+ batch_num]
        self.current_index += batch_num
        return [list,labels]

    def process_labels(self,label_path):
        with open(label_path) as f:
            label_file = csv.reader(f)
            label_dict = {}
            for row in label_file:
                break
            for row in label_file:
                label_dict[row[1]] = np.float32(row[2])/20.0

        self.label_dict = label_dict

    def crop_resize_image(self,img):

        width = int(self.frame_width - 1)
        y1 = 250
        new_height = 422
        crop_img = img[y1:y1 + new_height, 0:width]

        return cv2.resize(crop_img, (200, 66))
