import numpy as np
import cv2
class TeslaVideo:
    def __init__(self,path):
        self.cap = cv2.VideoCapture(path)
        self.epoch = 0  # how many epochs have been trained
        self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def next_batch(self,batch_num):
        next_frame_index = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        print('next frame index:',next_frame_index)
        if next_frame_index + batch_num > self.total_frames:
            self.epoch += 1
            self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)  # set the pointer to the beginning

        list = []
        # print('epoch: ',self.epoch)
        for i in range(0,batch_num):
            if self.cap.isOpened():
                # print('ratio: ',ratio)

                a,frame = self.cap.read()
                # frame = frame / 255.0  # normalize to 0~1
                list.append(frame)
        return list

    #
    # cap = cv2.VideoCapture('/Users/shidanlifuhetian/All/Tdevelop/myTesla/epochs/epoch01_front.mp4')
    # while(cap.isOpened()):
    #     a,frame = cap.read()
    #
    #     # # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     #
    #     # cv2.imshow('frame',gray)
    #     # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     #     break
    #
    #
    # cap.release()
    # cv2.destroyAllWindows()

