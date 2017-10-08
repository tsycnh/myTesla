import tensorflow as tf
import numpy as np

import model
import video_preprocess


tf.summary.scalar(name='loss',tensor=model.loss)

merged_summary = tf.summary.merge_all()

# train start
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter("./tmp/1008/1",sess.graph)

tv = video_preprocess.TeslaVideo('/Users/shidanlifuhetian/All/Tdevelop/myTesla/epochs/epoch01_front.mp4')
i = 1
while(True):
    out = tv.next_batch(100)
    print('epoch: ',tv.epoch,'a batch:',len(out),' total:',i*len(out))
    i = i+1