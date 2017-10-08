import tensorflow as tf
import numpy as np
import random
import model
import video_preprocess


tf.summary.scalar(name='loss',tensor=model.loss)
tf.summary.histogram(name='conv1',values=model.conv1)
merged_summary = tf.summary.merge_all()

# train start
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter("./tmp/1008/17",sess.graph)
saver = tf.train.Saver()

tv = video_preprocess.TeslaVideo('./epochs/epoch01_front.mp4','./epochs/epoch01_steering.csv')
i = 0
while(True):
    batch, label = tv.next_batch2(200)
    batch = np.array(batch)
    label = np.array(label)

    print('epoch: ',tv.epoch,'a batch:',len(batch),' total images:',i*len(batch))
    sess.run(model.train_step,feed_dict={model.x_input:batch,model.y_input:label})
    if i % 1 == 0:
        s = sess.run(merged_summary, feed_dict={model.x_input: batch, model.y_input: label})
        writer.add_summary(s, i)

    i = i+1
    saver.save(sess, 'my_test_model', global_step=1000)

