import cv2
import numpy as np
import tensorflow.compat.v1 as tf


class SalienceModel:

    def open(self):
        self.sess = tf.Session()
        self.sess.__enter__()
        self.saver = tf.train.import_meta_graph('./meta_graph/my-model.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint('./salience_model'))
        self.image_batch = tf.get_collection('image_batch')[0]
        self.pred_mattes = tf.get_collection('mask')[0]

    def close(self):
        self.sess.__exit__(None, None, None)

    @staticmethod
    def preprocessing(im):
        im = cv2.resize(im, (320, 320), interpolation=0)
        im = im.astype('float32')
        return np.expand_dims(im[..., ::-1] - [126.88, 120.24, 112.19], 0)

    def predict(self, im):
        h, w = im.shape[:2]
        im = self.preprocessing(im)
        feed_dict = {self.image_batch: im}
        pred_alpha = self.sess.run(self.pred_mattes, feed_dict=feed_dict)
        final_alpha = cv2.resize(np.squeeze(pred_alpha), (w, h))
        return final_alpha

    def find_rect(self, im):
        org_h, org_w = im.shape[:2]
        inputs = self.preprocessing(im)
        cur_h, cur_w = inputs.shape[1:3]
        feed_dict = {self.image_batch: inputs}
        pred_alpha = self.sess.run(self.pred_mattes, feed_dict=feed_dict)
        pred_alpha = np.squeeze(pred_alpha)
        pred_alpha = (pred_alpha * 255).astype('uint8')
        ret, pred_alpha = cv2.threshold(pred_alpha, 127, 255, cv2.THRESH_BINARY)

        cnts, _ = cv2.findContours(pred_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return

        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = -float('inf'), -float('inf')
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            x1 = x * org_w // cur_w
            y1 = y * org_h // cur_h
            x2 = (x + w) * org_w // cur_w
            y2 = (y + h) * org_h // cur_h
            x_min, y_min = min(x_min, x1), min(y_min, y1)
            x_max, y_max = max(x_max, x2), max(y_max, y2)

        cv2.rectangle(im, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        return x_min, y_min, x_max, y_max


def main():
    import os, sys
    m = SalienceModel()
    m.open()
    if os.path.isfile(sys.argv[1]):
        ifn = sys.argv[1]
        if ifn.endswith('.MP4'):
            cap = cv2.VideoCapture(ifn)
            ret, im = cap.read()
            cap.release()
        else:
            im = cv2.imread(ifn)
        final_alpha = m.predict(im)
        rect = m.find_rect(im)
        if len(sys.argv) > 2:
            ofn = sys.argv[2]
            cv2.imwrite(ofn, im)
        else:
            cv2.imshow('a', im)
            cv2.waitKey()
    elif os.path.isdir(sys.argv[1]):
        path = sys.argv[1]
        for fn in os.listdir(path):
            ifn = os.path.join(path, fn)
            ofn = os.path.join('test_output', fn)
            im = cv2.imread(ifn)
            final_alpha = m.predict(im)
            cv2.imwrite(ofn, final_alpha * 255)
    m.close()


if __name__ == '__main__':
    main()
