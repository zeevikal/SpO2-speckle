from tqdm import tqdm
import numpy as np
import glob
import cv2

def model_predict(model, val_data):
    predictions = model.predict(val_data, batch_size=128, workers=-1, verbose=1)
    return predictions


def prediction_per_video(data_path, model):
    val_list = glob.glob('{0}/val/*.png'.format(data_path))
    vid_pred = dict()

    for img in tqdm(val_list):
        img_path_list = img.split(sep='_')
        vid = '{0}_{1}_{2}_{3}_{4}'.format(img_path_list[1], img_path_list[2], img_path_list[3],
                                           img_path_list[4], img_path_list[5])

        cls = '{0}_{1}'.format(img_path_list[1], img_path_list[2])

        im = cv2.imread(img)
        image = np.expand_dims(im, axis=0)
        pred = model.predict(image)

        if vid not in vid_pred:
            vid_pred[vid] = list()
            vid_pred[vid] += [pred.argmax(axis=1)[0]]
        else:
            vid_pred[vid] += [pred.argmax(axis=1)[0]]

    for k, v in vid_pred.items():
        vid_name = k.split(sep='_')
        print('{0}_{1}'.format(vid_name[0], vid_name[1]))
        v0 = v.count(0)
        v1 = v.count(1)
        print('0', v0)
        print('1', v1)
        if v1 > 0:
            print(int(v0 / v1 * 100))
        print()