import os
import cv2
import glob
import scipy
import numpy as np
import pandas as pd
from scipy import io
from tqdm import tqdm

np.random.seed(42)

raw_data_path = '../raw_data'
data_path = '../data'
dates = ['date1', 'date2', 'date3']
objectives = ['subject1', 'subject2']


def vid2frames(vid, namemeta, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    mat = scipy.io.loadmat(vid)
    for k, v in mat.items():
        if k.__contains__('Video_fps'):
            for i in range(v.shape[2]):
                pngname = '{0}/{1}_{2}_{3}.png'.format(outdir, namemeta, k, i)
                #                 print(pngname)
                if not os.path.exists(pngname):
                    cv2.imwrite(pngname, v[:, :, i])


# # vid2frames
# def run_vid2frames(raw_data_path, data_path, dates):
#     for subdir, dirs, files in os.walk(raw_data_path):
#         for file in tqdm(files):
#             filepath = subdir + os.sep + file
#             filepath_list = filepath.split(sep=os.sep)
#             label = '{0}_{1}'.format(filepath_list[2], filepath_list[3])
#             if filepath_list[1] == dates[1]:
#                 if filepath.endswith(".mat"):
#     #                 print (filepath, label, '{0}/train'.format(data_path))
#     #                 break
#                     vid2frames(filepath, label, '{0}/train'.format(data_path))
#             elif filepath_list[1] == dates[0]:
#                     vid2frames(filepath, label, '{0}/val'.format(data_path))


# split videos to train / test
def prep_train_test(dates, raw_data_path):
    low_vids = list()
    normal_vids = list()

    for d in dates:
        for obj in objectives:
            for video_path in glob.glob('{0}/{1}/{2}/*/*.mat'.format(raw_data_path, d, obj)):
                if '96' in video_path:
                    low_vids.append(video_path)
                else:
                    normal_vids.append(video_path)

    train_vids = low_vids[2:] + low_vids[:-2] + normal_vids[2:] + normal_vids[:-2]
    val_vids = low_vids[:2] + low_vids[-2:] + normal_vids[:2] + normal_vids[-2:]
    return train_vids, val_vids


def prep_data(train_vids, val_vids):
    print('prep train data...')
    for filepath in tqdm(train_vids):
        filepath_list = filepath.split(sep=os.sep)
        label = filepath_list[1]
        if filepath.endswith(".mat"):
            #         print (filepath, label, '{0}/train'.format(data_path))
            #         break
            vid2frames(filepath, label, '{0}/train'.format(data_path))

    print('prep validation data...')
    for filepath in tqdm(val_vids):
        filepath_list = filepath.split(sep=os.sep)
        label = filepath_list[1]
        if filepath.endswith(".mat"):
            #         print (filepath, label, '{0}/train'.format(data_path))
            #         break
            vid2frames(filepath, label, '{0}/val'.format(data_path))


def explore_data(data_path):
    l = list()
    for subdir, dirs, files in tqdm(os.walk(data_path)):
        for file in files:
            file_list = file.split(sep='_')
            l.append([file, file_list[0], '{0}_{1}'.format(file_list[1], file_list[2]),
                      '{0}_{1}_{2}'.format(file_list[3], file_list[4], file_list[5])])

    # %%

    df = pd.DataFrame(l, columns=['ImageName', 'Label', 'OxygenSaturation', 'VideoName'])
    # df.head()
    # df.groupby(['Label', 'OxygenSaturation']).count()
    return df
