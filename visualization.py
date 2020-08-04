from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np


def model_classification_report(val_labels, predictions):
    cm = confusion_matrix(val_labels, predictions.argmax(axis=1))
    df_cm = pd.DataFrame(cm, index=[i for i in ["LOW", "NORMAL"]],
                         columns=[i for i in ["LOW", "NORMAL"]])
    plt.figure(figsize=(4, 3))
    sn.heatmap(df_cm, annot=True, fmt='g')
    print(classification_report(val_labels, predictions.argmax(axis=1),
                                target_names=['low', 'normal']))  # target_names=lb.classes_))


def plot_training(H):
    # plot the training loss and accuracy
    N = 10
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()


def plot_model_out_per_subject(vid_pred):
    x = list()
    low = list()
    normal = list()

    x_subject1_low = 1
    x_subject2_low = 1
    x_subject1_normal = 1
    x_subject2_normal = 1

    for k, v in vid_pred.items():
        vid_name = k.split(sep='_')
        #     print('{0}_{1}'.format(vid_name[0], vid_name[1]))
        if 58 <= int(vid_name[2]) <= 65 or 72 <= int(vid_name[2]) <= 76:
            if int(vid_name[2]) in [75, 76]:
                print('subject2', 'video name: ', k, '|', 'saturation level: ', 'LOW')
                #             x.append('subject2_{0}_saturation level: LOW'.format(k))
                x.append('subect1_low_video_{0}'.format(x_subject2_low))
                x_subject2_low += 1
            else:
                print('subject1', 'video name: ', k, '|', 'saturation level: ', 'LOW')
                #             x.append('subject1_{0}_saturation level: LOW'.format(k))
                x.append('subect2_low_video_{0}'.format(x_subject1_low))
                x_subject1_low += 1
        else:
            if int(vid_name[2]) in [80, 81]:
                print('subject2', 'video name: ', k, '|', 'saturation level: ', 'NORMAL')
                #             x.append('subject2_{0}_saturation level: NORMAL'.format(k))
                x.append('subect1_normal_video_{0}'.format(x_subject2_normal))
                x_subject2_normal += 1
            else:
                print('subject1', 'video name: ', k, '|', 'saturation level: ', 'NORMAL')
                #             x.append('subject1_{0}_saturation level: NORMAL'.format(k))
                x.append('subect2_normal_video_{0}'.format(x_subject1_normal))
                x_subject1_normal += 1
        v0 = v.count(0)
        v1 = v.count(1)

        low.append(v0)
        normal.append(v1)

        print('0', v0)
        print('1', v1)
        #     print(int(v0/v1*100))
        print()

    xx = np.arange(len(x))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(xx - width / 2, low, width, label='Low')
    rects2 = ax.bar(xx + width / 2, normal, width, label='Normal')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Oxygen saturation level')
    ax.set_title('Oxygen saturation levels per subject videos')
    ax.set_xticks(xx)
    ax.set_xticklabels(x, rotation=20, ha='right')
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()
