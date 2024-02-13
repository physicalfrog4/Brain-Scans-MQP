import os
import numpy as np
import pandas as pd
import torch
import data
from words import makeClassifications
from data import normalize_fmri_data
from LEM import extract_data_features, predAccuracy


def main():
    if platform == 'jupyter_notebook':
        data_dir = '../MQP/algonauts_2023_challenge_data/'
        data_dir = '../MQP/algonauts_2023_challenge_data/'
        parent_submission_dir = 'C:\GitHub\Brain-Scans-MQP\submissiondir'
    subj = 5  # @param ["1", "2", "3", "4", "5", "6", "7", "8"] {type:"raw", allow-input: true}
    # args
    args = argObj(data_dir, parent_submission_dir, subj)
    fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))


    #lh_data_min = np.min(lh_fmri)
    #lh_data_max = np.max(lh_fmri)
    #rh_data_min = np.min(rh_fmri)
    #rh_data_max = np.max(rh_fmri)

    print("________ Process Data ________")
    # Normalize Data Before Split
    lh_fmri, lh_data_min, lh_data_max = normalize_fmri_data(lh_fmri)
    print(lh_fmri)
    print("- - - - - - - -")
    rh_fmri, rh_data_min, rh_data_max = normalize_fmri_data(rh_fmri)
    print(rh_fmri)

    print('LH training fMRI data shape:')
    print(lh_fmri.shape)
    print('(Training stimulus images × LH vertices)')

    print('\nRH training fMRI data shape:')
    print(rh_fmri.shape)
    print('(Training stimulus images × RH vertices)')

    hemisphere = 'left'  # @param ['left', 'right'] {allow-input: true}
    roi = "OPA"  # @param ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies",
    # "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2",
    # "mfs-words", "mTL-words", "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]
    # {allow-input: true}

    train_img_dir = os.path.join(args.data_dir, 'training_split', 'training_images')
    test_img_dir = os.path.join(args.data_dir, 'test_split', 'test_images')

    # Create lists will all training and test image file names, sorted
    train_img_list = os.listdir(train_img_dir)
    train_img_list.sort()
    test_img_list = os.listdir(test_img_dir)
    test_img_list.sort()
    print('\nTraining images: ' + str(len(train_img_list)))
    print('\nTest images: ' + str(len(test_img_list)))
    train_img_file = train_img_list[0]
    print('\nTraining image file name: ' + train_img_file)
    print('\n73k NSD images ID: ' + train_img_file[-9:-4])

    print("________ Split Data ________")

    idxs_train, idxs_val, idxs_test = data.splitdata(train_img_list, test_img_list, train_img_dir)

    lh_fmri_train = lh_fmri[idxs_train]
    rh_fmri_train = rh_fmri[idxs_train]
    lh_fmri_val = lh_fmri[idxs_val]
    rh_fmri_val = rh_fmri[idxs_val]

    print("________ Make Lists ________")
    print(idxs_train)
    train_images = data.makeList(train_img_dir, train_img_list, idxs_train)
    val_images = data.makeList(train_img_dir, train_img_list, idxs_val)
    test_images = data.makeList(test_img_dir, test_img_list, idxs_test)


    print("________ Create Dataframe For ROI ________")

    # lh_train_ROI = data.dfROI(args, 'left', idxs_train, lh_fmri, rh_fmri)
    # lh_val_ROI = data.dfROI(args, 'left', idxs_val, lh_fmri, rh_fmri)
    # lh_test_ROI = data.dfROI(args, 'left', idxs_test, lh_fmri, rh_fmri)

    # rh_train_ROI = data.dfROI(args, 'right', idxs_train, lh_fmri, rh_fmri)
    # rh_val_ROI = data.dfROI(args, 'right', idxs_val, lh_fmri, rh_fmri)
    # rh_test_ROI = data.dfROI(args, 'right', idxs_test, lh_fmri, rh_fmri)

    print("________ Create Dataframe for All Regions ________")

    # df_lh_train = data.createDataFrame(idxs_train, lh_fmri_train)
    # df_lh_val = data.createDataFrame(idxs_val, lh_fmri_val)
    # df_rh_train = data.createDataFrame(idxs_train, rh_fmri_train)
    # df_rh_val = data.createDataFrame(idxs_val, rh_fmri_val)

    torch.cuda.empty_cache()

    print("________ Make Classifications ________")


    lh_classifications_val = makeClassifications(val_images, idxs_val)
    rh_classifications_val = lh_classifications_val
    lh_classifications = makeClassifications(train_images, idxs_train)
    rh_classifications = lh_classifications
    torch.cuda.empty_cache()

    print("________ Extract Image Features ________")

    train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader = (
        data.transformData(train_img_dir, test_img_dir, idxs_train, idxs_val, idxs_test, 10))

    # Model for Images
    model_img = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model_img.to('cuda:1')  # send the model to the chosen device ('cpu' or 'cuda')

    features_train, features_val, features_test = (
        extract_data_features(model_img, train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader, 50))
    del model_img

    print("________ Combine Data ________")
    dftrainL, dftrainFL = data.learnmore(lh_classifications, features_train, lh_fmri_train)
    dfvalL, dfvalFL = data.learnmore(lh_classifications_val, features_val, lh_fmri_val)
    lh_fmri_val_pred = data.Predictions(dftrainL, dftrainFL, dfvalL, dfvalFL)
    dftrainR, dftrainFR = data.learnmore(rh_classifications, features_train, rh_fmri_train)
    dfvalR, dfvalFR = data.learnmore(rh_classifications_val, features_val, rh_fmri_val)
    rh_fmri_val_pred = data.Predictions(dftrainR, dftrainFR, dfvalR, dfvalFR)

    # lh_train_input = np.concatenate([lh_classifications, features_train], axis=1)
    # rh_train_input = np.concatenate([rh_classifications, features_train], axis=1)
    # lh_val_input = np.concatenate([lh_classifications_val, features_val], axis=1)
    # rh_val_input = np.concatenate([rh_classifications_val, features_val], axis=1)

    print("________ Make Predictions ________")

    # lh_fmri_val_pred = makePredictions(lh_train_input, lh_fmri_train, lh_val_input, lh_fmri_val)
    lh_fmri_val_pred = data.unnormalize_fmri_data(lh_fmri_val_pred, lh_data_min, lh_data_max)
    # rh_fmri_val_pred = makePredictions(rh_train_input, rh_fmri_train, rh_val_input, rh_fmri_val)
    rh_fmri_val_pred = data.unnormalize_fmri_data(rh_fmri_val_pred, rh_data_min, rh_data_max)

    # lh_fmri_ROI_pred = makePredictions(lh_train_input, lh_train_ROI, lh_val_input, lh_val_ROI)
    # rh_fmri_ROI_pred = makePredictions(rh_train_input, rh_train_ROI, rh_val_input, rh_val_ROI)

    print("________ Visualize ________")
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

    lh_correlation, rh_correlation = predAccuracy(lh_fmri_val_pred, lh_fmri_val, rh_fmri_val_pred, rh_fmri_val)


    print("________ END ________")

    lh_avg = np.average(lh_fmri_val_pred - lh_fmri_val)
    rh_avg = np.average(rh_fmri_val_pred - rh_fmri_val)

    print("LH AVG ", lh_avg)
    print("RH AVG ", rh_avg)



    print("________ End ________")


class argObj:
    def __init__(self, data_dir, parent_submission_dir, subj):
        self.subj = format(subj, '02')
        self.data_dir = os.path.join(data_dir, 'subj' + self.subj)
        self.parent_submission_dir = parent_submission_dir
        self.subject_submission_dir = os.path.join(self.parent_submission_dir,
                                                   'subj' + self.subj)

        # Create the submission directory if not existing
        # if not os.path.isdir(self.subject_submission_dir):
        # os.makedirs(self.subject_submission_dir)


if __name__ == "__main__":
    platform = 'jupyter_notebook'  # @param ['colab', 'jupyter_notebook'] {allow-input: true}
    device = 'cuda'  # @param ['cpu', 'cuda'] {allow-input: true}
    device = torch.device(device)

    main()
