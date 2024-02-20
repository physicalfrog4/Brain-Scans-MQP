import os
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

import data
from words import makeClassifications, predictions
from data import normalize_fmri_data, learnmore, transformData
from LEM import extract_data_features, predAccuracy


def main():
    if platform == 'jupyter_notebook':

        data_dir = '../MQP/algonauts_2023_challenge_data/'
        parent_submission_dir = 'C:\GitHub\Brain-Scans-MQP\submissiondir'
    subj = 5  # @param ["1", "2", "3", "4", "5", "6", "7", "8"] {type:"raw", allow-input: true}
    SuperDuperModel = LinearRegression()

    all_features_LH = []
    all_fmri_LH = []
    all_features_RH = []
    all_fmri_RH = []
    for subj in range(1, 9):
        args = argObj(data_dir, parent_submission_dir, subj)
        fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
        lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
        rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

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

        hemisphere = 'left'
        roi = "OPA"
        train_img_dir = os.path.join(args.data_dir, 'training_split', 'training_images')
        test_img_dir = os.path.join(args.data_dir, 'test_split', 'test_images')

        # Create lists will all training and test image file names, sorted
        train_img_list = os.listdir(train_img_dir)
        train_img_list = train_img_list[: 1000]
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
        # print(train_images)

        torch.cuda.empty_cache()

        print("________ Make Classifications ________")

        lh_classifications_val = makeClassifications(val_images, idxs_val, device)
        rh_classifications_val = lh_classifications_val
        lh_classifications = makeClassifications(train_images, idxs_train, device)
        rh_classifications = lh_classifications
        torch.cuda.empty_cache()

        print("________ Extract Image Features ________")

        train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader = \
            transformData(train_img_dir, test_img_dir, idxs_train, idxs_val, idxs_test, 64)
        # model_img = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        model_img = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model_img.eval()
        model_img.to(device)

        features_train, features_val, features_test = \
            extract_data_features(model_img, train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader, 64)
        del model_img

        print("________ LEARN MORE ________")
        # model = LinearRegression()
        dftrainL, dftrainFL = learnmore(lh_classifications, features_train, lh_fmri_train)
        dfvalL, dfvalFL = learnmore(lh_classifications_val, features_val, lh_fmri_val)
        dftrainR, dftrainFR = learnmore(rh_classifications, features_train, rh_fmri_train)
        dfvalR, dfvalFR = learnmore(rh_classifications_val, features_val, rh_fmri_val)

        features_combined = np.concatenate([dftrainL, dfvalL], axis=0)
        fmri_combined = np.concatenate([dftrainFL, dfvalFL], axis=0)
        print(fmri_combined.shape)
        features_combined.reshape(-1, 66)
        print(features_combined)
        dummy = []
        dummy.extend(features_combined)


        features_combined2 = np.concatenate([dftrainR, dfvalR], axis=0)
        fmri_combined2 = np.concatenate([dftrainFR, dfvalFR], axis=0)

        # Perform k-fold cross-validation for training your model
        kf = KFold(n_splits=10, shuffle=True)

        for train_index, val_index in kf.split(features_combined):
            X_train, X_val = features_combined[train_index], features_combined[val_index]
            y_train, y_val = fmri_combined[train_index], fmri_combined[val_index]
            # Train your model
            model = train_model(X_train, y_train, SuperDuperModel)
            y_val_pred = model.predict(X_val)
            accuracy = model.score(X_val, y_val)
            print("Validation Accuracy1:", accuracy)

            X_train2, X_val2 = features_combined2[train_index], features_combined2[val_index]
            y_train2, y_val2 = fmri_combined2[train_index], fmri_combined2[val_index]
            model = train_model(X_train2, y_train2, SuperDuperModel)
            y_val_pred2 = model.predict(X_val2)
            accuracy2 = model.score(X_val2, y_val2)
            print("Validation Accuracy2:", accuracy2)
        final_model = train_model(features_combined, fmri_combined, SuperDuperModel)
        final_model = train_model(features_combined2, fmri_combined2, SuperDuperModel)

        print("________ Predictions ________")
        lh_fmri_val_pred = predictions(dftrainL, dftrainFL, dfvalL, dfvalFL, final_model)
        rh_fmri_val_pred = predictions(dftrainR, dftrainFR, dfvalR, dfvalFR, final_model)

        print("________ Normalize ________")
        lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
        rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

        lh_fmri_val = lh_fmri[idxs_val]
        rh_fmri_val = rh_fmri[idxs_val]

        print(lh_data_min, lh_data_max)
        print(rh_data_min, rh_data_max)

        lh_fmri_val_pred = data.unnormalize_fmri_data(lh_fmri_val_pred, lh_data_min, lh_data_max)
        # rh_fmri_val_pred = makePredictions(rh_train_input, rh_fmri_train, rh_val_input, rh_fmri_val)
        rh_fmri_val_pred = data.unnormalize_fmri_data(rh_fmri_val_pred, rh_data_min, rh_data_max)

        print("________ Results ________")
        words = ['furniture', 'food', 'kitchenware', 'appliance', 'person', 'animal', 'vehicle', 'accessory',
                 'electronics', 'sports', 'traffic', 'outdoor', 'home', 'clothing', 'hygiene', 'toy', 'plumbing',
                 'safety', 'luggage', 'computer', 'fruit', 'vegetable', 'tool']

        lh_correlation, rh_correlation = predAccuracy(lh_fmri_val_pred, lh_fmri_val, rh_fmri_val_pred, rh_fmri_val)
        lh_avg = np.average(lh_fmri_val_pred - lh_fmri_val)
        rh_avg = np.average(rh_fmri_val_pred - rh_fmri_val)

        print("LH AVG ", lh_avg)
        print("RH AVG ", rh_avg)
        length = len(words)

        for clss in range(words):
            print(clss, words[clss])
            avg = []
            avgg = []
            avg2 = []
            avgg2 = []
            for i in range(len(lh_classifications_val)):

                # vehicle = 7
                if lh_classifications_val[i][1] == clss:
                    # print(lh_classifications_val[i])
                    avg.append(lh_fmri_val_pred[i])
                    avgg.append(lh_fmri_val[i])
                    # print(lh_fmri_val[i])
                if rh_classifications[i][1] == clss:
                    avg2.append(rh_fmri_val_pred[i])
                    avgg2.append(lh_fmri_val[i])
            # print(avg)

            lh = np.mean(avg, axis=0)
            rh = np.mean(avg2, axis=0)
            print("MEAN PRED:\n", lh)
            # print("MEAN:\n", rh)
            # visualize.anotherOne(args, lh, rh)
            lh2 = np.mean(avgg, axis=0)
            rh2 = np.mean(avgg2, axis=0)
            print("MEAN REAL:\n", lh2)
            # print("MEAN:\n", rh2)
            # visualize.anotherOne(args, lh2, rh2)
            # print(np.mean((lh2 - lh)))
            corr = np.corrcoef(avg, avgg)
            print(corr)
            print("Corre ", np.mean(corr))

        print("________ END " + str(subj) + " ________")




    X_train, X_val = all_features_LH[train_index], all_features_LH[val_index]
    y_train, y_val = all_fmri_LH[train_index], all_fmri_LH[val_index]
    exit()

    # args
    args = argObj(data_dir, parent_submission_dir, subj)
    fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

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
    # train_img_list = train_img_list[: 1000]
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
    # print(train_images)

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

    train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader = \
        transformData(train_img_dir, test_img_dir, idxs_train, idxs_val, idxs_test, 64)
    # model_img = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model_img = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    # or any of these variants
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    model_img.eval()
    model_img.to('cuda:0')

    features_train, features_val, features_test = \
        extract_data_features(model_img, train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader, 64)
    del model_img

    print("________ LEARN MORE ________")
    # model = LinearRegression()
    dftrainL, dftrainFL = learnmore(lh_classifications, features_train, lh_fmri_train)
    dfvalL, dfvalFL = learnmore(lh_classifications_val, features_val, lh_fmri_val)
    dftrainR, dftrainFR = learnmore(rh_classifications, features_train, rh_fmri_train)
    dfvalR, dfvalFR = learnmore(rh_classifications_val, features_val, rh_fmri_val)

    dftrainL = np.array(dftrainL)
    dftrainFL = np.array(dftrainFL)
    dfvalL = np.array(dfvalL)
    dfvalFL = np.array(dfvalFL)

    dftrainR = np.array(dftrainR)
    dftrainFR = np.array(dftrainFR)
    dfvalR = np.array(dfvalR)
    dfvalFR = np.array(dfvalFR)

    features_combined = np.concatenate([dftrainL, dfvalL], axis=0)
    fmri_combined = np.concatenate([dftrainFL, dfvalFL], axis=0)

    features_combined2 = np.concatenate([dftrainR, dfvalR], axis=0)
    fmri_combined2 = np.concatenate([dftrainFR, dfvalFR], axis=0)

    # Perform k-fold cross-validation for training your model
    kf = KFold(n_splits=100, shuffle=True)

    for train_index, val_index in kf.split(features_combined):
        X_train, X_val = features_combined[train_index], features_combined[val_index]
        y_train, y_val = fmri_combined[train_index], fmri_combined[val_index]
        # Train your model
        model = train_model(X_train, y_train)
        y_val_pred = model.predict(X_val)
        accuracy = model.score(X_val, y_val)
        print("Validation Accuracy1:", accuracy)

        X_train2, X_val2 = features_combined2[train_index], features_combined2[val_index]
        y_train2, y_val2 = fmri_combined2[train_index], fmri_combined2[val_index]
        model = train_model(X_train2, y_train2)
        y_val_pred2 = model.predict(X_val2)
        accuracy2 = model.score(X_val2, y_val2)
        print("Validation Accuracy2:", accuracy2)
    final_model = train_model(features_combined, fmri_combined)
    final_model = train_model(features_combined2, fmri_combined2)

    print("________ Predictions ________")
    lh_fmri_val_pred = predictions(dftrainL, dftrainFL, dfvalL, dfvalFL, final_model)
    rh_fmri_val_pred = predictions(dftrainR, dftrainFR, dfvalR, dfvalFR, final_model)

    print("________ Normalize ________")
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

    lh_fmri_val = lh_fmri[idxs_val]
    rh_fmri_val = rh_fmri[idxs_val]

    print(lh_data_min, lh_data_max)
    print(rh_data_min, rh_data_max)

    lh_fmri_val_pred = data.unnormalize_fmri_data(lh_fmri_val_pred, lh_data_min, lh_data_max)
    # rh_fmri_val_pred = makePredictions(rh_train_input, rh_fmri_train, rh_val_input, rh_fmri_val)
    rh_fmri_val_pred = data.unnormalize_fmri_data(rh_fmri_val_pred, rh_data_min, rh_data_max)

    print("________ Results ________")

    lh_correlation, rh_correlation = predAccuracy(lh_fmri_val_pred, lh_fmri_val, rh_fmri_val_pred, rh_fmri_val)
    lh_avg = np.average(lh_fmri_val_pred - lh_fmri_val)
    rh_avg = np.average(rh_fmri_val_pred - rh_fmri_val)

    print("LH AVG ", lh_avg)
    print("RH AVG ", rh_avg)

    print("________ END ________")


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


def train_model(X, y, model):
    # Define your model here (modify as needed)
    # Change this to the appropriate model
    model = SuperDuperModel
    model.fit(X, y)
    return model


if __name__ == "__main__":
    platform = 'jupyter_notebook'  # @param ['colab', 'jupyter_notebook'] {allow-input: true}
    device = 'cuda:0'  # @param ['cpu', 'cuda'] {allow-input: true}
    device = torch.device(device)
    SuperDuperModel = LinearRegression()
    main()
