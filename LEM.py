import torch
from numpy import average
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from tqdm import tqdm
from scipy.stats import pearsonr as corr


def extract_data_features(modelGN, train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader, batch_size):
    train_nodes, _ = get_graph_node_names(modelGN)
    model_layer = "classifier.0"
    feature_extractor = create_feature_extractor(modelGN, return_nodes=[model_layer])
    pca = fit_pca(feature_extractor, train_imgs_dataloader, batch_size)

    features_train = extract_features(feature_extractor, train_imgs_dataloader, pca)
    features_val = extract_features(feature_extractor, val_imgs_dataloader, pca)
    features_test = extract_features(feature_extractor, test_imgs_dataloader, pca)

    print('\nTraining images features:')
    print(features_train.shape)
    print('(Training stimulus images × PCA features)')

    print('\nValidation images features:')
    print(features_val.shape)
    print('(Validation stimulus images × PCA features)')

    print('\nTest images features:')
    print(features_test.shape)
    print('(Test stimulus images × PCA features)')

    del pca
    return features_train, features_val, features_test


def linearMap(features_train, lh_fmri_train, rh_fmri_train, features_val, features_test, lh_fmri_val, rh_fmri_val):
    modelLH = LinearRegression()
    modelLH.fit(features_train, lh_fmri_train)

    # Use the fitted model to predict the validation and test fMRI data
    lh_fmri_val_pred = modelLH.predict(features_val)
    lh_fmri_test_pred = modelLH.predict(features_test)

    mae = mean_absolute_error(lh_fmri_val, lh_fmri_val_pred)
    print("Mean Absolute Error on Validation Data:", mae)
    modelRH = LinearRegression()
    modelRH.fit(features_train, rh_fmri_train)

    # Use fitted linear regressions to predict the validation and test fMRI data
    rh_fmri_val_pred = modelRH.predict(features_val)
    rh_fmri_test_pred = modelRH.predict(features_test)

    mae = mean_absolute_error(rh_fmri_val, rh_fmri_val_pred)
    print("Mean Absolute Error on Validation Data:", mae)

    return lh_fmri_val_pred, rh_fmri_val_pred, lh_fmri_test_pred, rh_fmri_test_pred


def predAccuracy(lh_fmri_val_pred, lh_fmri_val, rh_fmri_val_pred, rh_fmri_val):
    print("Start PredAccuracy")
    print("\npredicted\n", lh_fmri_val_pred, "\nactual\n", lh_fmri_val)

    # Empty correlation array of shape: (LH vertices)
    lh_correlation = np.zeros(lh_fmri_val_pred.shape[1])

    # Correlate each predicted LH vertex with the corresponding ground truth vertex
    for v in tqdm(range(lh_fmri_val_pred.shape[1])):
        lh_correlation[v] = corr(lh_fmri_val_pred[:, v], lh_fmri_val[:, v])[0]

    # Empty correlation array of shape: (RH vertices)
    rh_correlation = np.zeros(rh_fmri_val_pred.shape[1])

    # Correlate each predicted RH vertex with the corresponding ground truth vertex
    for v in tqdm(range(rh_fmri_val_pred.shape[1])):
        rh_correlation[v] = corr(rh_fmri_val_pred[:, v], rh_fmri_val[:, v])[0]

    print('average lh ', average(lh_correlation) * 100, 'average rh ', average(rh_correlation) * 100)
    return lh_correlation, rh_correlation


def extract_features(feature_extractor, dataloader, pca):
    features = []
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader), desc="Extracting Features"):
        # ...
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Apply PCA transform
        ft = pca.transform(ft.cpu().detach().numpy())
        # ft = pca.transform(ft.cuda().detach().numpy())
        features.append(ft)
    return np.vstack(features)




def fit_pca(feature_extractor, dataloader, batch_size):
    # torch.device = 'cuda:1'
    # Define PCA parameters
    pca = IncrementalPCA(batch_size=batch_size)

    # Fit PCA to batch
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Fit PCA to batch
        pca.partial_fit(ft.detach().cpu().numpy())
    return pca
