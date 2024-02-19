import os
from tqdm import tqdm

import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.models import vgg19
from torchvision import transforms
from torchvision import ops

from ultralytics import YOLO, settings
settings.update({
    "clearml" : False,          
    "comet" : False,           
    "dvc" : False,             
    "hub" : False,     
    "mlflow" : False,          
    "neptune" : False,         
    "raytune" : False,         
    "tensorboard" : False,     
    "wandb" : False     
})

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder


def getClassImages(dataDir:str, imgDataFolerDir: str, subj: int, className: str):
    pickleFilePath = os.path.join(imgDataFolerDir, f"subj0{subj}ImgData.pkl")
    imgData = pd.read_pickle(pickleFilePath)
    imageIDsInClass = imgData[imgData["classLabel"] == "person"]["nsdId"].values
    trainingImagesPath = os.path.join(dataDir, f"subj0{1}/training_split/training_images/")
    subjImages = os.listdir(trainingImagesPath)
    imagesInClass = []
    trainingIDsInClass = []
    for image in subjImages:
        imageSplits = image.split("-") # ['train', '0004_nsd', '00085.png']
        trainingID = int(imageSplits[1].rstrip("_nsd")) - 1
        nsdID = imageSplits[-1] # removes 'train-0001_nsd-' and keeps '00013.png'
        nsdID = nsdID.rstrip(".png") #removes png file type
        nsdID = int(nsdID)
        if nsdID in imageIDsInClass:
            imagesInClass.append(os.path.join(trainingImagesPath, image))
            trainingIDsInClass.append(trainingID)
    return np.array(imagesInClass), np.array(trainingIDsInClass)

    
def getAvgROI(parentFolderDir: str, subj: int, fmriData, hemi: str = "l"):
    rois = np.array(["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words", "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"])
    avgRoiValues = np.zeros((len(fmriData), len(rois)))
    for i in range(len(rois)):
        roi = rois[i]
        if roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
            roi_class = 'prf-visualrois'
        elif roi in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
            roi_class = 'floc-bodies'
        elif roi in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
            roi_class = 'floc-faces'
        elif roi in ["OPA", "PPA", "RSC"]:
            roi_class = 'floc-places'
        elif roi in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
            roi_class = 'floc-words'
        elif roi in ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]:
            roi_class = 'streams'
        roiMap = np.load(f"{parentFolderDir}/subj0{subj}/roi_masks/mapping_{roi_class}.npy", allow_pickle=True).item()
        challenge_roi_class = np.load(f"{parentFolderDir}/subj0{subj}/roi_masks/{hemi}h.{roi_class}_challenge_space.npy")
        # Select the vertices corresponding to the ROI of interest
        roi_mapping = list(roiMap.keys())[list(roiMap.values()).index(roi)]
        challenge_roi = np.asarray(challenge_roi_class == roi_mapping, dtype=int)
        # print(roi_mapping)       
        vals = fmriData[:,np.where(challenge_roi)[0]].mean(axis = 1)
        avgRoiValues[:, i] = vals
    mask = np.arange(len(avgRoiValues[0]))
    print(mask)
    mask = mask[~np.isnan(avgRoiValues.max(axis=0))]
    print(mask)
    return rois[mask], avgRoiValues[:, mask]

class NSDDatasetClassSubset(Dataset):
    def __init__(self, parentFolderDir: str, imgDataFolderDir: str, subj: int, className: str, idxs: list = None, tsfms = None):
        self.imgPaths, self.trainingIDs = getClassImages(parentFolderDir, imgDataFolderDir, subj, className)
        self.lhFMRI = np.load(os.path.join(parentFolderDir, f"subj0{subj}/training_split/training_fmri/lh_training_fmri.npy"))[self.trainingIDs]
        self.rhFMRI = np.load(os.path.join(parentFolderDir, f"subj0{subj}/training_split/training_fmri/rh_training_fmri.npy"))[self.trainingIDs]
        self.tsfms = tsfms
        self.lhROIs, self.lhAvgROIs = getAvgROI(parentFolderDir, subj, self.lhFMRI)
        self.rhROIs, self.rhAvgROIs = getAvgROI(parentFolderDir, subj, self.rhFMRI, "r")
        if idxs is not None:
            self.imgPaths = self.imgPaths[idxs]
            self.lhFMRI = self.lhFMRI[idxs]
            self.rhFMRI = self.rhFMRI[idxs]
            self.lhAvgROIs = self.lhAvgROIs[idxs]
            self.rhAvgROIs = self.rhAvgROIs[idxs]
    def __len__(self):
        return len(self.imgPaths)
    def __getitem__(self, idx):
        img = Image.open(self.imgPaths[idx])
        lh = self.lhFMRI[idx]
        rh = self.rhFMRI[idx]
        lhAvg = self.lhAvgROIs[idx]
        rhAvg = self.rhAvgROIs[idx]
        if self.tsfms:
            img = self.tsfms(img)
        return img, self.imgPaths[idx], torch.from_numpy(lh), torch.from_numpy(rh), torch.tensor(lhAvg, dtype = torch.float32), torch.tensor(rhAvg, dtype = torch.float32)


class BalancedCocoSuperClassDataset(Dataset):
    def __init__(self, parentDir: str, metaDataDir: str, idxs: list = None, tsfms = None):
        superClasses = ['accessory', 'animal', 'appliance', 'electronic', 'food',
       'furniture', 'indoor', 'kitchen', 'outdoor', 'person', 'sports',
       'vehicle']
        self.tsfms = tsfms  
        self.parentDir = parentDir      
        self.imgPaths = np.load(os.path.join(metaDataDir, "newDatasetImagesPath.npy"))
        self.labels = np.load(os.path.join(metaDataDir, "newDatasetImagesLabel.npy"))
        self.labelEncoder = LabelEncoder().fit(superClasses)
        if idxs is not None:
            self.imgPaths = self.imgPaths[idxs]
            self.labels = self.labels[idxs]
    def __len__(self):
        return len(self.imgPaths)
    def __getitem__(self, idx):
        filePath = self.imgPaths[idx].split("C:/Users/josem/Documents/schoolWork/MQP/algonauts2023_transformers#2Leader/algonauts_2023_challenge_data/")[-1]
        img = Image.open(os.path.join(self.parentDir, filePath))
        label = self.labels[idx]
        if self.tsfms:
            img = self.tsfms(img)
        return img, torch.tensor(self.labelEncoder.transform([label]), dtype=torch.long).squeeze()

class CocoVGG (torch.nn.Module):
    def __init__(self, numClasses):
        super(CocoVGG, self).__init__()
        self.vgg = vgg19(weights="DEFAULT")
        self.features = self.vgg.features
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7,7))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features = 25088, out_features = 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features = 4096, out_features = 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features = 1024, out_features = numClasses),
            # torch.nn.Softmax(dim=1)
        )
        for layer in self.classifier:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.constant_(layer.bias, 0)
    def forward(self, img):
        x = self.features(img)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    def __str__(self):
        return ""
    def __repr__(self):
        return ""

class roiVGG(torch.nn.Module):
    def __init__(self, numClasses: int, cocoVGGWeights, numROIs: int, device):
        super(roiVGG, self).__init__()
        self.cocoVgg19 = CocoVGG(numClasses)
        self.cocoVgg19.load_state_dict(torch.load(cocoVGGWeights, map_location=device))
        self.convFeatures = self.cocoVgg19.features[:36]
        self.maxPool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False)
        self.adaptiveAvgPool = torch.nn.AdaptiveAvgPool2d(output_size = (7, 7))
        self.cocoClassifier = self.cocoVgg19.classifier
        self.roiClassifier = torch.nn.Sequential(
            torch.nn.Linear(in_features = 25088, out_features = 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features = 4096, out_features = 1024),#1024
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features = 1024, out_features = numROIs)
        )
        self.gradients = None
    def activations_hook(self, grad):
        self.gradients = grad
    def forward(self, img):
        extractedFeatures = self.convFeatures(img)
        # hook = extractedFeatures.register_hook(self.activations_hook)
        intermediateOutput = self.maxPool(extractedFeatures)
        intermediateOutput = self.adaptiveAvgPool(intermediateOutput)
        print(intermediateOutput.shape)
        intermediateOutput = torch.flatten(intermediateOutput, 1)
        return self.cocoClassifier(intermediateOutput), self.roiClassifier(intermediateOutput)
    def get_activation_gradient(self):
        return self.gradients
    def get_activations(self, img):
        return self.convFeatures(img)
    def __str__(self):
        return ""
    def __repr__(self):
        return ""

class YoloModel(YOLO):
    def __init__(self, *args, **kwargs):
        super(YoloModel, self).__init__(*args, **kwargs)
    def __str__(self):
        return ""
    def __repr__(self):
        return ""


class roiVGGYolo(torch.nn.Module):
    def __init__(self, numROIs: int, tsfms):
        super(roiVGGYolo, self).__init__()
        self.vgg = vgg19(weights = "DEFAULT")
        for params in self.vgg.parameters():
            params.requires_grad = False
        self.yolo = YoloModel("yolov8n.pt")
        for params in self.yolo.parameters():
            params.requires_grad = False
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(25088, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, numROIs),
        )
        self.tsfms = tsfms
    def forward(self, img, imgPaths):
        pooling = self.vgg.features(img)
        pooling = self.vgg.avgpool(pooling)
        yoloInput = [self.tsfms(Image.open(image)) for image in imgPaths]
        yoloResults = [results.boxes for results in self.yolo.predict(torch.stack(yoloInput), verbose=False)]
        boundingBoxDataAllImages = self.getMappedBoundingBox(yoloResults)
        finalFMRIs = []
        indices = []
        count = 0
        for boundingBoxData in boundingBoxDataAllImages:
            # print(boundingBoxData)
            # print(imgPaths[count])
            if len(boundingBoxData) == 0:
                continue
            objectROIPools = ops.roi_pool(pooling, [boundingBoxData], output_size = (7,7))
            fmriPieces = []
            for objectROIPool in objectROIPools:
                # print("objROIP")
                # print(objectROIPool.shape)
                # print(objectROIPool)
                input = torch.flatten(objectROIPool)
                fmriPieces.append(self.MLP(input))
            totalFMRI = torch.sum(torch.stack(fmriPieces), dim=0)
            finalFMRIs.append(totalFMRI)
            indices.append(count)
            count+=1
            # print("fmriPieces")
            # print(f"Shape: ({len(fmriPieces)}, {fmriPieces[0].shape})")
            # print(fmriPieces)
        return torch.stack(finalFMRIs), indices 
    def getMappedBoundingBox(self, yoloResults):
        mappedBoxes = []
        for result in yoloResults:
            boundingBoxData = result.xyxyn
            boundingBoxStartX, boundingBoxStartY, boundingBoxEndX, boundingBoxEndY = boundingBoxData[:, 0], boundingBoxData[:, 1], boundingBoxData[:, 2], boundingBoxData[:, 3]
            transformedBoundingBoxStartX, transformedBoundingBoxStartY, transformedBoundingBoxEndX, transformedBoundingBoxEndY = boundingBoxStartX * 7, boundingBoxStartY * 7, boundingBoxEndX * 7, boundingBoxEndY * 7
            startCellX = torch.floor(transformedBoundingBoxStartX)
            startCellY = torch.floor(transformedBoundingBoxStartY)
            endCellX = torch.ceil(transformedBoundingBoxEndX)
            endCellY = torch.ceil(transformedBoundingBoxEndY)
            mappedBoxes.append(torch.hstack((startCellX.reshape(-1,1), startCellY.reshape(-1,1), endCellX.reshape(-1,1), endCellY.reshape(-1,1))))
        # print("mapped boxes")
        # print(f"Shape: ({len(mappedBoxes)}, {mappedBoxes[0].shape})")
        # print(mappedBoxes)
        return mappedBoxes
    def __str__(self):
        return ""
    def __repr__(self):
        return ""


def cocoVGGTrain():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    subj = 1
    parentDir = "./algonauts_2023_challenge_data/"
    metaDataDir = "./subjCocoImgData/"

    tsfms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    numImages = 13200
    trainIdxs, validIdxs = train_test_split(range(numImages), train_size=0.9)

    trainingDataset = BalancedCocoSuperClassDataset(parentDir, metaDataDir, idxs=trainIdxs,tsfms = tsfms)
    trainDataLoader = DataLoader(trainingDataset, batch_size = 64, shuffle = True)

    validDataset = BalancedCocoSuperClassDataset(parentDir, metaDataDir, idxs = validIdxs, tsfms = tsfms)
    validDataLoader = DataLoader(validDataset, batch_size = 64, shuffle = True)



    numClasses = 12
    model = CocoVGG(numClasses).to(device)
    learningRate = 0.0001
    weightDecay = 1e-4
    optim = torch.optim.Adam(model.parameters(), learningRate, weight_decay=1e-4)#,  weight_decay=5e-4
    scheduler = lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()


    epochs = 30
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        avgTrainingLoss = 0
        avgEvalLoss = 0
        numRight = 0
        for data in tqdm(trainDataLoader, desc="Training", unit="batch"):  # for data in trainDataLoader: #
            img, label = data
            img = img.to(device)
            label = label.to(device)
            optim.zero_grad()
            pred = model(img)
            loss = criterion(pred, label)
            loss.backward()
            optim.step()  
            # numRight += (torch.argmax(pred, 1) == label).sum().item()
            avgTrainingLoss += loss.item()
        model.eval()
        with torch.no_grad():
            for data in tqdm(validDataLoader, desc="Evaluating", unit="batch"): 
                img, label = data
                img = img.to(device)
                label = label.to(device)
                pred = model(img)
                # print(torch.argmax(pred, 1))
                # print(label)
                evalLoss = criterion(pred, label)
                numRight += (torch.argmax(pred, 1) == label).sum().item()
                avgEvalLoss += evalLoss.item()
        print(f"Epoch {epoch} using lr {learningRate} TrainingCE: {avgTrainingLoss / len(trainDataLoader)}, ValidCE: {avgEvalLoss / len(validDataLoader)}, ValidAcc: {numRight / len(validDataset)}, got {numRight} right")
        learningRate = 0.0000001
        if epoch == 0:
            learningRate = 0.0000001
            for g in optim.param_groups:
                g["lr"] = learningRate
        scheduler.step()
        if epoch % 5 == 4:
            # learningRate = 0.00000001
            weightDecay *= 2
            for g in optim.param_groups:
                # g["lr"] = learningRate
                g["weight_decay"] = weightDecay
        model.train()


    torch.save(model.state_dict(), './cocoVGGModel.pth')


def roiTrain():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    subj = 1
    parentDir = "./algonauts_2023_challenge_data/"
    metaDataDir = "./subjCocoImgData/"

    tsfms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    numImages = len(getClassImages(parentDir, metaDataDir, subj, "person")[0])
    trainIdxs, validIdxs = train_test_split(range(numImages), train_size=0.9)

    trainingDataset = NSDDatasetClassSubset(parentDir, metaDataDir, subj, "person", idxs=trainIdxs, tsfms = tsfms)
    trainDataLoader = DataLoader(trainingDataset, batch_size = 64, shuffle = True)

    validDataset = NSDDatasetClassSubset(parentDir, metaDataDir, subj, "person", idxs = validIdxs, tsfms = tsfms)
    validDataLoader = DataLoader(validDataset, batch_size = 64, shuffle = True)

    numClasses = 12
    numROIs = len(trainingDataset.lhROIs)
    model = roiVGG(numClasses, "./cocoVGGModel.pth", numROIs, device).to(device)
    learningRate = 0.00001
    optim = torch.optim.Adam(model.parameters(), learningRate, weight_decay=1e-2)#,  weight_decay=1e-4
    # scheduler = lr_scheduler.StepLR(optim, step_size=3, gamma=0.5)
    criterion = torch.nn.MSELoss()


    epochs = 30
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        avgTrainingLoss = 0
        avgEvalLoss = 0
        avgTrainingR2Score = 0
        avgEvalR2Score = 0
        for data in tqdm(trainDataLoader, desc="Training", unit="batch"):  # for data in trainDataLoader: #
            img, _, _, _, avgFMRI, _ = data
            img = img.to(device)
            avgFMRI = avgFMRI.to(device)
            optim.zero_grad()
            pred = model(img)[1]
            loss = criterion(pred, avgFMRI)
            loss.backward()
            optim.step()  
            # numRight += (torch.argmax(pred, 1) == label).sum().item()
            avgTrainingLoss += loss.item()
            # print(r2_score(pred.detach().cpu().numpy(), avgFMRI.detach().cpu().numpy()))
            avgTrainingR2Score += r2_score(pred.detach().cpu().numpy(), avgFMRI.detach().cpu().numpy())
        # print(f"pred shape {pred.shape} avgFMRI shape {avgFMRI.shape}")
        model.eval()
        with torch.no_grad():
            for data in tqdm(validDataLoader, desc="Evaluating", unit="batch"): 
                img, _, _, _, avgFMRI, _ = data
                img = img.to(device)
                avgFMRI = avgFMRI.to(device)
                pred = model(img)[1]
                # print(f"pred shape {pred.shape} avgFMRI shape {avgFMRI.shape}")
                evalLoss = criterion(pred, avgFMRI)
                avgEvalLoss += evalLoss.item()
                # print(r2_score(pred.detach().cpu().numpy(), avgFMRI.detach().cpu().numpy()))
                avgEvalR2Score += r2_score(pred.detach().cpu().numpy(), avgFMRI.detach().cpu().numpy())
        print(f"Epoch {epoch} using lr = {learningRate} TrainingMSE: {avgTrainingLoss / len(trainDataLoader)}, ValidMSE: {avgEvalLoss / len(validDataLoader)}, trainR2 = {avgTrainingR2Score / len(trainDataLoader)}, evalR2= {avgEvalR2Score / len(validDataLoader)}")
    # scheduler.step()
    # learningRate = 0.000001
    # if epoch == 10:
    #     first = False
    #     learningRate = 0.0000001
    #     for g in optim.param_groups:
    #         g["lr"] = learningRate
        # if epoch == 5:
        #     learningRate = 0.00000001
        #     for g in optim.param_groups:
        #         g["lr"] = learningRate
        #         # g["weight_decay"] = 1e-3
        # model.train()


    torch.save(model.state_dict(), './cocoVGGROIModel.pth')

# def vggYoloTrain():
device = "cuda:0" if torch.cuda.is_available() else "cpu"
subj = 1
parentDir = "./algonauts_2023_challenge_data/"
metaDataDir = "./subjCocoImgData/"

tsfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

yoloTsfms = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])


numImages = len(getClassImages(parentDir, metaDataDir, subj, "person")[0])
trainIdxs, validIdxs = train_test_split(range(numImages), train_size=0.9)

trainingDataset = NSDDatasetClassSubset(parentDir, metaDataDir, subj, "person", idxs=trainIdxs, tsfms = tsfms)
trainDataLoader = DataLoader(trainingDataset, batch_size = 64, shuffle = True)

validDataset = NSDDatasetClassSubset(parentDir, metaDataDir, subj, "person", idxs = validIdxs, tsfms = tsfms)
validDataLoader = DataLoader(validDataset, batch_size = 64, shuffle = True)

# numClasses = 12
numROIs = len(trainingDataset.lhROIs)
model = roiVGGYolo(numROIs, yoloTsfms).to(device)
learningRate = 0.00001
optim = torch.optim.Adam(model.parameters(), learningRate)#,  weight_decay=1e-4
# scheduler = lr_scheduler.StepLR(optim, step_size=3, gamma=0.5)
criterion = torch.nn.MSELoss()

epochs = 30
for epoch in range(epochs):
    print(f"Epoch {epoch}")
    avgTrainingLoss = 0
    avgEvalLoss = 0
    avgTrainingR2Score = 0
    avgEvalR2Score = 0
    for data in tqdm(trainDataLoader, desc="Training", unit="batch"):  # for data in trainDataLoader: #
        img, imgPaths, _, _, avgFMRI, _ = data
        img = img.to(device)
        avgFMRI = avgFMRI.to(device)
        optim.zero_grad()
        # print("start")
        # print(imgPaths)
        # print("end")
        pred, indices = model(img, imgPaths)
        avgFMRI = avgFMRI[indices]
        loss = criterion(pred, avgFMRI)
        loss.backward()
        optim.step()  
        # numRight += (torch.argmax(pred, 1) == label).sum().item()
        avgTrainingLoss += loss.item()
        # print(r2_score(pred.detach().cpu().numpy(), avgFMRI.detach().cpu().numpy()))
        avgTrainingR2Score += r2_score(pred.detach().cpu().numpy(), avgFMRI.detach().cpu().numpy())
    # print(f"pred shape {pred.shape} avgFMRI shape {avgFMRI.shape}")
    # model.eval()
    with torch.no_grad():
        for data in tqdm(validDataLoader, desc="Evaluating", unit="batch"): 
            img, imgPaths, _, _, avgFMRI, _ = data
            img = img.to(device)
            avgFMRI = avgFMRI.to(device)
            pred, indices = model(img, imgPaths)
            avgFMRI = avgFMRI[indices]
            # print(f"pred shape {pred.shape} avgFMRI shape {avgFMRI.shape}")
            evalLoss = criterion(pred, avgFMRI)
            avgEvalLoss += evalLoss.item()
            # print(r2_score(pred.detach().cpu().numpy(), avgFMRI.detach().cpu().numpy()))
            avgEvalR2Score += r2_score(pred.detach().cpu().numpy(), avgFMRI.detach().cpu().numpy())
    print(f"Epoch {epoch} using lr = {learningRate} TrainingMSE: {avgTrainingLoss / len(trainDataLoader)}, ValidMSE: {avgEvalLoss / len(validDataLoader)}, trainR2 = {avgTrainingR2Score / len(trainDataLoader)}, evalR2= {avgEvalR2Score / len(validDataLoader)}")

# vggYoloTrain()

Epoch 0 using lr = 1e-05 TrainingMSE: 0.24650328377118477, ValidMSE: 0.19375486175219217, trainR2 = -7.625267733626556, evalR2= -12.356705763269654
Epoch 1 using lr = 1e-05 TrainingMSE: 0.18768594356683585, ValidMSE: 0.22018982470035553, trainR2 = -12.214109572149878, evalR2= -8.384807029140058
Epoch 2 using lr = 1e-05 TrainingMSE: 0.1760681735781523, ValidMSE: 0.17714243630568186, trainR2 = -16.73048399197644, evalR2= -20.757941338389095
Epoch 3 using lr = 1e-05 TrainingMSE: 0.16484679396335894, ValidMSE: 0.16567627092202505, trainR2 = -22.1752094720062, evalR2= -45.04963277928502
Epoch 4 using lr = 1e-05 TrainingMSE: 0.1575962551511251, ValidMSE: 0.16889413197835287, trainR2 = -32.81867493022851, evalR2= -27.80360850972889
Epoch 5 using lr = 1e-05 TrainingMSE: 0.15647497200048888, ValidMSE: 0.16427722573280334, trainR2 = -35.09076421157616, evalR2= -35.641196329953694
Epoch 6 using lr = 1e-05 TrainingMSE: 0.15564923790784982, ValidMSE: 0.16318556666374207, trainR2 = -35.96132734104747, evalR2= -39.145821150269136
Epoch 7 using lr = 1e-05 TrainingMSE: 0.1537240118934558, ValidMSE: 0.16099330286184946, trainR2 = -42.269807393690584, evalR2= -47.73509331838227
Epoch 8 using lr = 1e-05 TrainingMSE: 0.15184346873026627, ValidMSE: 0.16702192525068918, trainR2 = -45.19708969757432, evalR2= -33.47447206920615
Epoch 9 using lr = 1e-05 TrainingMSE: 0.1529601623232548, ValidMSE: 0.1609935313463211, trainR2 = -49.078086689815564, evalR2= -55.8409053744091
Epoch 11 using lr = 1e-05 TrainingMSE: 0.15181895574698082, ValidMSE: 0.1595749408006668, trainR2 = -49.72943311658242, evalR2= -64.03805876632208
Epoch 12 using lr = 1e-05 TrainingMSE: 0.15166527663285917, ValidMSE: 0.1594913750886917, trainR2 = -67.7119690115551, evalR2= -55.474116812966635
Epoch 13 using lr = 1e-05 TrainingMSE: 0.15293799913846529, ValidMSE: 0.1594665745894114, trainR2 = -56.15792784313115, evalR2= -55.58497980355838
Epoch 14 using lr = 1e-05 TrainingMSE: 0.15033111090843493, ValidMSE: 0.1610065996646881, trainR2 = -56.71758709772686, evalR2= -53.218392007138895
Epoch 15 using lr = 1e-05 TrainingMSE: 0.149626782307258, ValidMSE: 0.1597311000029246, trainR2 = -60.92608237818389, evalR2= -41.714277238061605
Epoch 16 using lr = 1e-05 TrainingMSE: 0.14969483189857924, ValidMSE: 0.16075542569160461, trainR2 = -59.08405926474505, evalR2= -50.44897151484258
Epoch 17 using lr = 1e-05 TrainingMSE: 0.14929455490066454, ValidMSE: 0.15939724445343018, trainR2 = -57.58491004859822, evalR2= -57.94167399290697