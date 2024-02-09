import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from ultralytics import YOLO
from data import makeList


def makePredictions(train, train_fmri, val, val_fmri):

    model = LinearRegression()
    model.fit(train, train_fmri)
    random_forest_predictions = model.predict(val)

    print(val_fmri, "\n _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n", random_forest_predictions)
    random_forest_mse = mean_squared_error(val_fmri, random_forest_predictions)
    print(f'Mean Squared Error: {random_forest_mse}')
    accuracy_score = model.score(val, val_fmri)
    print("Accuracy Score", accuracy_score)


    return random_forest_predictions


def  makeClassifications(image_list, idxs, batch_size=100):
    # w2v = api.load("word2vec-google-news-300")
    modelYOLO = YOLO('yolov8n.pt')

    print(image_list)
    results = []
    index = 0

    for start_idx in range(0, len(image_list), batch_size):
        end_idx = start_idx + batch_size
        batch_imgs = image_list[start_idx:end_idx]

        # Perform predictions on the batch of images
        image_results = modelYOLO.predict(batch_imgs, stream=True)
        # data = []

        for result in image_results:
            # print(idxs[index])
            index = index + 1
            # print(result)
            # temp = []  # Use a set to store unique items
            detection_count = result.boxes.shape[0]

            for i in range(min(detection_count, 3)):  # Limit to a maximum of 5 items

                cls = int(result.boxes.cls[i].item())
                name = result.names[cls]
                confidence = float(result.boxes.conf[i].item())
                # print(confidence)
                # print(name, cls, confidence)
                if confidence > 0.9:
                    # temp.append(name)
                    # temp.append(cls)
                    bounding_box = result.boxes.xyxy[i].cpu().numpy()

                    # print("Bounding Box Coordinates:", bounding_box)

                    x1, y1, x2, y2 = map(int, bounding_box)
                    results.append([index, name, cls, x1, y1, x2, y2])
                    # print("Top-Left Corner (x1, y1):", x1, y1)
                    # print("Bottom-Right Corner (x2, y2):", x2, y2)
                    # print('\n')
            # results.append(temp)
        torch.cuda.empty_cache()

    del modelYOLO
    print(results)
    data = pd.DataFrame(results)
    data.fillna(-1, inplace=True)
    print(data)
    return data
