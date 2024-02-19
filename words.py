import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from ultralytics import YOLO


def predictions(train, train_fmri, val, val_fmri, model):
    print("PREDICTIONS")
    # train = train.to_numpy()
    # train_fmri = train_fmri.to_numpy()
    # val = val.to_numpy()
    # val_fmri = val_fmri.to_numpy()
    # input train data

    random_forest_model = model
    random_forest_model.fit(train, train_fmri)
    random_forest_predictions = random_forest_model.predict(val)
    print(random_forest_predictions)

    random_forest_mse = mean_squared_error(val_fmri, random_forest_predictions)
    print(f'Random Forest Mean Squared Error: {random_forest_mse}')

    accuracy_score = random_forest_model.score(val, val_fmri)
    print("accuracy score", accuracy_score)

    return random_forest_predictions



def makeClassifications(image_list,idxs,device,batch_size=300):
    modelYOLO = YOLO('yolov8n.pt')
    modelYOLO.to(device)
    results = []
    words = ['furniture', 'food', 'kitchenware', 'appliance', 'person', 'animal', 'vehicle', 'accessory',
             'electronics', 'sports', 'traffic', 'outdoor', 'home', 'clothing', 'hygiene', 'toy', 'plumbing',
             'safety', 'luggage', 'computer', 'fruit', 'vegetable', 'tool']

    for start_idx in range(0, len(image_list), batch_size):
        end_idx = start_idx + batch_size
        batch_imgs = image_list[start_idx:end_idx]
        batch_idxs = idxs[start_idx:end_idx]  # Ensure batch size matches

        # Perform predictions on the batch of images
        image_results = modelYOLO.predict(batch_imgs, stream=True)
        names = modelYOLO.names
        print(names)

        for i, result in enumerate(image_results):
            detection_count = result.boxes.shape[0]
            image_idx = batch_idxs[i]

            # Initialize count for the current image if not present in results
            # if image_idx not in results:
            #    results[image_idx] = {'classifications': 0, 'data': []}

            best_confidence = 0.3  # Adjust as needed
            best_item = None

            for j in range(len(result.boxes)):
                # print(result.boxes[j].scores)
                confidence = float(result.boxes.conf[j].item())
                # print(confidence)
                if confidence > best_confidence:
                    cls = int(result.boxes.cls[j].item())
                    name = names[cls]
                    # print(name)
                    name = class_mapping.get(name)
                    num = words.index(name)
                    # print(name)
                    # print(cls, num, confidence)
                    best_confidence = confidence

                    if best_item is None or confidence > best_item['confidence']:
                        best_item = {
                            'cls': cls,
                            'name': name,
                            'num': num,
                            'confidence': confidence
                            # Add more fields as needed
                        }

            # Print or use the best_item
            if best_item:
                # print("Best Item:", best_item)
                results.append([cls, num])
            else:

                results.append([-1, -1])
                # print("No item meets the criteria.")

    df = pd.DataFrame(results)
    df = df.fillna(-1)
    print(df)
    final = df.to_numpy()
    return final


def makeClassifications2(image_list, idxs, device, batch_size=64):
    modelYOLO = YOLO('yolov8n.pt')
    modelYOLO.to(device)

    results = {}

    for start_idx in range(0, len(image_list), batch_size):
        end_idx = start_idx + batch_size
        batch_imgs = image_list[start_idx:end_idx]
        batch_idxs = idxs[start_idx:end_idx]  # Ensure batch size matches

        # Perform predictions on the batch of images
        image_results = modelYOLO.predict(batch_imgs, stream=True)

        for i, result in enumerate(image_results):
            detection_count = result.boxes.shape[0]
            image_idx = batch_idxs[i]

            # Initialize count for the current image if not present in results
            if image_idx not in results:
                results[image_idx] = {'classifications': 0, 'data': []}

            for j in range(detection_count):
                confidence = float(result.boxes.conf[j].item())

                if confidence > 0.5 and results[image_idx]['classifications'] < 2:
                    cls = int(result.boxes.cls[j].item())
                    bounding_box = result.boxes.xyxy[j].cpu().numpy()
                    results[image_idx]['data'].append(cls)
                    results[image_idx]['classifications'] += 1

    # Convert the dictionary to a list of tuples for the final output
    final_results = [(image_idx, *data['data']) for image_idx, data in results.items()]
    df = pd.DataFrame(final_results)
    df = df.fillna(-1)
    print(df)
    final = df.to_numpy()

    return final

class_mapping = {
    'chair': 'furniture',
    'bowl': 'kitchenware',
    'dining table': 'furniture',
    'person': 'person',
    'bird': 'animal',
    'knife': 'kitchenware',
    'sink': 'appliance',
    'horse': 'animal',
    'cake': 'food',
    'giraffe': 'animal',
    'car': 'vehicle',
    'umbrella': 'accessory',
    'refrigerator': 'appliance',
    'cow': 'animal',
    'dog': 'animal',
    'tv': 'electronics',
    'surfboard': 'sports',
    'cat': 'animal',
    'stop sign': 'traffic',
    'train': 'vehicle',
    'zebra': 'animal',
    'carrot': 'vegetable',
    'bicycle': 'vehicle',
    'sports ball': 'sports',
    'sheep': 'animal',
    'motorcycle': 'vehicle',
    'bottle': 'kitchenware',
    'sandwich': 'food',
    'clock': 'home',
    'bear': 'animal',
    'truck': 'vehicle',
    'traffic light': 'traffic',
    'cell phone': 'electronics',
    'oven': 'appliance',
    'cup': 'kitchenware',
    'couch': 'furniture',
    'airplane': 'vehicle',
    'boat': 'vehicle',
    'bus': 'vehicle',
    'broccoli': 'vegetable',
    'tennis racket': 'sports',
    'elephant': 'animal',
    'parking meter': 'traffic',
    'tie': 'clothing',
    'bed': 'furniture',
    'toaster': 'appliance',
    'banana': 'fruit',
    'toothbrush': 'hygiene',
    'kite': 'toy',
    'skateboard': 'sports',
    'potted plant': 'home',
    'bench': 'outdoor',
    'donut': 'food',
    'spoon': 'kitchenware',
    'toilet': 'plumbing',
    'baseball bat': 'sports',
    'pizza': 'food',
    'scissors': 'tool',
    'fire hydrant': 'outdoor',
    'teddy bear': 'toy',
    'remote': 'electronics',
    'apple': 'fruit',
    'suitcase': 'luggage',
    'vase': 'home',
    'skis': 'sports',
    'hot dog': 'food',
    'frisbee': 'toy',
    'backpack': 'luggage',
    'microwave': 'appliance',
    'wine glass': 'kitchenware',
    'snowboard': 'sports',
    'baseball glove': 'sports',
    'book': 'toy',
    'orange': 'fruit',
    'fork': 'kitchenware',
    'laptop': 'electronics',
    'handbag': 'accessory',
    'keyboard': 'computer',
    'mouse': 'computer'
}