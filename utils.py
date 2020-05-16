import numpy as np

'''

Preprocessing Utils

'''

def normalize(image):
    return image/255.

def preprocess(image):
    image = image.reshape(48,48) 
    image = normalize(image)
    image = np.expand_dims(image, axis=0)
    return image



'''

Evaluation Utils

'''
def get_batch_evaluation_metrics(eval_dict, predicted, target, loss):

    eval_dict["loss"] += loss.item()
    pred_classes = predicted.max(dim = 1)[1]
    eval_dict["accuracy"] += (pred_classes == target).float().mean().item() 

    return eval_dict


def get_image_predictions(image, predicted, class_mapping):
    pred_classes = predicted.max(dim = 1)[1]
    
    image_predictions = []
    for idx in range(image.shape[0]):
        image_predictions.append({"Image": image[idx], "Emotion": class_mapping[pred_classes[idx]]})