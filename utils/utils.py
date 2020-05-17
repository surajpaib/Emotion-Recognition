import numpy as np
import wandb
import torch

'''

Preprocessing Utils

'''

def normalize(image):
    return image/255.

def preprocess(image):
    dim = int(np.sqrt(image.size))
    image = image.reshape(dim, dim) 
    image = normalize(image)
    image = np.expand_dims(image, axis=0)
    return image



'''

Training Utils

'''

def get_loss(args, class_weights):
    loss_str = eval('torch.nn.{}'.format(args.loss))

    if args.balanced_loss:
        criterion = loss_str(weight=class_weights)
    else:
        criterion = loss_str()

    return criterion


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')    


def wandb_log(image_predictions, metrics):

    images = []
    for prediction in image_predictions:
        images.append(wandb.Image(prediction["Image"], caption="Predicted: {}, True: {}".format(prediction["Predicted"], prediction["GT"])))
    
    metrics["Examples"] = images

    wandb.log(metrics)

'''

Evaluation Utils

'''

def get_batch_evaluation_metrics(eval_dict, predicted, target, loss):

    eval_dict["loss"] += loss.item()
    pred_classes = predicted.max(dim = 1)[1]
    eval_dict["accuracy"] += (pred_classes == target).float().mean().item() 

    return eval_dict


def get_image_predictions(image, target, predicted, class_mapping):
    pred_classes = predicted.max(dim = 1)[1]
    
    image_predictions = []
    for idx in range(image.shape[0]):
        image_predictions.append({"Image": image[idx], "Predicted": class_mapping[pred_classes[idx]], "GT": class_mapping[target[idx]]})

    return image_predictions

def concatenate_metrics(train_metrics, val_metrics):
    train_metrics = {k+'@train': v for k, v in train_metrics.items()}
    val_metrics = {k+'@val': v for k, v in val_metrics.items()}

    return {**train_metrics, **val_metrics}


def convertModel(model_path, model):
    if not(torch.cuda.is_available()):
        checkpoint = torch.load(model_path,  map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']
        new_dict = {}
        for key in state_dict:
            new_key = key.replace('module.', '')
            new_dict[new_key] = state_dict[key]
        print(checkpoint['optimizer'].keys())
        model.load_state_dict(new_dict)
    else:
        checkpoint = torch.load(model_path,  map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']        
        new_dict = {}
        for key in state_dict:
            new_key = key.replace('module.', '')
            new_dict[new_key] = state_dict[key]
        print(checkpoint['optimizer'].keys())

        model.load_state_dict(new_dict)

    print("Loaded Model: {} successfully".format(model_path))
    return model