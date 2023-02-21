import os.path
import sys
import os
import glob
import yaml
conf_path = os.getcwd()
sys.path.append(conf_path)

import click
import pandas as pd
import numpy as np
import tqdm
import PIL.Image
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
from segmentation_models_pytorch.losses import DiceLoss
from torchvision import transforms
from torchmetrics import Dice

from src.tools.train import get_model, load_embeddings
from src.utils.dataflow import get_data_transforms, ann_to_embedding, ann_to_one_hot
from src.utils.loss import nxn_cos_sim, CustomDiceLoss
from src.utils.transforms_utils import get_imagenet_mean_std


class ImageLoader:
    def __init__(self, device: torch.device):
        mean, std = get_imagenet_mean_std()
        self.normalizer = transforms.Normalize(mean, std)
        self.to_tensor = transforms.ToTensor()
        self.device = device

    def __call__(self, img_path: str):
        img = PIL.Image.open(img_path).convert("RGB")
        x = self.to_tensor(img)
        x = self.normalizer(x).to(self.device)
        return img, x


def predict(x: torch.Tensor, model: nn.Module, embeddings: torch.Tensor):
    with torch.no_grad():
        pred, _, _ = model(x.unsqueeze(0))
    bs, channels, h, w = pred.shape
    output = pred.permute(0, 2, 3, 1).view(-1, channels)
    cos_sim = nxn_cos_sim(output, embeddings) # [B*H*W, num_cls]
    # get the class indices for max values along the 'num_cls' dim
    _, labels = cos_sim.max(dim=-1)
    # get the most probable class for each pixel
    labels = labels.view(h, w)
    return pred.cpu(), labels.cpu()


def plot(
    img, labels, 
    pred_path, target=None
    ):
    labels = labels.numpy()
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(img)
    axs[0].set_title('Input')
    axs[0].axis('off')
    axs[1].imshow(labels)
    axs[1].axis('off')
    axs[1].set_title('Prediction')
    if target:
        axs[2].imshow(target)
        axs[2].axis('off')
        axs[2].set_title('Target')
    plt.savefig(pred_path)
    # plt.show()
    plt.close()


@click.command()
@click.argument("run-dir")
@click.option("--img-dir", default=None, multiple=True)
def test(run_dir, img_dir):

    try: 
        with open (f"{run_dir}/config_yaml", "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    if img_dir:
        data_dirs = img_dir
        config['files']['test_data_dirs'] = data_dirs
    else:
        data_dirs = config['files']['test_data_dirs']
    checkpoint_weights = config['files']['save_checkpoint_weights']
    output_folder = config['files']['output_folder']
    batch_size = int(config['hyperparams']['batch_size'])

    print("CONFIG: \n", config, "\n")
    

    device = torch.device(config['device'])
    img_loader = ImageLoader(device)
    device = torch.device(device)

    pred_dir = 'predictions'
    os.mkdir('predictions')
    print(f"Predictions will be saved in {run_dir}/{pred_dir}/")

    # load label embeddings and model
    embs = load_embeddings(config['files']['labels_path'])
    embs = embs.to(device)
    model = get_model(num_classes=512, checkpoint_weights=checkpoint_weights).to(device).eval()

    # find all test images
    test_images = []
    for dir in data_dirs:
        test_images.extend(glob.glob(dir + "*.jpg"))

    
    dice_loss_fn = CustomDiceLoss(embs)
    dice_losses = []
    prediction_paths = []
    # run prediction on all test images
    for i, test_image_path in enumerate(tqdm.tqdm(test_images)):
        img, x = img_loader(test_image_path)

        pred, labels = predict(x, model, embs)

        tgt_path = test_image_path.replace('.jpg', '.png')

        pred_path = pred_dir + "/" + test_image_path.split('/')[-1].split('.')[0] + "_pred_vis.png"
        prediction_paths.append(pred_path)
        
        if os.path.isfile(tgt_path):

            target = PIL.Image.open(tgt_path)

            _, ann_transform = get_data_transforms()
            ann = ann_transform(target)#.squeeze(0)
            ann = ann - 1  # change indexing
            ann_one_hot = ann_to_one_hot(ann, embs.shape[0])

            dice_loss = dice_loss_fn(pred.unsqueeze(0), target, ann_one_hot)
            dice_losses.append(dice_loss)

            plot(img, labels, pred_path, target=target)
        else:
            plot(img, labels, pred_path)
    
    df = pd.DataFrame(
        list(zip(test_images, prediction_paths, dice_losses)),
        columns =['image_path', 'prediction_path', 'dice_loss']
        )
    performance_csv = f"{run_dir}/test_out.csv"
    df.to_csv(performance_csv)
    print(f"Model performance saved to {performance_csv}")

    result = {"mean": np.mean(dice_losses), "std":np.std(dice_losses)}
    print(f"Model: {checkpoint_weights}\nDice Loss {result}")

    config['performance']['dice_loss'] = result

    with open (f"{run_dir}/config_yaml", "w") as file:
        config = yaml.dump(file)


if __name__ == "__main__":
    test()
