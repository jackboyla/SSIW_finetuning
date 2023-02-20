import os.path
import sys
import os
import glob
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
    img, labels, save_folder, 
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
    if pred_path:
        plt.savefig(pred_path)
    # plt.show()
    plt.close()


@click.command()
@click.argument("img-dir")
@click.argument("checkpoint-path")
@click.option("--device", default="cuda:0")
@click.option("--labels-path", default="src/configs/labels.yaml")
@click.option("--save-folder", default=None)
def test(img_dir, checkpoint_path, device, labels_path, save_folder):

    device = torch.device(device)
    img_loader = ImageLoader(device)
    device = torch.device(device)

    exp_path = '/'.join(checkpoint_path.split('/')[:-1])

    if save_folder:
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        print(f"Predictions will be saved in {save_folder}")
    else:
        save_folder = "."

    # load label embeddings and model
    embs = load_embeddings(labels_path)
    embs = embs.to(device)
    model = get_model(num_classes=512, checkpoint_weights=checkpoint_path).to(device).eval()

    # find all test images
    test_images = glob.glob(img_dir + "*.jpg")

    
    dice_loss_fn = CustomDiceLoss(embs)
    dice_losses = []
    prediction_paths = []
    # run prediction on all test images
    for i, test_image_path in enumerate(tqdm.tqdm(test_images)):
        img, x = img_loader(test_image_path)

        pred, labels = predict(x, model, embs)

        tgt_path = test_image_path.replace('.jpg', '.png')
        if save_folder:
            pred_path = save_folder + "/" + test_image_path.split('/')[-1].split('.')[0] + "_pred_vis.png"
            prediction_paths.append(pred_path)
        else:
            pred_path = None
        
        if os.path.isfile(tgt_path):


            target = PIL.Image.open(tgt_path)

            _, ann_transform = get_data_transforms()
            ann = ann_transform(target)#.squeeze(0)
            ann = ann - 1  # change indexing

            ann_one_hot = ann_to_one_hot(ann, embs.shape[0])

            dice_loss = dice_loss_fn(pred.unsqueeze(0), target, ann_one_hot)
            dice_losses.append(dice_loss)

            plot(img, labels, save_folder, pred_path, target=target)
        else:
            plot(img, labels, save_folder, pred_path)
    
    df = pd.DataFrame(list(zip(test_images, prediction_paths, dice_losses)),
                columns =['image_path', 'prediction_path', 'dice_loss'])
    df.to_csv(f"{exp_path}/test_out.csv")
    print(f"Model performance saved to {exp_path}/test_out.csv")

    result = {"mean": np.mean(dice_losses), "std":np.std(dice_losses)}
    print(f"Model: {checkpoint_path}\n Dice Loss {result}")


if __name__ == "__main__":
    test()
