
import sys
import os
conf_path = os.getcwd()
sys.path.append(conf_path)

import json
import logging
import datetime
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Dict

import click
import livelossplot
import torch
import yaml
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, global_step_from_engine, EarlyStopping
from ignite.metrics import Loss, RunningAverage
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F

from src.utils.dataflow import CMPDataset, ann_to_embedding, ann_to_one_hot
from src.utils.get_class_emb import create_embs_from_names
from src.utils.loss import HDLoss, CustomDiceLoss
from src.utils.segformer import get_configured_segformer


def loss_running_average(
    engine: Engine,
    monitoring_metrics: Sequence[str] = ("loss",),
    running_average_decay: float = 0.98,
    prefix: str = "batch_",
) -> List[str]:
    """partial instead of lambda - see https://github.com/pytorch/ignite/issues/639"""
    from functools import partial

    def output_transform(x, key: str):
        return x.get(key, 1.0)

    for m in monitoring_metrics:
        RunningAverage(
            alpha=running_average_decay,
            output_transform=partial(output_transform, key=m),
        ).attach(engine, prefix + m)
    return [prefix + m for m in monitoring_metrics]


def load_labels(path):
    '''
    args: str YAML path

    returns: dictionary 
                {label: {'description': ... , 
                        'label': ..., 
                        'pixel': ...},
                ...
                }
    '''
    with open(path, "r") as f:
        labels = yaml.load(f, yaml.BaseLoader)
    return labels


def load_embeddings(labels_path: str):
    '''
    args: str YAML path

    returns: 
        torch.Size([C, embed_dim])
        where C = number of labels and 
        embed_dim = embedding dimension for CLIP model (when using ViT-B/32 this is 512)
    '''
    labels = load_labels(labels_path)
    keys = labels.keys()
    descriptions = {k: v["description"] for k, v in labels.items()}
    embs = create_embs_from_names(keys, descriptions, device="cpu")
    return embs


def create_loaders(
    data_dirs: Sequence[str],
    train_ratio: float = 0.8,
    batch_size: int = 4,
    img_size: Tuple[int, int] = (400, 400),
):
    dataset = CMPDataset(data_dirs, img_size=img_size)
    train_len = int(len(dataset) * train_ratio)
    train_dataset, valid_dataset = random_split(
        dataset, lengths=[train_len, len(dataset) - train_len]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader


def get_model(num_classes: int, checkpoint_weights: str, freeze: bool = False):
    model = get_configured_segformer(num_classes, None, load_imagenet_model=False)
    state_dict = torch.load(checkpoint_weights, map_location=torch.device("cpu"))
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict.keys() else state_dict["model"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    if "criterion.0.logit_scale" in state_dict:
        del state_dict["criterion.0.logit_scale"]
    model.load_state_dict(state_dict)
    if freeze:
        for param in model.segmodel.encoder.parameters():
            param.requires_grad = False
        model.segmodel.encoder.patch_embed1.requires_grad = True  # unfreeze stump
        model.segmodel.encoder.patch_embed2.requires_grad = True  # unfreeze stump
        model.segmodel.encoder.patch_embed3.requires_grad = True  # unfreeze stump
        model.segmodel.encoder.patch_embed4.requires_grad = True  # unfreeze stump
    return model


def get_logit_scale(checkpoint_weights: str):
    state_dict = torch.load(checkpoint_weights, map_location=torch.device("cpu"))
    state_dict = state_dict["state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    logit_scale = state_dict["criterion.0.logit_scale"]
    logging.info(f"Load logit scale {logit_scale}")
    return logit_scale


def create_step_fn(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    embeddings: torch.Tensor,
    device: Optional[Union[str, torch.device]] = None,
):
    def update(
        _: Engine, batch: Sequence[torch.Tensor]
    ) -> Union[Any, Tuple[torch.Tensor]]:
        optimizer.zero_grad()
        model.train()
        x, ann = batch
        x = x.to(device)
        ann = ann.to(device)
        target = ann_to_embedding(ann, embeddings)
        ann_one_hot = ann_to_one_hot(ann, embeddings.shape[0])
        y_pred, _, _ = model(x)
        loss = loss_fn(y_pred, ann_one_hot)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}

    return update


def create_evaluation_step_fn(
    model: torch.nn.Module,
    embeddings: torch.Tensor,
    device: Optional[Union[str, torch.device]] = None,
    output_transform: Callable[
        [Any, Any, Any, Any], Any
    ] = lambda x, y, y_pred, ann_one_hot: (y_pred, ann_one_hot),
) -> Callable:
    def evaluate_step(
        _: Engine, batch: Sequence[torch.Tensor]
    ) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            x, ann = batch
            x = x.to(device)
            ann = ann.to(device)
            target = ann_to_embedding(ann, embeddings)
            ann_one_hot = ann_to_one_hot(ann, embeddings.shape[0])
            y_pred, _, _ = model(x)

            return output_transform(x, target, y_pred, ann_one_hot)

    return evaluate_step


def create_log_fn(trainer: Engine, event: Events, train_dir: str):
    logger = livelossplot.PlotLosses(outputs=["ExtremaPrinter"])

    def log_metrics(engine: Engine):
        current_step = global_step_from_engine(trainer)(None, event)
        logger.update(engine.state.metrics)
        for k, v in engine.state.metrics.items():
            print(f"Step: {current_step} | {k}: {v}")
        logger.send()

        with open(f"{train_dir}/logs.json", "w") as fp:
            json.dump(logger.logger.log_history, fp)
    return log_metrics


# def eval_score_function(engine, watch_metric):
#     eval_score = engine.state.metrics['val_loss']
#     # Objects with highest scores will be retained.
#     return eval_score

def create_eval_score_function(
    _: Engine, 
    watch_metric: str
) -> Callable:
    def eval_score_function(engine: Engine, watch_metric: str = watch_metric):
        eval_score = engine.state.metrics[watch_metric]
        # Objects with highest scores will be retained.
        return eval_score

    return eval_score_function


@click.command()
@click.option("--config-yaml", nargs=1)
def train(config_yaml):

    try: 
        with open (config_yaml, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    print("CONFIG: \n", config, "\n")

    data_dirs = config['files']['data_dirs']
    checkpoint_weights = config['files']['load_checkpoint_weights']
    output_folder = config['files']['output_folder']
    run_name = config['hyperparams']['run_name']
    train_dir = f"{output_folder}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{run_name}"
    max_epochs = int(config['hyperparams']['max_epochs'])
    batch_size = int(config['hyperparams']['batch_size'])
    device = torch.device(config['device'])
    embs = load_embeddings(config['files']['labels_path'])

    os.makedirs(train_dir)
    print(f"All run files will be saved to : {train_dir}")
    train_loader, val_loader = create_loaders(data_dirs, batch_size=batch_size)
    model = get_model(num_classes=512, checkpoint_weights=checkpoint_weights).to(device)
    embs = embs.to(device)
    logit_scale = get_logit_scale(checkpoint_weights=checkpoint_weights)
    loss_fn = HDLoss(embs, logit_scale).to(device)
    dice_loss_fn = CustomDiceLoss(embs).to(device)
    params = list(model.parameters())
    # add learnable temp parameter from HDLoss
    # params.extend(list(loss_fn.parameters()))
    optimizer = optim.Adam(lr=float(config['hyperparams']['learning_rate']), params=params)

    # TRAINER
    step_fn = create_step_fn(model, optimizer, loss_fn, embs, device)
    trainer = Engine(step_fn)
    print_metrics = create_log_fn(trainer, Events.ITERATION_COMPLETED, train_dir)
    loss_running_average(trainer, prefix="train_")
    trainer.add_event_handler(Events.EPOCH_COMPLETED, print_metrics)
    ProgressBar().attach(trainer)
    to_save = {"model": model, "optimizer": optimizer, "trainer": trainer}
    checkpointer = Checkpoint(to_save, f"{train_dir}/", n_saved=int(config['hyperparams']['checkpointing']['n_saved']))
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=int(config['hyperparams']['checkpointing']['save_every'])), checkpointer)
    trainer.add_event_handler(Events.COMPLETED, checkpointer)

    # EVALUATOR
    metrics = {
        "val_loss": Loss(loss_fn),
        "val_dice_loss": Loss(dice_loss_fn)
        }
    eval_step_fn = create_evaluation_step_fn(model, embs, device)
    evaluator = Engine(eval_step_fn)
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    evaluator.add_event_handler(Events.COMPLETED, print_metrics)
    eval_score_function = create_eval_score_function(evaluator, config['hyperparams']['early_stopping']['watch_metric'])
    early_stopping = EarlyStopping(
        patience=int(config['hyperparams']['early_stopping']['patience']), 
        score_function=eval_score_function, trainer=trainer
        )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, early_stopping)
    ProgressBar().attach(evaluator)

    def eval(_: Engine):
        evaluator.run(val_loader)

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=int(config['hyperparams']['eval_every'])), eval)

    trainer.run(train_loader, max_epochs=max_epochs)

    sd = model.state_dict()
    torch.save({"state_dict": sd}, f"{train_dir}/{config['files']['save_checkpoint_weights']}")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    train()
