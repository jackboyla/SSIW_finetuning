# About
Finetuning the SSIW model from the paper: [Wei Yin, Yifan Liu, Chunhua Shen, Anton van den Hengel, Baichuan Sun, The devil is in the labels: Semantic segmentation from sentences](https://arxiv.org/abs/2202.02002).

This fork contains additional code for training, evaluation and loss computation using the CMP Facade Dataset, with credit to previous work done by [Bartolo1024](https://github.com/Bartolo1024).

I have also added more suitable descriptions for the CMP Facade labels.

![](src\ann_imgs\cmp_x0083_pred_vis.png)

# Data

- Original Label Embedding: https://cloudstor.aarnet.edu.au/plus/s/gXaGsZyvoUwu97t
- Original Model CKPT: https://cloudstor.aarnet.edu.au/plus/s/AtYYaVSVVAlEwve

The model has been finetuned on the [CMP Facade Dataset](https://cmp.felk.cvut.cz/~tylecr1/facade/). The `base` version has been used for training while the `extended` version is reserved for testing.


# Train & Evaluate

Training and testing can be carried out in this [Kaggle notebook](https://www.kaggle.com/jackkboylan/ssiw-finetuning).

The following experiments wer carried out:

1. A base model was trained with the frozen layers using the pretrained model and original HD Loss.

2. An almost identical model to 1 was trained, but this time a learnable temperature paramter was added to the HD Loss function.

3. Due to the tendency for large models such as the Segformer to overfit on small datasets, all parameters in the model have been frozen apart from the patch embedding layer.

4. Model 2 was trained with early stopping now watching the dice loss metric during validation.

5. Model 4 was used with frozen model parameters.

Batch size and image size are chosen to suit my available resources.


# Results & Future Work

| Experiment      | Dice Loss Mean | Dice Loss Std     |
| :---        |    :----   |          :--- |
| 1      | 0.33610681215660615       | 0.10716346525468866   |
| 2   | 0.3210149120800851        | 0.10403830232390819      |
| 3  | 0.39679140630259846      | 0.12017719375905421     |
| 4      | 0.3386301629637417       | 0.10906255159950705   |
| 5  | 0.4029399963389886      | 0.11731406566119605      |


In all cases, the model performs similarly in terms of dice loss on the test set as it had on the validation set during training. Adding the learnable temperature parameter to the HD Loss function marginally improves perfromance. Freezing model parameters shows a negative impact on test performance.

Possible improvements could come from:
- Non-destructive image augmentations
- Combined Dice & HD Loss Function
- Hyperparameter search for learning rate
