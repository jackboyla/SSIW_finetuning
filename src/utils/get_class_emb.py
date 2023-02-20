import os
import os.path as osp

import numpy as np
import torch

from src.utils.labels_dict import ALL_LABEL2ID, UNAME2EM_NAME, UNI_UID2UNAME

UNI_UNAME2ID = {v: i for i, v in UNI_UID2UNAME.items()}


def create_embs_from_names(labels, other_descriptions=None, device=None):
    import clip

    DEVICE = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = "ViT-B/32"
    print(f"\nloading CLIP model {clip_model}...")
    CLIP_TEXT_MODEL, PREPROCESS = clip.load(clip_model, device=DEVICE)
    print(f"\nCLIP model {clip_model} loaded successfully")
    u_descrip_dir = "data/clip_descriptions"
    embs = []
    for name in labels:
        # get the label description from txt file or passed from other_descriptions
        if name in UNAME2EM_NAME.keys() and os.path.isfile(
            os.path.join(u_descrip_dir, UNAME2EM_NAME[name] + ".txt")
        ):
            with open(
                os.path.join(u_descrip_dir, UNAME2EM_NAME[name] + ".txt"), "r"
            ) as f:
                description = f.readlines()[0]
        elif name in other_descriptions:
            description = other_descriptions[name]

        # tokenize description
        text = clip.tokenize(
            [
                description,
            ]
        ).to(DEVICE)

        # extract features from tokens
        with torch.no_grad():
            text_features = CLIP_TEXT_MODEL.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        embs.append(text_features)
    embs = torch.stack(embs, dim=0).squeeze()
    return embs
