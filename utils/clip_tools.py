import sys
import os
import time
from tqdm import tqdm
import torch

sys.path.append('/disk/work/hjwang/osrd/wise-ft/src')

import templates as templates
import datasets as datasets
# import clip as clip


from transformers import CLIPModel, CLIPTokenizer, CLIPVisionModelWithProjection, AutoProcessor



class CLIP_ft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        # self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, inputs, texts=None):
        image_features = self.model.get_image_features(pixel_values=inputs).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        if texts is None:
            return image_features
        
        text_features = self.model.get_text_features(input_ids=texts['input_ids'].cuda(), 
                                            attention_mask=texts['attention_mask'].cuda()).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)   

        logits_per_image = self.model.logit_scale * image_features @ text_features.T
        logits_per_text = self.model.logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

class CLIP_lp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
        self.linear = torch.nn.Linear(512, 1000)

    def forward(self, inputs, text_inputs=None):
        outputs = self.model(inputs)
        outputs = self.linear(outputs.image_embeds)
        return outputs