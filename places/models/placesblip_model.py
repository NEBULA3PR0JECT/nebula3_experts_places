import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms
from torch.nn import functional as F
import os
import sys
import numpy as np
import cv2
from PIL import Image
from nebula3_experts_places.places.models.places_model import PlacesModel
from nebula3_experts_places.places.config import PLACES_CONF
from nebula3_experts_places.places.models.blip_scripts.blip_itm import blip_itm
import requests
from torchvision.transforms.functional import InterpolationMode


class PlacesBlip365Model(PlacesModel):

    def __init__(self):
        super().__init__()
        self.config = PLACES_CONF()
        # load the labels
        self.classes, self.labels_IO, self.labels_attribute, self.W_attribute = self.load_labels()
        # load the model
        self.model, self.device, self.image_size = self.load_model()


    def load_model(self):
        device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth'
        image_size = 384
        vit = 'large'
        model = blip_itm(pretrained=model_url, image_size=image_size, vit=vit)
        model.eval()
        model = model.to(device)
        return model, device, image_size

    def process_frame(self, raw_image):

        raw_image = raw_image.convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]) 
        image = transform(raw_image).unsqueeze(0).to(self.device)   
        return image


    def load_labels(self):
        # prepare all the labels
        # scene category relevant
        file_name_category = 'categories_places365.txt'
        if not os.access(file_name_category, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            os.system('wget ' + synset_url)
        classes = list()
        with open(file_name_category) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:].split("/")[0])
        classes = np.unique(classes)

        # indoor and outdoor relevant
        file_name_IO = 'IO_places365.txt'
        if not os.access(file_name_IO, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
            os.system('wget ' + synset_url)
        with open(file_name_IO) as f:
            lines = f.readlines()
            labels_IO = []
            for line in lines:
                items = line.rstrip().split()
                labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
        labels_IO = np.array(labels_IO)

        # scene attribute relevant
        file_name_attribute = 'labels_sunattribute.txt'
        if not os.access(file_name_attribute, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
            os.system('wget ' + synset_url)
        with open(file_name_attribute) as f:
            lines = f.readlines()
            labels_attribute = [item.rstrip() for item in lines]
        file_name_W = 'W_sceneattribute_wideresnet18.npy'
        if not os.access(file_name_W, os.W_OK):
            synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
            os.system('wget ' + synset_url)
        W_attribute = np.load(file_name_W)

        return classes, labels_IO, labels_attribute, W_attribute

    def forward(self, image: Image, metadata=None):
        # Compute cosine similarity between image feature and text feature
        image = self.process_frame(image)
        text = list(self.classes)
        text = [f"A photo of {place}" for place in text]
        itc_output = self.model(image, text, match_head='itc')
        # Check if its dotproduct
        itc_scores = itc_output.cpu().detach().numpy()[0]
        places_to_scores = [(text[idx], itc_score) for idx, itc_score in enumerate(itc_scores)]
        places_to_scores = sorted(places_to_scores, key=lambda x: x[1], reverse=True)
        return itc_scores

def main():
    places_blip_model = PlacesBlip365Model()

    url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    image = Image.open(requests.get(url, stream=True).raw)  
    similarity_scores = places_blip_model.forward(image)
    top_5_scores = sorted(similarity_scores, reverse=True)
    print(f"BLIP_ITC top 5 outputs: {top_5_scores}")

if __name__ == "__main__":
    main()
    