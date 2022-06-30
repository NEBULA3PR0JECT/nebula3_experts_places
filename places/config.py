from os import getenv

"""backend & models

"""

class PLACES_CONF:
    def __init__(self) -> None:

        # self.MODEL_FILE = getenv('MODEL_FILE','wideresnet18_places365.pth.tar')
        self.PLACES_EXPERT_NAME = getenv('PLACES_EXPERT_NAME', 'places365')
        self.PLACES_MODEL_NAME = getenv('PLACES_MODEL_NAME', 'resnet152')
        self.PLACES_SCENE_CATS  = int(getenv('PLACES_SCENE_CATS', '5'))
        self.PLACES_SCENE_ATTRS =  int(getenv('PLACES_SCENE_ATTRS', '5'))

    def get_places_expert(self):
        return (self.PLACES_EXPERT_NAME)
    def get_places_model_name(self):
        return (self.PLACES_MODEL_NAME)
    def get_places_scene_cats(self):
        return (self.PLACES_SCENE_CATS)
    def get_places_scene_attrs(self):
        return (self.PLACES_SCENE_ATTRS)
