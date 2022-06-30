# from typing import Optional
from locale import currency
from fastapi import FastAPI

from typing import Optional
import sys
from numpy import number
from PIL import Image
from nebula3_experts.nebula3_pipeline.nebula3_database.movie_tokens import TokenEntry

sys.path.append("/notebooks/nebula3_experts")
sys.path.append("/notebooks/nebula3_experts/nebula3_pipeline")
sys.path.append("/notebooks/nebula3_experts/nebula3_pipeline/nebula3_database")

from nebula3_experts.experts.common.constants import OUTPUT_DB
from nebula3_experts.experts.service.base_expert import BaseExpert
from nebula3_experts.experts.app import ExpertApp
from nebula3_experts.experts.common.models import ExpertParam, TokenRecord
from nebula3_experts.experts.common.defines import OutputStyle
from places.config import PLACES_CONF
from places.models.model_factory import create_places_model

# sys.path.remove(".")

class PlacesExpert(BaseExpert):
    def __init__(self):
        super().__init__()
        self.config = PLACES_CONF()
        self.model = self.load_model()
        self.dispatch_dict = {}
        # after init all
        self.set_active()

    def load_model(self):
        model = create_places_model(self.config.get_places_expert())
        return model

    def get_name(self):
        return "PlacesExpert"

    def add_expert_apis(self, app: FastAPI):
        @app.get("/my-expert")
        def get_my_expert(q: Optional[str] = None):
            return {"expert": "places" }

    def predict(self, expert_params: ExpertParam):
        """ handle new movie """
        print(f'Predicting movie: {expert_params.movie_id}')
        # get movie, extract frames, activate the places on the frames
        result, error = self.handle_movie(expert_params)
        if not error and expert_params.output == OUTPUT_DB:
            result, error = self.save_to_db(expert_params.movie_id, result)
        return { 'result': result, 'error': error }

    def handle_movie(self, params: ExpertParam):
        """handling prediction on movie:
        getting the movie, extracting the specified frames and predicting
        Args:
            params (ExpertParam): _description_

        Returns:
            result_: token entries
            error: error message
        """
        error_msg = None
        result = None
        if params.movie_id is None:
            self.logger.error(f'missing movie_id')
            return { 'error': f'movie frames not found: {params.movie_id}'}
        try:
            self.dispatch_dict[params.movie_id] = {}
            movie_fetched = self.download_video_file(params.movie_id)
            if movie_fetched:
                # now calling action function
                prediction_result, error = self.handle_predictions(params)
                # now transforming results data
                if prediction_result and not error:
                    result = self.transform_predictions(prediction_result, params)
            else:
                error_msg = f'movie: {params.movie_id} cannot be fetched or missing'
                self.logger.warning(error_msg)
        except Exception as e:
            error_msg = f'exception: {e} on movie: {params.movie_id}'
            self.logger.error(error_msg)
        finally:
            self.dispatch_dict.pop(params.movie_id)
        return result, error_msg

    def handle_predictions(self, params: ExpertParam):
        """getting the specific frames

        Args:
            params (ExpertParam): _description_
        """
        prediction_result = None
        error = None

        scene_element = params.scene_element if params.scene_element else 0
        movie = self.movie_db.get_movie(params.movie_id)
        if scene_element > len(movie['scene_elements']):
            return None, f'scene_element: {scene_element} is bigger than movie scene elements'

        frame_numbers = movie['mdfs'][scene_element]
        # now creating those frames from movie
        cur_frame_no = 0
        scene_places = []
        frames = self.divide_movie_into_frames(frame_numbers)
        for frame_file in frames:
            img = Image.open(frame_file)
            frame_model = self.model.forward(img)
            if frame_model:
                scene_places.append({frame_numbers[cur_frame_no]: frame_model})
            cur_frame_no += 1

        return scene_places, error

    def transform_predictions(self, predictions, params: ExpertParam):
        results = list()
        labels = dict()
        for frame in predictions:
            # get fist item (and only item)
            dict_pairs = frame.items()
            pairs_iterator = iter(dict_pairs)
            first_pair = next(pairs_iterator)
            cur_frame = first_pair[0]
            io = first_pair[1]['io']
            self.add_frame_to_label(labels, io, str(cur_frame))
            categories = first_pair[1]['cat']
            for category in categories:
                for key, value in category.items():
                    self.add_frame_to_label(labels, value, str(cur_frame), key)
            if 'attr' in first_pair[1]:
                attributes = first_pair[1]['attr']
                for attribute in attributes:
                    self.add_frame_to_label(labels, attribute, str(cur_frame))
        for label_key, label_value in labels.items():
            tr = TokenRecord(params.movie_id,
                    params.scene_element, params.scene_element, self.get_name(),
                    None, label_key, label_value, 0)
            results.append(tr)

        return results

    def add_frame_to_label(self, labels, label, cur_frame, probability = None):
        if label not in labels:
            labels[label] = {}
        label_data = labels[label]
        if cur_frame not in label_data:
            label_frame_data = {}
            label_data[cur_frame] = label_frame_data
        label_frame_data = label_data[cur_frame]
        if probability:
            label_frame_data['probability'] = probability

my_expert = PlacesExpert()
expert_app = ExpertApp(expert=my_expert)
app = expert_app.get_app()
expert_app.run()
