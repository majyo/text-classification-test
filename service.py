import os
import sys
import argparse
from typing import Optional
from typing import Any
from typing import Dict
from allennlp.commands.train import train_model_from_args
from allennlp.commands.train import train_model_from_file
from allennlp.commands.evaluate import evaluate_from_args
from allennlp.common import logging as common_logging
from allennlp.predictors import Predictor
from allennlp.commands.predict import _predict
from allennlp.commands.predict import _get_predictor
from allennlp.commands.predict import _PredictManager
from allennlp.common.util import import_module_and_submodules

class TextClassificationApp:
    def construct_params_for_predict(self) -> argparse.Namespace:
        allennlp_args = argparse.Namespace()
        allennlp_args.archive_file = "result/nanodata/model.tar.gz"
        allennlp_args.input_file = "data/nanotext/test.jsonl"
        allennlp_args.output_file = "testOut/outx.jsonl"
        allennlp_args.weights_file = None
        allennlp_args.batch_size = 1
        allennlp_args.silent = False
        allennlp_args.cuda_device = 0
        allennlp_args.use_dataset_reader = True
        allennlp_args.dataset_reader_choice = None
        allennlp_args.overrides = None
        allennlp_args.predictor = "sentence_classifier"
        allennlp_args.predictor_args = ""
        allennlp_args.file_friendly_logging = False
        allennlp_args.include_package = ["my_project"]
        return allennlp_args

    def restore_predictor(self, args: argparse.Namespace) -> Predictor:
        common_logging.FILE_FRIENDLY_LOGGING = args.file_friendly_logging

        predictor = _get_predictor(args)

        if args.silent and not args.output_file:
            print("--silent specified without --output-file.")
            print("Exiting early because no output will be created.")
            sys.exit(0)

        return predictor

    def predict(self, args: argparse.Namespace, predictor: Predictor):
        manager = _PredictManager(
            predictor,
            args.input_file,
            args.output_file,
            args.batch_size,
            not args.silent,
            args.use_dataset_reader,
        )
        manager.run()

    def predict_json(self, input_json: Dict[str, Any], predictor:Predictor):
        input_json = input_json if input_json else {"sentence": "A good movie!"}
        output = predictor.predict_json(input_json)
        return output

    def predict_text_instance(self, input_text: Dict[str, Any], predictor: Predictor):
        input_text = input_text if input_text else "{\"sentence\": \"A good movie!\"}"
        dataset_reader = predictor._dataset_reader
        instance = dataset_reader.text_to_instance(input_text)
        output = predictor.predict_instance(instance)
        return output


class TextClassificationService:
    def __init__(self, app: TextClassificationApp):
        import_module_and_submodules("my_project")
        self.app = app
        args = self.app.construct_params_for_predict()
        self.predictor = app.restore_predictor(args)

    def predict(self, text):
        return self.app.predict_json(text, self.predictor)

    def __call__(self, text):
        return self.app.predict_json(text, self.predictor)


if __name__ == "__main__":
    # import_module_and_submodules("my_project")
    app = TextClassificationApp()
    service = TextClassificationService(app)
    # args = app.construct_params_for_predict()
    # predictor = app.restore_predictor(args)
    text = {"sentence": "TEM images revealed well dispersed NPs of the following dimensions: AuNRs of 40x11 and 70x15nm and AuNSs of 40 and 70nm external diameter (see Experimental section for details)"}
    result = service.predict(text)
    # app.predict(args, predictor)
    print(result)
    # restore_and_predict()
