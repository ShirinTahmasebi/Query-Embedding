from enum import Enum

from constants import Constants


class FineTuningTasks(Enum):
    MLM = 1
    SIAMESE = 2
    TRIPLET = 3


class Tokenizer(Enum):
    BERT_BASED = 1
    SQL_BASED = 2


class BaseModel(Enum):
    BERT = 1
    CUBERT = 2
    CODE_BERT = 3


class ModelTaskItem:
    def __init__(
            self,
            model: BaseModel,
            task: FineTuningTasks,
            weights_path: str,
            prediction_results_path: dict
    ):
        self._model = model
        self._model_name = self._set_model_name(model)
        self._task = task
        self._weights_path = weights_path
        self._prediction_results_path = prediction_results_path

    def _set_model_name(self, model: BaseModel):
        if model == BaseModel.BERT:
            return Constants.BERT_MODEL_NAME_BASE_UNCASED
        if model == BaseModel.CODE_BERT:
            return Constants.BERT_MODEL_NAME_CODE_BERT

    def get_model(self):
        return self._model

    def get_model_name(self):
        return self._model_name

    def get_task(self):
        return self._task

    def get_weights_path(self):
        return self._weights_path

    def get_prediction_results_path(self):
        return self._prediction_results_path


class AllModelTaskItems(Enum):
    # BERT
    BERT_MLM = ModelTaskItem(
        BaseModel.BERT,
        FineTuningTasks.MLM,
        Constants.PATH_FINE_TUNED_MODEL_WEIGHTS_MLM_BERT,
        Constants.MLM_BERT_RESULT_PATH_DIC
    )

    BERT_SIAMESE = ModelTaskItem(
        BaseModel.BERT,
        FineTuningTasks.SIAMESE,
        Constants.PATH_FINE_TUNED_MODEL_WEIGHTS_SIAMESE_BERT,
        Constants.SIAMESE_BERT_RESULT_PATH_DIC
    )

    BERT_TRIPLET = ModelTaskItem(
        BaseModel.BERT,
        FineTuningTasks.TRIPLET,
        Constants.PATH_FINE_TUNED_MODEL_WEIGHTS_TRIPLET_BERT,
        Constants.TRIPLET_BERT_RESULT_PATH_DIC
    )

    # CUBERT
    CUBERT_MLM = ModelTaskItem(
        BaseModel.CUBERT,
        FineTuningTasks.MLM,
        Constants.PATH_FINE_TUNED_MODEL_WEIGHTS_MLM_CUBERT,
        Constants.MLM_CUBERT_RESULT_PATH_DIC
    )

    CUBERT_SIAMESE = ModelTaskItem(
        BaseModel.CUBERT,
        FineTuningTasks.SIAMESE,
        Constants.PATH_FINE_TUNED_MODEL_WEIGHTS_SIAMESE_CUBERT,
        Constants.SIAMESE_CUBERT_RESULT_PATH_DIC
    )

    CUBERT_TRIPLET = ModelTaskItem(
        BaseModel.CUBERT,
        FineTuningTasks.TRIPLET,
        Constants.PATH_FINE_TUNED_MODEL_WEIGHTS_TRIPLET_CUBERT,
        Constants.TRIPLET_CUBERT_RESULT_PATH_DIC
    )

    # CodeBERT
    CODEBERT_MLM = ModelTaskItem(
        BaseModel.CODE_BERT,
        FineTuningTasks.MLM,
        Constants.PATH_FINE_TUNED_MODEL_WEIGHTS_MLM_CODE_BERT,
        Constants.MLM_CODE_BERT_RESULT_PATH_DIC
    )

    CODEBERT_SIAMESE = ModelTaskItem(
        BaseModel.CODE_BERT,
        FineTuningTasks.SIAMESE,
        Constants.PATH_FINE_TUNED_MODEL_WEIGHTS_SIAMESE_CODE_BERT,
        Constants.SIAMESE_CODE_BERT_RESULT_PATH_DIC
    )

    CODEBERT_TRIPLET = ModelTaskItem(
        BaseModel.CODE_BERT,
        FineTuningTasks.TRIPLET,
        Constants.PATH_FINE_TUNED_MODEL_WEIGHTS_TRIPLET_CODE_BERT,
        Constants.TRIPLET_CODE_BERT_RESULT_PATH_DIC
    )

    CODEBERT_TRIPLET_TEMP = ModelTaskItem(
        BaseModel.CODE_BERT,
        FineTuningTasks.TRIPLET,
        Constants.PATH_FINE_TUNED_MODEL_WEIGHTS_TEMP_CODE_BERT,
        Constants.TRIPLET_CODE_BERT_RESULT_PATH_DIC_TEMP
    )

    BERT_TRIPLET_TEMP = ModelTaskItem(
        BaseModel.BERT,
        FineTuningTasks.TRIPLET,
        Constants.PATH_FINE_TUNED_MODEL_WEIGHTS_TEMP_BERT,
        Constants.TRIPLET_BERT_RESULT_PATH_DIC_TEMP
    )
