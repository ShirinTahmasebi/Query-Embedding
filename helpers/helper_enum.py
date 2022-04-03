from enum import Enum


class FineTuningTasks(Enum):
    MLM = 1
    SIAMESE = 2
    TRIPLET = 3


class Tokenizer(Enum):
    BERT_BASED = 1
    SQL_BASED = 2


class BaseModel(Enum):
    BERT = 1
    CU_BERT = 2
    CODE_BERT = 3


class ModelTaskItem:
    def __init__(self, model: BaseModel, task: FineTuningTasks, prefix_model_name: str):
        self._model = model
        self._task = task
        self._prefix_model_name = prefix_model_name

    def get_model(self):
        return self._model

    def get_task(self):
        return self._task

    def get_prefix_model_name(self):
        return self._prefix_model_name


class AllModelTaskItems(Enum):
    BERT_MLM = ModelTaskItem(BaseModel.BERT, FineTuningTasks.MLM, 'bert_mlm')
    BERT_SIAMESE = ModelTaskItem(BaseModel.BERT, FineTuningTasks.SIAMESE, 'bert_siamese')
    BERT_TRIPLET = ModelTaskItem(BaseModel.BERT, FineTuningTasks.TRIPLET, 'bert_triplet')

    CUBERT_MLM = ModelTaskItem(BaseModel.CU_BERT, FineTuningTasks.MLM, 'cubert_mlm')
    CUBERT_SIAMESE = ModelTaskItem(BaseModel.CU_BERT, FineTuningTasks.SIAMESE, 'cubert_siamese')
    CUBERT_TRIPLET = ModelTaskItem(BaseModel.CU_BERT, FineTuningTasks.TRIPLET, 'cubert_triplet')

    CODEBERT_MLM = ModelTaskItem(BaseModel.CODE_BERT, FineTuningTasks.MLM, 'codebert_mlm')
    CODEBERT_SIAMESE = ModelTaskItem(BaseModel.CODE_BERT, FineTuningTasks.SIAMESE, 'codebert_siamese')
    CODEBERT_TRIPLET = ModelTaskItem(BaseModel.CODE_BERT, FineTuningTasks.TRIPLET, 'codebert_triplet')
