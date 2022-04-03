import abc
import torch

from helpers.helper_enum import FineTuningTasks


class DataPreparation(metaclass=abc.ABCMeta):

    @staticmethod
    def init_processor(fine_tuning_task):
        if fine_tuning_task == FineTuningTasks.MLM:
            return DataPreparationMLM()
        if fine_tuning_task == FineTuningTasks.SIAMESE:
            return DataPreparationSiamese()
        if fine_tuning_task == FineTuningTasks.TRIPLET:
            return DataPreparationTriplet()

    @abc.abstractmethod
    def prepare(self, tokenized_dataset):
        return


class DataPreparationMLM(DataPreparation):

    def prepare(self, tokenized_dataset):
        tokenized_dataset['labels'] = tokenized_dataset['input_ids'].detach().clone()

        rand = torch.rand(tokenized_dataset['input_ids'].shape)
        mask_arr = (rand < 0.15) * (tokenized_dataset['input_ids'] != 101) * (tokenized_dataset['input_ids'] != 102) * (
                tokenized_dataset['input_ids'] != 0)

        selection = []

        for i in range(tokenized_dataset['input_ids'].shape[0]):
            selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())

        for i in range(tokenized_dataset['input_ids'].shape[0]):
            tokenized_dataset['input_ids'][i, selection[i]] = 103

        return tokenized_dataset


class DataPreparationSiamese(DataPreparation):

    def prepare(self, tokenized_dataset):
        pass


class DataPreparationTriplet(DataPreparation):

    def prepare(self, tokenized_dataset):
        pass
