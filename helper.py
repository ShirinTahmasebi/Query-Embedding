import torch
import pandas as pd
from transformers import BertForMaskedLM, AutoModelForMaskedLM

from constants import Constants
from helper_enum import BaseModel
from torch_dataset import TorchDataset


def load_from_disk_or_call_func(path: str, function, *args):
    import pickle
    try:
        loaded_object = pickle.load(open(path, 'rb'))
        if loaded_object:
            return loaded_object
    except Exception:
        print("File ", path, " did not exist. Start creating the file ...")

    calculated_object = function(*args)
    pickle.dump(calculated_object, open(path, 'wb'))
    return calculated_object

def tensor_torch_to_tf(tensor_torch):
    import tensorflow as tf
    return tf.convert_to_tensor(tensor_torch.numpy())

def initialize_sql_base_tokenizer(df: pd.DataFrame):
    import tensorflow_datasets as tfds
    import ast
    list_of_tokens = df['tokens'].map(lambda x: ast.literal_eval(
        x.replace(" ''", " '\\'")
            .replace("'',", "\\'',")
            .replace("'']", "\\'']")
    ))

    import itertools
    all_df_tokens = list(itertools.chain(*list_of_tokens))

    sql_tokenizer = tfds.deprecated.text.SubwordTextEncoder(set(all_df_tokens))
    return sql_tokenizer


def sql_based_tokenizer(input_col: pd.Series, sql_tokenizer):
    input_list = input_col.tolist()

    tokenized = {
        'input_ids': torch.tensor([]),
        'token_type_ids': torch.tensor([]),
        'attention_mask': torch.tensor([])
    }

    for query in input_list:
        input_id_tokenized = sql_tokenizer.encode(query)
        padding_len = 512 - len(input_id_tokenized)

        input_id = torch.nn.functional.pad(
            input=torch.tensor(input_id_tokenized), pad=(0, padding_len), mode='constant', value=0
        )

        mask = torch.ones(len(input_id_tokenized), dtype=torch.int)
        mask = torch.nn.functional.pad(input=mask, pad=(0, padding_len), mode='constant', value=0)
        token_type = torch.zeros(512, dtype=torch.int)

        tokenized['input_ids'] = torch.stack([*tokenized['input_ids'], input_id])
        tokenized['attention_mask'] = torch.stack([*tokenized['attention_mask'], mask])
        tokenized['token_type_ids'] = torch.stack([*tokenized['token_type_ids'], token_type])

    return sql_tokenizer, tokenized


def create_torch_dataset(tokenized, data_processor):
    train_set = data_processor.prepare(tokenized)

    dataset = TorchDataset(train_set)

    split_threshold = round(Constants.PARAMETERS_SPLIT_TRAIN_EVAL * len(dataset))

    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset,
        [split_threshold, len(dataset) - split_threshold],
        generator=torch.Generator().manual_seed(42)
    )

    return train_dataset, eval_dataset


def fetch_model(model_enum: str):
    model = None
    if model_enum == BaseModel.BERT:
        model = BertForMaskedLM.from_pretrained(Constants.BERT_MODEL_NAME_BASE_UNCASED)
    elif model_enum == BaseModel.CODE_BERT:
        model = AutoModelForMaskedLM.from_pretrained(Constants.BERT_MODEL_NAME_CODE_BERT)
    if not model:
        raise RuntimeError('Model initialization was not successful.')

    return model


def training_loop(model, train_dataset):
    # Initialize optimizer
    from transformers import AdamW
    optimizer = AdamW(model.parameters(), lr=5e-5)

    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Constants.PARAMETERS_BATCH_SIZE, shuffle=True)

    # Select a device and move the model over it
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Activate training mode
    model.train()

    torch.cuda.empty_cache()

    from tqdm import tqdm  # For the progress bar

    for epoch in range(Constants.PARAMETERS_EPOCHS):
        # Setup loop with TQDM and dataloader
        loop = tqdm(data_loader, leave=True)
        for batch in loop:
            # Initialize calculated gradients (from prev step)
            optimizer.zero_grad()

            # Pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Process
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            # Extract loss
            loss = outputs.loss

            # Calculate loss for every parameter that needs grad update
            loss.backward()

            # Update parameters
            optimizer.step()

            # Print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
