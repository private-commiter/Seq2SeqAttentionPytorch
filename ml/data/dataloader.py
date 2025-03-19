import numpy as np
from dataclasses import dataclass

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from ml.data.dataloader_utils import TextPreprocessor


@dataclass
class Config:
    data_dir: str
    batch_size: int
    max_length: int
    sos_token: int
    eos_token: int
    device: torch.device


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


# def tensorFromSentence(lang, sentence, eos_token, device):
#     indexes = indexesFromSentence(lang, sentence)
#     indexes.append(eos_token)
#     return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


# def tensorsFromPair(pair):
#     input_tensor = tensorFromSentence(input_lang, pair[0])
#     target_tensor = tensorFromSentence(output_lang, pair[1])
#     return (input_tensor, target_tensor)


def get_dataloader(config: Config):
    text_proc = TextPreprocessor(
        config.data_dir, config.max_length, config.sos_token, config.eos_token)
    input_lang, output_lang, pairs \
        = text_proc.prepareData('eng', 'fra', True)

    n = len(pairs)
    input_ids = np.zeros((n, config.max_length), dtype=np.int32)
    target_ids = np.zeros((n, config.max_length), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(config.eos_token)
        tgt_ids.append(config.eos_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(
        torch.LongTensor(input_ids).to(config.device),
        torch.LongTensor(target_ids).to(config.device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.batch_size)
    return input_lang, output_lang, train_dataloader
