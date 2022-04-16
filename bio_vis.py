# TODO 查看句子slot是不是按照BIO 聚类
# %%
import pickle
from matplotlib import cm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from fastNLP import DataSet, Instance, Vocabulary, DataSetIter, SequentialSampler
from typing import List, Tuple,  Dict
import torch
from utils.data_utils import read_all_format_all_data_type
from sklearn.decomposition import PCA
from utils.bio2bmes import bio2bmes
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import pandas as pd


def create_dataset(data_tuple: Tuple[List[List[str]], List[str], List[List[str]]],  encoder2decoder=False) -> DataSet:
    dataset = DataSet()
    raw_words_list = data_tuple[0]
    raw_intent_list = data_tuple[1]
    raw_slot_list = data_tuple[2]
    for words, intent, slots in zip(raw_words_list, raw_intent_list, raw_slot_list):
        dataset.append(
            Instance(raw_words=" ".join(words),
                     words=words,
                     intent=intent,
                     slots=slots,
                     seq_len=len(words),
                     token_intent=[intent] * len(words),
                     bio_tags=[s[0] for s in bio2bmes(slots)]
                     ))
    return dataset


def get_single_instance_dataset(sentence: str, slots: List, intent: str) -> Instance:
    dataset = DataSet()
    words = sentence.split()
    dataset.append(
        Instance(raw_words=" ".join(words),
                 words=words,
                 intent=intent,
                 slots=slots,
                 seq_len=len(words),
                 token_intent=[intent] * len(words)
                 ))
    return dataset


def get_slot_embedding_from_instance(use_bio=False, token_level_intent=False, force_bio=False):
    model = torch.load('bio_model.pth')
    train_tuple, valid_tuple, test_tuple = read_all_format_all_data_type(
        'snips')
    test_dataset = create_dataset(test_tuple)
    test_dataset = prepare_test_dataset(test_dataset)
    test_batch = DataSetIter(batch_size=1,
                             dataset=test_dataset, sampler=SequentialSampler())
    x = []
    y_idx = []
    y = []
    bio_idx = []
    bio_label = []
    bio_map = {'O': 0, "B": 1, "I": 2}
    device = torch.device("cuda")
    with torch.no_grad():
        model = model.eval()
        for batch_x, batch_y in tqdm(test_batch, desc="test is Running"):
            batch_x['words_idx'] = batch_x['words_idx'].to(device)
            batch_x['seq_len'] = batch_x['seq_len'].to(device)
            if force_bio:
                batch_y['bio_tags_idx'] = batch_y['bio_tags_idx'].to(device)
                result = model(batch_x['words_idx'], batch_x['seq_len'],
                               token_level_intent=token_level_intent, force_bio=batch_y['bio_tags_idx'])
            else:
                result = model(batch_x['words_idx'], batch_x['seq_len'],
                               token_level_intent=token_level_intent)
            for idx, slot in enumerate(batch_y['slots'][0]):
                x.append(result['pred_slots'][0][idx].cpu().numpy())
                y_idx.append(batch_y['slots_idx'][0][idx].item())
                y.append(slot)
                bio_idx.append(bio_map[slot[0]])
                bio_label.append(slot[0])
    return x, y,  y_idx, bio_label, bio_idx


def prepare_test_dataset(test_dataset):
    word_vocab = Vocabulary.load('word_vocab.txt')
    intent_vocab = Vocabulary.load('intent_vocab.txt')
    slot_vocab = Vocabulary.load('slot_vocab.txt')
    bio_vocab = Vocabulary.load('bio_vocab.txt')
    word_vocab.index_dataset(
        test_dataset, field_name='words', new_field_name='words_idx')
    intent_vocab.index_dataset(
        test_dataset, field_name='intent', new_field_name='intent_idx')
    intent_vocab.index_dataset(
        test_dataset, field_name='token_intent', new_field_name='token_intent_idx')
    slot_vocab.index_dataset(
        test_dataset, field_name='slots', new_field_name='slots_idx')
    bio_vocab.index_dataset(test_dataset, field_name='bio_tags',
                            new_field_name='bio_tags_idx')
    test_dataset.set_input('seq_len')
    test_dataset.set_input('words')
    test_dataset.set_input('words_idx')
    test_dataset.set_target('slots_idx')
    test_dataset.set_target('slots')
    test_dataset.set_target('bio_tags_idx')
    test_dataset.set_target('token_intent_idx')
    return test_dataset


def plot_with_labels(lowDWeights,  y,  y_idx, bio_label, bio_idx, i):
    plt.cla()
    # 降到二维了，分别给x和y
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    # 遍历每个点以及对应标签
    idx = 0
    for x1, x2, s in zip(X, Y, y_idx):
        # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
        c = cm.rainbow(int(255/9 * s))
        plt.text(x1, x2, y[idx], backgroundcolor=c, fontsize=4)
        idx += 1
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    plt.savefig("{}.jpg".format(i) , dpi=500,bbox_inches = 'tight' )
    plt.cla()
    idx = 0
    for x1, x2, s in zip(X, Y, bio_idx):
        # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
        c = cm.rainbow(int(255/9 * s))
        plt.text(x1, x2, bio_label[idx], backgroundcolor=c, fontsize=4)
        idx += 1
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    plt.savefig("bio_{}.jpg".format(i), dpi=500,bbox_inches = 'tight')


x, y,  y_idx, bio_label, bio_idx = get_slot_embedding_from_instance()
tsne = TSNE(n_components=2)  # TSNE降维，降到2
# 只需要显示前500个
plot_only = 500
# 降维后的数据
low_dim_embs = tsne.fit_transform(x)
plot_with_labels(low_dim_embs,   y,  y_idx, bio_label, bio_idx, "bio_model")



# %%
