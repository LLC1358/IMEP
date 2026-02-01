import argparse
import json
import os
import pickle
import random

import numpy as np

from nltk.tokenize import word_tokenize

###################################################################################################################################################

np.random.seed(2023)
random.seed(2023)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='MIND-small', help='MIND-large')
parser.add_argument('--max_title_length', type=int, default=30)
parser.add_argument('--max_abstract_length', type=int, default=30)
parser.add_argument('--max_news_length', type=int, default=50)
parser.add_argument('--split_ratio', type=int, default=0.95)
parser.add_argument('--np_ratio', type=int, default=4)
opt = parser.parse_args()

if opt.dataset == 'MIND-small':
    dataset_name = 'MIND/MINDsmall/MINDsmall'
    output_dataset = 'datasets\MIND-small'
    with open(dataset_name + '_train/behaviors.tsv', encoding='utf-8') as f:
        train_behaviors = f.readlines()
    with open(dataset_name + '_dev/behaviors.tsv', encoding='utf-8') as f:
        dev_behaviors = f.readlines()
    with open(dataset_name + '_train/news.tsv', encoding='utf-8') as f:
        train_newsMetadata = f.readlines()
    with open(dataset_name + '_dev/news.tsv', encoding='utf-8') as f:
        dev_newsMetadata = f.readlines()
        newsMetadata = train_newsMetadata + dev_newsMetadata
elif opt.dataset == 'MIND-large':
    dataset_name = 'MIND/MINDlarge/MINDlarge'
    output_dataset = 'datasets\MIND-large'
    with open(dataset_name + '_train/behaviors.tsv', encoding='utf-8') as f:
        train_behaviors = f.readlines()
    with open(dataset_name + '_validation/behaviors.tsv', encoding='utf-8') as f:
        validation_behaviors = f.readlines()
    with open(dataset_name + '_test/behaviors.tsv', encoding='utf-8') as f:
        test_behaviors = f.readlines()
    with open(dataset_name + '_train/news.tsv', encoding='utf-8') as f:
        train_newsMetadata = f.readlines()
    with open(dataset_name + '_validation/news.tsv', encoding='utf-8') as f:
        validation_newsMetadata = f.readlines()
    with open(dataset_name + '_test/news.tsv', encoding='utf-8') as f:
        test_newsMetadata = f.readlines()
        newsMetadata = train_newsMetadata + validation_newsMetadata + test_newsMetadata

###################################################################################################################################################

### news['N55528'] = ['lifestyle', 'lifestyleroyals', 'the brands queen elizabeth, prince charles, and prince philip swear by', 'shop the notebooks, jackets, and more that the royals can't live without.'] ###
### category['lifestyle'] = 1
### subcategory['lifestyleroyals'] = 1
news = {}
category = {'NULL': 0}
subcategory = {'NULL': 0}
for i in newsMetadata:
    line = i.strip('\n').split('\t')
    if line[0] not in news:
        news[line[0]] = [line[1], line[2], word_tokenize(line[3].lower()), word_tokenize(line[4].lower())]
        '''
        print(line[0])   # News ID: N55528
        print(line[1])   # Category: lifestyle
        print(line[2])   # Subcategory: lifestyleroyals
        print(line[3])   # Title: The Brands Queen Elizabeth, Prince Charles, and Prince Philip Swear By
        print(line[4])   # Abstract: Shop the notebooks, jackets, and more that the royals can't live without.
        '''
    if line[1] not in category:
        category[line[1]] = len(category)         # category 從 1 開始編號
    if line[2] not in subcategory:
        subcategory[line[2]] = len(subcategory)   # subcategory 從 1 開始編號
'''
### MIND-small ###
print('Number of news:', len(news))                     # 65238
print('Number of categories:', len(category)-1)         # 18
print('Number of subcategories:', len(subcategory)-1)   # 270

### MIND-large ###
print('Number of news:', len(news))                     # 130379
print('Number of categories:', len(category)-1)         # 18
print('Number of subcategories:', len(subcategory)-1)   # 293
'''

###############################################################

### news_index['N55528'] = 1
news_index = {'NULL': 0}
for i in news:
    news_index[i] = len(news_index)   # news_index 從 1 開始編號

###############################################################

### word_dict['the'] = [1, 148255]   # [index, frequency]
word_dict = {'PADDING': [0, 999999]}
for i in news:
    # title
    for j in news[i][2]:
        if j in word_dict:
            word_dict[j][1] += 1
        else:
            word_dict[j] = [len(word_dict), 1]
    # abstract
    for j in news[i][3]:
        if j in word_dict:
            word_dict[j][1] += 1
        else:
            word_dict[j] = [len(word_dict), 1]
'''
### MIND-small ###
print('Number of words:', len(word_dict))   # 80141

### MIND-large ###
print('Number of words:', len(word_dict))   # 115767
'''

###################################################################################################################################################

### word_vector['the'] = [0.27204, -0.06203, -0.1884, 0.023225, -0.018158, 0.0067192, -0.13877, 0.17708, 0.17709, 2.5882, -0.35179, -0.17312, 0.43285, -0.10708, 0.15006, -0.19982, -0.19093, 1.1871, -0.16207, -0.23538, 0.003664, -0.19156, -0.085662, 0.039199, -0.066449, -0.04209, -0.19122, 0.011679, -0.37138, 0.21886, 0.0011423, 0.4319, -0.14205, 0.38059, 0.30654, 0.020167, -0.18316, -0.0065186, -0.0080549, -0.12063, 0.027507, 0.29839, -0.22896, -0.22882, 0.14671, -0.076301, -0.1268, -0.0066651, -0.052795, 0.14258, 0.1561, 0.05551, -0.16149, 0.09629, -0.076533, -0.049971, -0.010195, -0.047641, -0.16679, -0.2394, 0.0050141, -0.049175, 0.013338, 0.41923, -0.10104, 0.015111, -0.077706, -0.13471, 0.119, 0.10802, 0.21061, -0.051904, 0.18527, 0.17856, 0.041293, -0.014385, -0.082567, -0.035483, -0.076173, -0.045367, 0.089281, 0.33672, -0.22099, -0.0067275, 0.23983, -0.23147, -0.88592, 0.091297, -0.012123, 0.013233, -0.25799, -0.02972, 0.016754, 0.01369, 0.32377, 0.039546, 0.042114, -0.088243, 0.30318, 0.087747, 0.16346, -0.40485, -0.043845, -0.040697, 0.20936, -0.77795, 0.2997, 0.2334, 0.14891, -0.39037, -0.053086, 0.062922, 0.065663, -0.13906, 0.094193, 0.10344, -0.2797, 0.28905, -0.32161, 0.020687, 0.063254, -0.23257, -0.4352, -0.017049, -0.32744, -0.047064, -0.075149, -0.18788, -0.015017, 0.029342, -0.3527, -0.044278, -0.13507, -0.11644, -0.1043, 0.1392, 0.0039199, 0.37603, 0.067217, -0.37992, -1.1241, -0.057357, -0.16826, 0.03941, 0.2604, -0.023866, 0.17963, 0.13553, 0.2139, 0.052633, -0.25033, -0.11307, 0.22234, 0.066597, -0.11161, 0.062438, -0.27972, 0.19878, -0.36262, -1.0006e-05, -0.17262, 0.29166, -0.15723, 0.054295, 0.06101, -0.39165, 0.2766, 0.057816, 0.39709, 0.025229, 0.24672, -0.08905, 0.15683, -0.2096, -0.22196, 0.052394, -0.01136, 0.050417, -0.14023, -0.042825, -0.031931, -0.21336, -0.20402, -0.23272, 0.07449, 0.088202, -0.11063, -0.33526, -0.014028, -0.29429, -0.086911, -0.1321, -0.43616, 0.20513, 0.0079362, 0.48505, 0.064237, 0.14261, -0.43711, 0.12783, -0.13111, 0.24673, -0.27496, 0.15896, 0.43314, 0.090286, 0.24662, 0.066463, -0.20099, 0.1101, 0.03644, 0.17359, -0.15689, -0.086328, -0.17316, 0.36975, -0.40317, -0.064814, -0.034166, -0.013773, 0.062854, -0.17183, -0.12366, -0.034663, -0.22793, -0.23172, 0.239, 0.27473, 0.15332, 0.10661, -0.060982, -0.024805, -0.13478, 0.17932, -0.37374, -0.02893, -0.11142, -0.08389, -0.055932, 0.068039, -0.10783, 0.1465, 0.094617, -0.084554, 0.067429, -0.3291, 0.034082, -0.16747, -0.25997, -0.22917, 0.020159, -0.02758, 0.16136, -0.18538, 0.037665, 0.57603, 0.20684, 0.27941, 0.16477, -0.018769, 0.12062, 0.069648, 0.059022, -0.23154, 0.24095, -0.3471, 0.04854, -0.056502, 0.41566, -0.43194, 0.4823, -0.051759, -0.27285, -0.25893, 0.16555, -0.1831, -0.06734, 0.42457, 0.010346, 0.14237, 0.25939, 0.17123, -0.13821, -0.066846, 0.015981, -0.30193, 0.043579, -0.043102, 0.35025, -0.19681, -0.4281, 0.16899, 0.22511, -0.28557, -0.1028, -0.018168, 0.11407, 0.13015, -0.18317, 0.1323]
word_vector = {}
with open('glove.840B.300d.txt', 'rb') as f:
    while True:
        line = f.readline()   # line: 一個詞的 300 維向量
        if len(line) == 0:
            break
        line = line.split()
        word = line[0].decode()
        if len(word) != 0:
            vector = [float(x) for x in line[1:]]
            if word in word_dict:
                word_vector[word] = vector

word_embeddings = [0] * len(word_dict)
in_dictionary_embeddings = []
for i in word_vector.keys():
    word_embeddings[word_dict[i][0]] = np.array(word_vector[i], dtype='float32')
    in_dictionary_embeddings.append(word_embeddings[word_dict[i][0]])
in_dictionary_embeddings = np.array(in_dictionary_embeddings, dtype='float32')
mean = np.mean(in_dictionary_embeddings, axis=0)
variance = np.cov(in_dictionary_embeddings.T)
norm = np.random.multivariate_normal(mean, variance, 1)   # 產生一個 300 維的向量，符合前面算出的高斯分布（mean, variance），用於初始化那些在 Glove 中找不到詞時的預設分布

for i in range(len(word_embeddings)):
    if type(word_embeddings[i]) == int:
        word_embeddings[i] = np.reshape(norm, 300)   # 初始化在 Glove 中找不到的詞
word_embeddings[0] = np.zeros(300, dtype='float32')
word_embeddings = np.array(word_embeddings, dtype='float32')
'''
### MIND-small ###
print('Shape of word embeddings:', word_embeddings.shape)   # (80141, 300)

### MIND-large ###
print('Shape of word embeddings:', word_embeddings.shape)   # (115767, 300)
'''

###############################################################

### news_title[1] = [ 1  2  3  4  5  6  7  5  8  6  9 10 11  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
news_title = [[0] * opt.max_title_length]
for i in news:
    title = []
    for word in news[i][2]:
        if word in word_dict:
            title.append(word_dict[word][0])
        else:
            print('Error!')
    title = title[:opt.max_title_length]
    news_title.append(title + [0] * (opt.max_title_length - len(title)))
news_title = np.array(news_title, dtype='int32')

###############################################################

### news_abstract[1] = [12  1 13  5 14  5  8 15 16  1 17 18 19 20 21 22  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
news_abstract = [[0] * opt.max_abstract_length]
for i in news:
    abstract = []
    for word in news[i][3]:
        if word in word_dict:
            abstract.append(word_dict[word][0])
        else:
            print('Error!')
    abstract = abstract[:opt.max_abstract_length]
    news_abstract.append(abstract + [0] * (opt.max_abstract_length - len(abstract)))
news_abstract = np.array(news_abstract, dtype='int32')

###############################################################

### news_category[1] = 1
news_category = [0]
for i in news:
    news_category.append(category[news[i][0]])
news_category = np.array(news_category, dtype='int32')

###############################################################

### news_subcategory[1] = 1
news_subcategory = [0]
for i in news:
    news_subcategory.append(subcategory[news[i][1]])
news_subcategory = np.array(news_subcategory, dtype='int32')

###################################################################################################################################################

def negative_sample(neg_news, np_ratio):
    if np_ratio > len(neg_news):
        return random.sample(neg_news * (np_ratio // len(neg_news) + 1), np_ratio)
    else:
        return random.sample(neg_news, np_ratio)

if opt.dataset == 'MIND-small':
    train = train_behaviors[:int(opt.split_ratio * len(train_behaviors))]
    validation = train_behaviors[int(opt.split_ratio * len(train_behaviors)):]
    test = dev_behaviors
elif opt.dataset == 'MIND-large':
    train = train_behaviors
    validation = validation_behaviors
    test = test_behaviors

###############################################################

### impression = ['1', 'U13740', '11/11/2019 9:05:58 AM', 'N55189 N42782 N34694 N45794 N18445 N63302 N10414 N19347 N31801', 'N55689-1 N35729-0']
train_history = []
train_candidate = []
train_candidate_label = []
for impression in train:
    impression = impression.replace('\n', '').split('\t')
    history = [news_index[x] for x in impression[3].split()][-50:]
    pos_news = [news_index[x.split('-')[0]] for x in impression[4].split() if x.split('-')[1] == '1']
    neg_news = [news_index[x.split('-')[0]] for x in impression[4].split() if x.split('-')[1] == '0']
    '''
    if history == []:
        continue
    '''
    for p_news in pos_news:
        candidate = negative_sample(neg_news, opt.np_ratio)
        candidate.append(p_news)
        candidate_label = [0] * opt.np_ratio + [1]
        candidate_order = list(range(opt.np_ratio + 1))
        random.shuffle(candidate_order)
        candidate_shuffle = []
        candidate_label_shuffle = []
        for i in candidate_order:
            candidate_shuffle.append(candidate[i])
            candidate_label_shuffle.append(candidate_label[i])
        train_history.append(history + [0] * (opt.max_news_length - len(history)))
        train_candidate.append(candidate_shuffle)
        train_candidate_label.append(candidate_label_shuffle)

###############################################################

validation_history = []
validation_candidate = []
validation_candidate_label = []
for impression in validation:
    impression = impression.replace('\n', '').split('\t')
    history = [news_index[x] for x in impression[3].split()][-opt.max_news_length:]
    candidate = [news_index[x.split('-')[0]] for x in impression[4].split()]
    candidate_label = [int(x.split('-')[1]) for x in impression[4].split()]
    '''
    if history == []:
        continue
    '''
    validation_history.append(history + [0] * (opt.max_news_length - len(history)))
    validation_candidate.append(candidate)
    validation_candidate_label.append(candidate_label)

###############################################################

test_history = []
test_candidate = []
test_candidate_label = []
test_index = []
for impression in test:
    if opt.dataset == 'MIND-small':
        impression = impression.replace('\n', '').split('\t')
        history = [news_index[x] for x in impression[3].split()][-opt.max_news_length:]
        candidate = [news_index[x.split('-')[0]] for x in impression[4].split()]
        candidate_label = [int(x.split('-')[1]) for x in impression[4].split()]
        test_history.append(history + [0] * (opt.max_news_length - len(history)))
        test_candidate.append(candidate)
        test_candidate_label.append(candidate_label)
    elif opt.dataset == 'MIND-large':
        impression = impression.replace('\n', '').split('\t')
        history = [news_index[x] for x in impression[3].split()][-opt.max_news_length:]
        candidate = [news_index[x] for x in impression[4].split()]
        test_history.append(history + [0] * (opt.max_news_length - len(history)))
        test_candidate.append(candidate)
        test_candidate_label.append([0] * len(candidate))

###############################################################

train = (train_candidate, train_candidate_label, train_history)
validation = (validation_candidate, validation_candidate_label, validation_history)
test = (test_candidate, test_candidate_label, test_history)
'''
### MIND-small ###
print('train:', len(train_history))             # 224536
print('validation:', len(validation_history))   # 7849
print('test:', len(test_history))               # 73152

### MIND-large ###
print('train:', len(train_history))             # 3383656
print('validation:', len(validation_history))   # 376471
print('test:', len(test_history))               # 2370727
'''

###################################################################################################################################################

if not os.path.exists(output_dataset):
    os.makedirs(output_dataset)

pickle.dump(train, open(output_dataset+'/train.txt', 'wb'))
pickle.dump(validation, open(output_dataset+'/validation.txt', 'wb'))
pickle.dump(test, open(output_dataset+'/test.txt', 'wb'))
pickle.dump(news_index, open(output_dataset+'/news_index.txt', 'wb'))
pickle.dump(word_embeddings, open(output_dataset+'/word_embeddings.txt', 'wb'))
pickle.dump(news_title, open(output_dataset+'/news_title_text.txt', 'wb'))
pickle.dump(news_abstract, open(output_dataset+'/news_abstract_text.txt', 'wb'))
pickle.dump(news_category, open(output_dataset+'/news2category.txt', 'wb'))
pickle.dump(news_subcategory, open(output_dataset+'/news2subcategory.txt', 'wb'))
pickle.dump(category, open(output_dataset+'/category.txt', 'wb'))
pickle.dump(subcategory, open(output_dataset+'/subcategory.txt', 'wb'))

with open(output_dataset + '/news_index.json', 'w', encoding='utf-8') as f:
    json.dump(news_index, f, ensure_ascii=False, indent=2)