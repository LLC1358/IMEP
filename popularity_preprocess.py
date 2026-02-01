import argparse
import json
import pickle

import numpy as np

from collections import Counter
from datetime import datetime

###################################################################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='MIND-small', help='MIND-large')
parser.add_argument('--semantic_similar_news', type=int, default=7)
parser.add_argument('--cold_start_news', type=int, default=0)
parser.add_argument('--popularity_level', type=int, default=8)
parser.add_argument('--recency_group', type=int, default=3)
parser.add_argument('--cold_start_news_aware', type=int, default=1)
parser.add_argument('--log_transformation', type=int, default=0)
opt = parser.parse_args()

news_index = pickle.load(open('datasets/' + opt.dataset + '/news_index.txt', 'rb'))
news_title = pickle.load(open('datasets/' + opt.dataset + '/news_title_text.txt', 'rb'))
word_embeddings = pickle.load(open('datasets/' + opt.dataset + '/word_embeddings.txt', 'rb'))

if opt.dataset == 'MIND-small':
    train_behaviors = 'MIND/MINDsmall/MINDsmall_train/behaviors.tsv'
    dev_behaviors = 'MIND/MINDsmall/MINDsmall_dev/behaviors.tsv'
elif opt.dataset == 'MIND-large':
    train_behaviors = 'MIND/MINDlarge/MINDlarge_train/behaviors.tsv'
    validation_behaviors = 'MIND/MINDlarge/MINDlarge_validation/behaviors.tsv'
    test_behaviors = 'MIND/MINDlarge/MINDlarge_test/behaviors.tsv'

###################################################################################################################################################

### Semantic Similar News ###
save_path = 'datasets/' + opt.dataset + f"/semantic_similar_news.pkl"

# DIGAT: similarity-10.json
similarity_path = 'datasets/' + opt.dataset + f'/similarity-10.json'
with open(similarity_path, "r", encoding="utf-8") as f:
    similarity_file = json.load(f)

num_news = max(news_index.values(), default=0)
similar_news = np.zeros((num_news + 1, opt.semantic_similar_news), dtype=np.int32)

for news_id, similar_news_list in similarity_file.items():
    news_idx = news_index.get(news_id, 0)
    similar_news_idx_list = [news_index.get(item[0], 0) for item in similar_news_list]

    while len(similar_news_idx_list) < opt.semantic_similar_news:
        similar_news_idx_list.append(0)

    similar_news[news_idx] = np.array(similar_news_idx_list[:opt.semantic_similar_news])

with open(save_path, "wb") as f:
    pickle.dump(similar_news, f)

#########################################################################

### Clicks-based Popularity ###
# 1. 計算各個新聞的點擊次數（train: histories & impressions）: news_click_num_0.txt
save_path = 'datasets/' + opt.dataset + f"/news_click_num_0.txt"

processed_users = set()
news_click_count = Counter()

def Click_Num(file):
    with open(file, "r", encoding="utf-8") as behaviors:
        for behavior in behaviors:
            data = behavior.strip().split("\t")
            if len(data) < 5:
                continue

            user_id = data[1]
            histories = data[3].split()
            impressions = data[4].split()

            # histories
            '''
            if user_id not in processed_users:
                processed_users.add(user_id)
                for news_id in histories:
                    news_click_count[news_id] += 1
            '''
            news_click_count.update(histories)

            # impressions
            for news in impressions:
                news_id, label = news.rsplit("-", 1)
                if label == "1":
                    news_click_count[news_id] += 1

if opt.dataset == 'MIND-small':
    Click_Num(train_behaviors)
    #Click_Num(dev_behaviors)
elif opt.dataset == 'MIND-large':
    Click_Num(train_behaviors)
    Click_Num(validation_behaviors)
    #Click_Num(test_behaviors)

# news_click_num[1] = 4
news_click_num = np.zeros((max(news_index.values(), default=0) + 1), dtype=np.int32)
for news_id, count in news_click_count.items():
    news_idx = news_index[news_id]
    news_click_num[news_idx] = count

cold_start_news_num = int((news_click_num[1:] <= opt.cold_start_news).sum())
print(f"Number of cold-start news: {cold_start_news_num}")   # MIND-small: 25373, MIND-large: 33679

with open(save_path, "wb") as f:
    pickle.dump(news_click_num, f)

save_path = 'datasets/' + opt.dataset + f"/clicks.txt"
with open(save_path, 'w', encoding='utf-8') as f:
    f.write("News_ID\tClicks\n")
    for news_idx in range(len(news_click_num)):
        line = f"{news_idx}\t{news_click_num[news_idx]}\n"
        f.write(line)

###############################################################

'''
# 2. 計算各個新聞的流行度: news_clickNum_1.txt
save_path = 'datasets/' + opt.dataset + '/news_click_num_1.txt'

news_click_num_log = np.log2(news_click_num + 1)

with open(save_path, "wb") as f:
    pickle.dump(news_click_num_log, f)
'''

###############################################################

# 3. 計算冷啟動新聞（點擊次數為 opt.cold_start_news）的流行度: news_click_num_2.txt
save_path = 'datasets/' + opt.dataset + '/news_click_num_2.txt'

news_click_num_filled = news_click_num.copy()
if opt.cold_start_news_aware == 1:
    print('Cold-start News-aware!')
    for news_idx in range(len(news_click_num_filled)):
        if news_click_num[news_idx] <= opt.cold_start_news:
            similar_news_idx = similar_news[news_idx]
            similar_news_clicks = news_click_num[similar_news_idx]
            mean_clicks = np.mean(similar_news_clicks[similar_news_clicks > 0]) if np.any(similar_news_clicks > 0) else 0
            news_click_num_filled[news_idx] = mean_clicks

if opt.log_transformation == 1:
    print('Log transformation!')
    news_click_num_filled = np.log2(news_click_num_filled + 1)

with open(save_path, "wb") as f:
    pickle.dump(news_click_num_filled, f)

save_path = 'datasets/' + opt.dataset + f"/processed_clicks.txt"
with open(save_path, 'w', encoding='utf-8') as f:
    f.write("News_ID\tClicks\n")
    for news_idx in range(len(news_click_num_filled)):
        line = f"{news_idx}\t{news_click_num_filled[news_idx]}\n"
        f.write(line)

###############################################################

# 4. 分組並存成流行度等級: popularity_level.txt
save_path = 'datasets/' + opt.dataset + '/popularity_level.txt'

# sorted_indices = [    0 52719 52720 ... 14217 33900 11466]
sorted_indices = np.argsort(news_click_num_filled)   # news_id 依點擊次數由小到大排序
num_news = len(news_click_num_filled)
group_size = num_news // opt.popularity_level

popularity_levels = np.zeros_like(news_click_num_filled, dtype=np.int32)

for group_id in range(opt.popularity_level):
    start_idx = group_id * group_size
    end_idx = (group_id + 1) * group_size if group_id < opt.popularity_level-1 else num_news
    popularity_levels[sorted_indices[start_idx:end_idx]] = group_id

with open(save_path, "wb") as f:
    pickle.dump(popularity_levels, f)

###################################################################################################################################################

### Recency-based Popularity ###
# 1. 紀錄每篇新聞的最早出現時間: news_publish_time.txt
save_path = save_path = 'datasets/' + opt.dataset + '/news_publish_time.txt'

# history: 1st week ~ 4th week (10/12 ~ 11/8)
# training: 5th week 1st day ~ 6th day (11/9 ~ 11/14)
# validation: 5th week 7th day (11/15)
# test: 6th week (11/16 ~ 11/22)
start_time_str = "11/08/2019 11:59:59 PM"
start_time = datetime.strptime(start_time_str, "%m/%d/%Y %I:%M:%S %p")

news_publish_time = {news_id: None for news_id in news_index.keys()}

def Publish_Time(file):
    with open(file, "r", encoding="utf-8") as behaviors:
        for behavior in behaviors:
            data = behavior.strip().split("\t")
            if len(data) < 5:
                continue
            
            time_str = data[2]
            impressions = data[4].split()
            time = datetime.strptime(time_str, "%m/%d/%Y %I:%M:%S %p")

            for news in impressions:
                if '-' not in news:
                    news_id = news
                else:
                    news_id, label = news.rsplit("-", 1)
                if news_id in news_publish_time:
                    if news_publish_time[news_id] is None or time < news_publish_time[news_id]:
                        news_publish_time[news_id] = time

if opt.dataset == 'MIND-small':
    Publish_Time(train_behaviors)
    Publish_Time(dev_behaviors)
elif opt.dataset == 'MIND-large':
    Publish_Time(train_behaviors)
    Publish_Time(validation_behaviors)
    Publish_Time(test_behaviors)

for news_id in news_publish_time:
    if news_publish_time[news_id] is None:
        news_publish_time[news_id] = start_time

# news_recency_time[1] = 1573228799.0
num_news = max(news_index.values())
news_recency_time = [start_time] * (num_news + 1)
for news_id, idx in news_index.items():
    news_recency_time[idx] = news_publish_time[news_id]
news_recency_time = np.array([dt.timestamp() for dt in news_recency_time], dtype=np.float64)

with open(save_path, "wb") as f:
    pickle.dump(news_recency_time, f)

###############################################################

# 2. 分組並存成新近度組別: recency_group.txt
save_path = 'datasets/' + opt.dataset + '/recency_group.txt'

is_special = (news_recency_time == start_time)
special_indices = np.where(is_special)[0]
non_special_indices = np.where(~is_special)[0]

non_special_times = news_recency_time[non_special_indices]
sorted_indices = np.argsort(non_special_times)
group_size = len(non_special_times) // opt.recency_group

news_recency_group = np.full_like(news_recency_time, fill_value=-1, dtype=np.int32)
news_recency_group[special_indices] = 0
for group_id in range(1, opt.recency_group + 1):
    start_idx = (group_id - 1) * group_size
    end_idx = group_id * group_size if group_id < opt.recency_group else len(non_special_times)
    indices = non_special_indices[sorted_indices[start_idx:end_idx]]
    news_recency_group[indices] = group_id

with open(save_path, "wb") as f:
    pickle.dump(news_recency_group, f)

#########################################################################

### User Attention ###
# 計算每位使用者的歷史點擊新聞的點擊次數: history_click_num.txt
save_path = 'datasets/' + opt.dataset + '/history_click_num.txt'

processed_users = set()
history_news_click_count = Counter()

def History_Click_Num(file):
    with open(file, "r", encoding="utf-8") as behaviors:
        for behavior in behaviors:
            data = behavior.strip().split("\t")
            if len(data) < 5:
                continue

            user_id = data[1]
            clicked_news = data[3].split()

            if user_id not in processed_users:
                processed_users.add(user_id)
                history_news_click_count.update(clicked_news)

if opt.dataset == 'MIND-small':
    History_Click_Num(train_behaviors)
    #History_Click_Num(dev_behaviors)
elif opt.dataset == 'MIND-large':
    History_Click_Num(train_behaviors)
    History_Click_Num(validation_behaviors)
    #History_Click_Num(test_behaviors)

# history_news_click_num[1] = 4
history_news_click_num = np.zeros((max(news_index.values(), default=0) + 1), dtype=np.int32)
for news_id, count in history_news_click_count.items():
    if news_id in news_index:
        news_idx = news_index[news_id]
        history_news_click_num[news_idx] = count

'''
### MIND-small ###
print(history_news_click_num.max())        # max: 4802 (train), 9778 (train+dev)
print(history_news_click_num.mean())       # mean: 14.194852772114839 (train), 31.233311362835114 (train+dev)
print(np.median(history_news_click_num))   # median: 1 (train), 1 (train+dev)

### MIND-large ###
print(history_news_click_num.max())        # max: 69736 (train+validation), 77831 (train+validation+test)
print(history_news_click_num.mean())       # mean: 105.40663445313699 (train+validation), 114.56949685534592 (train+validation+test)
print(np.median(history_news_click_num))   # median: 1 (train+validation), 1 (train+validation+test)
'''

with open(save_path, "wb") as f:
    pickle.dump(history_news_click_num, f)