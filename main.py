import argparse
import datetime
import os
import pickle
import time
import torch
import tqdm

import numpy as np

from data import *
from evaluate import *
from model import *

###################################################################################################################################################

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='MIND-small', help='MIND-large')
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=16)   # 16 (MIND-small), 32 (MIND-large)
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty') 
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_epoch', type=list, default=[4], help='the epoch which the learning rate decay')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--Glove_dim', type=int, default=300)
parser.add_argument('--category_dim', type=int, default=100)
parser.add_argument('--subcategory_dim', type=int, default=100)
parser.add_argument('--popularity_dim', type=int, default=100)
parser.add_argument('--recency_dim', type=int, default=100)
parser.add_argument('--n_popularity_level', type=int, default=8)
parser.add_argument('--n_recency_group', type=int, default=1+3)   # 0: padding
parser.add_argument('--num_head_n', type=int, default=16)
parser.add_argument('--num_head_u', type=int, default=24)
parser.add_argument('--num_head_c', type=int, default=24)
parser.add_argument('--head_dim', type=int, default=25)
parser.add_argument('--num_extracted_interests', type=int, default=8)
parser.add_argument('--M', type=int, default=7, help='the scaling of the unique_category_counts to compute the num_selected_interests')
parser.add_argument('--cold_start_users', type=int, default=0, help='the number of clicked news for a cold-start user')
parser.add_argument('--cold_start_users_weight', type=float, default=0.3, help='the weight of a_u for cold-start users')
parser.add_argument('--cold_start_users_threshold', type=int, default=0, help='the number of clicked news (threshold) for a cold-start user')
parser.add_argument('--cold_start_news', type=int, default=0, help='the number of clicks for a cold-start news')
parser.add_argument('--interest_adaptive', type=int, default=1)
parser.add_argument('--candidate_aware', type=int, default=1)
parser.add_argument('--popularity_mode', type=int, default=0, help='0: Model, 1: w/o Clicks, 2: w/o Recency, 3: w/o Content')
parser.add_argument('--lambda_c', type=float, default=0.0125)
parser.add_argument('--aug', type=bool, default=True)
parser.add_argument('--test_only', type=bool, default=False)
parser.add_argument('--save_path', default='model_save')
parser.add_argument('--prediction_path', default='prediction.txt')
parser.add_argument('--ilad_k', type=int, default=10)
parser.add_argument('--cr_k', type=int, default=10)
opt = parser.parse_args()
#print(opt)

if opt.save_path is not None:
    save_path = opt.save_path + '/' + opt.dataset
    save_dir = save_path + '/' + 'aug=' + str(opt.aug) + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    print('save dir: ', save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

###################################################################################################################################################

def count_cold_users(data):
    user_clicks = [len(history) for history in data.histories]
    zero_click_users = sum([1 for c in user_clicks if c == opt.cold_start_users_threshold])

    return zero_click_users, len(user_clicks)

###################################################################################################################################################

def ILAD(topk_candidates):
    K = topk_candidates.shape[0]
    if K <= 1:
        return 0.0
    
    topk_candidates_norm = topk_candidates / (np.linalg.norm(topk_candidates, axis=1, keepdims=True) + 1e-12)
    cos_similarity = topk_candidates_norm @ topk_candidates_norm.T
    ilad = 1 - (1 / (K*(K-1))) * np.sum(np.triu(cos_similarity, k=1)) * 2
    
    return float(ilad)

###################################################################################################################################################

def CR(model, data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, device, K):
    model.eval()
    slices = data.generate_batch(opt.batch_size)

    seen_crs = []
    unseen_crs = []

    with torch.no_grad():
        for index in tqdm.tqdm(slices, desc=f'CR@{K}'):
            scores, labels, lengths, _, _, _ = forward(
                model, index, data,
                news_title_text, news_abstract_text,
                news2category, news2subcategory,
                news_popularity_label, news_recency_label,
                news_click_num, device
            )

            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            lengths = lengths.cpu().numpy()
            candidate_news = [data.candidates[i] for i in index]
            history_padding, _, _ = data.get_padding(index)

            for i in range(len(index)):
                length = int(lengths[i])
                if length < K:
                    continue

                score = scores[i][:length]
                sorted_idx = np.argsort(-score)[:K]
                topk_candidates = np.array(candidate_news[i][:length])[sorted_idx]
                topk_labels = labels[i][:length][sorted_idx]

                topk_categories = news2category[topk_candidates]
                history_categories = set(news2category[history_padding[i]].tolist()) - {0}

                seen_clicked_new_category = 0
                unseen_clicked_new_category = 0

                candidate_clicked_num = 0
                for category, label in zip(topk_categories, topk_labels):
                    if label == 1 and category != 0:
                        candidate_clicked_num += 1
                        if category not in history_categories:
                            unseen_clicked_new_category += 1
                        else:
                            seen_clicked_new_category += 1

                if candidate_clicked_num != 0:
                    seen_cr = seen_clicked_new_category / candidate_clicked_num
                    seen_crs.append(seen_cr)
                    unseen_cr = unseen_clicked_new_category / candidate_clicked_num
                    unseen_crs.append(unseen_cr)
                else:
                    seen_cr = 0
                    seen_crs.append(seen_cr)
                    unseen_cr = 0
                    unseen_crs.append(unseen_cr)                   

    return np.mean(seen_crs), np.mean(unseen_crs)

###################################################################################################################################################

def main():
    init_seed(2023)

    if opt.dataset == 'MIND-small':
        n_category = 19 + 1       # 18 + 1
        n_subcategory = 271 + 1   # 270 + 1
        max_history_news_click_num = 4802
        cold_start_users_evaluate = 1
        opt.batch_size = 16
    elif opt.dataset == 'MIND-large':
        n_category = 19 + 1       # 18 + 1
        n_subcategory = 293 + 1   # 286 + 1
        max_history_news_click_num = 69736
        cold_start_users_evaluate = 0
        opt.batch_size = 32

    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    valid_data = pickle.load(open('datasets/' + opt.dataset + '/validation.txt', 'rb'))
    test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))
    news2category = pickle.load(open('datasets/' + opt.dataset + '/news2category.txt', 'rb'))
    news2subcategory = pickle.load(open('datasets/' + opt.dataset + '/news2subcategory.txt', 'rb'))
    word_embeddings = pickle.load(open('datasets/' + opt.dataset + '/word_embeddings.txt', 'rb'))
    news_title_text = pickle.load(open('datasets/' + opt.dataset + '/news_title_text.txt', 'rb'))
    news_abstract_text = pickle.load(open('datasets/' + opt.dataset + '/news_abstract_text.txt', 'rb'))
    news_popularity_label = pickle.load(open('datasets/' + opt.dataset + f'/popularity_level.txt', 'rb'))
    news_recency_label = pickle.load(open('datasets/' + opt.dataset + f'/recency_group.txt', 'rb'))
    news_click_num = pickle.load(open('datasets/' + opt.dataset + f'/history_click_num.txt', 'rb'))
    news_click_num_0 = pickle.load(open('datasets/' + opt.dataset + f'/news_click_num_0.txt', 'rb'))

    train_data = Data(train_data, shuffle=True)
    valid_data = Data(valid_data, shuffle=False)
    test_data = Data(test_data, shuffle=False)

    cold_train, total_train = count_cold_users(train_data)
    cold_valid, total_valid = count_cold_users(valid_data)
    cold_test, total_test = count_cold_users(test_data)
    print("------------------------------------------------------------------")
    print(f"[Train]      Cold-start users: {cold_train} / {total_train} ({cold_train/total_train*100:.2f}%)")
    print(f"[Validation] Cold-start users: {cold_valid} / {total_valid} ({cold_valid/total_valid*100:.2f}%)")
    print(f"[Test]       Cold-start users: {cold_test} / {total_test} ({cold_test/total_test*100:.2f}%)")
    print("------------------------------------------------------------------")

    model = MODEL(device, opt, word_embeddings, n_category, n_subcategory, max_history_news_click_num)
    model.initialize()
    model = model.to(device)
    #print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_dc_epoch, gamma=opt.lr_dc)

    start = time.time()

    if opt.test_only:
        if opt.dataset == 'MIND-small':
            #model.load_state_dict(torch.load(os.path.join(f'model.pt')))
            model.load_state_dict(torch.load(os.path.join(r'C:\Users\88697\Downloads\IMEP\model_save\MIND-small\IMEP\model.pt')))
            (all, cold) = valid_test(model, test_data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, cold_start_users_evaluate)
            test_auc, test_mrr, test_ndcg5, test_ndcg10 = all
            cs_auc, cs_mrr, cs_ndcg5, cs_ndcg10 = cold
            print('[All Users]        AUC: %.4f   MRR: %.4f   NDCG@5: %.4f   NDCG@10: %.4f' % (test_auc, test_mrr, test_ndcg5, test_ndcg10))
            print('[Cold-start Users] AUC: %.4f   MRR: %.4f   NDCG@5: %.4f   NDCG@10: %.4f' % (cs_auc, cs_mrr, cs_ndcg5, cs_ndcg10))

            model.eval()
            slices = test_data.generate_batch(opt.batch_size)
            
            '''
            ### Cold-start News Recommendations ###
            output_all = os.path.join("recommendation_results.txt")
            output_cold = os.path.join("recommendation_results_cold.txt")

            all_news = set(np.where(news_click_num_0 >= opt.cold_start_news)[0])
            all_news_click_count = {nid: 0 for nid in all_news}
            all_news_ranks = []

            cold_start_news = set(np.where(news_click_num_0 == opt.cold_start_news)[0])
            cold_start_news_click_count = {nid: 0 for nid in cold_start_news}
            cold_start_news_ranks = []
            
            with open(output_all, 'w', encoding='utf-8') as f_all, \
                open(output_cold, 'w', encoding='utf-8') as f_cold:

                f_all.write("Impression_ID\tRank\tNews_ID\tScore\tLabel\n")
                f_cold.write("Impression_ID\tRank\tNews_ID\tScore\tLabel\n")

                with torch.no_grad():
                    for index in tqdm.tqdm(slices, desc='Saving Predictions'):
                        scores, labels, lengths, _, _, _ = forward(model, index, test_data, news_title_text,
                                                                                           news_abstract_text, news2category, news2subcategory,
                                                                                           news_popularity_label, news_recency_label, news_click_num, device)
                        scores = scores.detach().cpu().numpy()
                        labels = labels.detach().cpu().numpy()
                        lengths = lengths.detach().cpu().numpy()
                        candidate_news = [test_data.candidates[i] for i in index]

                        history_padding, _, _ = test_data.get_padding(index)
                        click_counts = np.sum(history_padding != 0, axis=1)

                        for i in range(len(index)):
                            impression_id = index[i] + 1
                            length = int(lengths[i])
                            score = scores[i][:length]
                            label = labels[i][:length]
                            news_ids = candidate_news[i][:length]

                            sorted_idx = np.argsort(-score)
                            for rank, idx_ in enumerate(sorted_idx):
                                line = f"{impression_id}\t{rank+1}\t{news_ids[idx_]}\t{score[idx_]:.4f}\t{label[idx_]}\n"
                                f_all.write(line)
                                if click_counts[i] == opt.cold_start_news:
                                    f_cold.write(line)

                                nid = news_ids[idx_]
                                if label[idx_] == 1:
                                    all_news_click_count[nid] += 1
                                    all_news_ranks.append(rank + 1)
                                    if nid in cold_start_news:
                                        cold_start_news_click_count[nid] += 1
                                        cold_start_news_ranks.append(rank + 1)
                
                clicked_news = sum([v > 0 for v in all_news_click_count.values()])
                all_avg_rank = np.mean(all_news_ranks) if all_news_ranks else float('nan')

                clicked_cold_news = sum([v > 0 for v in cold_start_news_click_count.values()])
                cold_avg_rank = np.mean(cold_start_news_ranks) if cold_start_news_ranks else float('nan')

                print('--------------------------------------------------')
                print(f"num of news: {len(all_news)}")
                print(f"num of clicked news: {clicked_news}")
                print(f"average rank of clicked news: {all_avg_rank:.2f}")
                print('--------------------------------------------------')
                print(f"num of cold-start news: {len(cold_start_news)}")
                print(f"num of clicked cold-start news: {clicked_cold_news}")
                print(f"average rank of clicked cold-start news: {cold_avg_rank:.2f}")
                print('--------------------------------------------------')
            '''
            
            '''
            ### ILAD (Intra List Average Distance) ###
            for k in range(1, opt.ilad_k+1):
                all_ilads = []

                with torch.no_grad():
                    for index in tqdm.tqdm(slices, desc='ILAD Evaluation'):
                        scores, _, lengths, candidate_representations, _, _ = forward(
                            model, index, test_data, news_title_text, news_abstract_text,
                            news2category, news2subcategory, news_popularity_label,
                            news_recency_label, news_click_num, device,
                        )

                        scores = scores.cpu().numpy()
                        candidate_representations = candidate_representations.cpu().numpy()
                        lengths = lengths.cpu().numpy()

                        for i in range(len(index)):
                            length = int(lengths[i])
                            if length < k:
                                continue
                            score = scores[i][:length]
                            sorted_idx = np.argsort(-score)[:k]
                            topk_candidates = candidate_representations[i][sorted_idx]
                            ilad = ILAD(topk_candidates)
                            all_ilads.append(ilad)

                avg_ilad = np.mean(all_ilads) if all_ilads else 0.0
                print(f"ILAD@{k}: {avg_ilad:.4f}")
            '''
            
            '''
            ### CR (Category Ratio) ###
            for k in range(1, opt.cr_k+1):
                seen_cr, unseen_cr = CR(
                    model, test_data, news_title_text, news_abstract_text,
                    news2category, news2subcategory, news_popularity_label,
                    news_recency_label, news_click_num, device, K=k
                )
                print(f"[Seen   & Clicked]   CR@{k}: {seen_cr:.4f}")
                print(f"[Unseen & Clicked]   CR@{k}: {unseen_cr:.4f}")
            '''

        elif opt.dataset == 'MIND-large':
            model.load_state_dict(torch.load(os.path.join(f'model.pt')))
            prediction(model, test_data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, opt.prediction_path)

        return

    for epoch in range(opt.epoch):

        print('-------------------------------------------')
        print(f'epoch: {epoch}')
        
        # Training
        train(model, train_data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, optimizer)

        # Validation
        (all, cold) = valid_test(model, valid_data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, cold_start_users_evaluate)
        val_auc, val_mrr, val_ndcg5, val_ndcg10 = all
        cs_auc, cs_mrr, cs_ndcg5, cs_ndcg10 = cold
        print('[All Users]        AUC: %.4f   MRR: %.4f   NDCG@5: %.4f   NDCG@10: %.4f' % (val_auc, val_mrr, val_ndcg5, val_ndcg10))
        print('[Cold-start Users] AUC: %.4f   MRR: %.4f   NDCG@5: %.4f   NDCG@10: %.4f' % (cs_auc, cs_mrr, cs_ndcg5, cs_ndcg10))
        scheduler.step()

    if opt.save_path is not None:
        save_model = os.path.join(save_dir, f'model.pt')
        torch.save(model.state_dict(), save_model)
        print('Model saved.')

     # Test
    print('--------------------------------------------------')
    model.load_state_dict(torch.load(os.path.join(save_dir, f'model.pt')))
    if opt.dataset == 'MIND-small':
        (all, cold) = valid_test(model, test_data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, cold_start_users_evaluate)
        test_auc, test_mrr, test_ndcg5, test_ndcg10 = all
        cs_auc, cs_mrr, cs_ndcg5, cs_ndcg10 = cold
        print('[All Users]        AUC: %.4f   MRR: %.4f   NDCG@5: %.4f   NDCG@10: %.4f' % (test_auc, test_mrr, test_ndcg5, test_ndcg10))
        print('[Cold-start Users] AUC: %.4f   MRR: %.4f   NDCG@5: %.4f   NDCG@10: %.4f' % (cs_auc, cs_mrr, cs_ndcg5, cs_ndcg10))
    elif opt.dataset == 'MIND-large':
        prediction(model, test_data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, opt.prediction_path)
        
    end = time.time()
    print("Run time: %f s" % (end - start))

###################################################################################################################################################

def train(model, train_data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, optimizer):
    model.train()
    train_slices = train_data.generate_batch(opt.batch_size)

    for index in tqdm.tqdm(train_slices, desc='Training'):
        optimizer.zero_grad()
        scores, labels, _, _, category_preds, category_labels = forward(model, index, train_data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, device)
        labels = torch.where(labels != 0)[1]
        
        ### News Recommendation Task ###
        loss_r = model.loss_function(scores, labels)

        #'''
        ### News Category Classification Task ###
        category_preds_flat = category_preds.view(-1, category_preds.size(-1))   # [batch_size*(np_ratio+1), n_category]
        category_labels_flat = category_labels.view(-1)                          # [batch_size*(np_ratio+1)]
        mask = category_labels_flat != 0                                         # [batch_size*(np_ratio+1)]
        if mask.sum() > 0:
            loss_c = F.cross_entropy(category_preds_flat[mask], category_labels_flat[mask])
        else:
            loss_c = torch.tensor(0.0, device=device)
        #'''

        ###############################################################
        #'''
        ### Gradient Surgery ###
        # 1. Backward: News Recommendation Task (Main Task)
        loss_r.backward(retain_graph=True)
        g_main = []
        params = []
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)
                if param.grad is not None:
                    g_main.append(param.grad.detach().clone().flatten())
                else:
                    g_main.append(torch.zeros_like(param.data).flatten())
        g_main = torch.cat(g_main)

        # 2. Backward: News Category Classification Task (Auxiliary Task)
        loss_c.backward(retain_graph=True)
        g_aux = []
        for param in params:
            if param.grad is not None:
                g_aux.append(param.grad.detach().clone().flatten())
            else:
                g_aux.append(torch.zeros_like(param.data).flatten())
        g_aux = torch.cat(g_aux)

        # 3. Gradient Surgery
        dot = torch.dot(g_main, g_aux)
        if dot < 0:
            g_aux_norm = g_aux.norm() ** 2 + 1e-8
            proj = (dot / g_aux_norm) * g_aux
            g_main_proj = g_main - proj
        else:
            g_main_proj = g_main

        # 4. Update
        idx = 0
        for param in params:
            numel = param.numel()
            param.grad = g_main_proj[idx:idx+numel].view_as(param).clone()
            idx += numel
        
        ###############################################################
        #'''

        loss = loss_r + opt.lambda_c * loss_c
        loss.backward()
        optimizer.step()

        #break

    print(f"Loss_r: {loss_r.item():.6f}, Loss_c: {loss_c.item():.6f}")

###################################################################################################################################################

def valid_test(model, data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, cold_start_users_evaluate):
    model.eval()
    slices = data.generate_batch(opt.batch_size)
    
    click_counts_all = []
    total_scores, total_labels, total_lengths = [], [], []
    total_category_preds, total_category_labels = [], []

    with torch.no_grad():
        for index in tqdm.tqdm(slices, desc='Evaluating'):
            history_padding, _, _ = data.get_padding(index) 
            history_mask_np = np.where(history_padding == 0, 0, 1)
            click_counts_batch = history_mask_np.sum(axis=1).tolist()
            click_counts_all.extend(click_counts_batch)
            
            scores, labels, lengths, _, category_preds, category_labels = forward(model, index, data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, device)
            total_scores.extend(scores.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())
            total_lengths.extend(lengths.cpu().numpy())

            total_category_preds.append(category_preds.view(-1, category_preds.size(-1)))
            total_category_labels.append(category_labels.view(-1))

            #break
        
    all_preds = torch.cat(total_category_preds, dim=0).argmax(dim=-1).view(-1)
    all_labels = torch.cat(total_category_labels, dim=0).view(-1)
    mask = all_labels != 0
    acc = (all_preds[mask] == all_labels[mask]).float().mean().item()
    print(f"[Topic Classification Accuracy]: {acc:.4f}")

    ### All Users ###
    preds, labels = [], []
    for score, label, length in zip(total_scores, total_labels, total_lengths):
        score = np.asarray(score[:int(length)])
        label = np.asarray(label[:int(length)])
        rank = np.argsort(-score)   # 依分數由大到小排序
        labels.append(label[rank])
        preds.append(list(range(1, len(rank)+1)))
    auc, mrr, ndcg5, ndcg10 = evaluate(preds, labels)

    ### Cold-start Users ###
    if cold_start_users_evaluate == 1:
        preds_cs, labels_cs = [], []
        for score, label, length, cc in zip(total_scores, total_labels, total_lengths, click_counts_all):
            if cc == opt.cold_start_users_threshold:
                score = np.asarray(score[:int(length)])
                label = np.asarray(label[:int(length)])
                rank = np.argsort(-score)
                labels_cs.append(label[rank])
                preds_cs.append(list(range(1, len(rank)+1)))

        if len(preds_cs) > 0:
            auc_cs, mrr_cs, ndcg5_cs, ndcg10_cs = evaluate(preds_cs, labels_cs)
        else:
            auc_cs = mrr_cs = ndcg5_cs = ndcg10_cs = float('nan')
    else:
        auc_cs = mrr_cs = ndcg5_cs = ndcg10_cs = float('nan')

    all = (auc*100, mrr*100, ndcg5*100, ndcg10*100)
    cold = (auc_cs*100, mrr_cs*100, ndcg5_cs*100, ndcg10_cs*100)
   
    return all, cold

###################################################################################################################################################

def prediction(model, data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, output_path):
    model.eval()
    slices = data.generate_batch(opt.batch_size)

    with open(output_path, 'w', encoding='utf-8') as f:
        for index in tqdm.tqdm(slices, desc='Predicting'):
            scores, _, lengths, _, _, _ = forward(model, index, data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, device)
            scores = scores.detach().cpu().numpy()
            lengths = lengths.detach().cpu().numpy()

            batch_size = len(index)
            for i in range(batch_size):
                impression_id = index[i] + 1
                length = int(lengths[i])
                sorted_indices = np.argsort(-scores[i][:length])
                
                ranking = [0] * length
                for rank, idx_ in enumerate(sorted_indices):
                    ranking[idx_] = rank + 1
                
                f.write(f"{impression_id} [{','.join(map(str, ranking))}]\n")    

            #break

###################################################################################################################################################

def forward(model, index, data, news_title_text, news_abstract_text, news2category, news2subcategory, news_popularity_label, news_recency_label, news_click_num, device):
    history_padding, candidate_padding, label_padding = data.get_padding(index)
    history_mask = np.where(history_padding == 0, np.zeros_like(history_padding), np.ones_like(history_padding))

    history_category = news2category[history_padding]
    candidate_category = news2category[candidate_padding]

    history_subcategory = news2subcategory[history_padding]
    candidate_subcategory = news2subcategory[candidate_padding]

    history_title_word = news_title_text[history_padding]
    history_title_word_mask = np.where(history_title_word == 0, np.zeros_like(history_title_word), np.ones_like(history_title_word))
    candidate_title_word = news_title_text[candidate_padding]
    candidate_title_word_mask = np.where(candidate_title_word == 0, np.zeros_like(candidate_title_word), np.ones_like(candidate_title_word))

    history_abstract_word = news_abstract_text[history_padding]
    history_abstract_word_mask = np.where(history_abstract_word == 0, np.zeros_like(history_abstract_word), np.ones_like(history_abstract_word))
    candidate_abstract_word = news_abstract_text[candidate_padding]
    candidate_abstract_word_mask = np.where(candidate_abstract_word == 0, np.zeros_like(candidate_abstract_word), np.ones_like(candidate_abstract_word))

    candidate_popularity_label = news_popularity_label[candidate_padding]
    candidate_recency_label = news_recency_label[candidate_padding]
    history_click_num = news_click_num[history_padding]

    ###############################################################

    candidate_padding = torch.LongTensor(candidate_padding).to(device)
    label_padding = torch.LongTensor(label_padding).to(device)
    history_mask = torch.FloatTensor(history_mask).to(device)

    history_category = torch.LongTensor(history_category).to(device)
    candidate_category = torch.LongTensor(candidate_category).to(device)

    history_subcategory = torch.LongTensor(history_subcategory).to(device)
    candidate_subcategory = torch.LongTensor(candidate_subcategory).to(device)

    history_title_word = torch.LongTensor(history_title_word).to(device)
    history_title_word_mask = torch.LongTensor(history_title_word_mask).to(device)
    candidate_title_word = torch.LongTensor(candidate_title_word).to(device)
    candidate_title_word_mask = torch.LongTensor(candidate_title_word_mask).to(device)

    history_abstract_word = torch.LongTensor(history_abstract_word).to(device)
    history_abstract_word_mask = torch.LongTensor(history_abstract_word_mask).to(device)
    candidate_abstract_word = torch.LongTensor(candidate_abstract_word).to(device)
    candidate_abstract_word_mask = torch.LongTensor(candidate_abstract_word_mask).to(device)

    candidate_popularity_label = torch.LongTensor(candidate_popularity_label).to(device)
    candidate_recency_label = torch.LongTensor(candidate_recency_label).to(device)
    history_click_num = torch.LongTensor(history_click_num).to(device)

    ###############################################################

    scores, candidate_lengths, candidate_representations, category_preds = model(
        history_mask, candidate_padding,
        history_title_word, history_title_word_mask,
        candidate_title_word, candidate_title_word_mask,
        history_abstract_word, history_abstract_word_mask,
        candidate_abstract_word, candidate_abstract_word_mask,
        history_category, candidate_category,
        history_subcategory, candidate_subcategory,
        candidate_popularity_label, candidate_recency_label, history_click_num
    )
    return scores, label_padding, candidate_lengths, candidate_representations, category_preds, candidate_category

###################################################################################################################################################

if __name__ == '__main__':
    main()