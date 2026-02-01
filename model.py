import math
import torch

import torch.nn as nn
import torch.nn.functional as F

###################################################################################################################################################

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, h: int, d_model: int, d_k: int, d_v: int):
        super(MultiHeadSelfAttention, self).__init__()
        self.h = h
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.out_dim = self.h * self.d_v
        self.attention_scalar = math.sqrt(float(self.d_k))
        self.W_K = nn.Linear(d_model, self.h*self.d_k, bias=False)
        self.W_Q = nn.Linear(d_model, self.h*self.d_k, bias=True)
        self.W_V = nn.Linear(d_model, self.h*self.d_v, bias=True)

    def initialize(self):
        nn.init.zeros_(self.W_Q.bias)
        nn.init.zeros_(self.W_V.bias)

    def forward(self, Q, K, V, mask):
        batch_size = Q.size(0)
        len_k = K.size(1)
        batch_h_size = batch_size * self.h
        Q = self.W_Q(Q).view([batch_size, -1, self.h, self.d_k])                      # [batch_size, len_q, h, d_k]
        K = self.W_K(K).view([batch_size, -1, self.h, self.d_k])                      # [batch_size, len_k, h, d_k]
        V = self.W_V(V).view([batch_size, -1, self.h, self.d_v])                      # [batch_size, len_k, h, d_v]
        Q = Q.transpose(1, 2).contiguous().view([batch_h_size, -1, self.d_k])         # [batch_size * h, len_q, d_k]
        K = K.transpose(1, 2).contiguous().view([batch_h_size, -1, self.d_k])         # [batch_size * h, len_k, d_k]
        V = V.transpose(1, 2).contiguous().view([batch_h_size, -1, self.d_v])         # [batch_size * h, len_k, d_v]
        A = torch.bmm(Q, K.transpose(1, 2).contiguous()) / self.attention_scalar      # [batch_size * h, len_q, len_k]
        if mask is not None:                                                          # [batch_size, len_q]
            mask = mask.repeat(self.h, len_k).view(batch_h_size, -1, len_k)           # [batch_size * h, len_q, len_k]
            alpha = F.softmax(A.masked_fill(mask == 0, -1e9), dim=2)
        else:
            alpha = F.softmax(A, dim=2)                                               # [batch_size * h, len_q, len_k]
        out = torch.bmm(alpha, V).view([batch_size, self.h, -1, self.d_v])            # [batch_size, h, len_q, d_v]
        out = out.transpose(1, 2).contiguous().view([batch_size, -1, self.out_dim])   # [batch_size, len_q, h * d_v]
        
        return out

###################################################################################################################################################

class Attention(nn.Module):
    def __init__(self, feature_dim: int, attention_dim: int):
        super(Attention, self).__init__()
        self.affine1 = nn.Linear(feature_dim, attention_dim, bias=True)
        self.affine2 = nn.Linear(attention_dim, 1, bias=False)

    def initialize(self):
        nn.init.xavier_uniform_(self.affine1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, feature, mask):
        attention = torch.tanh(self.affine1(feature))                                   # [batch_size, length, attention_dim]
        a = self.affine2(attention).squeeze(dim=2)                                      # [batch_size, length]
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1).unsqueeze(dim=1)   # [batch_size, 1, length]
        else:
            alpha = F.softmax(a, dim=1).unsqueeze(dim=1)                                # [batch_size, 1, length]
        out = torch.bmm(alpha, feature).squeeze(dim=1)                                  # [batch_size, feature_dim]
        
        return out

###################################################################################################################################################

class PolyAttention(nn.Module):
    def __init__(self, news_dim: int, num_extracted_interests: int, context_code_dim: int):
        super(PolyAttention, self).__init__()
        self.linear = nn.Linear(news_dim, context_code_dim, bias=False)
        self.num_extracted_interests = num_extracted_interests
        self.context_codes = nn.Parameter( nn.init.xavier_uniform_(torch.empty(num_extracted_interests, context_code_dim), gain=nn.init.calculate_gain('tanh')))

    def forward(self, history_news_representations: torch.Tensor, attn_mask: torch.Tensor, num_extracted_interests: torch.Tensor, bias: torch.Tensor = None):
        proj = torch.tanh(self.linear(history_news_representations))   # [bs, L, context_code_dim]
        bs, L, d = proj.shape

        context_code_mask = (torch.arange(self.num_extracted_interests, device=history_news_representations.device).unsqueeze(0).expand(bs, -1) < num_extracted_interests.unsqueeze(1))   # [batch_size, num_extracted_interests]
        
        if bias is None:
            weights = torch.matmul(proj, self.context_codes[:self.num_extracted_interests].T)   # [batch_size, max_news_length, num_extracted_interests]
        else:
            bias = bias.mean(dim=2).unsqueeze(dim=2)
            weights = torch.matmul(proj, self.context_codes[:self.num_extracted_interests].T) + bias
        weights = weights.permute(0, 2, 1)                                                      # [batch_size, num_extracted_interests, max_news_length]
        weights = weights.masked_fill(~context_code_mask.unsqueeze(-1), float('-1e9'))
        weights = F.softmax(weights, dim=2)                                                     # [batch_size, num_extracted_interests, max_news_length]
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

        interest_representations = torch.matmul(weights, history_news_representations)          # [batch_size, num_extracted_interests, interest_dim]
        
        return interest_representations

###################################################################################################################################################

class TextEncoder(nn.Module):
    def __init__(self, opt, word_embeddings):
        super(TextEncoder, self).__init__()
        self.word_embeddings = nn.Embedding(num_embeddings=word_embeddings.shape[0], embedding_dim=opt.Glove_dim)
        self.word_embeddings.weight.data.copy_(torch.FloatTensor(word_embeddings))
        self.MHSA = MultiHeadSelfAttention(opt.num_head_n, opt.Glove_dim, opt.head_dim, opt.head_dim)
        self.ATT = Attention(opt.num_head_n * opt.head_dim, opt.num_head_n * opt.head_dim)
        self.dropout = nn.Dropout(p=opt.dropout)

    def initialize(self):
        self.MHSA.initialize()
        self.ATT.initialize()

    def forward(self, text, mask):
        e_t = self.dropout(self.word_embeddings(text))
        h_t = F.relu(self.MHSA(e_t, e_t, e_t, mask=mask))
        text_representation = self.ATT(h_t, mask=mask)

        return text_representation
    
###################################################################################################################################################

class InterestCandidateAttention(nn.Module):
    def __init__(self, M):
        super(InterestCandidateAttention, self).__init__()
        self.M = M

    def forward(self, interest_representations: torch.Tensor, candidate_news_representation: torch.Tensor, unique_category_counts: torch.Tensor, interest_adaptive, candidate_aware):
        bs, K, d = interest_representations.shape                                                                      # [batch_size, num_extracted_interests, interest_dimension]
        attention_weights = torch.sum(interest_representations * candidate_news_representation.unsqueeze(1), dim=-1)   # [batch_size, num_extracted_interests]
        
        if interest_adaptive:
            #dynamic_K = torch.ceil(self.M * unique_category_counts.float()).long().clamp(min=1, max=K)                # [batch_size]
            #dynamic_K = torch.ceil(torch.sqrt(self.M * unique_category_counts.float())).long().clamp(min=1, max=K)    # [batch_size]
            dynamic_K = torch.ceil(torch.log2(self.M * unique_category_counts.float())).long().clamp(min=1, max=K)     # [batch_size]
            topK_mask = torch.zeros_like(attention_weights)                                                            # [batch_size, num_extracted_interests]
            for i in range(bs):
                cur_K = dynamic_K[i]
                topK_idx = attention_weights[i].topk(cur_K.item(), largest=True, sorted=False).indices
                topK_mask[i, topK_idx] = 1

            # interest_adaptive + candidate_aware
            if candidate_aware:
                masked_weights = attention_weights * topK_mask  # [batch_size, num_extracted_interests]
                masked_weights = masked_weights.unsqueeze(-1)   # [batch_size, num_extracted_interests, 1]
                user_representations = torch.sum(masked_weights * interest_representations, dim=1)   # [batch_size, user_dim]

            # interest_adaptive + w/o candidate_aware
            else:
                masked_weights = topK_mask                      # [batch_size, num_extracted_interests]
                masked_weights = masked_weights.unsqueeze(-1)   # [batch_size, num_extracted_interests, 1]
                user_representations = torch.sum(masked_weights * interest_representations, dim=1)   # [batch_size, user_dim]

        else:
            # w/o interest_adaptive + candidate_aware
            if candidate_aware:
                masked_weights = attention_weights.unsqueeze(-1)
                user_representations = torch.sum(masked_weights * interest_representations, dim=1)   # [batch_size, user_dim]

            # w/o interest_adaptive + w/o candidate_aware
            else: 
                user_representations = torch.sum(interest_representations, dim=1) / K   # [batch_size, user_dim]

        return user_representations

###################################################################################################################################################

class UserEncoder(nn.Module):
    def __init__(self, news_dim, interest_dim, num_extracted_interests, M):
        super(UserEncoder, self).__init__()
        self.poly_attention = PolyAttention(news_dim, num_extracted_interests, interest_dim)
        self.interest_candidate_attention = InterestCandidateAttention(M)

    def initialize(self):
        pass

    def forward(self, history_news_representations, history_mask, candidate_news_representations, num_extracted_interests, unique_category_counts, interest_adaptive, candidate_aware):
        interest_representations = self.poly_attention(history_news_representations, history_mask, num_extracted_interests)   # [batch_size, num_extracted_interests, interest_dimension]

        user_representations = []
        bs, N, d = candidate_news_representations.size()                                                                                                                                   # [batch_size, np_ratio+1, news_dimension]
        for i in range(N):
            candidate_news_representation = candidate_news_representations[:, i, :]                                                                                                        # [batch_size, news_dimension]
            user_representation = self.interest_candidate_attention(interest_representations, candidate_news_representation, unique_category_counts, interest_adaptive, candidate_aware)   # [batch_size, user_dimension]
            user_representations.append(user_representation.unsqueeze(1))
        user_representations = torch.cat(user_representations, dim=1)                                                                                                                      # [batch_size, np_ratio+1, news_dimension]
        
        return user_representations

###################################################################################################################################################

class UserAttentionNetwork(nn.Module):
    def __init__(self, d_model):
        super(UserAttentionNetwork, self).__init__()
        #self.dense = nn.Linear(d_model, 1, bias=True)

    #def forward(self, candidate_news_representations, s_i, s_p, a_u):
    def forward(self, s_i, s_p, a_u):
        #w = torch.sigmoid(self.dense(candidate_news_representations)).squeeze(-1)

        #s_r = s_i
        #s_r = s_p
        #s_r = s_i + s_p
        #s_r = (1-w) * s_i + w * s_p
        s_r = s_i + (1+a_u)*s_p
        #s_r = w * s_i + ((1-w)*(1+a_u)) * s_p
        
        return s_r
    
###################################################################################################################################################

class MODEL(nn.Module):
    def __init__(self, device, opt, word_embeddings, n_category, n_subcategory, max_history_news_click_num):
        super(MODEL, self).__init__()
        self.device = device
        self.loss_function = nn.CrossEntropyLoss()

        self.text_encoder = TextEncoder(opt, word_embeddings)
        self.user_encoder = UserEncoder(news_dim=opt.num_head_n*opt.head_dim, interest_dim=opt.num_head_n*opt.head_dim, num_extracted_interests=opt.num_extracted_interests, M=opt.M)

        self.category_dim = opt.category_dim
        self.subcategory_dim = opt.subcategory_dim
        self.news_dim = opt.num_head_n * opt.head_dim
        self.popularity_dim = opt.popularity_dim
        self.recency_dim = opt.recency_dim
        self.num_extracted_interests = opt.num_extracted_interests
        self.max_history_news_click_num = max_history_news_click_num
        self.cold_start_users = opt.cold_start_users
        self.cold_start_users_weight = opt.cold_start_users_weight
        self.interest_adaptive = opt.interest_adaptive
        self.candidate_aware = opt.candidate_aware
        self.popularity_mode = opt.popularity_mode

        self.category_embedding = nn.Embedding(n_category, self.category_dim, padding_idx=0)
        self.subcategory_embedding = nn.Embedding(n_subcategory, self.subcategory_dim, padding_idx=0)
        self.news_clicks_embedding = nn.Embedding(opt.n_popularity_level, self.popularity_dim)
        self.news_recency_embedding = nn.Embedding(opt.n_recency_group, self.recency_dim)

        self.category_dense_layer_1 = nn.Linear(self.category_dim, self.category_dim, bias=True)
        self.subcategory_dense_layer_1 = nn.Linear(self.subcategory_dim, self.subcategory_dim, bias=True)
        self.category_dense_layer_2 = nn.Linear(self.category_dim, self.news_dim, bias=True)
        self.subcategory_dense_layer_2 = nn.Linear(self.subcategory_dim, self.news_dim, bias=True)
        self.clicks_dense_layer_1 = nn.Linear(self.popularity_dim, self.popularity_dim, bias=True)
        self.recency_dense_layer_1 = nn.Linear(self.recency_dim, self.recency_dim, bias=True)

        self.additive_attention = Attention(self.news_dim, self.news_dim)
        self.user_attention = UserAttentionNetwork(self.news_dim)

        self.clicks_dense_layer_2 = nn.Linear(self.popularity_dim, self.news_dim, bias=True)
        self.recency_dense_layer_2 = nn.Linear(self.recency_dim, self.news_dim, bias=True)
        self.recency_content_dense_layer = nn.Linear(self.news_dim * 2, 1, bias=True)
        
        self.category_prediction_layer = nn.Linear(self.news_dim, n_category)

    ###############################################################

    def initialize(self):
        self.text_encoder.initialize()
        self.user_encoder.initialize()

    ###############################################################

    def IACA_Attention(self, interest_representations: torch.Tensor, candidate_representations: torch.Tensor, unique_category_counts: torch.Tensor, interest_adaptive: int, candidate_aware: int) -> torch.Tensor:
        bs, K, d = interest_representations.shape
        N = candidate_representations.size(1)

        attention_weights = torch.einsum('bkd,bnd->bkn', interest_representations, candidate_representations)

        if interest_adaptive == 1:
            #selected_K = torch.ceil(unique_category_counts.float()).long().clamp_(1, K)
            #selected_K = torch.ceil(self.user_encoder.interest_candidate_attention.M * unique_category_counts.float()).long().clamp_(1, K)
            #selected_K = torch.ceil(torch.sqrt(unique_category_counts.float())).long().clamp_(1, K)
            #selected_K = torch.ceil(torch.sqrt(self.user_encoder.interest_candidate_attention.M * unique_category_counts.float())).long().clamp_(1, K)
            selected_K = torch.ceil(torch.log2(self.user_encoder.interest_candidate_attention.M * unique_category_counts.float())).long().clamp_(1, K)

            sort_idx = torch.argsort(attention_weights, dim=1, descending=True)
            rank_template = torch.arange(K, device=attention_weights.device).view(1, K, 1).expand(bs, K, N)
            ranks = torch.empty_like(sort_idx)
            ranks.scatter_(1, sort_idx, rank_template)
            masked_K = (ranks < selected_K.view(-1, 1, 1)).to(attention_weights.dtype)
        else:
            masked_K = attention_weights.new_ones(attention_weights.shape)

        # (w/ or w/o interest_adaptive) + w/ candidate_aware
        if candidate_aware == 1:
            interest_scores = (masked_K * attention_weights).sum(dim=1)

        else:
            # w/ interest_adaptive + w/o candidate_aware
            if interest_adaptive == 1:
                interest_mask = masked_K.any(dim=2).float()
                interest_mask = interest_mask.unsqueeze(-1)
                user_representation = torch.sum(interest_mask * interest_representations, dim=1) / (interest_mask.sum(dim=1) + 1e-8)
                interest_scores = torch.einsum('bd,bnd->bn', user_representation, candidate_representations)
            # w/o interest_adaptive + w/o candidate_aware
            else:
                user_representation = interest_representations.mean(dim=1)
                interest_scores = torch.einsum('bd,bnd->bn', user_representation, candidate_representations)

        return interest_scores

    ###############################################################

    def forward(self, history_mask, candidate_padding,
                history_title_word, history_title_word_mask,
                candidate_title_word, candidate_title_word_mask,
                history_abstract_word, history_abstract_word_mask,
                candidate_abstract_word, candidate_abstract_word_mask,
                history_category, candidate_category,
                history_subcategory, candidate_subcategory,
                candidate_popularity_label, candidate_recency_label, history_click_num,
                return_representations: bool = False):

        bs = history_title_word.shape[0]
        L = history_title_word.shape[1]
        W = history_title_word.shape[2]
        K = candidate_title_word.shape[1]

        bs_a = history_abstract_word.shape[0]
        L_a = history_abstract_word.shape[1]
        W_a = history_abstract_word.shape[2]
        K_a = candidate_abstract_word.shape[1]

        d = self.news_dim

        ### Title Representation ###
        history_title_word = history_title_word.view(bs * L, W)
        history_title_word_mask = history_title_word_mask.view(bs * L, W)
        history_news_title_representation = self.text_encoder(history_title_word, history_title_word_mask)
        history_news_title_representation = history_news_title_representation.view(bs, L, -1)
        candidate_title_word = candidate_title_word.view(bs * K, W)
        candidate_title_word_mask = candidate_title_word_mask.view(bs * K, W)
        candidate_news_title_representation = self.text_encoder(candidate_title_word, candidate_title_word_mask)
        candidate_news_title_representation = candidate_news_title_representation.view(bs, K, -1)

        ### Abstract Representation ###
        history_abstract_word = history_abstract_word.view(bs_a * L_a, W_a)
        history_abstract_word_mask = history_abstract_word_mask.view(bs_a * L_a, W_a)
        history_news_abstract_representation = self.text_encoder(history_abstract_word, history_abstract_word_mask)
        history_news_abstract_representation = history_news_abstract_representation.view(bs_a, L_a, -1)
        candidate_abstract_word = candidate_abstract_word.view(bs_a * K_a, W_a)
        candidate_abstract_word_mask = candidate_abstract_word_mask.view(bs_a * K_a, W_a)
        candidate_news_abstract_representation = self.text_encoder(candidate_abstract_word, candidate_abstract_word_mask)
        candidate_news_abstract_representation = candidate_news_abstract_representation.view(bs_a, K_a, -1)

        ### Category Representation ###
        category_embedding = F.relu(self.category_dense_layer_1(self.category_embedding.weight))
        history_category_representation = self.category_dense_layer_2(category_embedding[history_category])
        candidate_category_representation = self.category_dense_layer_2(category_embedding[candidate_category])

        ### Subcategory Representation ###
        subcategory_embedding = F.relu(self.subcategory_dense_layer_1(self.subcategory_embedding.weight))
        history_subcategory_representation = self.subcategory_dense_layer_2(subcategory_embedding[history_subcategory])
        candidate_subcategory_representation = self.subcategory_dense_layer_2(subcategory_embedding[candidate_subcategory])
        
        ### News Representation ###
        history_news = torch.stack([
            history_news_title_representation,                                            # [batch_size, max_news_length, news_dimension]
            history_news_abstract_representation,                                         # [batch_size, max_news_length, news_dimension]
            history_category_representation,                                              # [batch_size, max_news_length, news_dimension]
            history_subcategory_representation                                            # [batch_size, max_news_length, news_dimension]
        ], dim=2)                                                                         # [batch_size, max_news_length, 4, news_dimension]
        history_news = history_news.view(bs * L, 4, d)                                    # [batch_size*max_news_length, 4, news_dimension]
        history_news_representations = self.additive_attention(history_news, mask=None)   # [batch_size*max_news_length, news_dimension]
        history_news_representations = history_news_representations.view(bs, L, d)        # [batch_size, max_news_length, news_dimension]
 
        candidate_news = torch.stack([
            candidate_news_title_representation,
            candidate_news_abstract_representation,     
            candidate_category_representation,                                    
            candidate_subcategory_representation                               
        ], dim=2)                                              
        candidate_news = candidate_news.view(bs * K, 4, d)
        candidate_news_representations = self.additive_attention(candidate_news, mask=None)
        candidate_news_representations = candidate_news_representations.view(bs, K, d)                

        ### Interest Score ###
        x = history_category.clone()
        x[x == 0] = -1                         
        x_sorted, _ = torch.sort(x, dim=1)
        valid = x_sorted != -1
        difference  = torch.ones_like(x_sorted, dtype=torch.bool)
        difference[:, 1:] = x_sorted[:, 1:] != x_sorted[:, :-1]
        unique_category_counts = (valid & difference).sum(dim=1)

        extracted_K = torch.full((bs,), self.num_extracted_interests, dtype=torch.long, device=history_news_representations.device)
        interest_representations = self.user_encoder.poly_attention(history_news_representations, history_mask, extracted_K)
        interest_scores = self.IACA_Attention(interest_representations, candidate_news_representations, unique_category_counts, self.interest_adaptive, self.candidate_aware)

        ### Popularity Score ###
        news_clicks_embedding = F.relu((self.clicks_dense_layer_1(self.news_clicks_embedding.weight)))
        news_recency_embedding = F.relu((self.recency_dense_layer_1(self.news_recency_embedding.weight)))
        candidate_news_clicks = news_clicks_embedding[candidate_popularity_label]
        candidate_news_recency = news_recency_embedding[candidate_recency_label]

        p_n = candidate_news_representations                       # [batch_size, np_ratio+1, news_dim]
        p_r = self.recency_dense_layer_2(candidate_news_recency)   # [batch_size, np_ratio+1, news_dim]
        p_c = self.clicks_dense_layer_2(candidate_news_clicks)     # [batch_size, np_ratio+1, news_dim]

        recency_content_cat = torch.cat([p_n, p_r], dim=-1)                         # [batch_size, np_ratio+1, news_dim*2]
        mu = torch.sigmoid(self.recency_content_dense_layer(recency_content_cat))   # [batch_size, np_ratio+1, 1]
        p_rn = mu * p_n + (1 - mu) * p_r                                            # [batch_size, np_ratio+1, news_dim]
        
        # Model
        if self.popularity_mode == 0:
            popularity_scores = torch.sum(p_c * p_rn, dim=-1)   # [batch_size, np_ratio+1]
        # w/o Clicks
        elif self.popularity_mode == 1:
            popularity_scores = torch.sum(p_r * p_n)
        # w/o Recency
        elif self.popularity_mode == 2:
            popularity_scores = torch.sum(p_c * p_n)
        # w/o Content
        elif self.popularity_mode == 3:
            popularity_scores = torch.sum(p_c * p_r)

        ### Ranking Score ###
        a_u = torch.sum(history_click_num, dim=1) / ((torch.sum(history_click_num != 0, dim=1) + 1e-8) * self.max_history_news_click_num)
        a_u = a_u.unsqueeze(1)
        
        user_click_counts = torch.sum(history_click_num != 0, dim=1) 
        cold_start_users_mask = (user_click_counts <= self.cold_start_users).float().unsqueeze(1) 
        a_u = a_u + (self.cold_start_users_weight * cold_start_users_mask)

        candidate_mask_inf = torch.where(candidate_padding == 0, float('-inf') * torch.ones_like(candidate_padding), torch.zeros_like(candidate_padding).float())
        #ranking_scores = self.user_attention(candidate_news_representations, interest_scores, popularity_scores, a_u) + candidate_mask_inf
        ranking_scores = self.user_attention(interest_scores, popularity_scores, a_u) + candidate_mask_inf
        '''
        ### Popularity-based Recommendations for Cold-start Users (opt.cold_start_users) ###
        ranking_scores_full = self.user_attention(interest_scores, popularity_scores, a_u)
        ranking_scores_cold = popularity_scores
        user_click_counts = torch.sum(history_click_num != 0, dim=1)
        cold_start_mask = (user_click_counts <= self.cold_start_users).unsqueeze(1)
        ranking_scores = torch.where(cold_start_mask, ranking_scores_cold, ranking_scores_full)
        ranking_scores = ranking_scores + candidate_mask_inf
        '''
        '''
        ### Random Recommendations for Cold-start Users (opt.cold_start_users) ###
        ranking_scores_full = self.user_attention(interest_scores, popularity_scores, a_u)
        user_click_counts = torch.sum(history_click_num != 0, dim=1)
        cold_start_mask = (user_click_counts <= self.cold_start_users).unsqueeze(1)
        random_scores = torch.rand_like(popularity_scores)
        ranking_scores = torch.where(cold_start_mask, random_scores, ranking_scores_full)
        ranking_scores = ranking_scores + candidate_mask_inf
        '''

        candidate_mask = torch.where(candidate_padding == 0, torch.zeros_like(candidate_padding), torch.ones_like(candidate_padding))
        candidate_lengths = torch.sum(candidate_mask, dim=-1)

        ### News Category Classification Task ###
        category_preds = self.category_prediction_layer(candidate_news_representations)   # [batch_size, np_ratio+1, n_category]

        return ranking_scores, candidate_lengths, candidate_news_representations, category_preds