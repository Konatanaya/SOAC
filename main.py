import argparse
import os
import pandas as pd
import torch
import time
import numpy as np
from collections import defaultdict
from models import baseline, SOAC
from utility import calculate_hit, pad_history
import csv


def parse_args():
    parser = argparse.ArgumentParser(description="RL+RS.")

    parser.add_argument('--epoch', type=int, default=200000, help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='RC15', help='data directory, i.e., RC15 and Kaggle')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--r_click', type=float, default=0.2, help='reward for the click behavior.')
    parser.add_argument('--r_buy', type=float, default=1, help='reward for the purchase behavior.')
    parser.add_argument('--r_negative', type=float, default=0.0, help='reward for the negative behavior.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--discount', type=float, default=0.5, help='Discount factor for RL.')
    parser.add_argument('--ips', type=float, default=10, help='Discount factor for RL.')
    parser.add_argument('--threshold', type=float, default=0.25, help='Discount factor for RL.')
    parser.add_argument('--neg', type=int, default=10, help='number of negative samples.')
    parser.add_argument('--rs_model', type=str, default='GRU', help='the base recommendation models')
    parser.add_argument('--rl_model', type=str, default='SOAC', help='the rl models, including SOAC, SAC, S2AC, SQN, SNQN')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--slate_size', type=int, default=100, help='length of a sampled recommendation list')
    parser.add_argument('--slate_num', type=int, default=100, help='number of sampled recommendation list')
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def evaluate(model, test_dict, step):
    model.eval()
    total_clicks = 0.0
    total_purchase = 0.0
    total_reward = [0, 0, 0, 0]
    hit_clicks = [0, 0, 0, 0]
    ndcg_clicks = [0, 0, 0, 0]
    hit_purchase = [0, 0, 0, 0]
    ndcg_purchase = [0, 0, 0, 0]

    for key, val in test_dict.items():
        states_tensor = torch.LongTensor(val[0])
        len_states = val[1]
        actions = val[2]
        is_buy = val[3]
        rewards = val[4]
        total_purchase += val[5]
        total_clicks += val[6]
        prediction = model.predict(states_tensor.to(device), torch.Tensor(len_states)).cpu().detach().numpy()
        calculate_hit(prediction, topk, actions, rewards, reward_click, total_reward, hit_clicks, ndcg_clicks, hit_purchase, ndcg_purchase)
    print('#############################################################')
    print('total clicks: %d, total purchase:%d' % (total_clicks, total_purchase))
    row = [step]
    for i in range(len(topk)):
        hr_click = hit_clicks[i] / total_clicks
        hr_purchase = hit_purchase[i] / total_purchase
        ng_click = ndcg_clicks[i] / total_clicks
        ng_purchase = ndcg_purchase[i] / total_purchase
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('cumulative reward @ %d: %f' % (topk[i], total_reward[i]))
        print('clicks hr ndcg @ %d : %f, %f' % (topk[i], hr_click, ng_click))
        print('purchase hr and ndcg @%d : %f, %f' % (topk[i], hr_purchase, ng_purchase))
        row += [hr_purchase, float(ng_purchase), hr_click, float(ng_click), total_reward[i]]

    with open(os.path.join(result_directory, result_filename + '.csv'), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)
    print('#############################################################')
    model.train()


def init_model(args):
    rs_model = args.rs_model
    rl_model = args.rl_model
    if rl_model == 'None':
        model = baseline.RS(state_size, args.hidden_dim, item_num, args.lr, args.discount, args.neg, rs_model, device)
    elif rl_model == 'SNQN':
        model = baseline.SNQN(state_size, args.hidden_dim, item_num, args.lr, args.discount, args.neg, rs_model, device)
    elif rl_model == 'SA2C':
        model = baseline.SA2C(state_size, args.hidden_dim, item_num, args.lr, args.discount, args.neg, rs_model, device)
    elif rl_model == 'SQN':
        model = baseline.SQN(state_size, args.hidden_dim, item_num, args.lr, args.discount, args.neg, rs_model, device)
    elif rl_model == 'SAC':
        model = baseline.SAC(state_size, args.hidden_dim, item_num, args.lr, args.discount, args.neg, rs_model, device)
    elif rl_model == 'SOAC':
        model = SOAC.SOAC(state_size, args.hidden_dim, item_num, args.lr, args.discount, args.neg, rs_model, device, threshold=args.threshold, ips=args.ips,
                          slate_sample_num=args.slate_num, slate_size=args.slate_size)
    return model


def preprocess_dataset(data):
    data_dict = defaultdict(list)
    states, len_states, actions, is_buy = [], [], [], []
    for index, row in data.iterrows():
        states.append(row['state'])
        len_states.append(row['len_state'])
        actions.append(row['action'])
        is_buy.append(row['is_buy'])
        if (index + 1) % 1024 == 0 or (index + 1) == len(data):
            rewards = [reward_buy if v == 1 else reward_click for v in is_buy]
            total_purchase = np.sum(is_buy)
            total_clicks = len(is_buy) - total_purchase
            data_dict[index] = [states, len_states, actions, is_buy, rewards, total_purchase, total_clicks]
            states, len_states, actions, is_buy = [], [], [], []
    return data_dict


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    setup_seed(args.seed)
    data_directory = 'data/' + args.data
    data_statis = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing state_size and item_num
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    reward_click = args.r_click
    reward_buy = args.r_buy
    reward_negative = args.r_negative
    topk = [5, 10, 15, 20]

    replay_buffer = pd.read_pickle(os.path.join(data_directory, 'replay_buffer.df'))
    print('#############################################################')
    print('Dataset: %s' % args.data)
    print('Training set statistic | State size: %i | Number of items: %i' % (state_size, item_num))
    print('Preparing test set.')

    # preprocess test_set
    test_set = pd.read_pickle(os.path.join(data_directory, 'testset.df'))
    test_dict = preprocess_dataset(test_set)
    print('Test set preparation done.')

    model = init_model(args)
    model = torch.compile(model)
    model.train()

    result_directory = os.path.join('results', args.data, args.rs_model)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    number = 1
    for filename in os.listdir(result_directory):
        if model.name + '_' + str(args.seed) in filename:
            number += 1

    result_filename = model.name + '_' + str(args.seed) + '_' + str(args.threshold) + '_' + str(args.ips) + '_' + str(number)

    print('Model name: %s_%d' % (model.name, number))
    with open(os.path.join(result_directory, result_filename + '.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Step', 'Purchase_HR@5', 'Purchase_NG@5', 'Click_HR@5', 'Click_NG@5', 'Reward@5',
                         'Purchase_HR@10', 'Purchase_NG@10', 'Click_HR@10', 'Click_NG@10', 'Reward@10',
                         'Purchase_HR@15', 'Purchase_NG@15', 'Click_HR@15', 'Click_NG@15', 'Reward@15',
                         'Purchase_HR@20', 'Purchase_NG@20', 'Click_HR@20', 'Click_NG@20', 'Reward@20'])
    print('#############################################################')

    num_rows = replay_buffer.shape[0]
    num_batches = int(num_rows / args.batch_size)
    print(num_rows, num_batches)
    total_step = 0
    starttime = time.time()

    for i in range(args.epoch):
        total_step += 1
        batch = replay_buffer.sample(n=args.batch_size).to_dict()

        state = torch.LongTensor(list(batch['state'].values())).to(device)
        len_state = torch.Tensor(list(batch['len_state'].values()))
        action = list(batch['action'].values())

        neg_action = []
        for index in range(args.batch_size):
            negative_list = []
            for i in range(args.neg):
                neg_index = np.random.randint(item_num)
                while neg_index == action[index]:
                    neg_index = np.random.randint(item_num)
                negative_list.append(neg_index)
            neg_action.append(negative_list)
        neg_action = torch.LongTensor(neg_action).to(device)

        action = torch.LongTensor(list(batch['action'].values())).to(device)
        next_action = torch.LongTensor(list(batch['next_action'].values())).to(device)
        next_state = torch.LongTensor(list(batch['next_state'].values())).to(device)
        len_next_state = torch.Tensor(list(batch['len_next_states'].values()))
        is_done = torch.LongTensor([0 if i else 1 for i in list(batch['is_done'].values())]).to(device)
        is_buy = list(batch['is_buy'].values())
        reward = torch.FloatTensor([reward_buy if i == 1 else reward_click for i in is_buy]).to(device)
        neg_reward = args.r_negative * torch.ones(args.batch_size).to(device)

        if total_step <= 15000:
            loss = model.train_model(state, len_state, action, next_state, len_next_state, is_done, reward, neg_action, neg_reward)
        else:
            loss = model.train_model(state, len_state, action, next_state, len_next_state, is_done, reward, neg_action, neg_reward, flag=False)

        if total_step % 200 == 0:
            print("Step: %i | Loss in %dth batch is: %f | Time: %f" % (total_step, total_step, loss.item(), time.time() - starttime))
            starttime = time.time()
        if total_step % 1000 == 0:
            evaluate(model, test_dict, total_step)
