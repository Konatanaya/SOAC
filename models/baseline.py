import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import rsmodel


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, state_size, hidden_size, item_num, learning_rate, negative_samples, rs_model, rl_model, device):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.learning_rate = learning_rate
        self.neg = negative_samples
        self.rs_model = rs_model
        self.rl_model = rl_model
        self.device = device

        self.state_embeddings = nn.Embedding(self.item_num + 1, self.hidden_size)
        self.state_embeddings.weight.data.normal_(0, 0.01)

        self.pos_embeddings = nn.Embedding(self.state_size, self.hidden_size)
        self.pos_embeddings.weight.data.normal_(0, 0.01)

        if self.rs_model == 'GRU':
            self.head = rsmodel.GRU(self.hidden_size, self.hidden_size)

        self.q_layer = nn.Linear(self.hidden_size, self.item_num)  # Q-values for all actions
        self.q_layer.apply(self.init_weights)
        self.logits_layer = nn.Linear(self.hidden_size, self.item_num)  # logits for all actions
        self.logits_layer.apply(self.init_weights)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, state, state_length):
        input_embeddings = self.state_embeddings(state)
        mask = torch.unsqueeze((state != self.item_num).float(), -1)
        if self.rs_model == 'GRU':
            state_hidden = self.head(input_embeddings, state_length)
        elif self.rs_model == 'Caser':
            input_embeddings *= mask
            input_embeddings = torch.unsqueeze(input_embeddings, 1)
            state_hidden = self.head(input_embeddings)
        elif self.rs_model == 'SASRec':
            pos_embeddings = self.pos_embeddings(torch.tile(torch.arange(state.shape[1]).unsqueeze(0), (state.shape[0], 1)).to(self.device))
            seq = input_embeddings + pos_embeddings
            state_hidden = self.head(seq, mask, state_length)
        elif self.rs_model == 'NItNet':
            input_embeddings *= mask
            state_hidden = self.head(input_embeddings, mask, state_length)

        logits = self.logits_layer(state_hidden)
        q_values = self.q_layer(state_hidden)
        return logits, q_values


def double_q_learning(Qs, action, r_t, discount, target_Qs, main_Qs):
    best_action = torch.argmax(main_Qs, dim=1)
    temp_q = torch.gather(target_Qs, dim=1, index=best_action.unsqueeze(1)).squeeze(1)
    target = r_t + discount * temp_q.detach()
    temp_q_current = torch.gather(Qs, dim=1, index=action.unsqueeze(1)).squeeze(1)

    td_error = target - temp_q_current
    loss = 0.5 * torch.square(td_error)
    return loss


class RS(nn.Module):
    def __init__(self, state_size, hidden_size, item_num, learning_rate, discount, negative_samples, rs_model, device):
        super(RS, self).__init__()
        self.name = rs_model
        self.q_net_1 = QNetwork(state_size, hidden_size, item_num, learning_rate, negative_samples, rs_model, 'None', device).to(device)
        self.device = device
        self.discount = discount

    def train_model(self, state, state_len, action, next_state, next_state_len, is_done, reward, neg_action, neg_reward, flag=True):
        logits, _ = self.q_net_1(state, state_len)
        ce_loss = F.cross_entropy(logits, action)
        self.q_net_1.optimizer.zero_grad()
        ce_loss.backward()
        self.q_net_1.optimizer.step()
        return ce_loss

    def predict(self, state, state_length):
        logits, _ = self.q_net_1(state, state_length)
        prediction = torch.argsort(logits)[:, -20:]
        return prediction

    def behavior_probability(self, state, state_length):
        logits, _ = self.q_net_1(state, state_length)
        prob = F.softmax(logits, dim=1)
        return prob


class SQN(nn.Module):
    def __init__(self, state_size, hidden_size, item_num, learning_rate, discount, negative_samples, rs_model, device):
        super(SQN, self).__init__()
        self.name = 'SQN'
        self.q_net_1 = QNetwork(state_size, hidden_size, item_num, learning_rate, negative_samples, rs_model, self.name, device).to(device)
        self.q_net_2 = QNetwork(state_size, hidden_size, item_num, learning_rate, negative_samples, rs_model, self.name, device).to(device)
        self.neg = negative_samples
        self.discount = discount
        self.item_num = int(item_num)
        self.device = device

    def train_model(self, state, state_len, action, next_state, next_state_len, is_done, reward, neg_action, neg_reward, flag=True):
        pointer = np.random.randint(0, 2)
        if pointer == 0:
            main_Q = self.q_net_1
            target_Q = self.q_net_2
        else:
            main_Q = self.q_net_2
            target_Q = self.q_net_1

        _, target_Qs_next = target_Q(next_state, next_state_len)
        _, main_Qs_next = main_Q(next_state, next_state_len)
        logits, main_Qs_current = main_Q(state, state_len)

        discount = self.discount * is_done
        q_loss = double_q_learning(main_Qs_current, action, reward, discount, target_Qs_next, main_Qs_next)

        ce_loss = F.cross_entropy(logits, action, reduction='none')
        loss = torch.mean(q_loss + ce_loss)
        main_Q.optimizer.zero_grad()
        loss.backward()
        main_Q.optimizer.step()

        return loss

    def predict(self, state, state_length):
        logits, _ = self.q_net_1(state, state_length)
        prediction = torch.argsort(logits)[:, -20:]
        return prediction


class SAC(nn.Module):
    def __init__(self, state_size, hidden_size, item_num, learning_rate, discount, negative_samples, rs_model, device):
        super(SAC, self).__init__()
        self.name = 'SAC'
        self.q_net_1 = QNetwork(state_size, hidden_size, item_num, learning_rate, negative_samples, rs_model, self.name, device).to(device)
        self.q_net_2 = QNetwork(state_size, hidden_size, item_num, learning_rate, negative_samples, rs_model, self.name, device).to(device)
        self.neg = negative_samples
        self.discount = discount
        self.item_num = int(item_num)
        self.device = device
        self.rs_model = rs_model

    def train_model(self, state, state_len, action, next_state, next_state_len, is_done, reward, neg_action, neg_reward, flag=True):
        pointer = np.random.randint(0, 2)
        if pointer == 0:
            main_Q = self.q_net_1
            target_Q = self.q_net_2
        else:
            main_Q = self.q_net_2
            target_Q = self.q_net_1

        _, target_Qs_next = target_Q(next_state, next_state_len)  # shape: batch_size x hidden_dim
        _, main_Qs_next = main_Q(next_state, next_state_len)
        logits, main_Qs_current = main_Q(state, state_len)

        discount = self.discount * is_done  # shape: batch_size
        q_loss = double_q_learning(main_Qs_current, action, reward, discount, target_Qs_next, main_Qs_next)

        q_indexed = main_Qs_current.gather(1, action.unsqueeze(1)).detach().squeeze(1)

        ce_loss = F.cross_entropy(logits, action, reduction='none')

        if flag:
            loss = torch.mean(q_loss + ce_loss)
            main_Q.optimizer.zero_grad()
            loss.backward()
            main_Q.optimizer.step()
        else:
            loss = torch.mean(q_loss + ce_loss * q_indexed)
            main_Q.optimizer.zero_grad()
            loss.backward()
            main_Q.optimizer.step()

        return loss

    def predict(self, state, state_length):
        logits, _ = self.q_net_1(state, state_length)
        prediction = torch.argsort(logits)[:, -20:]
        return prediction


class SNQN(nn.Module):
    def __init__(self, state_size, hidden_size, item_num, learning_rate, discount, negative_samples, rs_model, device):
        super(SNQN, self).__init__()
        self.name = 'SNQN'
        self.q_net_1 = QNetwork(state_size, hidden_size, item_num, learning_rate, negative_samples, rs_model, self.name, device).to(device)
        self.q_net_2 = QNetwork(state_size, hidden_size, item_num, learning_rate, negative_samples, rs_model, self.name, device).to(device)
        self.neg = negative_samples
        self.discount = discount
        self.item_num = int(item_num)
        self.device = device

    def train_model(self, state, state_len, action, next_state, next_state_len, is_done, reward, neg_action, neg_reward, flag=True):
        pointer = np.random.randint(0, 2)
        if pointer == 0:
            main_Q = self.q_net_1
            target_Q = self.q_net_2
        else:
            main_Q = self.q_net_2
            target_Q = self.q_net_1

        _, target_Qs_next = target_Q(next_state, next_state_len)
        _, main_Qs_next = main_Q(next_state, next_state_len)
        _, target_Qs_current = target_Q(state, state_len)
        logits, main_Qs_current = main_Q(state, state_len)

        discount = self.discount * is_done
        q_loss_positive = double_q_learning(main_Qs_current, action, reward, discount, target_Qs_next, main_Qs_next)
        q_loss_negative = 0
        for i in range(self.neg):
            index = i * torch.ones(neg_action.shape[0], dtype=torch.int64).to(self.device)
            negative = torch.gather(neg_action, index=index.unsqueeze(1), dim=1).squeeze(1)
            # discount should be considered as state does not change.
            q_loss_negative += double_q_learning(main_Qs_current, negative, neg_reward, self.discount, target_Qs_current, main_Qs_current)

        ce_loss = F.cross_entropy(logits, action, reduction='none')
        loss = torch.mean(q_loss_positive + q_loss_negative + ce_loss)
        main_Q.optimizer.zero_grad()
        loss.backward()
        main_Q.optimizer.step()

        return loss

    def predict(self, state, state_length):
        logits, _ = self.q_net_1(state, state_length)
        prediction = torch.argsort(logits)[:, -20:]
        return prediction


class SA2C(nn.Module):
    def __init__(self, state_size, hidden_size, item_num, learning_rate, discount, negative_samples, rs_model, device):
        super(SA2C, self).__init__()
        self.name = 'SA2C'
        self.q_net_1 = QNetwork(state_size, hidden_size, item_num, learning_rate, negative_samples, rs_model, self.name, device).to(device)
        self.q_net_2 = QNetwork(state_size, hidden_size, item_num, learning_rate, negative_samples, rs_model, self.name, device).to(device)
        self.neg = negative_samples
        self.discount = discount
        self.item_num = int(item_num)
        self.device = device

        self.lr2 = 0.001
        self.lr_changed = False

    def train_model(self, state, state_len, action, next_state, next_state_len, is_done, reward, neg_action, neg_reward, flag=True):
        pointer = np.random.randint(0, 2)
        if pointer == 0:
            main_Q = self.q_net_1
            target_Q = self.q_net_2
        else:
            main_Q = self.q_net_2
            target_Q = self.q_net_1

        _, target_Qs_next = target_Q(next_state, next_state_len)  # shape: batch_size x hidden_dim
        _, main_Qs_next = main_Q(next_state, next_state_len)
        _, target_Qs_current = target_Q(state, state_len)
        logits, main_Qs_current = main_Q(state, state_len)

        discount = self.discount * is_done  # shape: batch_size
        q_loss_positive = double_q_learning(main_Qs_current, action, reward, discount, target_Qs_next, main_Qs_next)
        q_loss_negative = 0
        q_indexed_positive = main_Qs_current.gather(1, action.unsqueeze(1)).detach().squeeze(1)
        q_indexed_negative = 0
        for i in range(self.neg):
            index = i * torch.ones(neg_action.shape[0], dtype=torch.int64).to(self.device)
            negative = torch.gather(neg_action, index=index.unsqueeze(1), dim=1).squeeze(1)
            q_loss_negative += double_q_learning(main_Qs_current, negative, neg_reward, self.discount, target_Qs_current,
                                                 main_Qs_current)  # discount should be considered as state does not change.
            q_indexed_negative += main_Qs_current.gather(1, negative.unsqueeze(1)).detach().squeeze(1)

        advantage = q_indexed_positive - (q_indexed_positive + q_indexed_negative) / (1 + self.neg)
        advantage = torch.clip(advantage, 0, 10)
        ce_loss = F.cross_entropy(logits, action, reduction='none')

        if flag:
            loss = torch.mean(q_loss_positive + q_loss_negative + ce_loss)
            main_Q.optimizer.zero_grad()
            loss.backward()
            main_Q.optimizer.step()
        else:
            if not self.lr_changed:
                for param_group in self.q_net_1.optimizer.param_groups:
                    param_group['lr'] = self.lr2
                for param_group in self.q_net_2.optimizer.param_groups:
                    param_group['lr'] = self.lr2
            loss = torch.mean(q_loss_positive + q_loss_negative + ce_loss * advantage)
            main_Q.optimizer.zero_grad()
            loss.backward()
            main_Q.optimizer.step()

        return loss

    def predict(self, state, state_length):
        logits, _ = self.q_net_1(state, state_length)
        prediction = torch.argsort(logits)[:, -20:]
        return prediction


if __name__ == '__main__':
    # a = torch.LongTensor([[0,1,2,3,4,5,6,7,8,9]])
    # l = 10

    state = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 9, 9],
                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 9],
                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]]).to('cuda:1')
    l = torch.LongTensor([8, 8, 8])
    net = QNetwork(10, 64, 10, 0.01, 10, 'NItNet', 'None', 'cuda:1').to('cuda:1')
    net(state, l)
