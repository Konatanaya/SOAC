import torch
import torch.nn as nn
import torch.nn.functional as F
from models import rsmodel
from torch.utils.tensorboard import SummaryWriter


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

        self.v = nn.Linear(self.hidden_size, 1).apply(self.init_weights)
        self.advantage = nn.Linear(self.hidden_size, item_num).apply(self.init_weights)
        self.logits = nn.Linear(self.hidden_size, self.item_num).apply(self.init_weights)
        self.beta_logits = nn.Linear(self.hidden_size, self.item_num).apply(self.init_weights)

        # weight
        params = torch.ones(2, requires_grad=True)
        self.weight = nn.Parameter(params)

        params = [{'params': self.head.parameters(), 'lr': learning_rate * 0.1},
                  {'params': self.state_embeddings.parameters(), 'lr': learning_rate * 0.1},
                  {'params': self.pos_embeddings.parameters(), 'lr': learning_rate * 0.1},
                  {'params': self.v.parameters()},
                  {'params': self.advantage.parameters()},
                  {'params': self.logits.parameters()},
                  {'params': self.beta_logits.parameters()},
                  {'params': self.weight, 'lr': 1e-5}]

        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, state, state_length):
        input_embeddings = self.state_embeddings(state)
        mask = torch.unsqueeze((state != self.item_num).float(), -1)
        if self.rs_model == 'GRU':
            state_hidden = self.head(input_embeddings, state_length)

        logits = self.logits(state_hidden)

        v = self.v(state_hidden)
        advantage = self.advantage(state_hidden)
        q_values = v + advantage - torch.mean(advantage, dim=1, keepdim=True)

        beta_logits = self.beta_logits(state_hidden.detach())

        return logits, F.log_softmax(logits, dim=1), beta_logits, F.log_softmax(beta_logits, dim=1), q_values


class SOAC(nn.Module):
    def __init__(self, state_size, hidden_size, item_num, learning_rate, discount, negative_samples, rs_model, device, threshold=0.5, ips=10, slate_sample_num=100, slate_size=100):
        super(SOAC, self).__init__()
        self.name = 'SOAC'
        self.main_net = QNetwork(state_size, hidden_size, item_num, learning_rate, negative_samples, rs_model, self.name, device).to(device)
        self.target_net = QNetwork(state_size, hidden_size, item_num, learning_rate, negative_samples, rs_model, self.name, device).to(device)
        self.neg = negative_samples
        self.discount = discount
        self.item_num = int(item_num)
        self.device = device

        self.lr1 = learning_rate
        self.lr2 = 0.001
        self.lr_changed = False

        self.threshold = threshold
        self.update_counter = 0
        self.tau = 0.001
        self.ips_upper = ips

        self.slate_sample_num = slate_sample_num
        self.slate_size = slate_size

    def to_numpy(self, tensor):
        return tensor.cpu().detach().numpy()

    def train_model(self, state, state_len, action, next_state, next_state_len, is_done, reward, neg_action, neg_reward, flag=True):
        self.update_counter += 1

        main_Q = self.main_net
        target_Q = self.target_net

        action = action.unsqueeze(-1)
        reward = reward.unsqueeze(-1)
        is_done = is_done.unsqueeze(-1)

        logits, log_probs, beta_logits, beta_log_probs, current_main_qs = main_Q(state, state_len)
        #
        with torch.no_grad():
            next_logits, next_log_probs, _, _, next_main_qs = main_Q(next_state, next_state_len)
            _, _, _, _, next_target_qs = target_Q(next_state, next_state_len)

            next_probs = next_log_probs.exp()
            next_probs = (next_probs / next_probs.max(1, keepdim=True)[0] > self.threshold).float()
            next_action = (next_probs * next_main_qs + (1 - next_probs) * -1e8).argmax(1, keepdim=True)
            next_q = next_target_qs.gather(1, next_action)
            target_q = reward + self.discount * is_done * next_q
        q_loss = F.smooth_l1_loss(current_main_qs.gather(1, action), target_q)

        with torch.no_grad():
            ips = log_probs.exp().gather(1, action) / beta_log_probs.exp().gather(1, action)
            ips = torch.where(ips > 0.01, ips, 0.01)
            ips = torch.where(ips < self.ips_upper, ips, self.ips_upper)
            ips /= torch.mean(ips)
            ips = ips * (current_main_qs.gather(1, action) - (current_main_qs * F.softmax(logits, dim=1)).sum(-1, keepdims=True))


        num_logits = logits.size(1)
        batch_size = logits.size(0)
        slate_sample_size = self.slate_sample_num
        slate_size = self.slate_size
        rls = torch.cat([logits.detach(), torch.FloatTensor([float('-inf')]).to(self.device).view(1, 1).expand(logits.size(0), 1)], dim=1)
        log_probs_ = torch.cat([log_probs, torch.zeros(log_probs.size(0), 1).to(self.device)], dim=1)
        samples = torch.multinomial(F.softmax(logits.detach(), dim=1), slate_sample_size * slate_size, replacement=True).view(-1, slate_sample_size, slate_size)
        samples = torch.cat([samples, action.view(-1, 1, 1).expand(samples.size(0), slate_sample_size, 1)], dim=-1)
        samples = unique_and_padding(samples.view(-1, slate_size + 1), num_logits).reshape(batch_size, -1)
        rp = F.softmax(rls.gather(1, samples).view(batch_size, slate_sample_size, -1), dim=-1)
        lp = torch.sum(log_probs_.gather(1, samples).view(batch_size, slate_sample_size, -1), dim=-1) - log_probs_.gather(1, action)
        samples = samples.view(batch_size, slate_sample_size, -1)
        act = action.view(-1, 1, 1).expand(batch_size, slate_sample_size, 1)
        sampled_slate_res = torch.mean(ips * (rp[samples == act]).view(batch_size, -1) * lp, dim=1, keepdim=True)
        log_action_res = torch.mean(ips * (rp[samples == act]).view(batch_size, -1), dim=1, keepdim=True)
        actor_loss = -torch.mean(log_action_res * log_probs_.gather(1, action))
        actor_loss += -torch.mean(sampled_slate_res) * 1e-3

        ce_loss = -(reward * log_probs.gather(1, action)).mean()
        beta_ce_loss = F.cross_entropy(beta_logits, action.squeeze(-1))

        if not flag and not self.lr_changed:
            for param_group in self.main_net.optimizer.param_groups:
                if param_group['lr'] == self.lr1:
                    param_group['lr'] = self.lr2
                elif param_group['lr'] == self.lr2:
                    param_group['lr'] = self.lr2 * 0.1
            self.lr_changed = True

        loss_list = [ce_loss, actor_loss]
        loss = beta_ce_loss + q_loss
        for i, l in enumerate(loss_list):
            loss += (0.5 * l / (main_Q.weight[i] ** 2) + torch.log(1 + main_Q.weight[i] ** 2))

        main_Q.optimizer.zero_grad()
        loss.backward()
        main_Q.optimizer.step()

        self.soft_update(self.main_net, self.target_net)

        return loss

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def predict(self, state, state_length):
        logits, _, _, _, _ = self.main_net(state, state_length)
        prediction = torch.argsort(logits)[:, -20:]
        return prediction


def unique_and_padding(mat, padding_idx, dim=-1):
    samples, _ = torch.sort(mat, dim=dim)
    samples_roll = torch.roll(samples, -1, dims=dim)
    samples_diff = samples - samples_roll
    samples_diff[:, -1] = 1  # deal with the edge case that there is only one unique sample in a row
    samples_mask = torch.bitwise_not(samples_diff == 0)  # unique mask
    samples *= samples_mask.to(dtype=samples.dtype)
    samples += (1 - samples_mask.to(dtype=samples.dtype)) * padding_idx
    samples, _ = torch.sort(samples, dim=dim)
    # shrink size to max unique length
    samples = torch.unique(samples, dim=dim)
    return samples
