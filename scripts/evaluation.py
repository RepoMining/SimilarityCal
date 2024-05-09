import csv
import pickle
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import utils


class EmbeddingMLP(nn.Module):
    def __init__(self, size=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768 * size, 900 * size),
            nn.BatchNorm1d(900 * size),
            nn.ReLU(),
            nn.Linear(900 * size, 300 * size)
        )

    def forward(self, data):
        res = self.net(data)
        return res


class PairClassifier(nn.Module):
    def __init__(self, size=4):
        super().__init__()
        self.encoder = EmbeddingMLP(size)
        self.net = nn.Sequential(
            nn.Linear(300 * size * 2, 3000),
            nn.ReLU(),
            nn.Linear(3000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2),
        )

    def forward(self, data):
        e1 = self.encoder(data[:, :768 * 4])
        e2 = self.encoder(data[:, 768 * 4:])
        twins = torch.cat([e1, e2], dim=1)
        res = self.net(twins)
        return res


class MergePairDataset(Dataset):
    def __init__(self, data, labels, repos, embedding_types):
        self.data = []
        for i in range(len(labels)):
            e1 = []
            e2 = []
            for t in embedding_types:
                e1.append(data[t][i][0])
                e2.append(data[t][i][1])
            row = torch.cat(e1 + e2, dim=0)
            self.data.append(row)
        self.labels = torch.LongTensor(labels)
        self.repos = repos

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index], self.repos[index], index


def mergeEmbedding(sets_list, labels_list, repos_list, embedding_types):
    all_embeddings_set = {}
    all_labels = []
    all_repos = []
    for t in embedding_types:
        all_embeddings_set[t] = []

    for embedding_set in sets_list:
        for t in embedding_types:
            all_embeddings_set[t] = all_embeddings_set[t] + embedding_set[t]
    for labels_set in labels_list:
        all_labels = all_labels + labels_set
    for repos_set in repos_list:
        all_repos = all_repos + repos_set

    return all_embeddings_set, all_labels, all_repos


def build_embedding_sets(filename, unique_labels=None):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    repo_names = list(data.keys())
    embedding_types = list(data[repo_names[0]].keys())[-5:]
    mean_embeddings = {}
    labels = []

    for embedding in embedding_types:
        mean_embeddings[embedding] = []

    for repo in repo_names:
        if unique_labels and data[repo]['topic'] not in unique_labels:
            continue
        labels.append(data[repo]['topic'])
        for embedding in embedding_types:
            mean_embeddings[embedding].append(utils.mean_embeddings(data[repo][embedding]))

    return mean_embeddings, labels, repo_names


def build_pair_data(data, labels, embedding_types, repo_names, balanced=1.0):
    pair_embeddings = {}
    pair_labels = []
    pair_repos = []

    def make_pair(idx1, idx2):
        pair_repos.append((repo_names[idx1], repo_names[idx2]))
        if labels[idx1] != labels[idx2]:
            pair_labels.append(0)
        else:
            pair_labels.append(1)
        for embedding_type in embedding_types:
            e1 = data[embedding_type][idx1]
            e2 = data[embedding_type][idx2]
            pair_embeddings[embedding_type].append((e1, e2))

    for embedding_type in embedding_types:
        pair_embeddings[embedding_type] = []

    for i in range(len(labels)):
        make_pair(i, i)
        for j in range(i + 1, len(labels)):
            make_pair(i, j)
            make_pair(j, i)

    if balanced:
        print("Before balanced:", len(pair_labels))
        c0_index = []
        c1_index = []
        for i in range(len(pair_labels)):
            if pair_labels[i] == 0:
                c0_index.append(i)
            else:
                c1_index.append(i)

        c0_num = len(c0_index)
        c1_num = len(c1_index)

        if c0_num > c1_num:
            c0_num = min(int(c1_num * balanced), c0_num)
            c0_index = random.sample(c0_index, c0_num)
        else:
            c1_num = min(int(c0_num * balanced), c1_num)
            c1_index = random.sample(c1_index, c1_num)

        print("C 0:", c0_num)
        print("C 1:", c1_num)

        balanced_index = c0_index + c1_index

        balanced_embeddings = {}
        balanced_labels = []
        balanced_repos = []
        for embedding_type in embedding_types:
            balanced_embeddings[embedding_type] = []
        for i in balanced_index:
            balanced_labels.append(pair_labels[i])
            balanced_repos.append(pair_repos[i])
            for embedding_type in embedding_types:
                balanced_embeddings[embedding_type].append(pair_embeddings[embedding_type][i])

        pair_embeddings = balanced_embeddings
        pair_labels = balanced_labels
        pair_repos = balanced_repos
        print("After balanced:", len(pair_labels))

    return pair_embeddings, pair_labels, pair_repos


embedding_types = ['codes_embeddings', 'docs_embeddings', 'structure_embeddings',
                   'readme_embeddings']
train_embeddings, trainLabels, trainRepos = build_embedding_sets("..\\Dataset\\repo_info_train_embeddings_reduce.pkl")
test_embeddings, testLabels, testRepos = build_embedding_sets("..\\Dataset\\repo_info_test_embeddings_reduce.pkl")
valid_embeddings, validLabels, validRepos = build_embedding_sets(
    "..\\Dataset\\repo_info_validation_embeddings_reduce.pkl")

# train_pair_embeddings, train_labels, train_repos = build_pair_data(train_embeddings, trainLabels, embedding_types, trainRepos, balanced=1)
# test_pair_embeddings, test_labels, test_repos = build_pair_data(test_embeddings, testLabels, embedding_types, testRepos, balanced=1)
# valid_pair_embeddings, valid_labels, valid_repos = build_pair_data(valid_embeddings, validLabels, embedding_types, validRepos, balanced=1)


all_embeddings, allLabels, all_repos = mergeEmbedding([train_embeddings, test_embeddings, valid_embeddings],
                                                      [trainLabels, testLabels, validLabels],
                                                      [trainRepos, testRepos, validRepos],
                                                      embedding_types)


all_pair_embeddings, all_labels, all_repos = build_pair_data(all_embeddings, allLabels, embedding_types, all_repos,
                                                             balanced=1.0)

print("Total Size", len(all_labels))
all_dataset = MergePairDataset(all_pair_embeddings, all_labels, all_repos, embedding_types)
dataloader = DataLoader(all_dataset, batch_size=2, shuffle=True, drop_last=False)

data = [["repo1", "repo2", "Similarity(No. 1)", "Similarity(No. 2)"]]

with torch.no_grad():
    no1 = PairClassifier()
    no2 = PairClassifier()
    softmax = nn.Softmax(dim=1)
    no1.load_state_dict(torch.load("..\\mean\\TWINS_MODEL\\Best_Param_2023-07-13 15-35-00.528684.pt"))
    no2.load_state_dict(torch.load("..\\mean\\TWINS_MODEL\\Best_Param_2023-07-24 20-50-00.157130.pt"))
    no1.cuda()
    no2.cuda()
    softmax.cuda()
    for x, _, repos, i in dataloader:
        x = x.cuda()
        p1 = softmax(no1(x))
        p2 = softmax(no2(x))
        data.append([repos[0][0], repos[1][0], round(p1.cpu().numpy()[0][0], 3), round(p2.cpu().numpy()[0][0], 3)])
        data.append([repos[0][1], repos[1][1], round(p1.cpu().numpy()[1][0], 3), round(p2.cpu().numpy()[1][0], 3)])

    with open('SimilarityCal_Evaluation_4484_dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)
