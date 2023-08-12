import argparse
import pickle
from itertools import combinations
from pathlib import Path

import pandas as pd
import torch
from torch import nn
"""
-t set the strategy
-i set the input
-m set the model
-o set the output directory
python calculateSim.py -t sequential -i output.pkl -m "E:\SimilarityCal\sequential\TWINS_MODEL\Best_Param_2023-07-24 16-54-35.596111.pt" -o sequential
"""

def mean_similarity(repo1, repo2, net):
    item = []
    matrix1 = []
    matrix2 = []
    net = net.cpu()
    with torch.no_grad():
        matrix1.append(torch.Tensor(repo1['mean_repo_embedding']).squeeze())
        matrix2.append(torch.Tensor(repo2['mean_repo_embedding']).squeeze())
        item.append(torch.cat(matrix1 + matrix2))
        item.append(torch.cat(matrix2 + matrix1))
        items = torch.stack(item)
        res = net(items)
        softmax = nn.Softmax(dim=1)
        similarity = torch.mean(softmax(res), dim=0)[1].cpu().numpy()
        dissimilarity = torch.mean(softmax(res), dim=0)[0].cpu().numpy()
    return similarity, dissimilarity


def sequential_embeddings(embeddings: torch.Tensor, length=50) -> torch.Tensor:
    if len(embeddings.shape) == 1:
        embeddings = torch.stack([embeddings])
    le = embeddings.shape[0]
    if le < length:
        zeros = torch.zeros(length - le, embeddings.shape[1])
        embeddings = torch.cat([embeddings, zeros])
    elif le > length:
        embeddings = embeddings[:length, :]
    return embeddings


def sequential_similarity(repo1, repo2, net):
    embedding_types = ['code_embeddings', 'doc_embeddings', 'requirement_embeddings', 'readme_embeddings']
    input_data = [[], [], [], []]
    matrix = [[], []]
    for t in embedding_types:
        matrix[0].append(sequential_embeddings(torch.Tensor(repo1[t])))
        matrix[1].append(sequential_embeddings(torch.Tensor(repo2[t])))
    states = net.begin_state("cpu", batch_size=2)

    with torch.no_grad():
        for j in range(4):
            input_data[j] = []
            input_data[j].append(torch.stack([matrix[0][j], matrix[1][j]]))
            input_data[j].append(torch.stack([matrix[1][j], matrix[0][j]]))
        res, _ = net(input_data, states)
        softmax = nn.Softmax(dim=1)
        similarity = torch.mean(softmax(res), dim=0)[1].cpu().numpy()
        dissimilarity = torch.mean(softmax(res), dim=0)[0].cpu().numpy()
    return similarity, dissimilarity


# mean model structure
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


# sequential model
class StatesGRU(nn.Module):
    def __init__(self, num_hiddens, num_layers, dropout, size=768):
        super().__init__()
        self.net = nn.GRU(size, num_hiddens, num_layers, dropout=dropout, bidirectional=True)

    def forward(self, inputs, states):
        inputs = inputs.permute(1, 0, 2)
        states, last_state = self.net(inputs, states)
        return states, last_state


class MergeClassifier(nn.Module):

    def __init__(self, num_hiddens, num_layers, dropout, size=768 * 4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.gru = StatesGRU(num_hiddens, num_layers, dropout, size=size)
        self.classifier = nn.Sequential(
            nn.Linear(num_hiddens * 2 * 100, num_hiddens * 2),
            nn.ReLU(),
            nn.BatchNorm1d(num_hiddens * 2),
            nn.Linear(num_hiddens * 2, 2),
        )

    def forward(self, input, states):
        e1 = []
        e2 = []
        for i in range(4):
            e1.append(input[i][0])
            e2.append(input[i][1])
        e1 = torch.cat(e1, dim=2)
        e2 = torch.cat(e2, dim=2)
        s1, last_states1 = self.gru(e1, states[0])
        s2, last_states2 = self.gru(e2, states[1])
        enc_output = torch.cat([s1, s2]).permute(1, 0, 2)
        enc_output = enc_output.reshape(enc_output.shape[0], -1)
        res = self.classifier(enc_output)
        return res, [last_states1, last_states2]

    def begin_state(self, device, batch_size=1):
        init_state = torch.zeros((2 * self.num_layers,
                                  batch_size, self.num_hiddens),
                                 device=device)
        return [init_state, init_state.clone().requires_grad_()]


def main():
    """
    The main method for running this program.
    :return: None
    """

    # Arguments rules
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-t",
        "--type",
        help="The type of strategy (mean or sequential)",
        required=True
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Input repository embeddings files",
        required=True,
    )
    parser.add_argument("-o", "--output", help="Output directory", required=True)
    parser.add_argument(
        "-m",
        "--model",
        help="Model used", required=True
    )
    args = parser.parse_args()

    # Building the output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Loading model
    similarity = None
    if args.type == "mean":
        model = PairClassifier()
        similarity = mean_similarity
    elif args.type == "sequential":
        model = MergeClassifier(256, 2, 0.2, size=768 * 4)
        similarity = sequential_similarity

    model_name = args.model
    model.load_state_dict(torch.load(model_name))
    model.eval()

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    rows_list = []
    for info1, info2 in combinations(data, 2):
        rows_list.append(
            {
                "repo1": info1["name"],
                "repo2": info2["name"],
                "topics1": info1["topics"],
                "topics2": info2["topics"],
                "model": args.model,
                "strategy": args.type,
                "repo_sim": float(similarity(
                    info1, info2, model
                )[0]),
            })

        # Saving the calculation result
        df = pd.DataFrame(rows_list)
        df = df.sort_values("repo_sim", ascending=False).reset_index(drop=True)
        df.to_csv(output_dir / "evaluation_result.csv", index=False)
        print(f"[+] Evaluation results saved to {output_dir}/evaluation_result.csv!")


if __name__ == "__main__":
    main()
