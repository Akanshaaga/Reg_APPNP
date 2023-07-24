from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv
import utils
import numpy as np
import pickle
import networkx as nx
import scipy.sparse as sp
import dgl
from sklearn.metrics import f1_score


def KMM(X, Xtest, _A=None, _sigma=1e1, beta=0.2):
    H = torch.exp(- 1e0 * pairwise_distances(X)) + torch.exp(- 1e-1 * pairwise_distances(X)) + torch.exp(
        - 1e-3 * pairwise_distances(X))
    f = torch.exp(- 1e0 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(X, Xtest)) + torch.exp(
        - 1e-3 * pairwise_distances(X, Xtest))
    z = torch.exp(- 1e0 * pairwise_distances(Xtest, Xtest)) + torch.exp(
        - 1e-1 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(Xtest, Xtest))
    # H /= 3
    # f /= 3
    # z /= 3
    MMD_dist = H.mean() - 2 * f.mean() + z.mean()

    nsamples = X.shape[0]
    f = - X.shape[0] / Xtest.shape[0] * f.matmul(torch.ones((Xtest.shape[0], 1)))
    G = - np.eye(nsamples)
    _A = _A[~np.all(_A == 0, axis=1)]
    b = _A.sum(1)
    h = - beta * np.ones((nsamples, 1))
    #print(torch.linalg.matrix_rank())
    '''
    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False
    sol = solvers.qp(matrix(H.numpy().astype(np.double)), matrix(f.numpy().astype(np.double)), matrix(G), matrix(h),
                     matrix(_A), matrix(b))
    return np.array(sol['x']), MMD_dist.item()
    '''
    return MMD_dist.item()


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def cmd(X, X_test, K=5):
    """
    central moment discrepancy (cmd)
    objective function for keras models (theano or tensorflow backend)

    - Zellinger, Werner, et al. "Robust unsupervised domain adaptation for
    neural networks via moment alignment.", TODO
    - Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning.", ICLR, 2017.
    """
    x1 = X
    x2 = X_test
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = l2diff(mx1, mx2)
    scms = [dm]
    for i in range(K - 1):
        # moment diff of centralized samples
        scms.append(moment_diff(sx1, sx2, i + 2))
        # scms+=moment_diff(sx1,sx2,1)
    return sum(scms)

def MMD(X, Xtest):
    H = torch.exp(- 1e0 * pairwise_distances(X)) + torch.exp(- 1e-1 * pairwise_distances(X)) + torch.exp(
        - 1e-3 * pairwise_distances(X))
    f = torch.exp(- 1e0 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(X, Xtest)) + torch.exp(
        - 1e-3 * pairwise_distances(X, Xtest))
    z = torch.exp(- 1e0 * pairwise_distances(Xtest, Xtest)) + torch.exp(
        - 1e-1 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(Xtest, Xtest))
    MMD_dist = H.mean() - 2 * f.mean() + z.mean()

    return MMD_dist.item()


def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    return (x1 - x2).norm(p=2)


def moment_diff(sx1, sx2, k):
    """
    difference between moments
    """
    ss1 = sx1.pow(k).mean(0)
    ss2 = sx2.pow(k).mean(0)
    # ss1 = sx1.mean(0)
    # ss2 = sx2.mean(0)
    return l2diff(ss1, ss2)


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class GraphSAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GraphSAGE-mean
        self.layers.append(SAGEConv(in_size, hid_size, "gcn"))
        self.layers.append(SAGEConv(hid_size, out_size, "gcn"))
        self.dropout = nn.Dropout(0.5)

    def forward(self, graph, x):
        h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        self.h= h
        return h

    def output(self, features):
        h = features
        for layer in self.layers[:-1]:
            h = layer(self.g, h)
        return h

    def dann_output(self, idx_train, iid_train, alpha=1):
        reverse_feature = ReverseLayerF.apply(self.h, alpha)
        dann_loss = xent(self.disc(self.g, reverse_feature)[idx_train, :],
                         torch.ones_like(labels[idx_train])).mean() + xent(
            self.disc(self.g, reverse_feature)[iid_train, :], torch.zeros_like(labels[iid_train])).mean()
        return dann_loss

    def shift_robust_output(self, idx_train, iid_train, alpha=1):
        return alpha * cmd(self.h[idx_train, :], self.h[iid_train, :])

    def MMD_output(self, idx_train, iid_train, alpha=1):
        return alpha * MMD(self.h[idx_train, :], self.h[iid_train, :])


if __name__ == '__main__':
    DATASET = 'cora'
    EPOCH = 200
    # option of 'SRGNN' and None
    METHOD = 'SRGNN'
    # option of 'Baised' and None
    TRAININGSET = 'Baised'
    # option of 'gcn', 'mean' or 'pool'
    agg_type = 'gcn'
    # device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    adj, features, one_hot_labels, ori_idx_train, idx_val, ori_idx_test = utils.load_data(DATASET)
    nx_g = nx.Graph(adj + sp.eye(adj.shape[0]))
    g = dgl.from_networkx(nx_g).to(device)
    labels = torch.LongTensor([np.where(r == 1)[0][0] if r.sum() > 0 else -1 for r in one_hot_labels]).to(device)
    features = torch.FloatTensor(utils.preprocess_features(features)).to(device)
    xent = nn.CrossEntropyLoss(reduction='none')

    model = GraphSAGE(features.shape[1], 16, labels.max().item() + 1)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
    # an example of biased training data
    idx_train = torch.LongTensor(pickle.load(open('data/localized_seeds_{}.p'.format(DATASET), 'rb'))[0])
    # print(idx_train[:50])
    if False:
        # generating unbaised data (it create fluctuation in the accuracy)
        all_idx = set(range(g.number_of_nodes())) - set(idx_train)
        idx_test = torch.LongTensor(list(all_idx))
        perm = torch.randperm(idx_test.shape[0])
        iid_train = idx_test[perm[:idx_train.shape[0]]]
        pickle.dump({'iid_train': iid_train}, open('data_iid_train/{}_dump.p'.format(DATASET), 'wb'))
    else:
        # using already generated iid_train for stable results
        #train_dump = pickle.load(open('data_iid_train/{}_dump.p'.format(DATASET), 'rb'))
        #iid_train = train_dump['iid_train']
        iid_train, _, _, _, _, _ = utils.createDBLPTraining(one_hot_labels, ori_idx_train, idx_val, ori_idx_test,
                                                            max_train=20)
    Z_train = torch.FloatTensor(adj[idx_train.tolist(), :].todense())
    Z_test = torch.FloatTensor(adj[iid_train, :].todense())
    # Z_train = torch.FloatTensor(features[idx_train.tolist(), :])
    # Z_test = torch.FloatTensor(features[iid_train.tolist(), :])
    # embed()
    label_balance_constraints = np.zeros((labels.max().item() + 1, len(idx_train)))
    for i, idx in enumerate(idx_train):
        label_balance_constraints[labels[idx], i] = 1
    # embed()
    #kmm_weight, MMD_dist = KMM(Z_train, Z_test, label_balance_constraints, beta=0.2)
    MMD_dist = KMM(Z_train, Z_test, label_balance_constraints, beta=0.2)
    #print(kmm_weight.max(), kmm_weight.min())
    for epoch in range(EPOCH):
        model.train()
        optimiser.zero_grad()
        logits = model(g, features)
        #logits = F.log_softmax(logits1, 1)
        if TRAININGSET == 'Baised':
            loss = xent(logits[idx_train], labels[idx_train])
            all_idx = set(range(g.number_of_nodes())) - set(idx_train)
            idx_test = torch.LongTensor(list(all_idx))
        else:
            loss = xent(logits[iid_train], labels[iid_train])
            all_idx = set(range(g.number_of_nodes())) - set(iid_train)
            idx_test = torch.LongTensor(list(all_idx))
        if METHOD == 'SRGNN':
            # regularizer only: loss = loss.mean() + model.shift_robust_output(idx_train, iid_train)
            # instance-reweighting only: loss = (torch.Tensor(kmm_weight).reshape(-1).cuda() * (loss)).mean()
            # loss = (torch.Tensor(kmm_weight).reshape(-1) * (loss)).mean() + model.shift_robust_output(idx_train,
            #                                                                                           iid_train)
            loss = loss.mean() + model.shift_robust_output(idx_train, iid_train)
        elif METHOD == 'DANN':
            loss = loss.mean() + model.dann_output(idx_train, iid_train) + model.shift_robust_output(idx_train,
                                                                                                     iid_train) + (
                               torch.Tensor(kmm_weight).reshape(-1) * (loss)).mean()
        elif METHOD == 'MMD':
            loss = loss.mean() + model.MMD_output(idx_train, iid_train)
        elif METHOD is None:
            loss = loss.mean()
        loss.backward()
        optimiser.step()

    model.eval()
    embeds = model(g, features).detach()
    logits = embeds[idx_test]
    preds_all = torch.argmax(embeds, dim=1)

    print("Accuracy:{}".format(f1_score(labels[idx_test].cpu(), preds_all[idx_test].cpu(), average='micro')))
    filename = "graphsage_cmd_rslt.txt"
    # Open the file in append mode
    with open("graphsage_cmd_rslt.txt", "a") as file:
        for run in range(1):
            # Generate the result for each run (dummy example)
            result = format(f1_score(labels[idx_test].cpu(), preds_all[idx_test].cpu(), average='micro'))

            # Write the result to the file
            file.write(result + "\n")

    appnp_cmd = np.loadtxt("graphsage_cmd_rslt.txt")

    data = np.array(appnp_cmd)
    standard_error = np.std(data) / np.sqrt(len(data))
    print("Average accuracy:", data.mean(), "Standard Error:", standard_error, "n_run:", len(data))
