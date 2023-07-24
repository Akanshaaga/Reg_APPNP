from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import APPNPConv
import utils
import numpy as np
import pickle
import networkx as nx
import scipy.sparse as sp
import dgl
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn import manifold


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

def MMD(X, Xtest):
    H = torch.exp(- 1e0 * pairwise_distances(X)) + torch.exp(- 1e-1 * pairwise_distances(X)) + torch.exp(
        - 1e-3 * pairwise_distances(X))
    f = torch.exp(- 1e0 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(X, Xtest)) + torch.exp(
        - 1e-3 * pairwise_distances(X, Xtest))
    z = torch.exp(- 1e0 * pairwise_distances(Xtest, Xtest)) + torch.exp(
        - 1e-1 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(Xtest, Xtest))
    MMD_dist = H.mean() - 2 * f.mean() + z.mean()

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


def calc_ppr_exact(adj_matrix: sp.spmatrix, alpha: float) -> np.ndarray:
    nnodes = adj_matrix.shape[0]
    M = calc_A_hat(adj_matrix)
    A_inner = sp.eye(nnodes) - (1 - alpha) * M
    return alpha * np.linalg.inv(A_inner.toarray())


def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    A = adj_matrix + sp.eye(nnodes)
    D_vec = np.sum(A, axis=1)
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
    return D_invsqrt_corr @ A @ D_invsqrt_corr

def t_SNE(output, dimention):
    # output:待降维的数据
    # dimention：降低到的维度
    tsne = manifold.TSNE(n_components=dimention, init='pca', random_state=0)
    result = tsne.fit_transform(output)
    return result


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class APPNP(nn.Module):
    def __init__(
        self,
        g,
        in_feats,
        hiddens,
        n_classes,
        activation,
        feat_drop,
        edge_drop,
        alpha,
        k,
    ):
        super(APPNP, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.feat_drop = nn.Dropout(feat_drop)
        self.edge_drop = nn.Dropout(edge_drop)
        # input layer
        self.layers.append(nn.Linear(in_feats, hiddens[0]))
        # hidden layers
        for i in range(1, len(hiddens)):
            self.layers.append(nn.Linear(hiddens[i - 1], hiddens[i]))
        # output layer
        self.layers.append(nn.Linear(hiddens[-1], n_classes))
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, features):
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        self.h = h
        # propagation step
        h = self.propagate(self.g, h)
        return h

    def output(self, g, features):
        h = features
        for idx in range(len(self.layers)-1):
            h = self.layers[idx](g, h).flatten(1)
        return self.layers[-1](g, h).mean(1)

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
    DATASET = 'pubmed'
    EPOCH = 100
    # option of 'SRGNN' and None
    METHOD = 'SRGNN'
    # option of 'Baised' and None
    TRAININGSET = 'Baised'
    # biasing parameter alpha(I-(1-alpha)\hat(A))^{-1}
    alpha = .05
    # device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    adj, features, one_hot_labels, ori_idx_train, idx_val, ori_idx_test = utils.load_data(DATASET)
    nx_g = nx.Graph(adj + sp.eye(adj.shape[0]))
    g = dgl.from_networkx(nx_g).to(device)
    labels = torch.LongTensor([np.where(r == 1)[0][0] if r.sum() > 0 else -1 for r in one_hot_labels]).to(device)
    features = torch.FloatTensor(utils.preprocess_features(features)).to(device)
    xent = nn.CrossEntropyLoss(reduction='none')

    # Generating PPR matrix for biased data generation
    # ppr_vector = torch.FloatTensor(calc_ppr_exact(adj, alpha))
    # ppr_dist = pairwise_distances(ppr_vector)

    train_dump = pickle.load(open('intermediate/{}_dump.p'.format(DATASET), 'rb'))
    ppr_vector = train_dump['ppr_vector']
    ppr_dist = train_dump['ppr_dist']

    model = APPNP(g, features.shape[1], [64], labels.max().item() + 1, F.relu, feat_drop=0.5,
                  edge_drop=0.5, alpha=0.1, k=8)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

    n_repeats = 5
    max_train = 20
    # an example of biased training data
    for _run in range(n_repeats):
        # biased training data
        # generate biased sample
        if True:
            train_seeds, _, _, _, _, _ = utils.createDBLPTraining(one_hot_labels, ori_idx_train, idx_val,
                                                                      ori_idx_test, max_train=1)
            label_idx = []
            if DATASET == 'pubmed':
                num_pool = 10000
            elif DATASET == 'cora':
                num_pool = 1500
            else:
                num_pool = 1000
            for i in train_seeds:
                label_idx.append(torch.where(labels[:num_pool] == labels[i])[0])
            ppr_init = {}
            for i in train_seeds:
                ppr_init[i] = 1
            # print(train_seeds)
            idx_train = []
            for idx in range(len(train_seeds)):
                idx_train += label_idx[idx][
                    ppr_dist[train_seeds[idx], label_idx[idx]].argsort()[:max_train]].tolist()

    # idx_train = torch.LongTensor(pickle.load(open('data/localized_seeds_{}.p'.format(DATASET), 'rb'))[0])
    if False:
        # generating unbaised data (it create fluctuation in the accuracy)
        all_idx = set(range(g.number_of_nodes())) - set(idx_train)
        idx_test = torch.LongTensor(list(all_idx))
        perm = torch.randperm(idx_test.shape[0])
        iid_train = idx_test[perm[:idx_train.shape[0]]]
        pickle.dump({'iid_train': iid_train}, open('data_iid_train/{}_dump.p'.format(DATASET), 'wb'))
    else:
        # using already generated iid_train for stable results
        # train_dump = pickle.load(open('data_iid_train/{}_dump.p'.format(DATASET), 'rb'))
        # iid_train = train_dump['iid_train']
        iid_train, _, _, _, _, _ = utils.createDBLPTraining(one_hot_labels, ori_idx_train, idx_val, ori_idx_test,
                                                            max_train=20)
    label_balance_constraints = np.zeros((labels.max().item() + 1, len(idx_train)))
    for i, idx in enumerate(idx_train):
        label_balance_constraints[labels[idx], i] = 1
    # embed()
    #kmm_weight, MMD_dist = KMM(Z_train, Z_test, label_balance_constraints, beta=0.2)
    # MMD_dist = KMM(Z_train, Z_test, label_balance_constraints, beta=0.2)
    # MMD_dist = KMM(self.h[idx_train, :], self.h[iid_train, :], label_balance_constraints, beta=0.2)
    #print(kmm_weight.max(), kmm_weight.min())

    for epoch in range(EPOCH):
        model.train()
        optimiser.zero_grad()
        logits = model(features)
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
            #loss = (torch.Tensor(kmm_weight).reshape(-1) * (loss)).mean() + model.shift_robust_output(idx_train,
            #                                                                                          iid_train)
            loss = loss.mean() + 0.1*model.shift_robust_output(idx_train, iid_train) \
                   + 0.1*model.MMD_output(idx_train, iid_train)
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
    embeds = model(features).detach()
    # pre = t_SNE(embeds[idx_test], 2)
    # tsne = pre.fit_transform(model(features))
    # cmap = plt.cm.get_cmap('tab10', labels.max().item() + 1)
    # node_colors = [cmap(label) for label in range(labels.max().item() + 1)]
    # plt.figure(figsize=(8, 6))
    #
    # Step 4: Visualize the t-SNE embeddings
    # plt.scatter(tsne[:, 0], tsne[:, 1], c=node_colors, s=100)
    # plt.axis('off')  # Hide the axis
    # plt.title("t-SNE Visualization of GCN Output Representation with Node Colors")
    # plt.show()
    logits = embeds[idx_test]
    preds_all = torch.argmax(embeds, dim=1)

    print("Accuracy:{}".format(f1_score(labels[idx_test].cpu(), preds_all[idx_test].cpu(), average='micro')))
    # print(format(f1_score(labels[idx_test].cpu(), preds_all[idx_test].cpu(), average='micro')))

    filename = "appnp_cmd_pubmed.txt"
    # Open the file in append mode
    with open("appnp_cmd_pubmed.txt", "a") as file:
        for run in range(1):
            # Generate the result for each run (dummy example)
            result = format(f1_score(labels[idx_test].cpu(), preds_all[idx_test].cpu(), average='micro'))

            # Write the result to the file
            file.write(result + "\n")

    appnp_cmd = np.loadtxt("appnp_cmd_pubmed.txt")

    data = np.array(appnp_cmd)
    standard_error = np.std(data) / np.sqrt(len(data))
    print("Average accuracy:", data.mean(), "Standard Error:", standard_error, "n_run:", len(data))
    '''
    filename = "cmd_cora_Bias(wot_cmd).txt"
    # Open the file in append mode
    with open("cmd_cora_Bias(wot_cmd).txt", "a") as file:
        for run in range(1):
            # Generate the result for each run (dummy example)
            bias = cmd(ppr_vector[idx_train, :], ppr_vector[iid_train, :])

            # Write the result to the file
            file.write(str(bias.item()) + '\n')

    appnp_cmd = np.loadtxt("cmd_cora_Bias(wot_cmd).txt")
    '''


