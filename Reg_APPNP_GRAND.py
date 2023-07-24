from torch.autograd import Function
import torch
import torch.nn as nn
from torch.nn import functional as F, Parameter
import dgl.function as fn
from dgl.nn.pytorch.conv import GATv2Conv
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

def MMD(X, Xtest):
    H = torch.exp(- 1e0 * pairwise_distances(X)) + torch.exp(- 1e-1 * pairwise_distances(X)) + torch.exp(
        - 1e-3 * pairwise_distances(X))
    f = torch.exp(- 1e0 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(X, Xtest)) + torch.exp(
        - 1e-3 * pairwise_distances(X, Xtest))
    z = torch.exp(- 1e0 * pairwise_distances(Xtest, Xtest)) + torch.exp(
        - 1e-1 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(Xtest, Xtest))
    MMD_dist = H.mean() - 2 * f.mean() + z.mean()

    return MMD_dist.item()


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

def drop_node(feats, drop_rate, training):
    n = feats.shape[0]
    drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)

    if training:
        masks = torch.bernoulli(1.0 - drop_rates).unsqueeze(1)
        feats = masks.to(feats.device) * feats

    else:
        feats = feats * (1.0 - drop_rate)

    return feats


class MLP(nn.Module):
    def __init__(
        self, nfeat, nhid, nclass, input_droprate, hidden_droprate, use_bn=False
    ):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.input_dropout = nn.Dropout(input_droprate)
        self.hidden_dropout = nn.Dropout(hidden_droprate)
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn

    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()

    def forward(self, x):
        if self.use_bn:
            x = self.bn1(x)
        x = self.input_dropout(x)
        x = F.relu(self.layer1(x))

        if self.use_bn:
            x = self.bn2(x)
        x = self.hidden_dropout(x)
        x = self.layer2(x)

        return x


def GRANDConv(graph, feats, order):
    """
    Parameters
    -----------
    graph: dgl.Graph
        The input graph
    feats: Tensor (n_nodes * feat_dim)
        Node features
    order: int
        Propagation Steps
    """
    with graph.local_scope():
        """Calculate Symmetric normalized adjacency matrix   \hat{A}"""
        degs = graph.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5).to(feats.device).unsqueeze(1)

        graph.ndata["norm"] = norm
        graph.apply_edges(fn.u_mul_v("norm", "norm", "weight"))

        """ Graph Conv """
        x = feats
        y = 0 + feats

        for i in range(order):
            graph.ndata["h"] = x
            graph.update_all(fn.u_mul_e("h", "weight", "m"), fn.sum("m", "h"))
            x = graph.ndata.pop("h")
            y.add_(x)

    return y / (order + 1)


class GRAND(nn.Module):
    r"""

    Parameters
    -----------
    in_dim: int
        Input feature size. i.e, the number of dimensions of: math: `H^{(i)}`.
    hid_dim: int
        Hidden feature size.
    n_class: int
        Number of classes.
    S: int
        Number of Augmentation samples
    K: int
        Number of Propagation Steps
    node_dropout: float
        Dropout rate on node features.
    input_dropout: float
        Dropout rate of the input layer of a MLP
    hidden_dropout: float
        Dropout rate of the hidden layer of a MLPx
    batchnorm: bool, optional
        If True, use batch normalization.

    """

    def __init__(
        self,
        in_dim,
        hid_dim,
        n_class,
        S=4,
        K=8,
        node_dropout=0.5,
        input_droprate=0.5,
        hidden_droprate=0.5,
        batchnorm=False,
    ):
        super(GRAND, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.S = S
        self.K = K
        self.n_class = n_class

        self.mlp = MLP(
            in_dim, hid_dim, n_class, input_droprate, hidden_droprate, batchnorm
        )

        self.dropout = node_dropout
        self.node_dropout = nn.Dropout(node_dropout)

    def forward(self, graph, feats, training=True):
        X = feats
        S = self.S

        if training:  # Training Mode
            output_list = []
            h = []
            for s in range(S):
                drop_feat = drop_node(X, self.dropout, True)  # Drop node
                feat = GRANDConv(graph, drop_feat, self.K)  # Graph Convolution
                h.append(feat)
                output_list.append(
                    torch.log_softmax(self.mlp(feat), dim=-1)
                )  # Prediction
            self.h=h
            return output_list
        else:  # Inference Mode
            drop_feat = drop_node(X, self.dropout, False)
            X = GRANDConv(graph, drop_feat, self.K)
            return torch.log_softmax(self.mlp(X), dim=-1)


    def dann_output(self, idx_train, iid_train, alpha=1):
        reverse_feature = ReverseLayerF.apply(self.h, alpha)
        dann_loss = xent(self.disc(self.g, reverse_feature)[idx_train, :],
                         torch.ones_like(labels[idx_train])).mean() + xent(
            self.disc(self.g, reverse_feature)[iid_train, :], torch.zeros_like(labels[iid_train])).mean()
        return dann_loss

    def shift_robust_output(self, idx_train, iid_train, alpha=1):
        cmd_s = 0
        for k in range(self.S):
            cmd_s += cmd(self.h[k][idx_train, :], self.h[k][iid_train, :])
        return alpha*cmd_s / self.S
        # return alpha * cmd(self.h[idx_train, :], self.h[iid_train, :])

    def MMD_output(self, idx_train, iid_train, alpha=1):
        return alpha * MMD(self.h[idx_train, :], self.h[iid_train, :])
    
    
def consis_loss(logps, lam=1, temp=0.5):
    ps = [torch.exp(p) for p in logps]
    ps = torch.stack(ps, dim=2)

    avg_p = torch.mean(ps, dim=2)
    sharp_p = (
        torch.pow(avg_p, 1.0 / temp)
        / torch.sum(torch.pow(avg_p, 1.0 / temp), dim=1, keepdim=True)
    ).detach()

    sharp_p = sharp_p.unsqueeze(2)
    loss = torch.mean(torch.sum(torch.pow(ps - sharp_p, 2), dim=1, keepdim=True))

    loss = lam * loss
    return loss
    
    


if __name__ == '__main__':
    DATASET = 'cora'
    EPOCH = 100
    S = 4
    # option of 'SRGNN' and None
    METHOD = 'SRGNN'
    # option of 'Baised' and None
    TRAININGSET = 'Baised'
    # device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    adj, features, one_hot_labels, ori_idx_train, idx_val, ori_idx_test = utils.load_data(DATASET)
    nx_g = nx.Graph(adj + sp.eye(adj.shape[0]))
    g = dgl.from_networkx(nx_g).to(device)
    labels = torch.LongTensor([np.where(r == 1)[0][0] if r.sum() > 0 else -1 for r in one_hot_labels]).to(device)
    features = torch.FloatTensor(utils.preprocess_features(features)).to(device)
    xent = nn.CrossEntropyLoss(reduction='none')

    model = GRAND(in_dim=features.shape[1], hid_dim=32, n_class=labels.max().item() + 1)
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
        loss_sup = 0
        logits = model(g, features, True)
        #logits = F.log_softmax(logits1, 1)
        if TRAININGSET == 'Baised':
            for k in range(S):
                loss_sup += xent(logits[k][idx_train], labels[idx_train])
            loss_sup = loss_sup /S
            loss_consis = consis_loss(logits)

            loss = loss_sup + loss_consis
            all_idx = set(range(g.number_of_nodes())) - set(idx_train)
            idx_test = torch.LongTensor(list(all_idx))
        else:
            for k in range(S):
                loss_sup += xent(logits[k][iid_train], labels[iid_train])
            loss_sup = loss_sup /S
            loss_consis = consis_loss(logits)

            loss = loss_sup + loss_consis
            # loss = xent(logits[iid_train], labels[iid_train])
            all_idx = set(range(g.number_of_nodes())) - set(iid_train)
            idx_test = torch.LongTensor(list(all_idx))
        if METHOD == 'SRGNN':
            # regularizer only: loss = loss.mean() + model.shift_robust_output(idx_train, iid_train)
            # instance-reweighting only: loss = (torch.Tensor(kmm_weight).reshape(-1).cuda() * (loss)).mean()
            #loss = (torch.Tensor(kmm_weight).reshape(-1) * (loss)).mean() + model.shift_robust_output(idx_train,
            #                                                                                          iid_train)
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
    embeds = model(g, features, False).detach()
    logits = embeds[idx_test]
    preds_all = torch.argmax(embeds, dim=1)

    # print("Accuracy:{}".format(f1_score(labels[idx_test].cpu(), preds_all[idx_test].cpu(), average='micro')))
    
    filename = "GRAND_cmd_rslt.txt"
    # Open the file in append mode
    with open("GRAND_cmd_rslt.txt", "a") as file:
        for run in range(1):
            # Generate the result for each run (dummy example)
            result = format(f1_score(labels[idx_test].cpu(), preds_all[idx_test].cpu(), average='micro'))

            # Write the result to the file
            file.write(result + "\n")

    appnp_cmd = np.loadtxt("GRAND_cmd_rslt.txt")

    data = np.array(appnp_cmd)
    standard_error = np.std(data) / np.sqrt(len(data))
    print("Average accuracy:", data.mean(), "Standard Error:", standard_error, "n_run:", len(data))

