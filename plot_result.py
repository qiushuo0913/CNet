import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from constraint_experiment import test_model
import torch
from constraint_experiment import load_variable
from constraint_experiment import cal_sum_rate
from matplotlib.pyplot import MultipleLocator
import scipy.io as scio
from torch.utils.data import DataLoader
import time
from constraint_experiment import genete_B
from constraint_experiment import H_to_V
import seaborn as sns
from scipy.stats import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from brokenaxes import brokenaxes
from matplotlib.patches import ConnectionPatch


"""
    when you use plot_result to plot, you need to Comment out the main program
"""

Pmax = xxx
sigma = xxx
user = xxx
users1 = xxx
users2 = xxx
# if you want to plot the CDF, you need to set batch_size as 1
params = {'batch_size': 20, 'n_tests': 1}
# lamda = np.array([0.1, 0.2, 0.3, 0.4])
lamda = np.array([0.1, 0.2, 0.3, 0.4])
# restore the require rate, q, gama
require_rate = []
require_rate1 = []
require_rate2 = []
qq = []
qq1 = []
qq2 = []
gamama = []
gamama1 = []
gamama2 = []

feasible_1 = np.zeros(np.size(lamda))
feasible_2 = np.zeros(np.size(lamda))

sum_rate = np.zeros(np.size(lamda))
sum_rate_gener = np.zeros(np.size(lamda))
sum_rate_1 = np.zeros(np.size(lamda))
sum_rate_1_gener = np.zeros(np.size(lamda))
sum_rate_2 = np.zeros(np.size(lamda))
sum_rate_2_gener = np.zeros(np.size(lamda))

for i in range(np.size(lamda)):
    """
        req_rate , req_rate_1 represent the minimum rate requirement in different K
    """
    req_rate = ((torch.ones(user) * lamda[i]).view(1, user, 1)) * torch.ones(params['batch_size'], user, 1)
    req_rate_1 = ((torch.ones(users1) * lamda[i]).view(1, users1, 1)) * torch.ones(params['batch_size'], users1, 1)
    req_rate_2 = ((torch.ones(users2) * lamda[i]).view(1, users2, 1)) * torch.ones(params['batch_size'], users2, 1)
    q = (torch.pow(2.0, req_rate) - 1) * sigma
    q1 = (torch.pow(2.0, req_rate_1) - 1) * sigma
    q2 = (torch.pow(2.0, req_rate_2) - 1) * sigma
    gama = (torch.pow(2.0, req_rate) - 1)
    gama1 = (torch.pow(2.0, req_rate_1) - 1)
    gama2 = (torch.pow(2.0, req_rate_2) - 1)
    require_rate.append(req_rate)
    require_rate1.append(req_rate_1)
    require_rate2.append(req_rate_2)
    qq.append(q)
    qq1.append(q1)
    qq2.append(q2)
    gamama.append(gama)
    gamama1.append(gama1)
    gamama2.append(gama2)

# load testing data with different K
# H = load_variable('data_test_2.txt')
# H = torch.tensor(H)
# H1 = load_variable('data_test_1.txt')
# H1 = torch.tensor(H1)
# H2 = load_variable('data_test_3.txt')
# H2 = torch.tensor(H2)

H = load_variable('xxx_1.txt')
H = torch.tensor(H)
H1 = load_variable('xxx_2.txt')
H1 = torch.tensor(H1)
H2 = load_variable('xxx_3.txt')
H2 = torch.tensor(H2)

loss_fn = cal_sum_rate

# load model under different K and beta
# net1 = torch.load('constraint_network_0.1_2.pt')
# net2 = torch.load('constraint_network_0.2_2.pt')
# net3 = torch.load('constraint_network_0.3_2.pt')
# net4 = torch.load('constraint_network_0.4_2.pt')
# net = [net1, net2, net3, net4]
#
# net11 = torch.load('constraint_network_0.1.pt')
# net22 = torch.load('constraint_network_0.2.pt')
# net33 = torch.load('constraint_network_0.3.pt')
# net44 = torch.load('constraint_network_0.4.pt')
# nett = [net11, net22, net33, net44]
#
# net111 = torch.load('constraint_network_4_0.1_16.pt')
# net222 = torch.load('constraint_network_4_0.2_16.pt')
# net333 = torch.load('constraint_network_4_0.3_16.pt')
# net444 = torch.load('constraint_network_4_0.4_16.pt')
# nettt = [net111, net222, net333, net444]

net1 = torch.load('xxx_0.1.pt')
net2 = torch.load('xxx_0.2.pt')
net3 = torch.load('xxx_0.3.pt')
net4 = torch.load('xxx_0.4.pt')
net = [net1, net2, net3, net4]

net11 = torch.load('xxx_0.1.pt')
net22 = torch.load('xxx_0.2.pt')
net33 = torch.load('xxx_0.3.pt')
net44 = torch.load('xxx_0.4.pt')
nett = [net11, net22, net33, net44]

net111 = torch.load('xxx_0.1.pt')
net222 = torch.load('xxx_0.2.pt')
net333 = torch.load('xxx_0.3.pt')
net444 = torch.load('xxx_0.4.pt')
nettt = [net111, net222, net333, net444]

# def lambda_mismatch():
#     """
#        this function is used to plot the probability of constraints satisfaction in different cells K
#        in this function, test_model should return "d"
#     """
#     for k in range(np.size(lamda)):
#         feasible_1[k] = test_model(nett[0], loss_fn, params, H1, qq1[k], require_rate1[k], gamama1[k], users)
#         feasible_2[k] = test_model(nett[k], loss_fn, params, H1, qq1[k], require_rate1[k], gamama1[k], users)
#         feasible_1[k] = 1 - feasible_1[k]
#         feasible_2[k] = 1 - feasible_2[k]
#     plt.plot(lamda, feasible_2, ls='-', lw=2, label='CNet-β', color='red')
#     plt.plot(lamda, feasible_1, ls='--', lw=2, label='CNet-0.1', color='green')
#     plt.scatter(lamda, feasible_1, s=80, c='green', marker='*', alpha=0.5)
#     plt.scatter(lamda, feasible_2, s=80, c='red', marker='^', alpha=0.5)
#     plt.figure(num=1)
#     plt.legend()
#     plt.xlabel('β(bits/s/Hz)')
#     plt.ylabel('probability of constraint satisfaction')
#     y_major_locator = MultipleLocator(0.1)
#     ax = plt.gca()
#     ax.yaxis.set_major_locator(y_major_locator)
#     plt.ylim(0, 1.05)
#     plt.grid(ls='--')
#     plt.show()


# def plot_performance():
#     """
#            this function is used to plot the sum rate performance in different cells K, including plot the generalization
#            in this function, test_model should return "p"
#     """
#     for j in range(np.size(lamda)):
#         sum_rate[j] = test_model(net[j], loss_fn, params, H, qq[j], require_rate[j], gamama[j], user)
#         sum_rate_gener[j] = test_model(net[0], loss_fn, params, H, qq[j], require_rate[j], gamama[j], user)
#         sum_rate_1[j] = test_model(nettt[j], loss_fn, params, H1, qq2[j], require_rate2[j], gamama2[j], users2)
#         sum_rate_1_gener[j] = test_model(nettt[0], loss_fn, params, H1, qq2[j], require_rate2[j], gamama2[j], users2)
#         sum_rate_2[j] = test_model(nett[j], loss_fn, params, H2, qq1[j], require_rate1[j], gamama1[j], users1)
#         sum_rate_2_gener[j] = test_model(nett[0], loss_fn, params, H2, qq1[j], require_rate1[j], gamama1[j], users1)
#     # plt.xticks(fontsize=13)
#     # plt.yticks(fontsize=13)
#     # bax = brokenaxes(ylims=((17.6, 20.6), (24.6, 26.6)))
#     #
#     # bax.plot(lamda, sum_rate, ls='-', lw=2, label='CNet-β k=2', color='red')
#     # bax.scatter(lamda, sum_rate, s=60, c='red', marker='^', alpha=0.5)
#     # bax.plot(lamda, sum_rate_gener, ls='--', lw=2, label='CNet-0.1 k=2', color='red')
#     # bax.scatter(lamda, sum_rate_gener, s=60, c='red', marker='^', alpha=0.5)
#     #
#     # bax.plot(lamda, sum_rate_2, ls='-', lw=2, label='CNet-β k=3', color='blue')
#     # bax.scatter(lamda, sum_rate_2, s=60, c='blue', marker='D', alpha=0.5)
#     # bax.plot(lamda, sum_rate_2_gener, ls='--', lw=2, label='CNet-0.1 k=3', color='blue')
#     # bax.scatter(lamda, sum_rate_2_gener, s=60, c='blue', marker='D', alpha=0.5)
#
#     # bax.plot(lamda, sum_rate_1, ls='-', lw=2, label='CNet-β k=4', color='orange')
#     # bax.scatter(lamda, sum_rate_1, s=60, c='orange', marker='o', alpha=0.5)
#     # bax.plot(lamda, sum_rate_1_gener, ls='--', lw=2, label='CNet-0.1 k=4', color='orange')
#     # bax.scatter(lamda, sum_rate_1_gener, s=60, c='orange', marker='o', alpha=0.5)
#
#     plt.plot(lamda, sum_rate_1, ls='-', lw=2, label='CNet-β k=4', color='orange')
#     plt.scatter(lamda, sum_rate_1, s=60, c='orange', marker='o', alpha=0.5)
#     plt.plot(lamda, sum_rate_1_gener, ls='--', lw=2, label='CNet-0.1 k=4', color='orange')
#     plt.scatter(lamda, sum_rate_1_gener, s=60, c='orange', marker='o', alpha=0.5)
#     # bax.secondary_xaxis(labelpad=13)
#     # bax.yticks(fontsize=13)
#     plt.figure(num=1)
#
#     # bax.set_xlabel()
#     # bax.set_ylabel('Sum Rate(bits/s/Hz)')
#
#     # bax.set_xlabel('β(bits/s/Hz)', fontsize=13)
#     # bax.set_ylabel('Sum Rate(bits/s/Hz)', fontsize=13)
#     plt.xlabel('β(bits/s/Hz)', fontsize=13)
#     plt.ylabel('Sum Rate(bits/s/Hz)', fontsize=13)
#     # x_major_locator = MultipleLocator(2000)
#     # plt.plot.xaxis.set_major_locator(x_major_locator)
#     # y_major_locator = MultipleLocator(0.4)
#     # plt.plot.yaxis.set_major_locator(y_major_locator)
#     plt.legend(loc='upper right', fontsize=13)
#     y_major_locator = MultipleLocator(0.4)
#     ax = plt.gca()
#     ax.yaxis.set_major_locator(y_major_locator)
#     plt.grid(ls='--')
#
#     plt.show()

def plot_performance():
    """
           this function is used to plot the sum rate performance in different cells K, including plot the generalization
           in this function, test_model should return "p"
    """
    for j in range(np.size(lamda)):
        sum_rate[j] = test_model(net[j], loss_fn, params, H, qq[j], require_rate[j], gamama[j], user)
        sum_rate_gener[j] = test_model(net[0], loss_fn, params, H, qq[j], require_rate[j], gamama[j], user)
        sum_rate_1[j] = test_model(nett[j], loss_fn, params, H1, qq1[j], require_rate1[j], gamama1[j], users1)
        sum_rate_1_gener[j] = test_model(nett[0], loss_fn, params, H1, qq1[j], require_rate1[j], gamama1[j], users1)
        sum_rate_2[j] = test_model(nettt[j], loss_fn, params, H2, qq2[j], require_rate2[j], gamama2[j], users2)
        sum_rate_2_gener[j] = test_model(nettt[0], loss_fn, params, H2, qq2[j], require_rate2[j], gamama2[j], users2)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    bax = brokenaxes(ylims=((17.6, 20.6), (24.6, 26.6)))

    bax.plot(lamda, sum_rate, ls='-', lw=2, label='CNet-β k=2', color='red')
    bax.scatter(lamda, sum_rate, s=60, c='red', marker='^', alpha=0.5)
    bax.plot(lamda, sum_rate_gener, ls='--', lw=2, label='CNet-0.1 k=2', color='red')
    bax.scatter(lamda, sum_rate_gener, s=60, c='red', marker='^', alpha=0.5)

    bax.plot(lamda, sum_rate_1, ls='-', lw=2, label='CNet-β k=3', color='blue')
    bax.scatter(lamda, sum_rate_1, s=60, c='blue', marker='D', alpha=0.5)
    bax.plot(lamda, sum_rate_1_gener, ls='--', lw=2, label='CNet-0.1 k=3', color='blue')
    bax.scatter(lamda, sum_rate_1_gener, s=60, c='blue', marker='D', alpha=0.5)

    bax.plot(lamda, sum_rate_2, ls='-', lw=2, label='CNet-β k=4', color='orange')
    bax.scatter(lamda, sum_rate_2, s=60, c='orange', marker='o', alpha=0.5)
    bax.plot(lamda, sum_rate_2_gener, ls='--', lw=2, label='CNet-0.1 k=4', color='orange')
    bax.scatter(lamda, sum_rate_2_gener, s=60, c='orange', marker='o', alpha=0.5)

    # plt.plot(lamda, sum_rate_1, ls='-', lw=2, label='CNet-β k=4', color='orange')
    # plt.scatter(lamda, sum_rate_1, s=60, c='orange', marker='o', alpha=0.5)
    # plt.plot(lamda, sum_rate_1_gener, ls='--', lw=2, label='CNet-0.1 k=4', color='orange')
    # plt.scatter(lamda, sum_rate_1_gener, s=60, c='orange', marker='o', alpha=0.5)
    # # bax.secondary_xaxis(labelpad=13)
    # # bax.yticks(fontsize=13)
    # plt.figure(num=1)

    bax.set_xlabel('β(bits/s/Hz)', fontsize=13)
    bax.set_ylabel('Sum Rate(bits/s/Hz)', fontsize=13)

    # x_major_locator = MultipleLocator(2000)
    # plt.plot.xaxis.set_major_locator(x_major_locator)
    # y_major_locator = MultipleLocator(0.4)
    # plt.plot.yaxis.set_major_locator(y_major_locator)
    plt.legend(loc='upper right', fontsize=13)
    y_major_locator = MultipleLocator(0.4)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.grid(ls='--')

    plt.show()

"""
     test_model_bar is combined with plot-performance_bar to plot the CDF of the sum rate in different 
     cells and minimum rate. 
"""


def test_model_bar(model, loss_fn, params, data, q, require_rate, gama, users):
    print(model)
    test_data = data
    test_generator = DataLoader(test_data, batch_size=params['batch_size'], shuffle=False)
    model.eval()
    test_loss = []
    test_times = np.zeros(params['n_tests'])
    for i in range(params['n_tests']):
        start_test_time = time.time()
        for x in test_generator:
            # x = x.to(device)
            B = genete_B(x, params['batch_size'], gama, users)
            A = -torch.eye(users).view(1, users, users) * torch.ones(params['batch_size'], users, users)
            A = np.array(A)
            B = np.concatenate([B, A], axis=1)
            m = -torch.ones(users).view(1, users, 1) * torch.ones(params['batch_size'], users, 1) * Pmax
            m = np.array(m)
            n = np.concatenate([q, m], axis=1)
            B = torch.tensor(B)
            n = torch.tensor(n)
            V, V_11 = H_to_V(-B, -n, params['batch_size'], users)
            out = model(B, V, V_11)
            test_out = out * Pmax / (torch.max(out, 1, keepdim=True)[0])
            single_rate, loss = loss_fn(x, test_out, params['batch_size'], users, sigma)
            # loss = -loss
            test_loss.append(loss.item())
        end_test_time = time.time()
        test_time = end_test_time - start_test_time
        test_times[i] = test_time

    print('the mean testing time:', np.mean(test_times))
    print('the sample sumrate:', test_loss)
    test_loss = np.array(test_loss)
    probability, sumrate = np.histogram(test_loss, bins=80, range=None, density=False)
    return probability, sumrate, test_loss


# when draw the CDF ,the batch size should be set as 1
# def plot_performance_bar():
#     net_0_1 = torch.load('constraint_network_0.1_2.pt')
#     net_0_11 = torch.load('constraint_network_0.1.pt')
#     net_0_111 = torch.load('constraint_network_4_0.1_16.pt')
#     net_0_44 = torch.load('constraint_network_0.4.pt')
#     net_0_4 = torch.load('constraint_network_0.4_2.pt')
#     net_0_444 = torch.load('constraint_network_4_0.4_16.pt')
#     net_0_1 = torch.load('constraint_network_0.1_2.pt')
#     net_0_11 = torch.load('constraint_network_0.1.pt')
#     net_0_111 = torch.load('constraint_network_4_0.1_16.pt')
#     net_0_44 = torch.load('constraint_network_0.4.pt')
#     net_0_4 = torch.load('constraint_network_0.4_2.pt')
#     net_0_444 = torch.load('constraint_network_4_0.4_16.pt')
#     test_probability, _, _ = test_model_bar(net_0_1, loss_fn, params, H, qq[0], require_rate[0], gamama[0], user)
#     test_probability_4, _, _ = test_model_bar(net_0_4, loss_fn, params, H, qq[3], require_rate[3], gamama[3], user)
#     test_probability_11, _, _ = test_model_bar(net_0_11, loss_fn, params, H2, qq1[0], require_rate1[0], gamama1[0], users1)
#     test_probability_44, _, _ = test_model_bar(net_0_44, loss_fn, params, H2, qq1[3], require_rate1[3], gamama1[3], users1)
#     test_probability_111, _, _ = test_model_bar(net_0_111, loss_fn, params, H1, qq2[0], require_rate2[0], gamama2[0], users2)
#     test_probability_444, _, _ = test_model_bar(net_0_444, loss_fn, params, H1, qq2[3], require_rate2[3], gamama2[3], users2)
#     cdf1 = np.cumsum(test_probability/1000)
#     cdf2 = np.cumsum(test_probability_11/1000)
#     cdf3 = np.cumsum(test_probability_4/1000)
#     cdf4 = np.cumsum(test_probability_44 / 1000)
#     cdf5 = np.cumsum(test_probability_111 / 1000)
#     cdf6 = np.cumsum(test_probability_444 / 1000)
#     fig, ax = plt.subplots(1, 1)
#     plt.plot(cdf1, ls='-', lw=2, label='CNet-0.1 k=2', color='green')
#     plt.plot(cdf3, ls='--', lw=2, label='CNet-0.4 k=2', color='green')
#     plt.plot(cdf2, ls='-', lw=2, label='CNet-0.1 k=3', color='purple')
#     plt.plot(cdf4, ls='--', lw=2, label='CNet-0.4 k=3', color='purple')
#     plt.plot(cdf5, ls='-', lw=2, label='CNet-0.1 k=4', color='brown')
#     plt.plot(cdf6, ls='--', lw=2, label='CNet-0.4 k=4', color='brown')
#     plt.figure(num=1)
#     plt.legend(fontsize=13)
#     plt.xlabel('Sum Rate(bits/s/Hz)', fontsize=13)
#     plt.ylabel('Cumulative Probability', fontsize=13)
#     plt.xticks(fontsize=13)
#     plt.yticks(fontsize=13)
#     # ax.set_xlabel('Epoch', fontsize=13)
#     # ax.set_ylabel('Sum Rate(bits/s/Hz)', fontsize=13)
#     x_major_locator = MultipleLocator(20)
#     ax.xaxis.set_major_locator(x_major_locator)
#     # y_major_locator = MultipleLocator(0.4)
#     # ax.yaxis.set_major_locator(y_major_locator)
#     # ax.legend(fontsize=15)
#     plt.grid(ls='--')
#     plt.show()

def plot_performance_bar():
    net_0_1 = torch.load('xxx_0.1_1.pt')
    net_0_11 = torch.load('xxx_0.1_2.pt')
    net_0_111 = torch.load('xxx_0.1_3.pt')
    net_0_44 = torch.load('xxx_0.4.pt')
    net_0_4 = torch.load('xxx_0.4_2.pt')
    net_0_444 = torch.load('xxx_0.4_3.pt')
    test_probability, _, _ = test_model_bar(net_0_1, loss_fn, params, H, qq[0], require_rate[0], gamama[0], user)
    test_probability_4, _, _ = test_model_bar(net_0_4, loss_fn, params, H, qq[3], require_rate[3], gamama[3], user)
    test_probability_11, _, _ = test_model_bar(net_0_11, loss_fn, params, H1, qq1[0], require_rate1[0], gamama1[0], users1)
    test_probability_44, _, _ = test_model_bar(net_0_44, loss_fn, params, H1, qq1[3], require_rate1[3], gamama1[3], users1)
    test_probability_111, _, _ = test_model_bar(net_0_111, loss_fn, params, H2, qq2[0], require_rate2[0], gamama2[0], users2)
    test_probability_444, _, _ = test_model_bar(net_0_444, loss_fn, params, H2, qq2[3], require_rate2[3], gamama2[3], users2)
    cdf1 = np.cumsum(test_probability/1000)
    cdf2 = np.cumsum(test_probability_11/1000)
    cdf3 = np.cumsum(test_probability_4/1000)
    cdf4 = np.cumsum(test_probability_44 / 1000)
    cdf5 = np.cumsum(test_probability_111 / 1000)
    cdf6 = np.cumsum(test_probability_444 / 1000)
    fig, ax = plt.subplots(1, 1)
    plt.plot(cdf1, ls='-', lw=2, label='CNet-0.1 k=2', color='green')
    plt.plot(cdf3, ls='--', lw=2, label='CNet-0.4 k=2', color='green')
    plt.plot(cdf2, ls='-', lw=2, label='CNet-0.1 k=3', color='purple')
    plt.plot(cdf4, ls='--', lw=2, label='CNet-0.4 k=3', color='purple')
    plt.plot(cdf5, ls='-', lw=2, label='CNet-0.1 k=4', color='brown')
    plt.plot(cdf6, ls='--', lw=2, label='CNet-0.4 k=4', color='brown')
    plt.figure(num=1)
    plt.legend(fontsize=13)
    plt.xlabel('Sum Rate(bits/s/Hz)', fontsize=13)
    plt.ylabel('Cumulative Probability', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    # ax.set_xlabel('Epoch', fontsize=13)
    # ax.set_ylabel('Sum Rate(bits/s/Hz)', fontsize=13)
    x_major_locator = MultipleLocator(20)
    ax.xaxis.set_major_locator(x_major_locator)
    # y_major_locator = MultipleLocator(0.4)
    # ax.yaxis.set_major_locator(y_major_locator)
    # ax.legend(fontsize=15)
    plt.grid(ls='--')
    plt.show()


# def compare_performance_bar():
#     dataFile_1 = r"C:\Users\86151\Desktop\大学毕业设计\毕设代码\matlab_variables\sumrate_0.1.mat"
#     dataFile_2 = r"C:\Users\86151\Desktop\大学毕业设计\毕设代码\matlab_variables\sumrate_0.4.mat"
#
#     sumrate = scio.loadmat(dataFile_1)['sumrate']
#     sumrate_2 = scio.loadmat(dataFile_2)['sumrate']
#     sumrate = np.array(sumrate.tolist())
#     sumrate_2 = np.array(sumrate_2.tolist())
#     net_0_11 = torch.load('constraint_network_0.1.pt')
#     net_0_44 = torch.load('constraint_network_0.4.pt')
#     _, _, test_loss_11 = test_model_bar(net_0_11, loss_fn, params, H2, qq1[0], require_rate1[0], gamama1[0], users1)
#     _, _, test_loss_44 = test_model_bar(net_0_44, loss_fn, params, H2, qq1[3], require_rate1[3], gamama1[3], users1)
#
#     sns.kdeplot(test_loss_11, color='orange', label='CNet β=0.1', ls='-', lw=2)
#     sns.kdeplot(test_loss_44, color='orange', label='CNet β=0.4', ls='--', lw=2)
#     sns.kdeplot(sumrate[:, 0], color='blue', label='interior point β=0.1', ls='-', lw=2)
#     sns.kdeplot(sumrate_2[:, 0], color='blue', label='interior point β=0.4', ls='--', lw=2)
#     plt.figure(num=1)
#     plt.legend(fontsize=13)
#     plt.xlabel('Sum Rate(bits/s/Hz)', fontsize=13)
#     plt.ylabel('Density', fontsize=13)
#     plt.xticks(fontsize=13)
#     plt.yticks(fontsize=13)
#     # plt.xlabel('Sum Rate(bits/s/Hz)')
#     # plt.ylabel('Density')
#     plt.grid(ls='--')
#     plt.show()

def compare_performance_bar():
    dataFile_1 = r"path\sumrate_0.1.mat"
    dataFile_2 = r"path\sumrate_0.4.mat"

    sumrate = scio.loadmat(dataFile_1)['sumrate']
    sumrate_2 = scio.loadmat(dataFile_2)['sumrate']
    sumrate = np.array(sumrate.tolist())
    sumrate_2 = np.array(sumrate_2.tolist())
    net_0_11 = torch.load('xxx_0.1.pt')
    net_0_44 = torch.load('xxx_0.4.pt')
    _, _, test_loss_11 = test_model_bar(net_0_11, loss_fn, params, H1, qq1[0], require_rate1[0], gamama1[0], users1)
    _, _, test_loss_44 = test_model_bar(net_0_44, loss_fn, params, H1, qq1[3], require_rate1[3], gamama1[3], users1)

    sns.kdeplot(test_loss_11, color='orange', label='CNet β=0.1', ls='-', lw=2)
    sns.kdeplot(test_loss_44, color='orange', label='CNet β=0.4', ls='--', lw=2)
    sns.kdeplot(sumrate[:, 0], color='blue', label='interior point β=0.1', ls='-', lw=2)
    sns.kdeplot(sumrate_2[:, 0], color='blue', label='interior point β=0.4', ls='--', lw=2)
    plt.figure(num=1)
    plt.legend(fontsize=13)
    plt.xlabel('Sum Rate(bits/s/Hz)', fontsize=13)
    plt.ylabel('Density', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    # plt.xlabel('Sum Rate(bits/s/Hz)')
    # plt.ylabel('Density')
    plt.grid(ls='--')
    plt.show()


# def batch_size():
#     """
#         plot the training results under different batch size.
#     """
#     gap = 400
#     num_epochs = 8000
#     epoch_train = np.zeros(gap)
#     train_loss_11 = np.zeros(gap)
#     train_loss_22 = np.zeros(gap)
#     train_loss_33 = np.zeros(gap)
#
#     train_loss_1 = np.load("batch_size20.npy")
#     train_loss_2 = np.load("batch_size40.npy")
#     train_loss_3 = np.load("batch_size80.npy")
#
#     for i in range(gap):
#         epoch_train[i] = i*num_epochs/gap
#         train_loss_11[i] = train_loss_1[int(i * num_epochs/gap)]
#         train_loss_22[i] = train_loss_2[int(i * num_epochs/gap)]
#         train_loss_33[i] = train_loss_3[int(i * num_epochs/gap)]
#
#     fig, ax = plt.subplots(1, 1)
#     ax.plot(epoch_train, train_loss_11, label='Batchsize:20', ls='-', lw=2, color='green')
#     ax.plot(epoch_train, train_loss_22, label='Batchsize:40', ls='-', lw=2, color='orange')
#     ax.plot(epoch_train, train_loss_33, label='Batchsize:80', ls='-', lw=2, color='purple')
#     plt.xticks(fontsize=13)
#     plt.yticks(fontsize=13)
#     ax.set_xlabel('Epoch', fontsize=13)
#     ax.set_ylabel('Sum Rate(bits/s/Hz)', fontsize=13)
#     x_major_locator = MultipleLocator(2000)
#     ax.xaxis.set_major_locator(x_major_locator)
#     y_major_locator = MultipleLocator(0.4)
#     ax.yaxis.set_major_locator(y_major_locator)
#     ax.legend(fontsize=15)
#     plt.grid(ls='--')
#     plt.legend()
#     plt.xlabel('epoch')
#     plt.ylabel('Sum Rate(bits/s/Hz)')
#
#     # the following code is used to plot local enlarged drawing
#
#     axins = inset_axes(ax, width="25%", height="25%", loc='lower left',
#                        bbox_to_anchor=(0.3, 0.3, 1, 1),
#                        bbox_transform=ax.transAxes)
#     # choose the local magnification area
#     zone_left = 70
#     zone_right = 110
#
#     # choose the scale of magnification
#     x_ratio = 0
#     y_ratio = 0.05
#
#     # the region of x-axis
#     xlim0 = epoch_train[zone_left] - (epoch_train[zone_right] - epoch_train[zone_left]) * x_ratio
#     xlim1 = epoch_train[zone_right] + (epoch_train[zone_right] - epoch_train[zone_left]) * x_ratio
#
#     # the region of y-axis
#     y = np.hstack((train_loss_11[zone_left:zone_right], train_loss_22[zone_left:zone_right],
#                    train_loss_33[zone_left:zone_right]))
#     ylim0 = np.min(y) - (np.max(y) - np.min(y)) * y_ratio
#     ylim1 = np.max(y) + (np.max(y) - np.min(y)) * y_ratio
#
#     axins.set_xlim(xlim0, xlim1)
#     axins.set_ylim(ylim0, ylim1)
#
#     # plot rectangle in the original picture
#     tx0 = xlim0
#     tx1 = xlim1
#     ty0 = ylim0
#     ty1 = ylim1
#     sx = [tx0, tx1, tx1, tx0, tx0]
#     sy = [ty0, ty0, ty1, ty1, ty0]
#     ax.plot(sx, sy, "black")
#
#     axins.plot(epoch_train, train_loss_11, label='Batchsize:20', ls='-', lw=2, color='green')
#     axins.plot(epoch_train, train_loss_22, label='Batchsize:40', ls='-', lw=2, color='orange')
#     axins.plot(epoch_train, train_loss_33, label='Batchsize:80', ls='-', lw=2, color='purple')
#     plt.grid(ls='--')
#     # plot the arrow
#     ax.annotate('', xy=(3000, 19.15), xytext=(2000, 19.3), weight='bold',
#                 arrowprops=dict(facecolor='black', shrink=0.00005))
#     plt.xticks(fontsize=13)
#     plt.yticks(fontsize=13)
#     ax.set_xlabel('Epoch', fontsize=13)
#     ax.set_ylabel('Sum Rate(bits/s/Hz)', fontsize=13)
#     x_major_locator = MultipleLocator(2000)
#     ax.xaxis.set_major_locator(x_major_locator)
#     y_major_locator = MultipleLocator(0.4)
#     ax.yaxis.set_major_locator(y_major_locator)
#     ax.legend(fontsize=15)
#     plt.show()


# def lr():
#     """
#         plot the training results under different batch size.
#     """
#     gap = 400
#     num_epochs = 8000
#     epoch_train = np.zeros(gap)
#     train_loss_11 = np.zeros(gap)
#     train_loss_22 = np.zeros(gap)
#     train_loss_33 = np.zeros(gap)
#
#     train_loss_1 = np.load('lr_0.005.npy')
#     train_loss_2 = np.load('lr_0.01.npy')
#     train_loss_3 = np.load('lr_0.05.npy')
#
#     for i in range(gap):
#         epoch_train[i] = i*num_epochs/gap
#         train_loss_11[i] = train_loss_1[int(i*num_epochs/gap)]
#         train_loss_22[i] = train_loss_2[int(i*num_epochs/gap)]
#         train_loss_33[i] = train_loss_3[int(i*num_epochs/gap)]
#
#     fig, ax = plt.subplots(1, 1)
#     ax.plot(epoch_train, train_loss_11, label='Learning rate: 0.005', ls='-', lw=2, color='brown')
#     ax.plot(epoch_train, train_loss_22, label='Learning rate: 0.01', ls='-', lw=2, color='orange')
#     ax.plot(epoch_train, train_loss_33, label='Learning rate: 0.05', ls='-', lw=2, color='blue')
#     plt.grid(ls='--')
#     plt.legend()
#     plt.xticks(fontsize=13)
#     plt.yticks(fontsize=13)
#     ax.set_xlabel('Epoch', fontsize=13)
#     ax.set_ylabel('Sum Rate(bits/s/Hz)', fontsize=13)
#     x_major_locator = MultipleLocator(2000)
#     ax.xaxis.set_major_locator(x_major_locator)
#     y_major_locator = MultipleLocator(0.4)
#     ax.yaxis.set_major_locator(y_major_locator)
#     ax.legend(fontsize=15)
#     # plt.xlabel('epoch')
#     # plt.ylabel('Sum Rate(bits/s/Hz)')
#     plt.show()


# plot_performance_bar()
# lambda_mismatch()
# plot_performance()
# batch_size()
# lr()
# compare_performance_bar()

