import sys
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import cdd
from constraint_layer import LinearConstraint
import time
import argparse
import matplotlib.pyplot as plt
import pickle
from pytorchtools import EarlyStopping
from matplotlib.pyplot import MultipleLocator


torch.manual_seed(1)
# information of simulation
Pmax = xxx
sigma = xxx  # the noise power in value
users = xxx   # the numer of users
lamda = 0.1  # the  minimum rate requirement


def load_variable(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


# train_data = load_variable('data_train_4_1.txt')
# train_data = torch.tensor(train_data)
# val_data = load_variable('data_val_4_1.txt')
# val_data = torch.tensor(val_data)
train_data = load_variable('xxx_train.txt')
train_data = torch.tensor(train_data)
val_data = load_variable('xxx_val.txt')
val_data = torch.tensor(val_data)

# use double description to produce extreme points v
def H_to_V(A, b, batch_size, users):
    """
    Converts a polytope(bounded polyhedron) in H-representation to
    one in V-representation using pycddlib.
    the constraint set A.dot(x) <= b.
    """
    # define cdd problem and convert representation
    if len(b.shape) == 1:
        b = np.expand_dims(b, axis=1)
    mat_np = np.concatenate([b, -A], axis=2)
    if mat_np.dtype in [np.int32, np.int64]:
        nt = 'fraction'
    else:
        nt = 'float'

    """
       V restore the vector of v
       V_11 restore the indicator value v_1 in order to achieve the batch training since the number 
       of each sample's extreme points is different
     """

    V = []
    V_11 = []
    for k in range(batch_size):
        mat_list = mat_np.tolist()[k]
        mat_cdd = cdd.Matrix(mat_list, number_type=nt)
        mat_cdd.rep_type = cdd.RepType.INEQUALITY
        poly = cdd.Polyhedron(mat_cdd)
        gen = poly.get_generators()
        # convert the cddlib output data structure to numpy
        V_list = []
        R_list = []
        lin_set = gen.lin_set
        V_lin_idx = []
        R_lin_idx = []
        for i in range(gen.row_size):
            g = gen[i]
            g_type = g[0]  # the type of extreme point and extreme ray(0--extreme ray;1--extreme point)
            g_vec = g[1:]
            if i in lin_set:
                is_linear = True
            else:
                is_linear = False
            if g_type == 1:
                # check if the extreme point is the origin
                if not torch.equal(torch.tensor(g_vec), torch.FloatTensor(np.zeros(b.shape[1]))):
                    V_list.append(g_vec)
                    if is_linear:
                        V_lin_idx.append(len(V_list) - 1)
            elif g_type == 0:
                if not torch.equal(torch.tensor(g_vec), torch.FloatTensor(np.zeros(b.shape[1]))):
                    R_list.append(g_vec)
                    if is_linear:
                        R_lin_idx.append(len(R_list) - 1)
            else:
                raise ValueError('Generator data structure is not valid.')

        V_dd = np.asarray(V_list).reshape(-1, users)
        V_dd = np.float32(V_dd)
        V_1 = np.ones_like(V_dd)  # represent the true number of extreme points

        """
          produce the indicator vector
        """
        if (pow(2, users)-len(V_dd)) > 0:
            T = pow(2, users)-len(V_dd)
            V_dd = np.concatenate([V_dd, np.zeros((T, users))], axis=0)
            V_1 = np.concatenate([V_1, np.zeros((T, users))], axis=0)
        V_1 = np.mean(V_1, axis=1)
        V.append(V_dd)
        V_11.append(V_1)

    V = torch.tensor(V)
    V_11 = torch.tensor(V_11)

    return V, V_11


# calculate the sum of users' rate
def cal_sum_rate(H, P_output, numh, usernum, sigma):
    # numh: batch size
    h_square = torch.abs(H).view(numh, usernum, usernum)
    P_output_temp = torch.ones(numh, usernum, usernum) * P_output.view(numh, usernum, -1)
    P_output_mul = h_square * P_output_temp
    mask_1 = torch.eye(usernum).view(1, usernum, usernum)
    P_output_upper = torch.sum(P_output_mul * mask_1, dim=1)
    mask_2 = torch.subtract(torch.ones(numh, usernum, usernum), torch.eye(usernum).view(-1, usernum, usernum))
    P_output_down = torch.sum(P_output_mul * mask_2, dim=1)+sigma
    single_rate = torch.log2(1+torch.div(P_output_upper, P_output_down))
    sum_rate = torch.mean(torch.sum(single_rate, dim=1))
    return single_rate, sum_rate


# produce the matrix of B, m is the square of the H
def genete_B(m, batch_size, gama, users):
    H_input = abs(m).view(batch_size, users, users)
    H_trans = H_input.transpose(2, 1)
    diagH = H_input * torch.eye(users)
    B = diagH - (H_trans - diagH) * gama
    return B


def train_model(model, loss_fn, params,  q, gama, users):
    # device_id = params['device']
    # device = torch.device('cuda:{}'.format(device_id) if device_id >= 0 else 'cpu')
    # print('Using device:\n', device)
    # model = model.to(device)
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params['learning_rate'],
        betas=(0.9, 0.99)

    )

    print('train size:', len(train_data))
    print('val size:', len(val_data))

    training_generator = DataLoader(
        train_data, batch_size=params['batch_size'], shuffle=True)
    validation_generator = DataLoader(
        val_data, batch_size=params['batch_size'], shuffle=False)
    step_size = 1000
    # build a scheduler to get better learning rate(decrease in every step_size epoch)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.5, last_epoch=-1)

    training_metrics = {'train_loss': [], 'val_loss': [], 'cumulative_epoch_time': []}
    cumulative_epoch_time = 0
    final_epoch = []  # restore the epoch of early stopping
    # the setting of early stop
    early_stopping = EarlyStopping(patience=300, verbose=True)
    for epoch in range(params['num_epochs']):

        model.train()
        # train_loss is used to record the present sumrate, train_loss_1 is used to optimization of NN
        train_loss = 0
        train_loss_1 = 0
        # i = 0
        # train_loss_1 = np.zeros(1)
        # train_loss_1 = torch.tensor(train_loss_1)
        epoch_start = time.time()
        for x in training_generator:
            # x = x.to(device)
            # print('i', i)
            B = genete_B(x, params['batch_size'], gama, users)
            A = -torch.eye(users).view(1, users, users) * torch.ones(params['batch_size'], users, users)
            A = np.array(A)
            # the new matrix A(B and the requirement of P)
            A = np.concatenate([B, A], axis=1)
            m = -torch.ones(users).view(1, users, 1) * torch.ones(params['batch_size'], users, 1)*Pmax
            m = np.array(m)
            # the new matrix of q
            n = np.concatenate([q, m], axis=1)
            A = torch.tensor(A)
            n = torch.tensor(n)
            V, V_11 = H_to_V(-A, -n, params['batch_size'], users)
            out = model(A, V, V_11)
            # print(out)
            # scaling of P
            out = out*Pmax/(torch.max(out, 1, keepdim=True)[0])
            single_rate, loss = loss_fn(x, out, params['batch_size'], users, sigma)
            loss = -loss
            train_loss_1 += loss
            train_loss += loss.item()
            optimizer.zero_grad()
            train_loss_1.backward()
            optimizer.step()
            # print('{}: train loss: {}, validation loss: {}, lr: {:.2E}'.format(
            #     epoch,
            #     training_metrics['train_loss'][-1],
            #     training_metrics['val_loss'][-1],
            #     optimizer.param_groups[0]['lr'],
            #     ))
            train_loss_1 = 0
            # i += 1
        if epoch > 0:
            scheduler.step()
            # lr = scheduler.get_last_lr()
            # print('scheduler learning rate:', lr)

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        cumulative_epoch_time += epoch_time
        training_metrics['train_loss'].append(-train_loss / len(training_generator))
        training_metrics['cumulative_epoch_time'].append(cumulative_epoch_time)
        # validation
        if epoch % params['verbosity'] == 0:
            with torch.set_grad_enabled(False):
                model.eval()
                val_loss = 0
                for x in validation_generator:
                    # x = x.to(device)
                    B = genete_B(x, params['batch_size'], gama, users)
                    A = -torch.eye(users).view(1, users, users) * torch.ones(params['batch_size'], users, users)
                    A = np.array(A)
                    A = np.concatenate([B, A], axis=1)
                    m = -torch.ones(users).view(1, users, 1) * torch.ones(params['batch_size'], users, 1) * Pmax
                    m = np.array(m)
                    n = np.concatenate([q, m], axis=1)
                    A = torch.tensor(A)
                    n = torch.tensor(n)
                    V, V_11 = H_to_V(-A, -n, params['batch_size'], users)
                    out = model(A, V, V_11)
                    out = out * Pmax / (torch.max(out, 1, keepdim=True)[0])
                    _, loss = loss_fn(x, out, params['batch_size'], users, sigma)
                    loss = -loss
                    val_loss += loss.item()
                training_metrics['val_loss'].append(-val_loss / len(validation_generator))
                early_stopping(val_loss / len(validation_generator), model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    # if you want to show the early stop in the full training procedure,you can use the next code
                    final_epoch.append(epoch)
                    # if you want to training stop once early stopping ,you can add "break"
                    # break
            print('{}: train loss: {}, validation loss: {}, lr: {:.2E}, per epoch time: {}'.format(
                epoch,
                training_metrics['train_loss'][-1],
                training_metrics['val_loss'][-1],
                optimizer.param_groups[0]['lr'],
                epoch_time))
    # if early stopping do not happen in the process of training, we need add the final to the "final_epoch"
    if len(final_epoch) == 0:
        final_epoch.append(params['num_epochs'])
    return training_metrics, final_epoch[0]


def check_unfeasible(out, require_rate, single_rate, batch_size, users):
    # Check whether the requirements of output P are met; if not met, label+1
    # label1 - minimum sum rate requirement
    # label2 - The power P is less than Pmax
    # label3 - The power P is more than 0
    label1 = (torch.sum(((torch.greater(require_rate.view(batch_size, users), single_rate+1e-03)).int()), dim=1)).view(batch_size)
    label2 = (torch.sum((torch.greater(out, Pmax+1e-03)).int(), dim=1)).view(batch_size)
    label3 = (torch.sum((torch.less(out+1e-03, 0.0)).int(), dim=1)).view(batch_size)
    label = label1+label2+label3
    return label


def test_model(model, loss_fn, params, data, q, require_rate, gama, users):

    print(model)

    testing_metrics = {'test_loss': [], 'unfeasible': []}

    test_data = data

    test_generator = DataLoader(test_data, batch_size=params['batch_size'], shuffle=False)
    label = 0
    model.eval()
    test_loss = 0
    test_times = np.zeros(params['n_tests'])
    for i in range(params['n_tests']):
        start_test_time = time.time()
        for x in test_generator:

            B = genete_B(x, params['batch_size'], gama, users)
            A = -torch.eye(users).view(1, users, users) * torch.ones(params['batch_size'], users, users)
            A = np.array(A)
            A = np.concatenate([B, A], axis=1)
            m = -torch.ones(users).view(1, users, 1) * torch.ones(params['batch_size'], users, 1) * Pmax
            m = np.array(m)
            n = np.concatenate([q, m], axis=1)
            A = torch.tensor(A)
            n = torch.tensor(n)
            V, V_11 = H_to_V(-A, -n, params['batch_size'], users)
            out = model(A, V, V_11)
            test_out = out * Pmax / (torch.max(out, 1, keepdim=True)[0])
            single_rate, loss = loss_fn(x, test_out, params['batch_size'], users, sigma)
            loss = -loss
            test_loss += loss.item()
            # print('the target power:', -loss)
            """
              check the probability of the constraints satisfaction
            """
            if torch.sum(check_unfeasible(out, require_rate, single_rate, params['batch_size'], users)) != 0:
                label += 1

        end_test_time = time.time()
        test_time = end_test_time - start_test_time

        test_times[i] = test_time
        testing_metrics['test_loss'].append(-test_loss / len(test_generator))
    print('the mean testing time:', np.mean(test_times))
    print('the mean testing sumrate:', np.mean(testing_metrics['test_loss']))
    print('the mean unfeasible ratio:', label/(params['n_tests']*len(test_generator)*params['batch_size']))
    """
        d,p represent the unfeasibility of constraints and sum rate respectively, 
         which can be used in the plot_result
    """
    # d = label/(params['n_tests']*len(test_generator)*params['batch_size'])
    p = np.mean(testing_metrics['test_loss'])
    return p


def main(params):

    # default information
    req_rate = ((torch.ones(users) * lamda).view(1, users, 1)) * torch.ones(params['batch_size'], users, 1)
    q = (torch.pow(2.0, req_rate) - 1) * sigma
    gama = (torch.pow(2.0, req_rate) - 1)
    model = LinearConstraint(q)
    # optim = torch.optim.Adam(
    #     model.parameters(),
    #     lr=params['learning_rate'],
    #     betas=(0.9, 0.99)
    #
    # )
    # print('para:', [x.grad for x in optim.param_groups[0]['params']])
    # net = torch.load('constraint_network_0.2_2.pt')
    loss_fn = cal_sum_rate
    training_metrics, fin_epoch = train_model(model, loss_fn, params, q, gama, users)
    print(fin_epoch)
    # torch.save(model, 'constraint_network_4_0.4_16.pt')
    torch.save(model, 'xxx.pt')
    epoch_train = np.zeros(params['num_epochs'])
    epoch_val = []
    for i in range(params['num_epochs']):
        epoch_train[i] = i
        if i % params['verbosity'] == 0:
            epoch_val.append(i)
    epoch_val = np.array(epoch_val)
    """
       if you use 'break' for early stop, you can use the following code to picture
    """
    # epoch_train = []
    # epoch_val = []
    # for i in range(fin_epoch+1):
    #     epoch_train.append(i)
    #     if i % params['verbosity'] == 0:
    #         epoch_val.append(i)
    # epoch_val = np.array(epoch_val)
    # epoch_train = np.array(epoch_train)

    # np.save(r"path+name", training_metrics['train_loss'])
    plt.plot(epoch_train, training_metrics['train_loss'], label='train', ls='-', lw=2, color='purple')
    plt.plot(epoch_val, training_metrics['val_loss'], label='validation', ls='-', lw=2, color='blue')
    plt.scatter(fin_epoch-1, training_metrics['train_loss'][fin_epoch-1], c='r',  label='early stop point')

    y_major_locator = MultipleLocator(0.1)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlabel('epoch')
    plt.ylabel('Sum Rate')
    plt.legend()
    plt.grid(ls='--')
    plt.title('xxx')
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_tests', type=int, default=1)
    parser.add_argument('--verbosity', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=None)
    # parser.add_argument('--log_to_file', dest='log_to_file', action='store_true', default=False)
    args = parser.parse_args()
    params = vars(args)

    print('Parameters:\n', params)


# main({'num_epochs': , 'batch_size': , 'n_tests': , 'verbosity': ,
#       'learning_rate': })




