import numpy as np
import cdd
import math
import torch
import pickle
from sklearn.model_selection import train_test_split

num = xxx
w = xxx
W = xxx    # meters
data_size = 1000
Pmax = xxx
sigma = xxx
# being used to produce testing data
lamda = 0.4
lamda1 = 0.1
lamda2 = 0.2
lamda3 = 0.3
# np.random.seed(0)

BS_x = np.array([-W, W, 0, 0])
BS_y = np.array([0, 0, math.sqrt(3)*W, -math.sqrt(3)*W])

req_rate = (torch.ones(num) * lamda).view(num, 1)
req_rate1 = (torch.ones(num) * lamda1).view(num, 1)
req_rate2 = (torch.ones(num) * lamda2).view(num, 1)
req_rate3 = (torch.ones(num) * lamda3).view(num, 1)

q = (torch.pow(2.0, req_rate) - 1) * sigma
q1 = (torch.pow(2.0, req_rate1) - 1) * sigma
q2 = (torch.pow(2.0, req_rate2) - 1) * sigma
q3 = (torch.pow(2.0, req_rate3) - 1) * sigma

gama = (torch.pow(2.0, req_rate) - 1)
gama1 = (torch.pow(2.0, req_rate1) - 1)
gama2 = (torch.pow(2.0, req_rate2) - 1)
gama3 = (torch.pow(2.0, req_rate3) - 1)


def db_change(a):
    b = math.pow(10, a/10)
    return b


def creat_random_UE():
    UE_x= np.zeros(num)
    UE_y = np.zeros(num)
    for i in range(num):
        theta = 2*math.pi*np.random.random()
        r = (W - w)*np.random.random() + w
        x = r*math.cos(theta)
        y = r*math.sin(theta)
        UE_x[i] = BS_x[i] + x
        UE_y[i] = BS_y[i] + y
    return UE_x, UE_y


def channel_fading(UE_x, UE_y):
    d = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            d[i][j] = np.sqrt(np.square(BS_x[i]-UE_x[j]) + np.square(BS_y[i]-UE_y[j]))
    loss = np.zeros((num, num))
    alpha = np.zeros((num, num))
    g = np.zeros((num, num))
    for m in range(num):
        for n in range(num):
            loss[m][n] = 36.3 + 37.6*np.log10(d[m][n]/1000) + np.random.normal(0, 8)
            alpha[m][n] = db_change(-loss[m][n])
            g[m][n] = alpha[m][n]*np.random.exponential(1, 1)
    g = np.float32(g)
    return g


def genete_B(g, gama):
    H_input = abs(g).view(num, num)
    H_trans = H_input.transpose(1, 0)
    diagH = H_input * torch.eye(num)
    B = diagH - (H_trans - diagH) * gama
    return B


def concatenate(g):
    B = genete_B(g, gama)
    B1 = genete_B(g, gama1)
    B2 = genete_B(g, gama2)
    B3 = genete_B(g, gama3)

    A = -torch.eye(num).view(num, num)
    A = np.array(A)
    # the new matrix of A (B and the requirement of P)
    AA = np.concatenate([B, A], axis=0)
    A1 = np.concatenate([B1, A], axis=0)
    A2 = np.concatenate([B2, A], axis=0)
    A3 = np.concatenate([B3, A], axis=0)
    m = -torch.ones(num).view(num, 1) * Pmax
    m = np.array(m)
    n = np.concatenate([q, m], axis=0)
    n1 = np.concatenate([q1, m], axis=0)
    n2 = np.concatenate([q2, m], axis=0)
    n3 = np.concatenate([q3, m], axis=0)
    AA = torch.tensor(AA)
    A1 = torch.tensor(A1)
    A2 = torch.tensor(A2)
    A3 = torch.tensor(A3)
    n = torch.tensor(n)
    n1 = torch.tensor(n1)
    n2 = torch.tensor(n2)
    n3 = torch.tensor(n3)
    return AA, A1, A2, A3, n, n1, n2, n3


def H_to_V(A, b):
    """
    Converts a polyhedron in H-representation to
    one in V-representation using pycddlib.
    """
    # define cdd problem and convert representation
    if len(b.shape) == 1:
        b = np.expand_dims(b, axis=1)
    mat_np = np.concatenate([b, -A], axis=1)
    if mat_np.dtype in [np.int32, np.int64]:
        nt = 'fraction'
    else:
        nt = 'float'
    # R,V restore the vector of ri,vi

    R = []
    V = []
    flag = 0
    mat_list = mat_np.tolist()
    # print(mat_list)
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
        # print(g)
        if i in lin_set:
            is_linear = True
        else:
            is_linear = False
        if g_type == 1:
            if not torch.equal(torch.tensor(g_vec), torch.FloatTensor(np.zeros(b.shape[1]))):
                V_list.append(g_vec)
                if is_linear:
                    V_lin_idx.append(len(V_list) - 1)
        elif g_type == 0:
            R_list.append(g_vec)
            if is_linear:
                R_lin_idx.append(len(R_list) - 1)
        else:
            flag = 1

    V_dd = np.asarray(V_list)
    R_dd = np.asarray(R_list)
    V = np.float32(V_dd)
    R = np.float32(R_dd)
    return V, R, flag


def check_possible(g):

    AA, A1, A2, A3, n, n1, n2, n3 = concatenate(g)
    V, R, flag = H_to_V(-AA, -n)
    V1, R1, flag1 = H_to_V(-A1, -n1)
    V2, R2, flag2 = H_to_V(-A2, -n2)
    V3, R3, flag3 = H_to_V(-A3, -n3)
    if flag == 1 or R.any() or flag1 == 1 or R1.any() or flag2 == 1 or R2.any() or flag3 == 1 or R3.any():
            # or \
            # len(V) > pow(2, num/2) or len(V1) > pow(2, num/2) or len(V2) > pow(2, num/2) or len(V3) > pow(2, num/2):
        label = 1
    else:
        label = 0
    return label

# or \
#             len(V) > pow(2, num) or len(V1) > pow(2, num) or len(V2) > pow(2, num) or len(V3) > pow(2, num)
def generate_data():
    x = []
    i = 0
    while i < data_size:
        try:
            UE_x, UE_y = creat_random_UE()
            g = channel_fading(UE_x, UE_y)
            g1 = g.reshape(-1)
            g = g.reshape(-1).tolist()
            # print('g:', g)
            beta = torch.tensor(g1)
            # print(check_possible(beta))
            if check_possible(beta) != 1:
                x.append(g)
                i += 1
            else:
                i -= 0
        except RuntimeError:
            pass
    return x, len(x)


# data, true_data_size = generate_data()
# print(data)
# print(true_data_size)


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f, 0)
    f.close()
    return filename


def load_variable(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


# produce the data required by matlab
def save_variable_matlab(v, filename, mode='a'):
    f = open(filename, mode)
    for i in range(len(v)):
        f.write(str(v[i]) + '\n')
    f.close()
    return filename


data = load_variable('data_test_1.txt')
# filename_1 = open('data_test_1.txt', 'w').close()
# filename = save_variable(data, 'data_test_1.txt')
filename_2 = open('data_test_1_matlab.txt', 'w').close()
filename_matlab = save_variable_matlab(data, 'data_test_1_matlab.txt')


"""
   produce the different rate requirement data in the way mentioned above(choose different rate requirement,lamda)
   produce the training data and the validation data
"""

# H = load_variable('data_train_4.txt')
# H = torch.tensor(H)
# train_data,  val_data = train_test_split(H, test_size=0.2)
# train_data = train_data.tolist()
# val_data = val_data.tolist()
# file1 = open('data_train_4_1.txt', 'w').close()
# file2 = open('data_val_4_1.txt', 'w').close()
# print(len(train_data))
# print(len(val_data))
# filename_11 = save_variable(train_data, 'data_train_4_1.txt')
# filename_2 = save_variable(val_data, 'data_val_4_1.txt')

# H = load_variable('xxx_train.txt')
# H = torch.tensor(H)
# train_data,  val_data = train_test_split(H, test_size=0.2)
# train_data = train_data.tolist()
# val_data = val_data.tolist()
# filename_1 = save_variable(train_data, 'xxx.txt')
# filename_2 = save_variable(val_data, 'xxx.txt')


