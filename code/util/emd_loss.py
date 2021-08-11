import torch
import energyflow as ef
import numpy as np

from qpth.qp import QPFunction


# derived from https://github.com/icoz69/DeepEMD/blob/master/Models/models/emd_utils.py
def emd_inference_qpth(distance_matrix, weight1, weight2, device, form='QP', l2_strength=0.0001, add_energy_diff=True):
    """
    to use the QP solver QPTH to derive EMD (LP problem),
    one can transform the LP problem to QP,
    or omit the QP term by multiplying it with a small value,i.e. l2_strngth.
    :param distance_matrix: nbatch * element_number * element_number
    :param weight1: nbatch  * weight_number
    :param weight2: nbatch  * weight_number
    :return:
    emd distance: nbatch*1
    flow : nbatch * weight_number *weight_number
    """

    nbatch = distance_matrix.shape[0]
    nelement_distmatrix = distance_matrix.shape[1] * distance_matrix.shape[2]
    nelement_weight1 = weight1.shape[1]
    nelement_weight2 = weight2.shape[1]

    # reshape dist matrix too (nbatch, 1, n1 * n2)
    Q_1 = distance_matrix.view(-1, 1, nelement_distmatrix).double()
    # print(Q_1)

    if form == 'QP':  # converting to QP - after testing L2 reg performs marginally better than QP
        # version: QTQ
        Q = torch.bmm(Q_1.transpose(2, 1), Q_1).double() + 1e-4 * torch.eye(
            nelement_distmatrix).double().unsqueeze(0).repeat(nbatch, 1, 1)  # 0.00001 *
        p = torch.zeros(nbatch, nelement_distmatrix).double().to(device)
    elif form == 'L2':  # regularizing a trivial Q term with l2_strength
        # version: regularizer
        Q = (l2_strength * torch.eye(nelement_distmatrix).double()).unsqueeze(0).repeat(nbatch, 1, 1).to(device)
        p = distance_matrix.view(nbatch, nelement_distmatrix).double()
    else:
        raise ValueError('Unkown form')

    # h = [0 ... 0 w1 w2]
    h_1 = torch.zeros(nbatch, nelement_distmatrix).double().to(device)
    h_2 = torch.cat([weight1, weight2], 1).double()
    h = torch.cat((h_1, h_2), 1)

    G_1 = -torch.eye(nelement_distmatrix).double().unsqueeze(0).repeat(nbatch, 1, 1).to(device)
    G_2 = torch.zeros([nbatch, nelement_weight1 + nelement_weight2, nelement_distmatrix]).double().to(device)
    # sum_j(xij) = si
    for i in range(nelement_weight1):
        G_2[:, i, nelement_weight2 * i:nelement_weight2 * (i + 1)] = 1
    # sum_i(xij) = dj
    for j in range(nelement_weight2):
        G_2[:, nelement_weight1 + j, j::nelement_weight2] = 1

    # xij>=0, sum_j(xij) <= si,sum_i(xij) <= dj, sum_ij(x_ij) = min(sum(si), sum(dj))
    G = torch.cat((G_1, G_2), 1)
    A = torch.ones(nbatch, 1, nelement_distmatrix).double().to(device)
    b = torch.min(torch.sum(weight1, 1), torch.sum(weight2, 1)).unsqueeze(1).double()
    flow = QPFunction(verbose=-1)(Q, p, G, h, A, b)

    if add_energy_diff: energy_diff = torch.abs(torch.sum(weight1, dim=1) - torch.sum(weight2, dim=1))

    emd_score = torch.sum((Q_1).squeeze() * flow, 1)
    if add_energy_diff: emd_score += energy_diff

    return emd_score, flow.view(-1, nelement_weight1, nelement_weight2)



def emd(jets1, jets2, form='L2', l2_strength=0.0001):
    """
    calculate Energy Mover's Distance between each jet in jets1 and each in jet2
    :param jets1: [nbatch * ] num_particles * 3
        - 3 particle features [eta, phi, pt]
    :param jets2: [nbatch * ] num_particles * 3
    :return:
    emd distance: nbatch * nbatch
    flow : nbatch * nbatch * num_particles * num_particles
    """

    if len(jets1.shape) < 3:
        jets1 = jets1.unsqueeze(0)
        jets2 = jets2.unsqueeze(0)

    n = jets1.shape[0]

    x1 = jets1.unsqueeze(1).repeat(1, n, 1, 1).view(n * n, -1, 3)
    x2 = jets2.repeat(n, 1, 1)

    diffs = -(x1[:, :, :2].unsqueeze(2) - x2[:, :, :2].unsqueeze(1)) + 1e-12
    dists = torch.norm(diffs, dim=3)

    weight1 = jets1[:, :, 2].unsqueeze(1).repeat(1, n, 1).view(n * n, -1)
    weight2 = jets2[:, :, 2].repeat(n, 1)

    return emd_inference_qpth(dists, weight1, weight2, form=form, l2_strength=l2_strength)


def emd_loss(jets1, jets2, form='L2', l2_strength=0.0001, return_flow=False, device=torch.device('cpu')):
    """
    batched Energy Mover's Distance between jets1 and jets2
    :param jets1: [nbatch * ] num_particles * 3
        - 3 particle features [eta, phi, pt]
    :param jets2: [nbatch * ] num_particles * 3
    :param return_flow: _bool_
        - return the flow as well as the EMD score
    :return:
    emd distance: nbatch * 1
    flow : (if return_flow) nbatch * num_particles * num_particles
    """

    if len(jets1.shape) < 3:
        jets1 = jets1.unsqueeze(0)
        jets2 = jets2.unsqueeze(0)

    diffs = -(jets1[:, :, :2].unsqueeze(2) - jets2[:, :, :2].unsqueeze(1)) + 1e-12
    dists = torch.norm(diffs, dim=3)

    emd_score, flow = emd_inference_qpth(dists, jets1[:, :, 2], jets2[:, :, 2], device, form=form, l2_strength=l2_strength)

    return (emd_score, flow) if return_flow else emd_score
