# -*- coding: utf-8 -*-
import lda
import numpy as np

LDA = lda.LDA()


def para_estimation(W, alpha, beta, vacab):
    MAX_iter = 50
    epsilon = 10 ** -2
    for iter in range(MAX_iter):
        para_gamma_phi = E_step(W, alpha, beta)
        alpha_, beta_ = M_step(W, alpha, beta, vacab, para_gamma_phi)
        if (np.sum(np.sum(np.power(beta_ - beta, 2))) < epsilon):
            break
        alpha = alpha_
        beta = beta_
    return alpha_, beta_


def E_step(W, alpha, beta, get_gamma_phi=LDA.variational_inference()):
    """

    :param W:
    :param alpha:
    :param beta:
    :param get_gamma_phi:
    :return: given old para:alpha，beta， to get gamma phi。
    """
    para_gamma_phi = []
    M = len(W)
    for d in range(M):
        gamma, phi = get_gamma_phi(W[d], alpha, beta)
        para_gamma_phi.append((gamma, phi))  # turple's list.
    return para_gamma_phi


def M_step(W, alpha, beta, vacab, para_gamma_phi):
    """

    :param W:
    :param alpha:
    :param beta:
    :param vacab:
    :param para_gamma_phi:
    :return: given gamma，phi of every doc,to optimize （alpha，beta），which target function are  L_(alpha) and L_(beta)
    """
    K = len(alpha)
    beta_new = optimization_Beta(W, K, vacab, para_gamma_phi)
    alpha_new = optimization_alpha(alpha, beta, W, para_gamma_phi)
    return alpha_new, beta_new


def optimization_Beta(W, K, vacab, para_gamma_phi):
    M = len(W)
    V = len(vacab)
    beta = np.zeros((K, V), dtype='float64')
    for d in range(M):
        gamma, phi = para_gamma_phi[d]  # turple
        for n in range(W[d]):
            j = vacab.index(W[d][n])
            beta[:, j] += phi[n]
    return beta


def optimization_alpha(alpha_, beta_, W, para_gamma_phi, func=LDA.big_L_alpha, g_func=LDA.grad_alpha_negloglikelifunc,
                       H_component=LDA.Hession_components_ofalpha_loglikelifunc,
                       descent_derection=LDA.descent_derectionof_alpha):
    x = alpha_  # x is row vector.
    y = beta_
    max_out_iters = 100
    alpha = 0.4
    beta = 0.5
    epsilon = 10 ** -3

    for out_iter in range(max_out_iters):
        f_val = func(W, x, y, para_gamma_phi)
        grad_x = g_func(W, x, y, para_gamma_phi)
        h_H, z_H = H_component(W, x)
        delta_x = descent_derection(grad_x, h_H, z_H)

        t = 1.0
        while (f_val + alpha * t * np.dot(grad_x, delta_x) < func(W, x + t * delta_x)):
            t *= beta
        t *= beta
        x_new = x + t * delta_x
        if (np.sum(np.power(x_new - x, 2)) < epsilon):
            x = x_new
            break
        x = x_new
    return x
