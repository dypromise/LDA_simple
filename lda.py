#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation + variational inference + variational EM.


import numpy as np
import scipy.special as ss
import numpy.random as random


class LDA:
    def __init__(self, K, alpha, beta, docs, V, smartinit=True):
        self.K = K
        self.alpha = alpha  # parameter of topics prior
        self.beta = beta  # parameter of words prior
        self.docs = docs
        self.V = V

        self.z_m_n = []  # topics of words of documents
        self.n_m_z = np.zeros((len(self.docs), K)) + alpha  # word count of each document and topic
        self.n_z_t = np.zeros((K, V)) + beta  # word count of each topic and vocabulary
        self.n_z = np.zeros(K) + V * beta  # word count of each topic

        self.N = 0
        self.vacabulary = []

        for m, doc in enumerate(docs):
            self.N += len(doc)
            z_n = []
            for t in doc:
                if smartinit:
                    p_z = self.n_z_t[:, t] * self.n_m_z[m] / self.n_z  # 应该是P(z_n|w_n,theta)
                    z = np.random.multinomial(1, p_z / p_z.sum()).argmax()
                else:
                    z = np.random.randint(0, K)
                z_n.append(z)
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1
            self.z_m_n.append(np.array(z_n))

    def inference(self):
        """learning once iteration"""
        for m, doc in enumerate(self.docs):
            z_n = self.z_m_n[m]
            n_m_z = self.n_m_z[m]
            for n, t in enumerate(doc):
                # discount for n-th word t with topic z
                z = z_n[n]
                n_m_z[z] -= 1
                self.n_z_t[z, t] -= 1
                self.n_z[z] -= 1

                # sampling topic new_z for t
                p_z = self.n_z_t[:, t] * n_m_z / self.n_z
                new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()

                # set z the new topic and increment counters
                z_n[n] = new_z
                n_m_z[new_z] += 1
                self.n_z_t[new_z, t] += 1
                self.n_z[new_z] += 1

    def variational_inference(self, w, alpha, beta):
        """

        :param w: doc word vector
        :param alpha:
        :param beta:
        :return: variational approximation of log(P(w|alpha,beta)).
        """
        epsilon = 10 ** -2
        N = len(w)
        K = self.K
        phi = np.ones((N, K)) * 1.0 / K
        gamma = alpha + N / K
        gamma_ = np.zeros(K)
        phi_ = np.zeros((N, K))
        max_iter = 300

        for iter in range(max_iter):
            for n in range(N):
                tmp = 0.0
                for i in range(K):
                    phi_[n, i] = beta[i, self.vacabulary.index(w[n])] * np.exp(ss.psi(gamma[i]))
                    tmp += phi_[n, i]
                phi_[n] /= tmp
            gamma_ = alpha + np.sum(phi, 0)
            if (np.sum(np.sum(np.power(phi-phi_, 2)))< epsilon):
                break
            gamma = gamma_
            phi = phi_
        return gamma_, phi_

    def L_function_value(self, w, gamma, phi, alpha, beta):
        """

        :param w:
        :param gamma:
        :param phi:
        :param alpha:
        :param beta:
        :return:
        """
        # alpha is a colum vector.
        psi_rele = ss.psi(gamma) - ss.psi(np.sum(gamma))
        tmp = 0.0
        for n in range(len(w)):
            tmp += np.sum(np.dot(np.log(beta[:, self.vacabulary.index(w[n])]), phi[n]))

        L_val = np.log(ss.gamma(np.sum(alpha))) - np.sum(np.log(ss.gamma(alpha)) + np.dot(alpha - 1, psi_rele))
        + np.sum(np.mat(phi) * np.mat(alpha).T)
        + tmp
        - np.log(ss.gamma(np.sum(gamma))) + np.sum(np.log(ss.gamma(gamma)) + np.dot(gamma - 1, psi_rele))
        - np.sum(np.sum(np.dot(phi, np.log(phi))))

        return L_val

    def negloglikelifuncvalue(self, W, alpha, beta):
        M = len(W)
        tmp = 0.0
        for d in range(M):
            gamma, phi = self.variational_inference(W[d], alpha, beta)
            tmp += self.L_function_value(W[d], gamma, phi, alpha, beta)
        return -tmp

    def big_L_alpha(self, W, alpha, beta, para_gamma_phi):
        M = len(W)
        tmp = 0.0
        for d in range(M):
            gamma, phi = para_gamma_phi[d]
            psi_rele = ss.psi(gamma) - ss.psi(np.sum(gamma))
            tmp += np.log(ss.gamma(np.sum(alpha))) - np.sum(np.log(ss.gamma(alpha)) + np.dot(alpha - 1.0, psi_rele))
        return -tmp

    def grad_alpha_negloglikelifunc(self, W, alpha, beta, para_gamma_phi):
        M = len(W)
        tmp = 0.0
        for d in range(M):
            gamma, phi = para_gamma_phi[d]
            tmp += (ss.psi(gamma) - ss.psi(np.sum(gamma)) - ss.psi(alpha) + ss.psi(np.sum(alpha)))
        return -tmp

    def Hession_components_ofalpha_loglikelifunc(self, W, alpha):
        """

        :param W:
        :param alpha:
        :return: h_H and z_H. which are components of  H=diag(h) + 1z1^T.
        """
        M = len(W)
        h = ss.polygamma(1, alpha) * M
        z = ss.polygamma(1, np.sum(alpha))
        return -h, -z

    def descent_derectionof_alpha(self, grad, h_H, z_H):
        """

        :param grad:
        :param h_H:
        :param z_H:
        :return: descent direction of loglikelihood, which is Hession^-1*grad.
        """
        c = np.sum(grad / h_H) / (1 / z_H + np.sum(np.power(h_H, -1)))
        return -(grad - c) / h_H





def worddist(self):
    """get topic-word distribution"""
    return self.n_z_t / self.n_z[:, np.newaxis]


def perplexity(self, docs=None):
    if docs == None: docs = self.docs
    phi = self.worddist()
    log_per = 0
    N = 0
    Kalpha = self.K * self.alpha
    for m, doc in enumerate(docs):
        theta = self.n_m_z[m] / (len(self.docs[m]) + Kalpha)
        for w in doc:
            log_per -= np.log(np.inner(phi[:, w], theta))
        N += len(doc)
    return np.exp(log_per / N)


def lda_learning(lda, iteration, voca):
    pre_perp = lda.perplexity()
    print ("initial perplexity=%f" % pre_perp)
    for i in range(iteration):
        lda.inference()
        perp = lda.perplexity()
        print ("-%d p=%f" % (i + 1, perp))
        if pre_perp:
            if pre_perp < perp:
                output_word_topic_dist(lda, voca)
                pre_perp = None
            else:
                pre_perp = perp
    output_word_topic_dist(lda, voca)


def output_word_topic_dist(lda, voca):
    zcount = np.zeros(lda.K, dtype=int)
    wordcount = [dict() for k in range(lda.K)]
    for xlist, zlist in zip(lda.docs, lda.z_m_n):
        for x, z in zip(xlist, zlist):
            zcount[z] += 1
            if x in wordcount[z]:
                wordcount[z][x] += 1
            else:
                wordcount[z][x] = 1

    phi = lda.worddist()
    for k in range(lda.K):
        print ("\n-- topic: %d (%d words)" % (k, zcount[k]))
        for w in np.argsort(-phi[k])[:20]:
            print ("%s: %f (%d)" % (voca[w], phi[k, w], wordcount[k].get(w, 0)))


def main():
    import optparse
    import vocabulary
    parser = optparse.OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename")
    parser.add_option("-c", dest="corpus", help="using range of Brown corpus' files(start:end)")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.5)
    parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.5)
    parser.add_option("-k", dest="K", type="int", help="number of topics", default=20)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
    parser.add_option("-s", dest="smartinit", action="store_true", help="smart initialize of parameters", default=False)
    parser.add_option("--stopwords", dest="stopwords", help="exclude stop words", action="store_true", default=False)
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    parser.add_option("--df", dest="df", type="int", help="threshold of document freaquency to cut words", default=0)
    (options, args) = parser.parse_args()
    if not (options.filename or options.corpus): parser.error("need corpus filename(-f) or corpus range(-c)")

    if options.filename:
        corpus = vocabulary.load_file(options.filename)
    else:
        corpus = vocabulary.load_corpus(options.corpus)
        if not corpus: parser.error("corpus range(-c) forms 'start:end'")
    if options.seed != None:
        np.random.seed(options.seed)

    voca = vocabulary.Vocabulary(options.stopwords)
    docs = [voca.doc_to_ids(doc) for doc in corpus]
    if options.df > 0: docs = voca.cut_low_freq(docs, options.df)

    lda = LDA(options.K, options.alpha, options.beta, docs, voca.size(), options.smartinit)
    print (
        "corpus=%d, words=%d, K=%d, a=%f, b=%f" % (
            len(corpus), len(voca.vocas), options.K, options.alpha, options.beta))

    # import cProfile
    # cProfile.runctx('lda_learning(lda, options.iteration, voca)', globals(), locals(), 'lda.profile')
    lda_learning(lda, options.iteration, voca)


if __name__ == "__main__":
    main()
