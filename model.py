# -*- encoding: utf-8 -*-
import numpy as np

class LDAGibbsSampling():
    def __init__(self, documents, K, alpha, beta, iterations=100, save_step=10, beigin_save_iters=80):
        self.documents = documents
        # topic number
        self.K = K
        # doc-topic dirichlet prior parameter
        self.alpha = alpha
        # topic-word dirichlet prior parameter
        self.beta = beta
        # times of iteration
        self.iterations = iterations
        # interval between each saving
        self.save_step = save_step
        # begin save model
        self.begin_save_iters = beigin_save_iters
        # document number
        self.M = len(documents.docs)
        # vocabulary size
        self.V = len(documents.word2idx)
        # document m, count times of topic k
        self.nmk = np.zeros((self.M, K), dtype=np.int32)
        # sum for nmk
        self.nmk_sum = np.zeros(self.M, dtype=np.int32)
        # topic k, count time of word t
        self.nkv = np.zeros((K, self.V), dtype=np.int32)
        # sum for nkv
        self.nkv_sum = np.zeros(K, dtype=np.int32)
        # topic-vocabulary distribution
        self.phi = np.zeros((self.K, self.V))
        # document-topic distribution
        self.theta = np.zeros((self.M, self.K))
        # doc-words index
        self.doc_words = []
        # doc-(word's topic) index
        self.doc_words_topic = []
        # initialize model
        self.init_model()

    def init_model(self):
        for m, doc in enumerate(self.documents.docs):
            self.doc_words.append(np.zeros(len(doc.words), dtype=np.int32))
            self.doc_words_topic.append(np.zeros(len(doc.words), dtype=np.int32))
            for n, w in enumerate(doc.words):
                self.doc_words[m][n] = self.documents.word2idx.get(w)
                init_topic = np.random.randint(self.K)
                self.doc_words_topic[m][n] = init_topic
                self.nmk[m][init_topic] += 1
                self.nkv[init_topic][self.doc_words[m][n]] += 1
                self.nkv_sum[init_topic] += 1
            self.nmk_sum[m] = len(doc.words)

    def inference(self):
        if self.iterations < self.begin_save_iters + self.save_step:
            raise Exception('The number of iterations should be larger than '
                            + str(self.begin_save_iters + self.save_step))
        for i in range(self.iterations):
            print('iteration:', i)
            if i >= self.begin_save_iters and i % self.save_step == 0:
                self.update_distribution()
                self.save_model(i)
            else:
                for m in range(len(self.doc_words)):
                    for n in range(len(self.doc_words[m])):
                        # sample an new topic
                        new_topic = self.sample_topic(m, n)
                        self.doc_words_topic[m][n] = new_topic

    def update_distribution(self):
        # update topic-word distribution
        for k in range(self.K):
            for v in range(self.V):
                self.phi[k][v] = (self.nkv[k][v] + self.beta) / (self.nkv_sum[k] + self.V * self.beta)
        # update document-topic distribution
        for m in range(self.M):
            for k in range(self.K):
                self.theta[m][k] = (self.nmk[m][k] + self.alpha) / (self.nmk_sum[m] + self.K * self.alpha)


    def sample_topic(self, m, n):
        old_topic = self.doc_words_topic[m][n]
        self.nmk[m][old_topic] -= 1
        self.nkv[old_topic][self.doc_words[m][n]] -= 1
        self.nmk_sum[m] -= 1
        self.nkv_sum[old_topic] -= 1

        p = np.zeros(self.K)
        # using formula to compute probability
        for i in range(self.K):
            p[i] = (self.nkv[i][self.doc_words[m][n]] + self.beta) / (self.nkv_sum[i] + self.V * self.beta) \
                   * (self.nmk[m][i] + self.alpha) / (self.nmk_sum[m] + self.K * self.alpha)

        for i in range(1, self.K):
            p[i] += p[i-1]

        r = np.random.rand() * p[self.K - 1]
        new_topic = 0
        for _ in range(self.K):
            if r < p[new_topic]:
                break
            new_topic += 1

        self.nmk[m][new_topic] += 1
        self.nkv[new_topic][self.doc_words[m][n]] += 1
        self.nmk_sum[m] += 1
        self.nkv_sum[new_topic] += 1
        return new_topic

    def save_model(self, iteration):
        model_name = 'LDA_' + str(iteration)

        # lda.phi K * V
        f = open(model_name + '.phi', 'w', encoding='utf-8')
        for k in range(self.K):
            f.write('\t'.join(list(map(str, self.phi[k]))) + '\n')
        f.close()

        # lda.theta M * K
        f = open(model_name + '.theta', 'w', encoding='utf-8')
        for m in range(self.M):
            f.write('\t'.join(list(map(str, self.theta[m]))) + '\n')
        f.close()

        # lda.words_topic 每篇文章词语的主题
        f = open(model_name + '.words_topic', 'w', encoding='utf-8')
        for m in range(self.M):
            for n in range(len(self.doc_words[m])):
                f.write(self.documents.idx2word[self.doc_words[m][n]] +
                        ':' + str(self.doc_words_topic[m][n]) + '\t')
            f.write('\n')
        f.close()

        # lda.topic_words 每个主题下词语概率分布，逆序
        f = open(model_name + '.topic_words', 'w', encoding='utf-8')
        for k in range(self.K):
            f.write(str(k) + ':' + '\t')
            sorted_prob = sorted(enumerate(self.phi[k]), key=lambda x: x[1])
            for v in range(self.V):
                f.write(self.documents.idx2word[sorted_prob[v][0]] + ':' + str(sorted_prob[v][1]))
            f.write('\n')
        f.close()

    def predict(self, doc):
        mk = np.zeros(self.K, dtype=np.int32)
        _doc_words = []
        _doc_words_topic = []
        # init
        for w in doc.words:
            if w in self.documents.word2idx:
                _doc_words.append(self.documents.word2idx[w])
                init_topic = np.random.randint(self.K)
                _doc_words_topic.append(init_topic)
                mk[init_topic] += 1

        theta = np.zeros(self.K)

        for i in range(self.iterations):
            if i >= self.begin_save_iters and i % self.save_step == 0:
                for k in range(self.K):
                    theta[k] = (mk[k] + self.alpha) / (len(_doc_words) + self.K * self.alpha)
            for n in range(len(_doc_words)):
                new_topic = self.sample_topic_predict(n, mk, _doc_words, _doc_words_topic)
                _doc_words_topic[n] = new_topic
        return theta



    def sample_topic_predict(self, n, mk, _doc_words, _doc_words_topic):
        old_topic = _doc_words_topic[n]
        mk[old_topic] -= 1

        p = np.zeros(self.K)
        # freeze the part of topic-word distribution, only compute the part of document-topic distribution
        for i in range(self.K):
            p[i] = (self.nkv[i][_doc_words[n]] + self.beta) / (self.nkv_sum[i] + self.V * self.beta) \
                   * (mk[i] + self.alpha) / (len(_doc_words) - 1 + self.K * self.alpha)

        for i in range(1, self.K):
            p[i] += p[i - 1]

        r = np.random.rand() * p[self.K - 1]
        new_topic = 0
        for _ in range(self.K):
            if r < p[new_topic]:
                break
            new_topic += 1

        mk[new_topic] += 1
        return new_topic



