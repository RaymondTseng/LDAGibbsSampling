from preprocess import PreProcessor

preprocessor = PreProcessor('./resources/noise_words.txt')

class Documents(object):
    def __init__(self):
        self.docs = []
        self.word2idx = {}
        self.words_count = {}
        self.idx2word = []

    # add document
    def add_doc(self, doc):
        self.docs.append(doc)
        for w in doc.words:
            if w in self.word2idx:
                self.words_count[w] += 1
            else:
                self.word2idx[w] = len(self.word2idx)
                self.idx2word.append(w)
                self.words_count[w] = 1


class Document(object):
    def __init__(self, path):
        self.words = []
        self.__load_doc__(path)

    def __load_doc__(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.words.extend(preprocessor.process(line.strip()))





