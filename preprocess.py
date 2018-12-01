import jieba

class PreProcessor():

    def __init__(self, noise_words_path):
        self.noise_words = self.__load_noise_words__(noise_words_path)

    def __load_noise_words__(self, path):
        noise_words = set()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                noise_words.add(line.strip())
        return noise_words

    # clean noise words
    def process(self, text):
        words = jieba.lcut(text)
        _words = []
        for w in words:
            w = w.strip()
            if w and w not in self.noise_words:
                _words.append(w)
        return _words