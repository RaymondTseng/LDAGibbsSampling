import os
from load import *
from model import LDAGibbsSampling

train_path = './train_data/'
test_path = './test_data/'


docs = Documents()

# load train data
for root, dirs, files in os.walk(train_path):
    for name in files:
        file_path = os.path.join(root, name)
        doc = Document(file_path)
        docs.add_doc(doc)



model = LDAGibbsSampling(docs, 3, 0.5, 0.1)
# training
model.inference()

# load test data
doc = Document(test_path + '20.txt')
# return document-topics distribution
theta = model.predict(doc)
print(theta)
