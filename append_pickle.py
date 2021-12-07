import pickle

news_positive = ()

news_negative = ()

positive_path = 'data/version2/positive_trans_news.pickle'
negative_path = 'data/version2/negative_trans_news.pickle'

with open(positive_path, 'rb') as pos: 
    data_pickle_pos = pickle.load(pos)

with open(negative_path, 'rb') as neg: 
    data_pickle_neg = pickle.load(neg)

for npos in news_positive:
    data_pickle_pos.append(npos)

for nneg in news_negative:
    data_pickle_neg.append(nneg)

with open(positive_path, 'wb') as posw: 
    pickle.dump(data_pickle_pos, posw)

with open(negative_path, 'wb') as negw:
    pickle.dump(data_pickle_neg, negw)

with open(positive_path, 'rb') as pos: 
    data_pickle_pos_new = pickle.load(pos)

with open(negative_path, 'rb') as neg: 
    data_pickle_neg_new = pickle.load(neg)

print('POSITIVE LAST INDEX AFTER APPEND')
print(data_pickle_pos_new[-1])
print('NEGATIVE LAST INDEX AFTER APPEND')
print(data_pickle_neg_new[-1])