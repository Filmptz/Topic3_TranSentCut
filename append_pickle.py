import pickle

news_positive = []

news_negative = []

positive_path = 'data/version2/positive_trans_news.pickle'
negative_path = 'data/version2/negative_trans_news.pickle'

# CHECK DIMFORMAT
posError, negError = [], []
checkPosError, checkNegError = False, False
for i in news_positive :
    if(len(i) !=  2) :
        checkPosError = True
        posError.append(i)
for i in news_negative :
    if(len(i) !=  2) :
        checkNegError = True
        negError.append(i)
if(checkPosError or checkNegError) :
    print('------- Positive ERROR ------- ')
    print('-none-') if len(posError) == 0 else print(posError)
    print('------- Negative ERROR ------- ')
    print('-none-') if len(negError) == 0 else print(negError)
else :
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

    print('----- POSITIVE LAST INDEX AFTER APPEND -----')
    print(data_pickle_pos_new[-1])
    print('----- NEGATIVE LAST INDEX AFTER APPEND -----')
    print(data_pickle_neg_new[-1])

