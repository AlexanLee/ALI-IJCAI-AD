import ffm
import pandas as pd

# prepare the data
# (field, fea-index, value) format

train = pd.read_csv("input/train.txt", sep="\s+")
test = pd.read_csv("input/test.txt", sep="\s+")


def process(data):
    v = data.values
    d = []
    for i in range(len(v)):
        t = []
        for j in range(len(v[0])):
            if j < 10 and j not in [2, 3]:
                tmp = (1, j, v[i][j])
                t.append(tmp)
            elif 10 <= j < 16:
                tmp = (2, j, v[i][j])
                t.append(tmp)
            elif 16 <= j < 20 and j != 18:
                tmp = (3, j, v[i][j])
                t.append(tmp)
            elif 20 <= j:
                tmp = (4, j, v[i][j])
                t.append(tmp)
        d.append(t)
    return d


def ffm_model():
    X = process(train)
    Y = train['is_trade']

    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
    ffm_data = ffm.FFMData(X, Y)
    ffm_data_test = ffm.FFMData(X[418028:468028], Y[418028:468028])

    model = ffm.FFM(eta=0.1, lam=0.0001, k=4)
    model.fit(ffm_data, num_iter=200, val_data=ffm_data_test, metric='logloss', early_stopping=6,
              maximum=True)

    t = process(test)
    ffm_test = ffm.FFMData(t)
    pred = model.predict_proba(ffm_test)

    test['predicted_score'] = pred
    sub1 = test[['instance_id', 'predicted_score']]
    sub = pd.read_csv("input/test.txt", sep="\s+")
    sub = pd.merge(sub, sub1, on=['instance_id'], how='left')
    sub = sub.fillna(0)
    sub[['instance_id', 'predicted_score']].to_csv('result/result0422_ffm.txt', sep=" ",
                                                   index=False)


# model.save_model('result/res_ffm.bin')
# model = ffm.read_model('result/res_ffm.bin')
# print(model.predict(ffm_data_test))


if __name__ == '__main__':
    print('main')
    # process()
    ffm_model()
