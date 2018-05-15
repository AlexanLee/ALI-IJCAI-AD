import pandas as pd


def source():
    train = pd.read_csv('input/train_sample_a.txt', sep='\s+')
    d = train.sample(frac=0.2)
    d.to_csv('input/train_sample_b.txt', sep=' ', index=False)
    # test = pd.read_csv('input/test.txt', sep='\s+')

    # print(len(train))
    # print(len(test))

    # for c in train.columns:
    #     print(c)


def feature():
    fea = pd.read_csv('result/feature_a_0422.txt', sep='\s+')
    print(len(fea))
    print(fea.columns)
    col = [c for c in fea if
           c not in ['is_trade', 'item_category_list', 'item_property_list',
                     'predict_category_property', 'instance_id',
                     'context_id', 'realtime', 'context_timestamp']]
    data = fea[col]
    for c in fea.columns:
        print(c)
    print(len(fea['instance_id']))


if __name__ == '__main__':
    print('main')
    source()
    # feature()
