# coding: utf-8


import time
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from contextlib import contextmanager


@contextmanager
def timer(name):
    start_time = time.time()
    yield
    print('%s done in %.2f s' % (name, time.time() - start_time))


def add_count(df, cols, cname, value):
    df_count = pd.DataFrame(df.groupby(cols)[value].count()).reset_index()
    df_count.columns = cols + [cname]
    df = df.merge(df_count, on=cols, how='left')
    del df_count
    gc.collect()
    return df


def add_mean(df, cols, cname, value):
    df_mean = pd.DataFrame(df.groupby(cols)[value].mean()).reset_index()
    df_mean.columns = cols + [cname]
    df = df.merge(df_mean, on=cols, how='left')
    del df_mean
    gc.collect()
    return df


def add_std(df, cols, cname, value):
    df_std = pd.DataFrame(df.groupby(cols)[value].std()).reset_index()
    df_std.columns = cols + [cname]
    df = df.merge(df_std, on=cols, how='left')
    del df_std
    gc.collect()
    return df


def add_nunique(df, cols, cname, value):
    df_nunique = pd.DataFrame(df.groupby(cols)[value].nunique()).reset_index()
    df_nunique.columns = cols + [cname]
    df = df.merge(df_nunique, on=cols, how='left')
    del df_nunique
    gc.collect()
    return df


def add_cumcount(df, cols, cname):
    df[cname] = df.groupby(cols).cumcount() + 1
    return df


def true_predict_count(true_lst, pred_lst):
    items, cnt = true_lst.split(';'), 0
    for i in pred_lst:
        if i in items:
            cnt += 1
    return cnt


def true_predict_precision(true_lst, pred_lst):
    return true_predict_count(true_lst, pred_lst) / len(pred_lst)


def true_predict_recall(true_lst, pred_lst):
    return true_predict_count(true_lst, pred_lst) / len(true_lst.split(';'))


with timer('Read train and test'):
    train = pd.read_csv('input/train.txt', delimiter=' ')
    test = pd.read_csv('input/test.txt', delimiter=' ')

    train = train.sort_values(by='context_timestamp').reset_index().iloc[:, 1:]
    test['raw_index'] = np.arange(test.shape[0])
    test = test.sort_values(by='context_timestamp').reset_index().iloc[:, 1:]

    print(train.shape, test.shape)
    print(train.columns)

with timer('Feature engineering'):
    df_full = [train, test]
    df_full_processed = []

    # get the customers (users) gender ratio for each shop
    df_shop_gender_ratio = train.groupby(['shop_id'])['user_gender_id'] \
        .agg([lambda x: np.mean(x == 0)]) \
        .reset_index().rename(columns={'<lambda>': 'shop_user_gender_ratio'})

    # get the average age level of customers for each shop
    df_shop_avg_age_level = train.groupby(['shop_id'])['user_age_level'] \
        .mean() \
        .reset_index() \
        .rename(columns={'user_age_level': 'user_avg_age_level'})

    for df in df_full:
        df['item_category_1'] = df['item_category_list'].str.split(';').apply(
                lambda x: x[1]).astype(int)
        df['context_page_id'] = df['context_page_id'] % 4000

        # convert timestamp into 24 hours
        df['context_timestamp'] = (df['context_timestamp'] - df['context_timestamp'] // (
            3600 * 24) * (3600 * 24)) // 3600

        # count features
        df = add_count(df, cols=['user_id'], cname='user_count', value='item_id')
        df = add_count(df, cols=['user_id', 'shop_id'], cname='user_shop_count', value='item_id')
        df = add_count(df, cols=['user_id', 'item_id'], cname='user_item_count',
                       value='item_brand_id')
        df = add_count(df, cols=['user_id', 'shop_id', 'item_id'], cname='user_shop_item_count',
                       value='item_brand_id')

        # cumulative count features
        df = add_cumcount(df, cols=['user_id'], cname='user_cumcount')
        df = add_cumcount(df, cols=['user_id', 'shop_id'], cname='user_shop_cumcount')
        df = add_cumcount(df, cols=['user_id', 'item_id'], cname='user_item_cumcount')
        df = add_cumcount(df, cols=['user_id', 'shop_id', 'item_id'],
                          cname='user_shop_item_cumcount')

        # unique count features
        df = add_nunique(df, cols=['shop_id'], cname='shop_item_nunique', value='item_id')
        df = add_nunique(df, cols=['user_id'], cname='user_item_nunique', value='item_id')
        df = add_nunique(df, cols=['user_id'], cname='user_shop_nunique', value='shop_id')

        # average features
        df = add_mean(df, cols=['shop_id'], cname='shop_price_mean', value='item_price_level')
        df = add_mean(df, cols=['item_id'], cname='item_price_mean', value='item_price_level')
        df = add_mean(df, cols=['user_id'], cname='user_item_price_mean', value='item_price_level')
        df = add_mean(df, cols=['user_id'], cname='user_item_collected_mean',
                      value='item_collected_level')

        df = df.merge(df_shop_gender_ratio, on='shop_id', how='left')
        df = df.merge(df_shop_avg_age_level, on='shop_id', how='left')

        df['predict_category'] = df['predict_category_property'].str.split(';').apply(
                lambda x: [p[0] for p in [p.split(':') for p in x]])

        df['predict_category_count'] = df.apply(
                lambda x: true_predict_count(x['item_category_list'], x['predict_category']),
                axis=1)
        df['predict_category_recall'] = df.apply(
                lambda x: true_predict_recall(x['item_category_list'], x['predict_category']),
                axis=1)

        df_full_processed.append(df)

with timer('Prepare for LGBM'):
    # columns to excluded
    features_exclude = ['instance_id'
        , 'item_category_list'
        , 'item_property_list'
        , 'item_brand_id'
        , 'context_id'
        , 'is_trade'
        , 'predict_category_property'
        , 'user_gender_id'
        , 'shop_star_level'
        , 'user_id'
        , 'predict_category'
        , 'predict_property']

    # features
    features = [f for f in df_full_processed[0].columns if f not in features_exclude]

    lgb_params = {
        'application': 'binary'
        , 'learning_rate': 0.009
        , 'max_depth': 4
        , 'num_leaves': 15
        , 'min_child_samples': 20
        , 'subsample': 0.8
        , 'colsample_bytree': 0.3
        , 'metric': 'binary_logloss'
        , 'data_random_seed': 42
        , 'nthread': 4
    }

    X_train_lgb = lgb.Dataset(df_full_processed[0][features].values, free_raw_data=True)
    X_train_lgb.set_label(df_full_processed[0]['is_trade'].values)

with timer('Train LGBM'):
    clf_lgb = lgb.train(lgb_params
                        , train_set=X_train_lgb
                        , num_boost_round=4500
                        , feature_name=features)

with timer('Predict test'):
    test['predicted_score'] = clf_lgb.predict(df_full_processed[1][features],
                                              num_iteration=clf_lgb.best_iteration)
    test = test.sort_values(by='raw_index').reset_index().iloc[:, 1:]
    test[['instance_id', 'predicted_score']].to_csv('result/result0430.txt', index=False, sep=' ')
