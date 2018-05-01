import time
import warnings

import lightgbm as lgb
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt


def base_process(data):
    lbl = preprocessing.LabelEncoder()
    print(
        '--------------------------------------------------------------item--------------------------------------------------------------')
    data['len_item_category'] = data['item_category_list'].map(lambda x: len(str(x).split(';')))
    data['len_item_property'] = data['item_property_list'].map(lambda x: len(str(x).split(';')))
    for i in range(1, 3):
        data['item_category_list' + str(i)] = lbl.fit_transform(data['item_category_list'].map(
                lambda x: str(str(x).split(';')[i]) if len(
                        str(x).split(';')) > i else ''))
    for i in range(10):
        data['item_property_list' + str(i)] = lbl.fit_transform(data['item_property_list'].map(
                lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))
    for col in ['item_id', 'item_brand_id', 'item_city_id']:
        data[col] = lbl.fit_transform(data[col])
    print(
        '------------------------------------------------------------user--------------------------------------------------------------')
    for col in ['user_id']:
        data[col] = lbl.fit_transform(data[col])
    print('user 0,1 feature')
    data['gender0'] = data['user_gender_id'].apply(lambda x: 1 if x == -1 else 2)
    data['age0'] = data['user_age_level'].apply(
            lambda x: 1 if x == 1000 | x == 1001 | x == -1 else 2)
    data['occupation0'] = data['user_occupation_id'].apply(occupation)
    data['star0'] = data['user_star_level'].apply(use_star_level)
    print(
        '--------------------------------------------------------------context--------------------------------------------------------------')
    data['realtime'] = data['context_timestamp'].apply(timestamp_datetime)
    data['realtime'] = pd.to_datetime(data['realtime'])
    data['day'] = data['realtime'].dt.day
    data['hour'] = data['realtime'].dt.hour
    data['len_predict_category_property'] = data['predict_category_property'].map(
            lambda x: len(str(x).split(';')))
    for i in range(5):
        data['predict_category_property' + str(i)] = lbl.fit_transform(
                data['predict_category_property'].map(
                        lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))
    print('context 0,1 feature')
    data['context_page0'] = data['context_page_id'].apply(
            lambda x: 1 if x == 4001 | x == 4002 | x == 4003 | x == 4004 | x == 4005 else 2)
    print(
        '--------------------------------------------------------------shop--------------------------------------------------------------')
    for col in ['shop_id']:
        data[col] = lbl.fit_transform(data[col])
    # data['shop_score_delivery0'] = data['shop_score_delivery'].apply(
    #         lambda x: 0 if 0.98 >= x >= 0.96 else 1)
    return data


def occupation(x):
    if x == -1 | x == 2003:
        return 1
    elif x == 2002:
        return 2
    else:
        return 3


def use_star_level(x):
    if x == -1 | x == 3000:
        return 1
    elif x == 3009 | x == 3010:
        return 2
    else:
        return 3


def map_hour(x):
    if (x >= 7) & (x <= 12):
        return 1
    elif (x >= 13) & (x <= 22):
        return 2
    else:
        return 3


def deliver(x):
    step = 0.1
    for i in range(1, 20):
        if (x >= 4.1 + step * (i - 1)) & (x <= 4.1 + step * i):
            return i + 1
    if x == -5:
        return 1


def deliver1(x):
    if (x >= 2) & (x <= 4):
        return 1
    elif (x >= 5) & (x <= 7):
        return 2
    else:
        return 3


def review(x):
    step = 0.02
    for i in range(1, 30):
        if (x >= 0.714 + step * (i - 1)) & (x <= 0.714 + step * i):
            return i + 1
    if x == -1:
        return 1


def review1(x):
    if (x >= 2) & (x <= 12):
        return 1
    elif (x >= 13) & (x <= 15):
        return 2
    else:
        return 3


def service(x):
    step = 0.1
    for i in range(1, 20):
        if (x >= 3.93 + step * (i - 1)) & (x <= 3.93 + step * i):
            return i + 1
    if x == -1:
        return 1


def service1(x):
    if (x >= 2) & (x <= 7):
        return 1
    elif (x >= 8) & (x <= 9):
        return 2
    else:
        return 3


def describe(x):
    step = 0.1
    for i in range(1, 30):
        if (x >= 3.93 + step * (i - 1)) & (x <= 3.93 + step * i):
            return i + 1
    if x == -1:
        return 1


def describe1(x):
    if (x >= 2) & (x <= 8):
        return 1
    elif (x >= 9) & (x <= 10):
        return 2
    else:
        return 3


def encodeHour(data):
    data['hour_map'] = data['hour'].apply(map_hour)
    return data


def shop_fenduan(data):
    data['shop_score_delivery'] *= 5
    data = data[data['shop_score_delivery'] != -5]
    data['deliver_map'] = data['shop_score_delivery'].apply(deliver)
    data['deliver_map'] = data['deliver_map'].apply(deliver1)
    print(data.deliver_map.value_counts())

    data['shop_score_service'] = data['shop_score_service'] * 5
    data = data[data['shop_score_service'] != -5]
    data['service_map'] = data['shop_score_service'].apply(service)
    data['service_map'] = data['service_map'].apply(service1)
    # del data['shop_score_service']
    print(data.service_map.value_counts())

    data['shop_score_description'] = data['shop_score_description'] * 5
    data = data[data['shop_score_description'] != -5]
    data['de_map'] = data['shop_score_description'].apply(describe)
    data['de_map'] = data['de_map'].apply(describe1)
    # del data['shop_score_description']
    print(data.de_map.value_counts())

    data = data[data['shop_review_positive_rate'] != -1]
    data['review_map'] = data['shop_review_positive_rate'].apply(review)
    data['review_map'] = data['review_map'].apply(review1)
    print(data.review_map.value_counts())

    data['normal_shop'] = data.apply(
            lambda x: 1 if (x.deliver_map == 3) & (x.service_map == 3) & (x.de_map == 3) & (
                x.review_map == 3) else 0,
            axis=1)
    del data['de_map']
    del data['service_map']
    del data['deliver_map']
    del data['review_map']
    return data


def zuhe(data):
    for col in ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']:
        data[col] = data[col].apply(lambda x: 0 if x == -1 else x)

    for col in ['item_sales_level', 'item_price_level', 'item_collected_level',
                'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level',
                'shop_review_num_level', 'shop_star_level']:
        data[col] = data[col].astype(str)

    print('item')
    data['sale_price'] = data['item_sales_level'] + data['item_price_level']
    data['sale_collect'] = data['item_sales_level'] + data['item_collected_level']
    data['price_collect'] = data['item_price_level'] + data['item_collected_level']

    print('user')
    data['gender_age'] = data['user_gender_id'] + data['user_age_level']
    data['gender_occ'] = data['user_gender_id'] + data['user_occupation_id']
    data['gender_star'] = data['user_gender_id'] + data['user_star_level']

    print('shop')
    data['review_star'] = data['shop_review_num_level'] + data['shop_star_level']

    for col in ['item_sales_level', 'item_price_level', 'item_collected_level', 'sale_price',
                'sale_collect', 'price_collect',
                'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level',
                'gender_age', 'gender_occ', 'gender_star',
                'shop_review_num_level', 'shop_star_level', 'review_star']:
        data[col] = data[col].astype(int)

    del data['review_star']

    return data


def item(data):
    print('brand,price salse collected level')

    itemcnt = data.groupby(['item_id'], as_index=False)['instance_id'].agg({'item_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_id'], how='left')

    for col in ['item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
                'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_id'], as_index=False)['instance_id'].agg(
                {str(col) + '_item_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_id'], how='left')
        data[str(col) + '_item_prob'] = data[str(col) + '_item_cnt'] / data['item_cnt']
    del data['item_cnt']

    print('price salse collected level')

    itemcnt = data.groupby(['item_brand_id'], as_index=False)['instance_id'].agg(
            {'item_brand_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_brand_id'], how='left')

    for col in ['item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level',
                'item_pv_level']:
        itemcnt = data.groupby([col, 'item_brand_id'], as_index=False)['instance_id'].agg(
                {str(col) + '_brand_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_brand_id'], how='left')
        data[str(col) + '_brand_prob'] = data[str(col) + '_brand_cnt'] / data['item_brand_cnt']
    del data['item_brand_cnt']

    print('city:item_price_level,item_sales_level,item_collected_level,item_pv_level')

    itemcnt = data.groupby(['item_city_id'], as_index=False)['instance_id'].agg(
            {'item_city_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_city_id'], how='left')
    for col in ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_city_id'], as_index=False)['instance_id'].agg(
                {str(col) + '_city_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_city_id'], how='left')
        data[str(col) + '_city_prob'] = data[str(col) + '_city_cnt'] / data['item_city_cnt']
    del data['item_city_cnt']

    print('price:item_sales_level,item_collected_level,item_pv_level')

    itemcnt = data.groupby(['item_price_level'], as_index=False)['instance_id'].agg(
            {'item_price_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_price_level'], how='left')
    for col in ['item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_city_id'], as_index=False)['instance_id'].agg(
                {str(col) + '_price_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_city_id'], how='left')
        data[str(col) + '_price_prob'] = data[str(col) + '_price_cnt'] / data['item_price_cnt']
    del data['item_price_cnt']

    print('item_sales_level:item_collected_level,item_pv_level')

    itemcnt = data.groupby(['item_sales_level'], as_index=False)['instance_id'].agg(
            {'item_salse_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_sales_level'], how='left')
    for col in ['item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_sales_level'], as_index=False)['instance_id'].agg(
                {str(col) + '_salse_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_sales_level'], how='left')
        data[str(col) + '_salse_prob'] = data[str(col) + '_salse_cnt'] / data['item_salse_cnt']
    del data['item_salse_cnt']

    print('item_collected_level:item_pv_level')

    itemcnt = data.groupby(['item_collected_level'], as_index=False)['instance_id'].agg(
            {'item_coll_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_collected_level'], how='left')
    for col in ['item_pv_level']:
        itemcnt = data.groupby([col, 'item_collected_level'], as_index=False)['instance_id'].agg(
                {str(col) + '_coll_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_collected_level'], how='left')
        data[str(col) + '_coll_prob'] = data[str(col) + '_coll_cnt'] / data['item_coll_cnt']
    del data['item_coll_cnt']

    return data


def user(data):
    itemcnt = data.groupby(['user_id'], as_index=False)['instance_id'].agg({'user_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_id'], how='left')

    for col in ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']:
        itemcnt = data.groupby([col, 'user_id'], as_index=False)['instance_id'].agg(
                {str(col) + '_user_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'user_id'], how='left')
        data[str(col) + '_user_prob'] = data[str(col) + '_user_cnt'] / data['user_cnt']
    del data['user_cnt']

    itemcnt = data.groupby(['user_gender_id'], as_index=False)['instance_id'].agg(
            {'user_gender_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_gender_id'], how='left')

    for col in ['user_age_level', 'user_occupation_id', 'user_star_level']:
        itemcnt = data.groupby([col, 'user_gender_id'], as_index=False)['instance_id'].agg(
                {str(col) + '_user_gender_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'user_gender_id'], how='left')
        data[str(col) + '_user_gender_prob'] = data[str(col) + '_user_gender_cnt'] / data[
            'user_gender_cnt']
    del data['user_gender_cnt']

    print('user_age_level:user_occupation_id,user_star_level')
    itemcnt = data.groupby(['user_age_level'], as_index=False)['instance_id'].agg(
            {'user_age_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_age_level'], how='left')

    for col in ['user_occupation_id', 'user_star_level']:
        itemcnt = data.groupby([col, 'user_age_level'], as_index=False)['instance_id'].agg(
                {str(col) + '_user_age_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'user_age_level'], how='left')
        data[str(col) + '_user_age_prob'] = data[str(col) + '_user_age_cnt'] / data['user_age_cnt']
    del data['user_age_cnt']

    print('user_occupation_id user_star_level')
    itemcnt = data.groupby(['user_occupation_id'], as_index=False)['instance_id'].agg(
            {'user_occ_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_occupation_id'], how='left')
    for col in ['user_star_level']:
        itemcnt = data.groupby([col, 'user_occupation_id'], as_index=False)['instance_id'].agg(
                {str(col) + '_user_occ_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'user_occupation_id'], how='left')
        data[str(col) + '_user_occ_prob'] = data[str(col) + '_user_occ_cnt'] / data['user_occ_cnt']
    del data['user_occ_cnt']
    return data


def user_item(data):
    itemcnt = data.groupby(['user_id'], as_index=False)['instance_id'].agg({'user_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_id'], how='left')
    print('user:item_id,item_brand_id')
    for col in ['item_id',
                'item_brand_id', 'item_city_id', 'item_price_level',
                'item_sales_level', 'item_collected_level', 'item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_id'], as_index=False)['instance_id'].agg(
                {str(col) + '_user_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_id'], how='left')
        data[str(col) + '_user_prob'] = data[str(col) + '_user_cnt'] / data['user_cnt']

    print('user_gender:item_id,item_brand_id')
    itemcnt = data.groupby(['user_gender_id'], as_index=False)['instance_id'].agg(
            {'user_gender_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_gender_id'], how='left')
    for col in ['item_id',
                'item_brand_id', 'item_city_id', 'item_price_level',
                'item_sales_level', 'item_collected_level', 'item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_gender_id'], as_index=False)['instance_id'].agg(
                {str(col) + '_user_gender_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_gender_id'], how='left')
        data[str(col) + '_user_gender_prob'] = data[str(col) + '_user_gender_cnt'] / data[
            'user_gender_cnt']

    print('user_age_level:item_id,item_brand_id')
    itemcnt = data.groupby(['user_age_level'], as_index=False)['instance_id'].agg(
            {'user_age_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_age_level'], how='left')
    for col in ['item_id',
                'item_brand_id', 'item_city_id', 'item_price_level',
                'item_sales_level', 'item_collected_level', 'item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_age_level'], as_index=False)['instance_id'].agg(
                {str(col) + '_user_age_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_age_level'], how='left')
        data[str(col) + '_user_age_prob'] = data[str(col) + '_user_age_cnt'] / data['user_age_cnt']

    print('user_occupation_id:item_id,item_brand_id')
    itemcnt = data.groupby(['user_occupation_id'], as_index=False)['instance_id'].agg(
            {'user_occ_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_occupation_id'], how='left')
    for col in ['item_id',
                'item_brand_id', 'item_city_id', 'item_price_level',
                'item_sales_level', 'item_collected_level', 'item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_occupation_id'], as_index=False)[
            'instance_id'].agg({str(col) + '_user_occ_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_occupation_id'], how='left')
        data[str(col) + '_user_occ_prob'] = data[str(col) + '_user_occ_cnt'] / data['user_occ_cnt']

    return data


def user_shop(data):
    print('user:shop_id,shop_review_num_level')

    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_id'], as_index=False)['instance_id'].agg(
                {str(col) + '_user_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_id'], how='left')
        data[str(col) + '_user_prob'] = data[str(col) + '_user_cnt'] / data['user_cnt']
    del data['user_cnt']

    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_gender_id'], as_index=False)['instance_id'].agg(
                {str(col) + '_user_gender_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_gender_id'], how='left')
        data[str(col) + '_user_gender_prob'] = data[str(col) + '_user_gender_cnt'] / data[
            'user_gender_cnt']
    del data['user_gender_cnt']

    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_age_level'], as_index=False)['instance_id'].agg(
                {str(col) + '_user_age_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_age_level'], how='left')
        data[str(col) + '_user_age_prob'] = data[str(col) + '_user_age_cnt'] / data['user_age_cnt']
    del data['user_age_cnt']

    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_occupation_id'], as_index=False)[
            'instance_id'].agg(
                {str(col) + '_user_occ_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_occupation_id'], how='left')
        data[str(col) + '_user_occ_prob'] = data[str(col) + '_user_occ_cnt'] / data['user_occ_cnt']
    del data['user_occ_cnt']

    return data


def shop_item(data):
    itemcnt = data.groupby(['shop_id'], as_index=False)['instance_id'].agg({'shop_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['shop_id'], how='left')
    for col in ['item_id',
                'item_brand_id', 'item_city_id', 'item_price_level',
                'item_sales_level', 'item_collected_level', 'item_pv_level']:
        item_shop_cnt = data.groupby([col, 'shop_id'], as_index=False)['instance_id'].agg(
                {str(col) + '_shop_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'shop_id'], how='left')
        data[str(col) + '_shop_prob'] = data[str(col) + '_shop_cnt'] / data['shop_cnt']
    del data['shop_cnt']

    itemcnt = data.groupby(['shop_review_num_level'], as_index=False)['instance_id'].agg(
            {'shop_rev_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['shop_review_num_level'], how='left')
    for col in ['item_id',
                'item_brand_id', 'item_city_id', 'item_price_level',
                'item_sales_level', 'item_collected_level', 'item_pv_level']:
        item_shop_cnt = data.groupby([col, 'shop_review_num_level'], as_index=False)[
            'instance_id'].agg({str(col) + '_shop_rev_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'shop_review_num_level'], how='left')
        data[str(col) + '_shop_rev_prob'] = data[str(col) + '_shop_rev_cnt'] / data['shop_rev_cnt']
    del data['shop_rev_cnt']

    return data


def lgbCV(train, test):
    col = [c for c in train if
           c not in ['is_trade', 'item_category_list', 'item_property_list',
                     'predict_category_property', 'instance_id',
                     'context_id', 'realtime', 'context_timestamp']]
    X = train[col]
    y = train['is_trade'].values
    X_tes = test[col]
    y_tes = test['is_trade'].values
    print('Training LGBM model...')
    lgb0 = lgb.LGBMClassifier(
            objective='binary',
            num_leaves=35,
            max_depth=8,
            learning_rate=0.03,
            seed=2018,
            colsample_bytree=0.8,
            subsample=0.9,
            min_sum_hessian_in_leaf=100,
            n_estimators=20000)
    lgb_model = lgb0.fit(X, y, eval_set=[(X_tes, y_tes)], early_stopping_rounds=200)
    best_iter = lgb_model.best_iteration_
    predictors = [i for i in X.columns]
    feat_imp = pd.Series(lgb_model.feature_importances_, predictors).sort_values(ascending=False)
    print(feat_imp)
    print(feat_imp.shape)
    pred = lgb_model.predict_proba(test[col])[:, 1]
    test['pred'] = pred
    test['index'] = range(len(test))
    print('los:', log_loss(test['is_trade'], test['pred']))
    print('best_iter:', best_iter)
    return best_iter


def sub(train, test, best_iter):
    col = [c for c in train if
           c not in ['is_trade', 'item_category_list', 'item_property_list',
                     'predict_category_property', 'instance_id',
                     'context_id', 'realtime', 'context_timestamp']]
    X = train[col]
    y = train['is_trade'].values
    print('Training LGBM model...')
    lgb0 = lgb.LGBMClassifier(
            objective='binary',
            num_leaves=35,
            max_depth=8,
            learning_rate=0.03,
            seed=2018,
            colsample_bytree=0.8,
            subsample=0.9,
            min_sum_hessian_in_leaf=100,
            n_estimators=best_iter)
    lgb_model = lgb0.fit(X, y)
    predictors = [i for i in X.columns]
    feat_imp = pd.Series(lgb_model.feature_importances_, predictors).sort_values(ascending=False)
    print("feat_imp:", feat_imp)
    print("feat_imp_shape:", feat_imp.shape)
    pred = lgb_model.predict_proba(test[col])[:, 1]
    test['predicted_score'] = pred
    sub1 = test[['instance_id', 'predicted_score']]
    sub = pd.read_csv("input/test.txt", sep="\s+")
    sub = pd.merge(sub, sub1, on=['instance_id'], how='left')
    sub = sub.fillna(0)
    sub[['instance_id', 'predicted_score']].to_csv('result/result0502.txt', sep=" ", index=False)


def feature():
    train = pd.read_csv("input/train.txt", sep="\s+")
    test = pd.read_csv("input/round1_test_a.txt", sep="\s+")
    data = pd.concat([train, test])
    data = data.drop_duplicates(subset='instance_id')
    print('make feature')
    data = base_process(data)
    data = encodeHour(data)
    data = shop_fenduan(data)
    data = item(data)
    data = user(data)
    col = [c for c in data if
           c not in ['is_trade']]
    X = data[col]
    X['is_trade']=data['is_trade'].values
    X.to_csv('input/feature_test.txt', sep=" ", index=False)


def train():
    data = pd.read_csv("input/feature_test.txt", sep="\s+")
    train = data[data.is_trade.notnull()]
    X_train, X_test = train_test_split(train, test_size=0.2, random_state=0)
    best_iter = lgbCV(X_train, X_test)
    test = data[data.is_trade.isnull()]
    sub(train, test, best_iter)
    # result(LogisticRegression(C=10, n_jobs=-1), train, test)
    # model_log_loss(LogisticRegression(C=10, n_jobs=-1), train)


if __name__ == "__main__":
    # feature()
    train()
