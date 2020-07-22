import pandas as pd
import numpy as np
import sys
sys.path.append('../')

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, item_features, weighting=True):
        @staticmethod
        def prepare_matrix(data):
            user_item_matrix = pd.pivot_table(data,
                                              index='user_id', columns='item_id',
                                              values='quantity',
                                              aggfunc='count',
                                              fill_value=0
                                              )

            user_item_matrix = user_item_matrix.astype(float)

            return user_item_matrix

        @staticmethod
        def prepare_dicts(user_item_matrix):
            """Подготавливает вспомогательные словари"""

            userids = user_item_matrix.index.values
            itemids = user_item_matrix.columns.values

            matrix_userids = np.arange(len(userids))
            matrix_itemids = np.arange(len(itemids))

            id_to_itemid = dict(zip(matrix_itemids, itemids))
            id_to_userid = dict(zip(matrix_userids, userids))

            itemid_to_id = dict(zip(itemids, matrix_itemids))
            userid_to_id = dict(zip(userids, matrix_userids))

            return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

        @staticmethod
        def prepare_ctm(features):
            """Создаем словарь {item_id: 0/1}. 0/1 - факт принадлежности товара к СТМ"""

            own_items = features[['PRODUCT_ID', 'BRAND']]
            own_items['priznak'] = features['BRAND'].isin(['Private'])
            own_items = own_items.replace(to_replace=[False, True], value=[0, 1])

            return dict(zip(own_items['PRODUCT_ID'], own_items['priznak']))

        @staticmethod
        def fit_own_recommender(user_item_matrix):
            """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

            own_recommender = ItemItemRecommender(K=1, num_threads=4)
            own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

            return own_recommender

        @staticmethod
        def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
            """Обучает ALS"""

            model = AlternatingLeastSquares(factors=n_factors,
                                            regularization=regularization,
                                            iterations=iterations,
                                            num_threads=num_threads)
            model.fit(csr_matrix(user_item_matrix).T.tocsr())

            return model

        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()
        
        # Словарь {item_id: 0/1}. 0/1 - факт принадлежности товара к СТМ
        self.item_id_to_ctm = prepare_ctm.__func__(item_features)

        # Топ покупок каждого юзера с учётом CTM
        self.overall_top_purchases_w_ctm = data[data['item_id'].isin([int(x) for x in self.item_id_to_ctm.keys()])].groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases_w_ctm.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases_w_ctm = self.overall_top_purchases_w_ctm[self.overall_top_purchases_w_ctm['item_id'] != 999999]
        self.overall_top_purchases_w_ctm = self.overall_top_purchases_w_ctm.item_id.tolist()
        
        self.user_item_matrix = prepare_matrix.__func__(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = prepare_dicts.__func__(self.user_item_matrix)
        
        
        
        # Own recommender обучается до взвешивания матрицы
        self.own_recommender = fit_own_recommender.__func__(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = fit.__func__(self.user_item_matrix)


    def get_similar_items_recommendation(self, user, filter_ctm=True, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        # your_code
        # Практически полностью реализовали на прошлом вебинаре
        # Не забывайте, что нужно учесть параметр filter_ctm
        res = []
        if filter_ctm:
            dataset = self.overall_top_purchases_w_ctm
        else:
            dataset = self.overall_top_purchases
        recommends = self.model.similar_items(self.itemid_to_id[user], N)
        for item in recommends:
            res.append(item[0])

        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        res = []
        recommends = self.model.similar_users(self.userid_to_id[user], N)
        for item in recommends:
            res.append(item[0])

        return res


if __name__ == '__main__':
    data = pd.read_csv('../raw_data/retail_train.csv')
    item_features = pd.read_csv('../raw_data/product.csv')
    #print(item_features.columns)
    chelik = MainRecommender(data, item_features)
    print(chelik.get_similar_items_recommendation(33150))
    print(chelik.get_similar_users_recommendation(2375))