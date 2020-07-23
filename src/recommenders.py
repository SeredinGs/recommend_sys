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
    
    def __init__(self, data, item_features=None, weighting=True):
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
            if features is None:
                items = None
            else:
                own_items = features[['PRODUCT_ID', 'BRAND']]
                own_items['priznak'] = features['BRAND'].isin(['Private'])
                own_items = own_items.replace(to_replace=[False, True], value=[0, 1])
                items = dict(zip(own_items['PRODUCT_ID'], own_items['priznak']))

            return items

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
        if item_features is None:
            self.item_id_to_ctm = None
        else:
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

    def _update_dict(self, user_id):
        """Если появился новыю user / item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():

            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})
    
    def _get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)  # Товар похож на себя -> рекомендуем 2 товара
        top_rec = recs[1][0]  # И берем второй (не товар из аргумента метода)
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases[:N])
            recommendations = recommendations[:N]

        return recommendations
    
    def _get_recommendations(self, user, model, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
                                        user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                        N=N,
                                        filter_already_liked_items=False,
                                        filter_items=[self.itemid_to_id[999999]],
                                        recalculate_user=True)]

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    def get_als_recommendations(self, user, N=5):
        """Рекомендации через стандартные библиотеки implicit"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.model, N=N)

    def get_own_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.own_recommender, N=N)
    
    
    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == user].head(N)

        res = top_users_purchases['item_id'].apply(lambda x: self._get_similar_item(x)).tolist()
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        res = []
        recommends = self.model.similar_users(self.id_to_userid[user], N)
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