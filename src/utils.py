import numpy as np

def prefilter_items(data, take_n_popular=5000):
    """Предфильтрация товаров"""

    
    # 1. Удаление товаров, со средней ценой < 1$
    data['item_price'] = data['sales_value'] / (np.maximum(data['quantity'], 1)) # делим выручку на количество товара
    data = data[data['item_price'] > 1]
    
    # 2. Удаление товаров со соедней ценой > 30$
    data = data[data['item_price'] < 30]
    
    # 3. Придумайте свой фильтр
    # в качестве фильтра будут использоваться покупки течении последнего месяца
    data = data[data['week_no'] <= 4] 
    
    # 4. Выбор топ-N самых популярных товаров (N = take_n_popular)
    popular = data.groupby('item_id')['sales_value'].sum().reset_index()
    popular.sort_values('sales_value', ascending=False, inplace=True)
    
    popular = popular.head(take_n_popular).item_id
    
    data = data.loc[data['item_id'].isin(popular)]
    
    return data