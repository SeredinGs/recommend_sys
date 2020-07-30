import numpy as np

def hit_rate_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])
    
    flags = np.isin(bought_list, recommended_list)
    
    hit_rate = (flags.sum() > 0) * 1
    
    return hit_rate
    
def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
        
    bought_list = np.array(bought_list)
    bought_list_shape = bought_list.shape[0]
    recommended_list = np.array(recommended_list)
    prices_recommended = np.array(prices_recommended[:bought_list_shape])
    
    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]
    
    flags = np.isin(bought_list, recommended_list)
    
    priced_flags = flags * prices_recommended
    
    precision = priced_flags.sum() / prices_recommended[:k].sum()
    
    return precision
    
def recall_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])
    
    flags = np.isin(bought_list, recommended_list)  # [False, False, True, True]
    
    recall = flags.sum() / len(bought_list)
    
    return recall
    
def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    
    bought_list = np.array(bought_list)
    bought_list_shape = bought_list.shape[0]
    recommended_list = np.array(recommended_list[:k])
    prices_recommended = np.array(prices_recommended[:bought_list_shape])
    
    flags = np.isin(bought_list, recommended_list)  # [False, False, True, True]
    
    priced_flags = flags * prices_recommended
    
    recall = flags.sum() / len(bought_list)
    
    return recall
    
def reciprocal_rank(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    rank = np.where(flags)[0][0]
    
    result = (1/rank)
    
    return result

def precision_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    

    bought_list = bought_list  # Тут нет [:k] !!
    
    if k < len(recommended_list):
        recommended_list = recommended_list[:k]

    flags = np.isin(bought_list, recommended_list)

    precision = flags.sum() / len(recommended_list)

    return precision

    
if __name__ == '__main__':
    recommended_list = [143, 156, 1134, 991, 27, 1543, 3345, 533, 11, 43] #id товаров
    bought_list = [521, 32, 143, 991]

    prices_recommended = [100, 90, 10, 450, 50, 37, 99, 120, 34, 100]
    prices_bought = [110, 190, 100, 450]
    
    print(precision_at_k(recommended_list, bought_list, k=7))
    print(reciprocal_rank(recommended_list, bought_list))
    print(money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5))
    print(recall_at_k(recommended_list, bought_list, k=5))
    print(money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5))
    print(hit_rate_at_k(recommended_list, bought_list, k=5))
    
