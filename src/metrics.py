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

    if recommended_list.shape >= bought_list.shape:
        flags = np.isin(bought_list, recommended_list)
    else:
        flags = np.isin(recommended_list, bought_list)
    #print(flags, prices_recommended)
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
    #recommended_list = [143, 156, 1134, 991, 27, 1543, 3345, 533, 11, 43] #id товаров
    #bought_list = [521, 32, 143, 991]

    recommended_list = [839818, 1007512, 9297615, 5577022, 9803545]
    bought_list = [880007,   883616,   931136,   938004,   940947,   947267,   952924,  958046,
     959219,   961554,   962568,   965766,   976335,   979707,   986947,   990656,
     991024,  1004906,  1005186,  1037507,  1037863,  1045285,  1046816,  1049998,
     1062002,  1075074,  1087268,  1112333,  1131321,  1132231,  5582712,  9527558,
     12330539, 13877012, 15800711, 15830875]
    prices_recommended = [3.4299999999999997, 2.9900000000000007, 3.2337499999999997, 2.98, 3.4952631578947373]
    #prices_recommended = [100, 90, 10, 450, 50, 37, 99, 120, 34, 100]
    prices_bought = [110, 190, 100, 450]
    
    #print(precision_at_k(recommended_list, bought_list, k=7))
    #print(reciprocal_rank(recommended_list, bought_list))
    #print(money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5))
    #print(recall_at_k(recommended_list, bought_list, k=5))
    print(money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5))
    #print(hit_rate_at_k(recommended_list, bought_list, k=5))
    
