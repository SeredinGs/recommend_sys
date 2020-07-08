import numpy as np

def precision_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    bought_list = bought_list  # Тут нет [:k] !!
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
