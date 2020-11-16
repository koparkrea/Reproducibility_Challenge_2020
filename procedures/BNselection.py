import pickle, os




# pruning BatchNorm weigth 
def BNselection(start_index, num_conv = 1):
    ratio_path = os.path.join(os.getcwd(),'result', 'cifar10_criteria_PR_BN_last_2', 'which_ratio_epoch_599_600_best.pkl')
    criteria_path = os.path.join(os.getcwd(),'result', 'cifar10_criteria_PR_BN_last_2', 'which_criteria_epoch_599_600_best.pkl')

    with open(ratio_path, 'rb') as f:
        ratio_list = pickle.load(f)
    with open(criteria_path, 'rb') as f:
        criteria_list = pickle.load(f)
 
    width_selection = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cri_selection = ['l1-norm', 'l2-norm', 'l2-GM']
 
    ratio_list_l = ratio_list[start_index : start_index + num_conv]
    criteria_list_l = criteria_list[start_index : start_index +num_conv]

    index_ratio = []
    index_cri = []
    for ratio in ratio_list_l:
        index = [i for i,c in enumerate(width_selection) if c == ratio]
        index_ratio += index
    for cri in criteria_list_l:
        index = [i for i,c in enumerate(cri_selection) if c == cri]
        index_cri += index
    
    return index_ratio, index_cri
    
