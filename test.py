import pickle, os

result = os.path.join(os.getcwd(), 'result', 'cifar10_criteria_I2')

for i in range(600):
    result_path = os.path.join(result, 'gumbel_0.1_5.0_epoch' +'_' + str(i) + '_600_best.pkl')
    if os.path.exists(result_path):
        with open(result_path, 'rb') as f:
            fi = pickle.load(f)
            print(fi)
