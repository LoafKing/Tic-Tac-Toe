import pickle

with open('Q_table_dict.pkl', 'rb') as f:
    Q_table_pkl = pickle.load(f)
    print (Q_table_pkl)
    for one in Q_table_pkl:
        print (type(one))
