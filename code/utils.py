import pickle


def save_pickle(obj, file_name):
    with open(file_name, 'w') as f:
        s = pickle.dumps(obj, protocol=0)
        f.write(s.decode('latin-1'))

def load_pickle(file_name):
    with open(file_name, 'r') as f:
        s = f.read()
        obj = pickle.loads(s.encode('latin-1'))
    return obj