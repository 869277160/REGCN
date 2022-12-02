import json
import pickle

def process_json(path):
    """
    Reading json file
    :param path: Path to a JSON file.
    :return data: Dictionary with data.
    """
    data = json.load(open(path))

    return data

def generate_json(path, data):
    """
        Reading python object [list|dict|set]
        :param path: Path to a JSON file; data: python object.
    """
    with open(path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False)

def generate_pickle(path, data):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def process_pickle(path, encoding="latin1"):
    """
        Reading pickle file
        :param path: Path to a pickle file.
        :return data: Dictionary with data.
    """
    with open(path, 'rb') as handle:
        # data = pickle.load(open(path, "rb"), encoding=encoding)
        data = pickle.load(handle, encoding=encoding)
    return data

def process_text(path):
    """
        Reading pickle file
        :param path: Path to a pickle file.
        :return data: Dictionary with data.
    """
    data = {}
    with open(path, 'r') as lines:
        for i, key in enumerate(lines):
            key = key.strip("\n")
            data[key] = i
    return data




# def cosine_distance(x1, x2=None, eps=1e-8):
#     x2 = x1 if x2 is None else x2
#     w1 = x1.norm(p=2, dim=1, keepdim=True)
#     w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
#     return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
#
# def cosine_sim_score(x1, x2=None, eps=1e-8):
#     x2 = x1 if x2 is None else x2
#     w1 = x1.norm(p=2, dim=1, keepdim=True)
#     w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
#     return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)