def concat_dicts(d1, d2):
    """ Concatenates two dictionaries d1 and d2 which have the same keys. """
    combined = {}

    for key in d1.keys():
        combined[key] = np.append(d1[key], d2[key])

    return combined
