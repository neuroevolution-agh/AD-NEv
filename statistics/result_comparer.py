import numpy as np

wadi_anomalies_indexes = {+8: list(range(5103, 6603)) + list(range(59053, 59642)),
                          5: list(range(5103, 6603)) + list(range(59053, 59642)),
                          0: list(range(60903, 62543)) + list(range(163593, 164223)),
                          60: list(range(60903, 62543)) + list(range(163593, 164223)),
                          62: list(range(63043, 63893)) + list(range(70773, 71443)),
                          63: list(range(63043, 63893)) + list(range(70773, 71443)),
                          64: list(range(63043, 63893)),
                          65: list(range(63043, 63893)),
                          66: list(range(63043, 63893)),
                          67: list(range(63043, 63893)),
                          1: list(range(74918, 75600)),
                          70: list(range(74918, 75600)),
                          89: list(range(85203, 85783)) + list(range(149749, 150423)) + list(range(151144, 151503)),
                          39: list(range(85203, 85783)) + list(range(149749, 150423)) + list(range(151144, 151503)),
                          18: list(range(147303, 147390)),
                          9: list(range(148677, 149483)),
                          -1: list(range(152163, 152739))
                          }

swat_anomalies_indexes = {}

def getUniqueElementsForModel(list_2, list_1):
    # yields the elements in `list_2` that are NOT in `list_1`
    return np.setdiff1d(list_2, list_1)


def getCommonPartForModels(list2, list_1):
    return np.intersect1d(list2, list_1)
