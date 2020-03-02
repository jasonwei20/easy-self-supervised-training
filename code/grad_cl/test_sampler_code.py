import random

def get_class_sequence(n, num_classes):

    assert n % num_classes == 0

    sublist_num_to_sublist = {}
    for i in range(num_classes):
        sublist_start = int( n / num_classes ) * i 
        sublist_end = int( n / num_classes ) * (i+1) 
        if sublist_end > n:
            sublist_end = n
        sublist_num_to_sublist[i] = range(sublist_start, sublist_end)
    
    return_list = []
    for i in range(len(sublist_num_to_sublist[0])):

        indexes = list(sublist_num_to_sublist.keys())
        random.shuffle(indexes)
        for index in indexes:
            element = sublist_num_to_sublist[index][i]
            return_list.append(element)

    return return_list

print(get_class_sequence(12, 4))