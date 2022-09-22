l1 = [1, 5, 5, 5, 9]
l2 = [2, 3, 5, 5, 5, 7, 23]

def merge(l1, l2):
    ## Check if any list is empty:
    if l1 is None:
        return l2
    elif l2 is None:
        return l1

    l3 = []

    i, j = 0, 0

    ## Run till one list is finished:
    while i < len(l1) and j < len(l2):
        if l1[i] < l2[j]:
            l3.append(l1[i])
            i += 1
        else:
            l3.append(l2[j])
            j += 1

    print(i, j, len(l1), len(l2))
    assert (i >= len(l1)) or (j >= len(l2))


    ## Check if any items are pending in any list; copy all to result
    if i < len(l1):
        l3 = l3 + l1[i:]
    elif j < len(l2):
        l3 = l3 + l2[j:]

    return l3


print(merge(l1, l2))