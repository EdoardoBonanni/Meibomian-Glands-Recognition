import random

def balance_data_coordinate(box_features, label, value, x, y):
    counter1 = 0
    for i in range(len(label)):
        if(label[i] == 1):
            counter1 += 1
    perc1 = counter1/len(label) * 100

    while perc1 < value:
        index = random.randrange(len(label))
        if label[index] == 0:
            for j in range(0, len(box_features)):
                box_features[j].pop(index)
                #print('len box features', j, ':', len(box_features[j]))
            x.pop(index)
            y.pop(index)
            label.pop(index)
            perc1 = counter1/len(label) * 100

    return box_features, label, x, y
