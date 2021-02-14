
def abs_sales_diff(pred, target):
    """
    Calculates sum of absolute sales differences per day for an item
    E.g.
    pred = [1, 2, 3]
    target = [0, 1, 1]
    return 1 + 1 + 2 = 4
    :param pred: Array of predicted sales units for an item
    :param target: Array of actual sales units for an item
    :return: Sum of absolute sales differences per day
    """
    assert len(pred) == len(target)
    return sum([abs(pred[i] - target[i]) for i in range(len(pred))])

def soldout_day(pred, stock):
    """
    Calculates first day that stock hits 0 in a certain month for an item
    :param pred: Array of predicted sales units for an item
    :param stock: Stock at beginning of month for an item
    :return: Day of month that stock reaches 0
    """
    soldout_day = len(pred)
    for day in range(len(pred)):
        stock -= pred[day]
        print(stock)
        if stock <= 0:
            soldout_day = day+1
            break
    return soldout_day