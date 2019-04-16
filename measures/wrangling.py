import pandas as pd


def remove_price_cols(df, date_from, date_to):
    """
    Removes columns of prices within the date period specified, start and end dates inclusive
    Columns should have headers like '2018-01-31_price'
    :param df: dataframe containing columns with prices to remove
    :param date_from: Start date in 'YYYY-MM-DD' format
    :param date_to: End date in 'YYYY-MM-DD' format
    :return: dataframe with price columns removed
    """
    dates = ['{0}_price'.format(d) for d in pd.date_range(date_from, date_to).strftime('%Y-%m-%d')]
    result = df.drop([date for date in dates if date in df.columns], axis=1)
    return result



def sales_cols(df, date_from, date_to):
    """
    Returns columns of sales units within the date period specified, start and end dates inclusive
    Columns should have headers like '2018-01-31_sales'
    :param df: dataframe containing columns with sales units to keep
    :param date_from: Start date in 'YYYY-MM-DD' format
    :param date_to: End date in 'YYYY-MM-DD' format
    :return: dataframe with only key and sales columns
    """
    dates = ['{0}_sales'.format(d) for d in pd.date_range(date_from, date_to).strftime('%Y-%m-%d')]
    dates.append('key')
    result = df.drop([col for col in df.columns if col not in dates], axis=1)
    return result


def remove_sales_cols(df, date_from, date_to):
    """
    Removes columns of sales units within the date period specified, start and end dates inclusive
    Columns should have headers like '2018-01-31_sales'
    :param df: dataframe containing columns with sales units to remove
    :param date_from: Start date in 'YYYY-MM-DD' format
    :param date_to: End date in 'YYYY-MM-DD' format
    :return: dataframe with sales units columns removed
    """
    dates = ['{0}_sales'.format(d) for d in pd.date_range(date_from, date_to).strftime('%Y-%m-%d')]
    result = df.drop([date for date in dates if date in df.columns], axis=1)
    return result