import pandas as pd
import numpy as np
import os
import datetime
import calendar
import config
import gzip
from tqdm import tqdm

def process_amazon(cfg, force_run):
    if os.path.exists(cfg.path_amz_inter_pilot) and os.path.exists(cfg.path_amz_meta) \
        and os.path.exists(cfg.path_tra_amz_train) and not force_run:
        return
    g_inter = gzip.open(cfg.path_amz_inter_gz, 'rb')
    df, idx = {}, 0
    # print(g.readlines()[0])
    for line in tqdm(g_inter.readlines()):
        df[idx] = eval(line)
        idx += 1
    df_inter = (pd.DataFrame.from_dict(df, orient='index', dtype=str)
        .drop(columns=['reviewerName', 'helpful', 'reviewText', 'summary', 'reviewTime'])
        .rename(columns={'reviewerID': 'user', 'asin': 'item', 'unixReviewTime': 'time', 'overall': 'rate'})
        .astype({'time':int, 'rate':float}))

    # filter positive rate
    df_inter = df_inter[df_inter['rate']>=3]

    # filter duplicate interaction
    df_inter.drop_duplicates(subset=['user', 'item'], keep='first', inplace=True)
    df_inter.reset_index(drop=True, inplace=True)
    print('len_df: {}'.format(len(df_inter)))

    g_meta = gzip.open(cfg.path_amz_meta_gz, 'rb')
    df, idx = {}, 0
    # print(g.readlines()[0])
    for line in tqdm(g_meta.readlines()):
        df[idx] = eval(line)
        idx += 1
    df_meta = (pd.DataFrame.from_dict(df, orient='index', dtype=str)
        .drop(columns=['imUrl', 'related'])
        .rename(columns={ 'asin': 'item'}))
        # .astype({'time':int, 'rate':float}))
    df_meta = df_meta[df_meta['item'].isin(df_inter['item'])]
    df_meta = df_meta.drop_duplicates(subset=['item'], keep='first')
    df_meta.reset_index(drop=True, inplace=True)
    print(df_meta)
    print(df_meta.columns)
    if len(set(list(df_inter['item']))) != len(df_meta):
        print('Jing: item miss metainfo!')
        exit()
    #################################### Read Done, Start Process ##########################################


    start_time = min(df_inter['time'])
    end_time = max(df_inter['time'])
    dt_start = datetime.datetime.fromtimestamp(start_time)
    dt_end = datetime.datetime.fromtimestamp(end_time)
    print(f'Start time:{dt_start.strftime("%Y-%m-%d")}')
    print(f'End time:{dt_end.strftime("%Y-%m-%d")}\n')

    start_year, start_month, start_day = int(dt_start.strftime('%Y')), int(dt_start.strftime('%m')), int(dt_start.strftime('%d'))
    end_year, end_month, end_day = int(dt_end.strftime('%Y')), int(dt_end.strftime('%m')), int(dt_end.strftime('%d'))
    test_month = [end_year, end_month]
    valid_month = [end_year, end_month-1]
    if valid_month[1] == 0:
        valid_month[1] = 12
        valid_month[0] -= 1
    if (end_month != 2 and end_day < 30) or (end_month == 2 and end_day < 27):
        test_month[1] -= 1
        if test_month[1] == 0:
            test_month[1] = 12
            test_month[0] -= 1
        valid_month[1] -= 1
        if valid_month[1] == 0:
            valid_month[1] = 12
            valid_month[0] -= 1
    print(f'Test month: {test_month}')
    print(f'Valid month: {valid_month}\n')

    # Filter interactions later than last month
    test_last_day = calendar.monthrange(test_month[0], test_month[1])[1]
    test_last_time = int(datetime.datetime(test_month[0], test_month[1], test_last_day, 23, 59, 59).timestamp())
    print(f'Test last time: {datetime.datetime.fromtimestamp(test_last_time).strftime("%Y-%m-%d")}\n')

    df_inter = df_inter[df_inter['time'] <= test_last_time]
    print(f'Select interactions before the last time, number of interactions: {len(df_inter)}, '
          f'number of item:{len(set(df_inter["item"]))}')

    # Filter item not occur in training time later than last month
    train_last_time = int(datetime.datetime(valid_month[0], valid_month[1], 1, 0, 0, 0).timestamp()) - 1
    print(f'Train last time: {datetime.datetime.fromtimestamp(train_last_time).strftime("%Y-%m-%d")}\n')

    items_have_inter = set(df_inter[df_inter['time'] <= train_last_time]['item'])
    df_inter = df_inter[df_inter['item'].isin(items_have_inter)]
    print(f'Select items occurs in training set, number of interactions: {len(df_inter)}, '
          f'number of item:{len(set(df_inter["item"]))}\n')

    # Keep items with more than 5 interactions
    inter_count = df_inter['item'].value_counts()
    item_min_inters = cfg.item_min_inters
    df_inter = df_inter[df_inter['item'].isin(inter_count[inter_count >= item_min_inters].index)]
    # print(inter_count[inter_count >= item_min_inters])
    print(
        f'##################### Dataset Statistic #####################\n'
        f'Keep items with >= {item_min_inters} interactions, number of interactions: {len(df_inter)}, '
        f'number of item: {len(set(df_inter["item"]))},'
        f' number of users: {len(set(df_inter["user"]))}\n'

    )
    # Reset index for df_inter
    df_inter.reset_index(drop=True, inplace=True)

    ##################### Generate Train, Valid, and Test dataset #####################
    # Generate unique index for items
    dict_item_idx = {}

    for i in range(len(df_inter)):
        item_now = df_inter['item'][i]
        if item_now not in dict_item_idx:
            dict_item_idx[item_now] = len(dict_item_idx)
    print(f'Number of items in dict_item_idx: {len(dict_item_idx)}\n')

    num_item = len(dict_item_idx)

    inter_month_count = [[0 for _ in range((test_month[0] - start_year) * 12 + test_month[1]-start_month)] for _ in range(num_item)]
    inter_each = [[] for _ in range(num_item)]

    month_all = [0 for _ in range((test_month[0] - start_year) * 12 + test_month[1]-start_month)]

    time_all = list(df_inter['time'])

    dataset_tra = [[[], [], []], [[], [], []], [[], [], []]]
    # Split dataset and build traditional dataset
    dict_user_idx = {}

    for i in range(len(time_all)):
        set_type = 0 # 0: Train, 1: Valid, 2: Test
        item_now = df_inter['item'][i]
        time_now = time_all[i]
        idx_item_now = dict_item_idx[item_now]

        temp_time = datetime.datetime.fromtimestamp(time_now)
        year_now = int(temp_time.strftime('%Y'))
        month_now = int(temp_time.strftime('%m'))
        day_now = int(temp_time.strftime('%d'))
        # print(year_now,start_year, month_now, start_month)
        month_count_now = (year_now - start_year) * 12 + month_now - 1 - start_month
        inter_month_count[idx_item_now][month_count_now] += 1
        inter_each[idx_item_now].append(time_now)

        month_all[month_count_now] += 1

        user_now = df_inter['user'][i]
        if user_now not in dict_user_idx:
            dict_user_idx[user_now] = len(dict_user_idx)
        user_idx_now = dict_user_idx[user_now]

        if year_now == valid_month[0] and month_now == valid_month[1]:
            set_type = 1
        if year_now == test_month[0] and month_now == test_month[1]:
            set_type = 2

        dataset_tra[set_type][0].append(user_idx_now)
        dataset_tra[set_type][1].append(idx_item_now)
        dataset_tra[set_type][2].append(time_now)
    print(month_all)

    len_tra_train, len_tra_valid, len_tra_test = len(dataset_tra[0][0]), len(dataset_tra[1][0]), len(dataset_tra[2][0])
    print(f'Baseline dataset: Len_train: {len_tra_train}, Len_valid: {len_tra_valid}, Len_test: {len_tra_test}')

    train_users = dataset_tra[0][0]
    not_in = [0, 0]
    for user in dataset_tra[1][0]:
        if user not in train_users:
            # print(f'valid {user}')
            not_in[0] += 1
    for user in dataset_tra[2][0]:
        if user not in train_users:
            # print(f'test {user}')
            not_in[1] += 1
    print(not_in)

    df_tra_train = pd.DataFrame({
        'user': dataset_tra[0][0],
        'item': dataset_tra[0][1],
        'rating': [5 for _ in range(len_tra_train)],
        'time': dataset_tra[0][2]
    })
    df_tra_valid = pd.DataFrame({
        'user': dataset_tra[1][0],
        'item': dataset_tra[1][1],
        'rating': [5 for _ in range(len_tra_valid)],
        'time': dataset_tra[1][2]
    })
    df_tra_test = pd.DataFrame({
        'user': dataset_tra[2][0],
        'item': dataset_tra[2][1],
        'rating': [5 for _ in range(len_tra_test)],
        'time': dataset_tra[2][2]
    })
    df_tra_train.to_csv(cfg.path_tra_amz_train)
    df_tra_valid.to_csv(cfg.path_tra_amz_valid)
    df_tra_test.to_csv(cfg.path_tra_amz_test)

    for i in range(num_item):
        inter_each[i].sort()

    # Filter df_db_meta_old, df_db_meta_new
    df_meta = df_meta[df_meta['item'].isin(dict_item_idx)]
    df_meta.reset_index(drop=True, inplace=True)
    print(f'After filtering number of items in, number of items in df_meta: {len(df_meta)}\n')

    item_old = list(df_meta['item'])
    idx_old = [dict_item_idx[i] for i in item_old]
    df_meta['item_id'] = idx_old
    df_meta = df_meta.sort_values(by='item_id', ascending=True)
    df_meta.reset_index(drop=True, inplace=True)

    df_meta.to_csv(cfg.path_amz_meta)

    df_inter_pilot = pd.DataFrame({
        'item_id': list(range(num_item)),
        'inter_month_count': inter_month_count,
        'inter_each': inter_each
    })
    df_inter_pilot.to_csv(cfg.path_amz_inter_pilot)




def read_douban(cfg):
    path_db_inter = cfg.path_db_inter
    path_db_meta_old = cfg.path_db_meta_old
    path_db_meta_new = cfg.path_db_meta_new

    print('Reading df_db_inter...')
    df_db_inter = pd.read_csv(
        path_db_inter, header=0, usecols=['MovieID', 'UserID', 'Time'],
        dtype={'MovieID': str, 'UserID': str, 'Time': int}
    ).rename(columns={'MovieID': 'item', 'UserID': 'user', 'Time': 'time'})
    print(f'Length of df_db_inter:{len(df_db_inter)}, '
          f'number of items: {len(set(df_db_inter["item"]))}, '
          f'number of users: {len(set(df_db_inter["user"]))}\n')

    print('Reading df_db_meta_old...')
    df_db_meta_old = pd.read_csv(
        path_db_meta_old, header=0,
        usecols=['MovieID', 'IMDb', 'District', 'Also_Called', 'Length', 'Rate', 'Rate_Num'],
        dtype={'MovieID': str, 'IMDb': str, 'District': str, 'Also_Called': str,
               'Length': str, 'Rate': str, 'Rate_Num': str}
    ).rename(columns={'MovieID': 'item', 'IMDb': 'imdb'})


    print('Reading df_db_meta_new...\n')
    df_db_meta_new = pd.read_excel(
        path_db_meta_new, header=0,
        usecols=['movieid', 'imdb', 'movie_name', 'description', 'release_date', 'genre',
                 'language', 'director', 'writers', 'top5_actors', 'producers', 'music'],
        dtype={'movieid': str, 'imdb': str, 'movie_name': str, 'description': str, 'release_date': str, 'genre': str,
               'language': str, 'director': str, 'writers': str, 'top5_actors': str, 'producers': str, 'music': str}
    ).rename(columns={'movieid': 'item'})

    # Remove duplicated items
    print('Before remove duplicated items')
    print(f'Length of df_db_meta_new:{len(df_db_meta_new)}')
    print(f'Length of df_db_meta_old:{len(df_db_meta_old)}')
    df_db_meta_old = df_db_meta_old.drop_duplicates(subset=['item'], keep='first')
    df_db_meta_old.reset_index(drop=True, inplace=True)
    df_db_meta_new = df_db_meta_new.drop_duplicates(subset=['item'], keep='first')
    df_db_meta_new.reset_index(drop=True, inplace=True)
    print('After remove duplicated items')
    print(f'Length of df_db_meta_old:{len(df_db_meta_old)}')
    print(f'Length of df_db_meta_new:{len(df_db_meta_new)}')

    print('Reading douban datasets done!')
    return df_db_inter, df_db_meta_old, df_db_meta_new


def process_douban(df_db_inter, df_db_meta_old, df_db_meta_new):
    # df_db_inter = df_db_inter[df_db_inter['imdb'].isin(df_db_meta_new['imdb'])]
    print(f'In original dataset, number of interactions: {len(df_db_inter)}, '
          f'number of items: {len(set(df_db_inter["item"]))}\n')

    df_db_inter = df_db_inter[df_db_inter['item'].isin(df_db_meta_old['item'])]
    print(f'Filter item in df_db_meta_old, number of interactions: {len(df_db_inter)}, '
          f'number of item:{len(set(df_db_inter["item"]))}\n')

    df_db_inter = df_db_inter[df_db_inter['item'].isin(df_db_meta_new['item'])]
    print(f'Filter item in df_db_meta_new, number of interactions: {len(df_db_inter)}, '
          f'number of item:{len(set(df_db_inter["item"]))}\n')

    start_time = min(df_db_inter['time'])
    end_time = max(df_db_inter['time'])
    dt_start = datetime.datetime.fromtimestamp(start_time)
    dt_end = datetime.datetime.fromtimestamp(end_time)
    print(f'Start time:{dt_start.strftime("%Y-%m-%d")}')
    print(f'End time:{dt_end.strftime("%Y-%m-%d")}\n')

    start_year, start_month, start_day = int(dt_start.strftime('%Y')), int(dt_start.strftime('%m')), int(dt_start.strftime('%d'))
    end_year, end_month, end_day = int(dt_end.strftime('%Y')), int(dt_end.strftime('%m')), int(dt_end.strftime('%d'))
    test_month = [end_year, end_month]
    valid_month = [end_year, end_month-1]
    if valid_month[1] == 0:
        valid_month[1] = 12
        valid_month[0] -= 1
    if (end_month != 2 and end_day < 30) or (end_month == 2 and end_day < 27):
        test_month[1] -= 1
        if test_month[1] == 0:
            test_month[1] = 12
            test_month[0] -= 1
        valid_month -= 1
        if valid_month[1] == 0:
            valid_month[1] = 12
            valid_month[0] -= 1
    print(f'Test month: {test_month}')
    print(f'Valid month: {valid_month}\n')

    # Filter interactions later than last month
    test_last_day = calendar.monthrange(test_month[0], test_month[1])[1]
    test_last_time = int(datetime.datetime(test_month[0], test_month[1], test_last_day, 23, 59, 59).timestamp())
    print(f'Test last time: {datetime.datetime.fromtimestamp(test_last_time).strftime("%Y-%m-%d")}\n')

    df_db_inter = df_db_inter[df_db_inter['time'] <= test_last_time]
    print(f'Select interactions before the last time, number of interactions: {len(df_db_inter)}, '
          f'number of item:{len(set(df_db_inter["item"]))}')

    # Filter item not occur in training time later than last month
    train_last_time = int(datetime.datetime(valid_month[0], valid_month[1], 1, 0, 0, 0).timestamp()) - 1
    print(f'Train last time: {datetime.datetime.fromtimestamp(train_last_time).strftime("%Y-%m-%d")}\n')

    items_have_inter = set(df_db_inter[df_db_inter['time'] <= train_last_time]['item'])
    df_db_inter = df_db_inter[df_db_inter['item'].isin(items_have_inter)]
    print(f'Select items occurs in training set, number of interactions: {len(df_db_inter)}, '
          f'number of item:{len(set(df_db_inter["item"]))}\n')

    # Keep items with more than 5 interactions
    inter_count = df_db_inter['item'].value_counts()
    item_min_inters = cfg.item_min_inters
    df_db_inter = df_db_inter[df_db_inter['item'].isin(inter_count[inter_count >= item_min_inters].index)]
    # print(inter_count[inter_count >= item_min_inters])
    print(
        f'##################### Dataset Statistic #####################\n'
        f'Keep items with >= {item_min_inters} interactions, number of interactions: {len(df_db_inter)}, '
        f'number of item: {len(set(df_db_inter["item"]))},'
        f' number of users: {len(set(df_db_inter["user"]))}\n'

    )
    # Reset index for df_db_inter
    df_db_inter.reset_index(drop=True, inplace=True)

    ##################### Generate Train, Valid, and Test dataset #####################
    # Generate unique index for items
    dict_item_idx = {}

    for i in range(len(df_db_inter)):
        item_now = df_db_inter['item'][i]
        if item_now not in dict_item_idx:
            dict_item_idx[item_now] = len(dict_item_idx)
    print(f'Number of items in dict_item_idx: {len(dict_item_idx)}\n')

    num_item = len(dict_item_idx)

    inter_month_count = [[0 for _ in range((test_month[0] - start_year) * 12 + test_month[1])] for _ in range(num_item)]
    inter_each = [[] for _ in range(num_item)]

    month_all = [0 for _ in range((test_month[0] - start_year) * 12 + test_month[1])]

    time_all = list(df_db_inter['time'])

    dataset_tra = [[[], [], []], [[], [], []], [[], [], []]]
    # Split dataset and build traditional dataset
    dict_user_idx = {}

    for i in range(len(time_all)):
        set_type = 0 # 0: Train, 1: Valid, 2: Test
        item_now = df_db_inter['item'][i]
        time_now = time_all[i]
        idx_item_now = dict_item_idx[item_now]

        temp_time = datetime.datetime.fromtimestamp(time_now)
        year_now = int(temp_time.strftime('%Y'))
        month_now = int(temp_time.strftime('%m'))
        day_now = int(temp_time.strftime('%d'))

        month_count_now = (year_now - start_year) * 12 + month_now - 1
        inter_month_count[idx_item_now][month_count_now] += 1
        inter_each[idx_item_now].append(time_now)

        month_all[month_count_now] += 1

        user_now = df_db_inter['user'][i]
        if user_now not in dict_user_idx:
            dict_user_idx[user_now] = len(dict_user_idx)
        user_idx_now = dict_user_idx[user_now]

        if year_now == valid_month[0] and month_now == valid_month[1]:
            set_type = 1
        if year_now == test_month[0] and month_now == test_month[1]:
            set_type = 2

        dataset_tra[set_type][0].append(user_idx_now)
        dataset_tra[set_type][1].append(idx_item_now)
        dataset_tra[set_type][2].append(time_now)
    print(month_all)

    len_tra_train, len_tra_valid, len_tra_test = len(dataset_tra[0][0]), len(dataset_tra[1][0]), len(dataset_tra[2][0])
    print(f'Baseline dataset: Len_train: {len_tra_train}, Len_valid: {len_tra_valid}, Len_test: {len_tra_test}')

    train_users = dataset_tra[0][0]
    not_in = [0, 0]
    for user in dataset_tra[1][0]:
        if user not in train_users:
            print(f'valid {user}')
            not_in[0] += 1
    for user in dataset_tra[2][0]:
        if user not in train_users:
            print(f'test {user}')
            not_in[1] += 1
    print(not_in)

    df_tra_train = pd.DataFrame({
        'user': dataset_tra[0][0],
        'item': dataset_tra[0][1],
        'rating': [5 for _ in range(len_tra_train)],
        'time': dataset_tra[0][2]
    })
    df_tra_valid = pd.DataFrame({
        'user': dataset_tra[1][0],
        'item': dataset_tra[1][1],
        'rating': [5 for _ in range(len_tra_valid)],
        'time': dataset_tra[1][2]
    })
    df_tra_test = pd.DataFrame({
        'user': dataset_tra[2][0],
        'item': dataset_tra[2][1],
        'rating': [5 for _ in range(len_tra_test)],
        'time': dataset_tra[2][2]
    })
    df_tra_train.to_csv(cfg.path_tra_db_train)
    df_tra_valid.to_csv(cfg.path_tra_db_valid)
    df_tra_test.to_csv(cfg.path_tra_db_test)

    for i in range(num_item):
        inter_each[i].sort()

    # Filter df_db_meta_old, df_db_meta_new
    df_db_meta_old = df_db_meta_old[df_db_meta_old['item'].isin(dict_item_idx)]
    df_db_meta_new = df_db_meta_new[df_db_meta_new['item'].isin(dict_item_idx)]
    df_db_meta_old.reset_index(drop=True, inplace=True)
    df_db_meta_new.reset_index(drop=True, inplace=True)
    print(f'After filtering number of items in df_db_meta_old: {len(df_db_meta_old)}'
          f', number of items in df_db_meta_new: {len(df_db_meta_new)}\n')

    item_old = list(df_db_meta_old['item'])
    idx_old = [dict_item_idx[i] for i in item_old]
    df_db_meta_old['item_id'] = idx_old
    df_db_meta_old = df_db_meta_old.sort_values(by='item_id', ascending=True)
    df_db_meta_old.reset_index(drop=True, inplace=True)


    item_new = list(df_db_meta_new['item'])
    idx_new = [dict_item_idx[i] for i in item_new]
    df_db_meta_new['item_id'] = idx_new
    df_db_meta_new = df_db_meta_new.sort_values(by='item_id', ascending=True)
    df_db_meta_new.reset_index(drop=True, inplace=True)

    #'MovieID', 'IMDb', 'District', 'Also_Called', 'Length', 'Rate', 'Rate_Num'
    df_db_meta_new['District'] = df_db_meta_old['District']
    df_db_meta_new['Also_Called'] = df_db_meta_old['Also_Called']
    df_db_meta_new['Length'] = df_db_meta_old['Length']
    df_db_meta_new['Rate'] = df_db_meta_old['Rate']
    df_db_meta_new['Rate_Num'] = df_db_meta_old['Rate_Num']
    # Check if the imdb match in old and new metadata
    for i in range(num_item):
        if df_db_meta_new['imdb'][i] != df_db_meta_old['imdb'][i]:
            print(f"wrong: {df_db_meta_new['imdb'][i]}-{df_db_meta_old['imdb'][i]}")

    df_db_meta_new.to_csv(cfg.path_db_meta)

    df_db_inter_pilot = pd.DataFrame({
        'item_id': list(range(num_item)),
        'inter_month_count': inter_month_count,
        'inter_each': inter_each
    })
    df_db_inter_pilot.to_csv(cfg.path_db_inter_pilot)


def ft_prepare_dataset(cfg: config.Config):
    if os.path.exists(cfg.path_LLM_ft):
        df_data = pd.read_csv(cfg.path_LLM_ft, dtype={'item_id': str, 'gt': str}, index_col=0)
        return df_data
    print('Dataset not found! Preparing dataset')
    df_meta = pd.read_csv(cfg.path_db_meta, index_col=0, dtype={'item_id': int}, )
    df_train = pd.read_csv(cfg.path_db_train, index_col=0, dtype={'item_id': int, 'begin_month': int, 'max_month': int})
    df_train['inter_month_count'] = [eval(line) for line in df_train['inter_month_count']]
    df_train['train_month'] = [eval(line) for line in df_train['train_month']]

    # print(df_pt)
    # print(df_meta)
    # print(df_train)
    # print(f'len df_pt: {len(df_pt)}')
    # print(f'len df_meta: {len(df_meta)}')
    # print(f'len df_train: {len(df_train)}')
    #
    #
    # print(df_pt.columns)
    # print(df_meta.columns)
    # print(df_train.columns)

    dataset = []

    for i in range(len(df_train)):
        # if i>100:
        #     continue
        item_id_now = df_train['item_id'][i]

        item_meta_now = df_meta[df_meta['item_id'] == item_id_now]

        name_movie = f'The name of the movie is {item_meta_now["movie_name"][item_id_now]}. '
        director_movie = f'The director of the movie is {item_meta_now["director"][item_id_now]}. '
        actors_movie = f'The top5 actors of the movie are {item_meta_now["top5_actors"][item_id_now]}. '
        rate_movie = f'The Rate of the movie is {item_meta_now["Rate"][item_id_now]}. '
        genre_movie = f'The genre of the movie is {item_meta_now["genre"][item_id_now]}. '

        writers_movie = f'The writers of the movie are {item_meta_now["writers"][item_id_now]}. '
        producers_movie = f'The producers of the movie are {item_meta_now["producers"][item_id_now]}. '
        language_movie = f'The language of the movie is {item_meta_now["language"][item_id_now]}. '
        length_movie = f'The Length of the movie is {item_meta_now["Length"][item_id_now]}. '
        alias_movie = f'The alias of the movie is {item_meta_now["Also_Called"][item_id_now]}. '
        description_movie = f'The description of the movie is {item_meta_now["description"][item_id_now]}. '

        # Inters
        item_inter_now = df_train[df_train['item_id'] == item_id_now]

        inter_month_now = item_inter_now["inter_month_count"][item_id_now]
        list_train_month = item_inter_now["train_month"][item_id_now]
        begin_month_now = item_inter_now["begin_month"][item_id_now]

        for train_month_now in list_train_month:
            gt_now = inter_month_now[train_month_now]

            inter_month_all_train = inter_month_now[begin_month_now:train_month_now]

            if_json_inter = True
            if if_json_inter:
                json_inter_month_all_train = []
                for i_inter in range(len(inter_month_all_train)):
                    month_now = month_plus(cfg.start_month, begin_month_now + i_inter)
                    month_now[1] += 1
                    json_inter_month_all_train.append(
                        # {'Month': str(month_now), 'Monthly sales': inter_month_all_train[i_inter]}
                        f'{str(month_now)}: {inter_month_all_train[i_inter]}'
                    )
                inter_month_all_train = json_inter_month_all_train

            month_predict = month_plus(cfg.start_month, begin_month_now + len(inter_month_all_train))
            month_predict[1] += 1
            month_predict = str(month_predict)

            inter_raw_all = (
                # f'The monthly viewership numbers are'
                f' {str(inter_month_all_train)}. '
            )

            pre_user_sentence = ('Item features are:\n' + '1. ' + name_movie + '\n'
                                 + '2. ' + director_movie + '\n'
                                 + '3. ' + actors_movie + '\n'
                                 + '4. ' + rate_movie + '\n'
                                 + '5. ' + genre_movie + '\n'
                                 + '6. ' + writers_movie + '\n'
                                 + '7. ' + producers_movie + '\n'
                                 + '8. ' + language_movie + '\n'
                                 + '9. ' + length_movie + '\n'
                                 + '10. ' + alias_movie + '\n'
                                 + '11. ' + description_movie + '\n'
                                 + ' Monthly sales data from past to recent is:\n' + inter_raw_all + '\n'
                                 + 'The month to be predict is ' + month_predict
                                 # + '\nFrom left to the right means the chronological order\n'
                                 )

            dataset.append([item_id_now, pre_user_sentence, gt_now])
            # dataset.append({"instruction": pre_system_prompt, "input": pre_user_sentence})
    # print(len(dataset))
    # print(dataset[0])
    # gts = [line[-1] for line in dataset]
    # random.shuffle(gts)
    # top_all = 0
    # times = int(len(gts)/128)+1
    # for i in range(times):
    #     gts_now = gts[128*i:128*(i+1)]
    #     top10, _ = torch.topk(torch.tensor(gts_now), 10)
    #     print(top10)
    #     top_all += sum(top10)/10
    # print(top_all/times)
    df_data = pd.DataFrame(
        {'item_id': [record[0] for record in dataset],
         'user_sentence': [record[1] for record in dataset],
         'gt': [record[2] for record in dataset]}
    )
    df_data.to_csv(cfg.path_LLM_ft)
    return ft_prepare_dataset(cfg)


if __name__ == '__main__':
    if 'douban' in cfg.dataset:
        df_db_inter, df_db_meta_old, df_db_meta_new = read_douban(cfg)
        process_douban(df_db_inter, df_db_meta_old, df_db_meta_new)
    else:
        process_amazon(cfg)