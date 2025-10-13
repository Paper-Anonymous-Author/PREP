import math
import os
import time

import matplotlib.pyplot as plt
import openai
import torch

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HUGGINGFACE_HUB_TOKEN"] = ""

import pandas as pd
import numpy as np
from setup_llm import generate_gpt, prepare_gpt
from script import is_float, month_plus
from evaluate import evaluate_recommend
from matplotlib import pyplot
import random
import config

random.seed(2025)
np.random.seed(2025)

system_prompt = (
    '1. I would like to use a Large Language Model (LLM) to predict the sales of this product for the next month.\n'
    'Please output a prompt that can be used as input to the LLM and make the LLM to predict the item sales value.\n'
    'You should follow the following 6 steps:\n'
    
    '1.1: First, output "Instruction:"\n'
    '1.2: First, you should state that the LLM’s task is to predict the sales based on feature and popularity trend.\n'
    
    '1.3: Then, output "Features:"\n'
    
    '1.4: Then, you should select and describe the relevant item features.\n'
    'In this part, you should first filter for features that are relevant to the item\'s sales value,'
    ' and then present them in an appropriate manner. \n'
    'You may consider summarizing long features, '
    'or combining multiple features to generate more effective input. \n'
    
    '1.5: Then, output "Past monthly sales:"\n'
    
    '1.6: Finally, provide the item’s past monthly sales.\n'
    'In this part, you should output specific monthly sales numbers.'
    # ' Do not just output the popularity trend.'
    
    '2. Notes:\n'
    '2.1. Please directly output the prompt, and omit the output.\n'
    '2.2. You should complete all 6 steps 1.1-1.6'
    '\n'
)



def LLM_generate_no_batch(cfg:config.Config, llm_client, generate_for_test_set=False):

    if cfg.dataset == 'douban':
        df_meta = pd.read_csv(cfg.path_db_meta, index_col=0, dtype={'item_id': int},)
        df_train = pd.read_csv(cfg.path_db_train, index_col=0, dtype={'item_id': int, 'begin_month': int, 'max_month': int})
    else:
        df_meta = pd.read_csv(cfg.path_amz_meta, index_col=0, dtype={'item_id': int}, )
        df_train = pd.read_csv(cfg.path_amz_train, index_col=0,
                               dtype={'item_id': int, 'begin_month': int, 'max_month': int})
    df_train['inter_month_count'] = [eval(line) for line in df_train['inter_month_count']]
    df_train['train_month'] = [eval(line) for line in df_train['train_month']]

    for i in list(range(len(df_meta)))[:]:
        # if i > 100:
        #     exit()
        item_id_now = df_train['item_id'][i]
        print('\n')
        print(f'##### item id: {item_id_now} #####')
        # Features

        if cfg.dataset == 'douban':
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
        else:
            item_meta_now = df_meta[df_meta['item_id'] == item_id_now]
            title = f'The title of the item is {item_meta_now["title"][item_id_now]}. '
            price = f'The price of the item is {item_meta_now["price"][item_id_now]}. '
            salesRank = f'The salesRank of the item are {item_meta_now["salesRank"][item_id_now]}. '
            categories = f'The categories of the item is {"-".join(eval(item_meta_now["categories"][item_id_now])[0])}. '
            description = f'The description of the item is: {item_meta_now["description"][item_id_now]} '


        # Inters
        item_inter_now = df_train[df_train['item_id'] == item_id_now]

        inter_month_now = item_inter_now["inter_month_count"][item_id_now]
        list_train_month = item_inter_now["train_month"][item_id_now]
        begin_month_now = item_inter_now["begin_month"][item_id_now]

        if generate_for_test_set:
            train_month_now = len(inter_month_now)-1
        else:
            train_month_now = random.choice(list_train_month)

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

        if cfg.dataset == 'douban':
            user_sentence = ('Item features are:\n' + '1. ' + name_movie + '\n'
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
        else:
            user_sentence = ('Item features are:\n' + '1. ' + title + '\n'
                             + '2. ' + price + '\n'
                             + '3. ' + salesRank + '\n'
                             + '4. ' + categories + '\n'
                             + '5. ' + description + '\n'
                             + ' Monthly sales data from past to recent is:\n' + inter_raw_all + '\n'
                             + 'The month to be predict is ' + month_predict
                             # + '\nFrom left to the right means the chronological order\n'
                             )

        with torch.no_grad():
            if 'gpt' in cfg.llm_model:
                print('sleep')
                time.sleep(1.5)
            llm_generate = generate_gpt(
                cfg=cfg, llm_client=llm_client, system_prompt=system_prompt, user_sentence=user_sentence
            )
        print(f'### item: {i} prompt begin###')
        print(llm_generate)
        print(f'### item: {i} prompt end###')
        print(f'## gt: {inter_month_now[train_month_now]} ##')
        time_now = time.time()-cfg.start_time
        print(f'Time: {int(time_now/3600)}h-{int((time_now % 3600) / 60)}min-{int(time_now % 60)}s')


def LLM_generate(cfg:config.Config, llm_client, batch, generate_for_test_set=False):
    if cfg.dataset == 'douban':
        df_meta = pd.read_csv(cfg.path_db_meta, index_col=0, dtype={'item_id': int},)
        df_train = pd.read_csv(cfg.path_db_train, index_col=0, dtype={'item_id': int, 'begin_month': int, 'max_month': int})
    else:
        df_meta = pd.read_csv(cfg.path_amz_meta, index_col=0, dtype={'item_id': int},)
        df_train = pd.read_csv(cfg.path_amz_train, index_col=0, dtype={'item_id': int, 'begin_month': int, 'max_month': int})

    df_train['inter_month_count'] = [eval(line) for line in df_train['inter_month_count']]
    df_train['train_month'] = [eval(line) for line in df_train['train_month']]

    list_system_p, list_user_s, list_prompt, list_gt = [], [], [], []

    for i in list(range(len(df_meta)))[:]:
            # if i > 100:
            #     exit()
            # print(time_now)
            item_id_now = df_train['item_id'][i]
            print('\n')
            print(f'##### item id: {item_id_now} #####')
            # Features
            if cfg.dataset == 'douban':
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
            else:
                item_meta_now = df_meta[df_meta['item_id'] == item_id_now]
                title = f'The title of the item is {item_meta_now["title"][item_id_now]}. '
                price = f'The price of the item is {item_meta_now["price"][item_id_now]} dollars. '
                salesRank = f'The salesRank of the item are {item_meta_now["salesRank"][item_id_now]}. '
                categories = f'The categories of the item is {"-".join(eval(item_meta_now["categories"][item_id_now])[0])}. '
                description = f'The description of the item is: {item_meta_now["description"][item_id_now]} '

            # Inters
            item_inter_now = df_train[df_train['item_id'] == item_id_now]

            inter_month_now = item_inter_now["inter_month_count"][item_id_now]
            list_train_month = item_inter_now["train_month"][item_id_now]
            begin_month_now = item_inter_now["begin_month"][item_id_now]

            if generate_for_test_set:
                train_month_now = len(inter_month_now)-1
            else:
                train_month_now = random.choice(list_train_month)

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
            # print(inter_raw_all)

            if cfg.dataset == 'douban':
                user_sentence = ('Item features are:\n' + '1. ' + name_movie + '\n'
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
            else:
                user_sentence = ('Item features are:\n' + '1. ' + title + '\n'
                                 + '2. ' + price + '\n'
                                 + '3. ' + salesRank + '\n'
                                 + '4. ' + categories + '\n'
                                 + '5. ' + description + '\n'
                                 + ' Monthly sales data from past to recent is:\n' + inter_raw_all + '\n'
                                 + 'The month to be predict is ' + month_predict
                                 # + '\nFrom left to the right means the chronological order\n'
                                 )

            list_system_p.append(system_prompt)
            list_user_s.append(user_sentence)
            list_gt.append(inter_month_now[train_month_now])

            if (i+1) % batch == 0 or i+1 == len(df_meta):
                for j in range(len(list_system_p)):
                    list_prompt.append([
                        {"role": "system", "content": list_system_p[j]},
                        {"role": "user", "content": list_user_s[j]},
                    ])
                if cfg.llm_type == 'Llama':
                    llm_client.tokenizer.pad_token_id = llm_client.tokenizer.eos_token_id
                    llm_client.tokenizer.padding_side = 'left'
                    llm_generates = llm_client(
                        list_prompt,
                        pad_token_id=llm_client.tokenizer.eos_token_id,
                        max_new_tokens=3000,
                        batch_size=len(list_prompt)
                    )
                else:
                    llm_generates = llm_client.chat.completions.create(
                        model=cfg.llm_model,
                        messages=list_prompt
                    )
                    # llm_generates = openai.ChatCompletion.create(
                    #     model=cfg.llm_model,
                    #     messages=[
                    #         {"role": "system", "content": system_prompt},
                    #         {"role": "user", "content": user_sentence}
                    #     ]
                    # )
                    print(llm_generates)
                    exit()
                    # generation = completion.choices[0].message.content
                for j in range(len(list_prompt)):
                    print(f'### item: {i-len(list_prompt)+1+j} prompt begin###')
                    output_now = llm_generates[j][0]["generated_text"][-1]['content']
                    print(output_now)
                    print(f'# len: {len(output_now)} #')
                    print(f'### item: {i-len(list_prompt)+1+j} prompt end###')
                    print(f'### gt: {list_gt[j]} ###')

                list_system_p, list_user_s, list_prompt, list_gt = [], [], [], []
                time_now = time.time()-cfg.start_time
                print(f'Time: {int(time_now/3600)}h-{int((time_now % 3600) / 60)}min-{int(time_now % 60)}s, in total {time_now}s')


def evaluate_with_LLM_old(cfg, llm_client):
    llm_generate_path = '/home/jiazheng/jingjiazheng/model/Popularity+Prompt+LLM/main/result_new/'
    llm_generate_file = llm_generate_path + 'llama_3b.out'
    print(f'llama_path: {llm_generate_file}')

    list_sp, list_us, list_gt = [], [], []
    system_prompt = ('You should round the prediction result to one decimal place. '
                     'You should only output that result number.')

    with open(llm_generate_file, 'r') as f_in:
        print(f'file: {llm_generate_file}')
        file = f_in.readlines()
        print(f'len_file: {len(file)}')
        reading_prompt, reading_gt = False, False

        us_now = []
        for line in file:
            line = line.strip()
            print(line)
            if '### LLM output begin ###' in line[:25]:
                reading_prompt = True
            elif reading_prompt:
                us_now.append(line)

            if '### LLM output end ###' in line[:23]:
                reading_prompt = False
                reading_gt = True
                list_us.append(us_now)
                us_now = []
            elif reading_gt:
                reading_gt = False
                gt_now = int(line[7:-3])
                list_gt.append(gt_now)

        f_in.close()

    valid_num = 0
    mae = 0
    llm_generate = 0
    for i in range(len(list_us)):
        user_sentence_now = list_us[i]
        gt_now = list_gt[i]

        predict_correct = False
        try_num = 0

        while not predict_correct and try_num < 3:
            try:
                llm_generate = generate_gpt(
                    cfg=cfg, llm_client=llm_client, system_prompt=system_prompt, user_sentence=user_sentence_now
                )
                llm_generate = float(llm_generate)
                predict_correct = True
                valid_num += 1
                break
            except:
                predict_correct = False
                try_num += 1
        else:
            continue
        print(f'llm_generate; {llm_generate}, gt_now: {gt_now}')
        mae += abs(llm_generate-gt_now)
    print(f'mae: {mae}')


def evaluate_with_LLM(cfg, llm_client, llm_generate_name, type):
    if 'douban' in type:
        llm_generate_path = '/home/jiazheng/jingjiazheng/model/Popularity+Prompt+LLM/main/result_new/'
    else:
        llm_generate_path = '/home/jiazheng/jingjiazheng/model/Popularity+Prompt+LLM/main/result_amz/'
    llm_generate_file = llm_generate_path + llm_generate_name
    print(f'llama_path: {llm_generate_file}')

    list_sp, list_us, list_gt = [], [], []
    system_prompt = ('In this task, you should only output a single Arabic number! \n'
                     'You should round the number to one decimal place.\n'
                     'You should omit your thinking procedure!')

    with open(llm_generate_file, 'r') as f_in:
        print(f'file: {llm_generate_file}')
        file = f_in.readlines()
        print(f'len_file: {len(file)}')
        reading_prompt, reading_gt = False, False

        us_now = []
        for line in file:
            line = line.strip()
            # print(line)
            if '### item: 'in line[:11] and 'prompt begin###' in line:
                reading_prompt = True
                print(f'Processing: {line[10:-16]}')
            elif reading_prompt and 'prompt end###' not in line and '# len: ' not in line:
                us_now.append(line)

            if '### item: 'in line[:11] and 'prompt end###' in line:
                print(f'Processing: {line[10:-14]} Done')
                reading_prompt = False
                reading_gt = True
                list_us.append(us_now)
                us_now = []
            elif reading_gt:
                reading_gt = False
                gt_now = int(line[7:-3])
                list_gt.append(gt_now)
        f_in.close()




    valid_num, mae, llm_generate = 0, 0, 0
    have_think = 0
    for i in range(len(list_us)):
        user_sentence_now = '\n'.join(list_us[i])
        if '</think>' in user_sentence_now:
            user_sentence_now = user_sentence_now.split('</think>')[-1]
            have_think += 1
            print('</think> detected')
        gt_now = list_gt[i]

        predict_correct = False
        try_num = 0

        while not predict_correct and try_num < 10:
            try:
                print(f'\nprocessing: item {i} try_num: {try_num+1}\n')
                time_now = time.time()-cfg.start_time
                print(f'Time: {int(time_now/3600)}h-{int((time_now % 3600) / 60)}min-{int(time_now % 60)}s')
                llm_generate = generate_gpt(
                    cfg=cfg, llm_client=llm_client, system_prompt=system_prompt, user_sentence=user_sentence_now
                )
                ori_llm_generate = llm_generate
                if try_num<5:
                    llm_generate = float(ori_llm_generate)
                else:
                    llm_generate = 0.0
                predict_correct = True
                valid_num += 1
                break
            except:
                time.sleep(3)
                print(f'Wrong output is:\n{ori_llm_generate}')
                print(f'gt_now: {gt_now}')
                predict_correct = False
                try_num += 1
        else:
            continue
        print(f'llm_generate: {llm_generate}, gt_now: {gt_now}')
        mae += abs(llm_generate-gt_now)
        # exit()
    print(f'mae: {mae}, valid_num: {valid_num}')
    print(f'ressult: {mae/valid_num}')


def eva_recommend(cfg, path, dataset_type):
    print(f'Processing path: {path}')
    list_item = []
    mae = 0
    valid = 0
    if dataset_type == 'douban':
        open_file = '/home/jiazheng/jingjiazheng/model/Popularity+Prompt+LLM/main/result_API_predict/' + path
    else:
        open_file = '/home/jiazheng/jingjiazheng/model/Popularity+Prompt+LLM/main/result_amz/' + path

    with open(
            open_file,
            'r') as f_in:
        item_now = -1
        lines = f_in.readlines()
        for line in lines:
            line = line.strip()
            if 'Processing: ' in line and 'item' not in line and 'Done' not in line:
                item_now = int(line[12:])
            if 'Processing: ' in line and 'item' not in line and 'Done' in line:
                if int(line[12:-5]) == item_now:
                    list_item.append(item_now)
        # list_item = list(range(3645))
        pred_pop = [-1 for _ in range(max(list_item)+1)]
        item_now = -1
        for line in lines:
            line = line.strip()
            if 'processing: ' in line and 'item' in line:
                item_now = int(line.split(' try_num')[0][17:])
            if 'llm_generate: ' in line or 'llm_generate; ' in line:
                pred_now = float(line.split(', gt_now:')[0][14:])
                gt_now = float(line.split(', gt_now:')[1])
                mae += abs(pred_now - gt_now)
                valid += 1
                pred_pop[list_item[item_now]] = pred_now
    print(f'Total mae: {mae}, valid: {valid}, final_mae: {mae/valid}')
    print(f'len pred_pop: {len(pred_pop)}')
    print(pred_pop)
    evaluate_recommend(cfg, 'test', pred_pop, False)
    return pred_pop



if __name__ == '__main__':
    args = 0
    cfg = config.Config(args)

    ####################### For PREP generate #######################
    cfg.llm_model = 'Llama'
    cfg.llm_type = 'Llama'
    llm_client = prepare_gpt(cfg)
    LLM_generate(cfg, llm_client, batch=8, generate_for_test_set=True)
    exit()
    #######################For API #######################
    # cfg.llm_model = 'gpt-4.1-mini'
    # llm_client = prepare_gpt(cfg)
    # llm_generate = generate_gpt(
    #     cfg=cfg, llm_client=llm_client, system_prompt='translate to Chinese', user_sentence='apple'
    # )
    # print(llm_generate)
    # exit()
    # LLM_generate_no_batch(cfg, llm_client, generate_for_test_set=True)
    # exit()

    ####################### Evaluate #######################
    # cfg.llm_model = 'gpt-4.1-mini'
    # cfg.llm_model = 'gpt-4.1'
    llm_client = prepare_gpt(cfg)

    # llm_generate_name = '0711_llama_temp0.out'
    evaluate_with_LLM(cfg, llm_client, llm_generate_name, 'douban')
    evaluate_with_LLM(cfg, llm_client, llm_generate_name, 'amz')

    ####################### Evaluation with recommendation metrics #######################
    eva_path = '0607-0606_llama3b-deepseek_6.out'
    eva_recommend(cfg, eva_path, 'douban')
    # eva_recommend(cfg, eva_path, 'amz')
    # exit()
