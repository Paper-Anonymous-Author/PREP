import sys
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HUGGINGFACE_HUB_TOKEN"] = ""

import numpy as np
import torch
import math
from datetime import datetime
import time
import random

random.seed(2025)
np.random.seed(2025)

class Config:
    def __init__(self, args):
        ################ Path ################
        self.dataset = 'douban'
        print(f'Dataset: {self.dataset}')
        # Beauty Baby douban

        #
        self.path_data = '/home/jiazheng/jingjiazheng/data/'
        # self.path_data = '/home/yinan/jing/data/'
        # self.path_data = '/Users/jingjiazheng/Library/CloudStorage/OneDrive-NanyangTechnologicalUniversity/jing_py/Popularity+Prompt+LLM/data/processed/'
        self.path_db = self.path_data + 'Douban/'
        self.path_db_inter = self.path_db + 'douban_movie_review.csv'
        self.path_db_meta_old = self.path_db + 'movie_meta_2w_transed_final.csv'
        self.path_db_meta_new = self.path_db + 'preprocessed_with_movie_v3.xlsx'

        self.path_amazon = self.path_data + 'Amazon/'
        self.path_amz_inter_gz = self.path_amazon + 'reviews_' + self.dataset + '_5.json.gz'
        self.path_amz_meta_gz = self.path_amazon + 'meta_' + self.dataset + '.json.gz'

        # Processsed path
        self.path_amz_processed = self.path_amazon + 'processed_llm/'
        self.path_amz_meta = self.path_amz_processed + self.dataset + '_meta.csv'
        self.path_amz_inter_pilot = self.path_amz_processed + self.dataset + '_inter_pilot.csv'
        self.path_amz_train = self.path_amz_processed + self.dataset + '_inter_train.csv'

        self.path_db_processed = self.path_db + 'processed_llm/'
        self.path_db_meta = self.path_db_processed + 'db_meta.csv'
        self.path_db_inter_pilot = self.path_db_processed + 'db_inter_pilot.csv'
        # self.path_db_inter_pilot_zhixi = self.path_db_processed + 'douban_inter_pilot_zhixi.csv'
        self.path_db_inter_pilot_zhixi = self.path_db_processed + 'douban_inter_train_original_zhixi.csv'

        self.path_db_train = self.path_db_processed + 'db_inter_train.csv'

        # Data for LLM pretrain
        self.path_LLM_pt_raw = self.path_db_processed + 'db_llm_pt_raw.csv'
        self.path_LLM_pt = self.path_db_processed + 'db_llm_pt.csv'
        self.path_LLM_pt_best = self.path_db_processed + 'db_llm_pt_best.csv'

        self.path_LLM_pt_json = self.path_db_processed + 'db_llm_pt_best.json'

        self.path_LLM_ft = self.path_db_processed + 'df_llm_ft.csv'

        # Dataset settings
        self.item_min_inters = 5

        # Dataset information
        if 'douban' in self.dataset:
            self.start_month = [2010, 0]
            self.valid_month = [2019, 11]
            self.test_month = [2020, 0]
        if 'Beauty' in self.dataset:
            self.start_month = [2002, 6]
            self.valid_month = [2014, 5]
            self.test_month = [2014, 6]
        if 'Baby' in self.dataset:
            self.start_month = [2001, 1]
            self.valid_month = [2014, 5]
            self.test_month = [2014, 6]


        # Traditional type dataset for baselines
        self.path_tra_db_train = self.path_db_processed + 'douban_train.csv'
        self.path_tra_db_valid = self.path_db_processed + 'douban_valid.csv'
        self.path_tra_db_test = self.path_db_processed + 'douban_test.csv'
        self.path_tra_db_test_zhixi = self.path_db_processed + 'douban_test_zhixi.csv'
        # self.path_tra_db_test_zhixi = self.path_db_processed + 'douban_zhixi_.csv'

        self.path_tra_amz_train = self.path_amz_processed + self.dataset + '_train.csv'
        self.path_tra_amz_valid = self.path_amz_processed + self.dataset + '_valid.csv'
        self.path_tra_amz_test = self.path_amz_processed + self.dataset + '_test.csv'

        ################ Training dataset config ################
        self.num_training_sample = 5  # Sample month number including the max popularity month
        self.num_template = 5  # template number for each training record
        self.rate_feature_select = 0.7

        ################ Time ################
        self.start_time = time.time()
        print()

        ################ Model ################
        # 'gpt-4.1-mini', 'deepseek-chat', 'gpt-4o-mini', 'gpt-4o', 'Llama', 'gemini-2.5-flash'
        # self.llm_model = 'deepseek-chat'
        self.llm_model = 'gpt-4o-mini'
        # self.llm_model = 'gemini-2.5-flash'
        self.llm_type = ''
        if 'gpt' in self.llm_model:
            self.llm_type = 'openai'
        elif 'deepseek' in self.llm_model:
            self.llm_type = 'deepseek'
        elif 'Llama' in self.llm_model:
            self.llm_type = 'Llama'
        elif 'gemini' in self.llm_model:
            self.llm_type = 'gemini'


        # Angel API Key
        self.deepseek_api_key = ''
        self.jing_llama_path = "/home/jiazheng/jingjiazheng/model/Llama-3.2-3B-Instruct"
        self.llama_path = self.jing_llama_path



        ################ Monitoring ################
        self.print_llm_input = True
        self.print_llm_output = True  # Please keep this config True

        ################ Others ################
        self.name_mon = ['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May.', 'Jun.',
                               'Jul.', 'Aug.', 'Sept.', 'Oct.', 'Nov.', 'Dec.']


        ################ Fine Tuning Setting ################
        self.ft_batch_size = 128
        self.ft_epoch = 20

        self.ft_system_prompt = (
            'I would like to use a Large Language Model (LLM) to predict the sales of this product for the next month.\n'
            'Please design a prompt that incorporates the product-related information and can be used as input to the LLM.\n'
            'The prompt you generate should consist of three parts:\n'
            '1. Clearly state that the LLM’s task is to predict the sales.\n'
            '2. Select and describe the relevant item features.\n'
            '3. Provide a reasonable description of the item’s monthly sales.\n'

            'I will provide you with the following information, '
            'both parts should be considered within the generating output: \n'

            '1. A product along with its relevant features. In this section,'
            ' you should first filter for features that are relevant to the item\'s popularity,'
            ' and then present them in an appropriate manner. '
            'You may consider summarizing or distilling the information from the features, '
            'or combining multiple features to generate more effective input. \n'

            '2. The product’s monthly sales data. '
            'You should present the item\'s monthly sales in an appropriate format. '
            # 'Your description should includes '
            'For example, use natural language to describe the popularity trend, key values,'
            ' and values of recent months, '
            'or provide only the sales data from the most recent 6 or 12 months to avoid unnecessary noise.'
            'Additionally, to ensure the informativeness of your output, '
            'please make sure to include enough concrete monthly sales values.'
            '\n'
        )

