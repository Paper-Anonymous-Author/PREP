# PREP

This is the implementation for our AAAI2025 paper:
> Ranking Tomorrow’s Hits: Recommendation via LLM Popularity Forecasts with Learned Prompt Generation

## Environment
We use Python language and Pytorch library to establish our model. 

For the detailed environment, please follow `requirements.txt`

## Dataset
We leverage the _Douban Movies_, _Amazon Baby_, and _Amazon Beauty_.

## Train *PREP*
### Dataset Preprocessing
We take dataset "douban" for an example, for Amazon Baby and Amazon Beauty dataset, please set
```
--dataset="Baby"    --dataset="Beauty"
```
Please preprocess the dataset for *PREP* with
```
python ./run_model.py --dataset="douban" --task="dataset"
```
### Model Training
First initialize *PREP* with 
```
python ./run_model.py --dataset="douban" --task="init"
```
Then Fine-tune *PREP* with
```
python ./run_model.py --dataset="douban" --task="finetune"
```
### Model Evaluation
First generate prompt with 
```
python ./run_model.py --dataset="douban" --task="generate"
```
Then evaluate *PREP* with
```
python ./run_model.py --dataset="douban" --task="evaluate"
```
