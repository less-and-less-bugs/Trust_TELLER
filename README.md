# Trustworthy Misinformation Detector: TELLER

The official implementation of our paper "TELLER: A Trustworthy Framework for Explainable, Generalizable and
Controllable Fake News Detection". 

## Getting Started

Step 1: Download the dataset folder from onedrive by [data.zip](https://portland-my.sharepoint.com/:u:/g/personal/liuhui3-c_my_cityu_edu_hk/EfApQlFP3PhFjUW4527STo0BALMdP16zs-HPMNgwQVFWsA?e=zoHlW2). Unzip this folder into the project  directory.  You can find four orginal datasets,  pre-processed datasets (i.e., val.jsonl, test.jsonl, train.jsonl in each dataset folder) and the files incuding questions and answers 

Step 2: Place you OpenAI key into the file named api_key.txt. 

```
openai.api_key = ""
```

# Running Our Codes 

To reproduce the results of in-domain experiments on four Dataset: 

python drive_liar.py

To reproduce the results of cross-domain experiments on three datasets:

python drive_dg.py
