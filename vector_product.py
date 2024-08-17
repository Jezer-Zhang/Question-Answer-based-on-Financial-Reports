# coding: UTF-8
import gc
import glob

import torch
import time
import os
import json
from collections import defaultdict
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from tqdm import tqdm
import config


text_path = config.BASE_PATH
output_path = "output"
vector_path = "output/vector"
embedding = config.EMBEDDING_MODEL


if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(vector_path):
    os.makedirs(vector_path)

file_df = glob.glob(f"{text_path}/*.txt")

# 获取公司名称
company_path = os.path.join(output_path, "company")
if not os.path.exists(company_path):
    os.makedirs(company_path)
    company_list = []
    fullname_shortname = {}
    fullname = []
    for file in os.listdir(text_path):
        company = file.split("__")[-3]
        company_list.append(company)
    company_list = list(set(company_list))
    print("公司数量是", len(company_list))
    json.dump(
        company_list,
        open(f"{company_path}/company.json", "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )

    # 创建公司全程-简称的字典
    fullname_shortname = {}
    for file in os.listdir(text_path):
        fullname = file[12:-4]
        shortname = file.split("__")[-3]
        fullname_shortname[fullname] = shortname
    json.dump(
        fullname_shortname,
        open(f"{company_path}/fullname_shortname.json", "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )
