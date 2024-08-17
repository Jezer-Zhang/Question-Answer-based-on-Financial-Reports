# coding: UTF-8
import os
import time
import torch
import pandas as pd
import gc
import json
import logging
from transformers import AutoTokenizer, AutoModel
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import re
import config

# Utility functions from utils
from utils import (
    content_product,
    load_prompt,
    _load_vector,
    load_prompt_vector,
    load_prompt_statistic,
)

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = config.MODEL_DIR
EMBEDDING_MODEL = config.EMBEDDING_MODEL
OUTPUT_DIR = config.OUT_DIR

# Paths
TEST_PATH = "output/question_type.json"
COMPANY_PATH = "output/company/company.json"
FULLNAME_SHORTNAME_PATH = "output/company/fullname_shortname.json"
TEXT_VECTOR_PATH = "output/vector"
TITLE_VECTOR_PATH = "output/title_vector"

# Constants
COMPANY_MARKER = "公司"
OF_MARKER = "的"

# Initialize
years = ["2019", "2020", "2021", 2022]
fout_temp = []

# Model and Tokenizer Setup
logger.info("Loading model and tokenizer.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True).half().cuda()
model.eval()

# Load full name and short name mappings
logger.info("Loading company mappings.")
with open(COMPANY_PATH, "r") as f:
    stock_mapping = json.load(f)

with open(FULLNAME_SHORTNAME_PATH, "r") as f:
    fullname_shortname = json.load(f)

# Load or create FAISS vector store for company names
docs_company = []
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

if not os.path.exists(TITLE_VECTOR_PATH):
    os.makedirs(TITLE_VECTOR_PATH)
    for idx, (fullname, shortname) in enumerate(fullname_shortname.items(), start=1):
        metadata = {"source": f"{idx}"}
        docs_company.append(Document(page_content=fullname, metadata=metadata))
    vector_company = FAISS.from_documents(docs_company, embeddings)
    vector_company.save_local(TITLE_VECTOR_PATH)
else:
    vector_company = FAISS.load_local(TITLE_VECTOR_PATH, embeddings=embeddings)


def load_test_questions(test_path):
    """Load and parse test questions from a JSON file."""
    with open(test_path) as f:
        return [json.loads(line) for line in f.readlines()]


def truncate_question(question, truncate_index):
    """Truncate question by removing content up to the truncate_index and clean specific patterns."""
    question = question[truncate_index:]
    cleaned_text = re.sub(r"在\d{4}年|\d{4}年", "", question)
    return (
        cleaned_text[1:]
        if cleaned_text and cleaned_text[0] == OF_MARKER
        else cleaned_text
    )


def truncate_opening_question(question, truncate_index):
    """Truncate opening questions while cleaning them based on specific patterns."""
    tmp_question = question.split("，")
    if len(tmp_question) == 2 and re.search("20\d{2}", tmp_question[1]):
        return tmp_question[0]

    return truncate_question(question, truncate_index)


def handle_statistics_question(row_question):
    """Process 'statistics' question type."""
    prompt_statistic, model_product = load_prompt_statistic(row_question["question"])
    if prompt_statistic and model_product:
        answer, _ = model.chat(tokenizer, prompt_statistic, history=[])
    else:
        answer = prompt_statistic if prompt_statistic else row_question["question"]

    row_question["answer"] = answer
    torch.cuda.empty_cache()
    gc.collect()
    return row_question


def find_company_and_year(question):
    """Find years and stock names in the question."""
    years_list = re.findall("20(?:\d{2})", question)
    company_list = [
        (stock, question.find(stock), question.find(stock) + len(stock))
        for stock in stock_mapping
        if stock in question
    ]
    company_list.sort(
        key=lambda x: x[1]
    )  # Sort companies by their appearance in the text
    return years_list, company_list


def process_content_for_years_and_companies(
    question, question_type, years_list, company, cleaned_text
):
    """Process content for the given years and companies."""
    content_list = []
    use_excel = False
    for year in years_list:
        content_year, use_excel = content_product(
            question, company, year, cleaned_text, question_type
        )
        content_list.extend(content_year)
        if question_type in {"basic", "peplenum"}:
            break  # Exit early if basic or peplenum type

    content_list = sorted(
        set(content_list), key=content_list.index
    )  # Remove duplicates while maintaining order
    return content_list, use_excel


def generate_answer_from_vector_store(
    row_question, question, prompt, faiss_vector_store
):
    """Generate answer using FAISS vector store."""
    answer, _ = model.chat(tokenizer, prompt, history=[])
    row_question["answer"] = answer
    torch.cuda.empty_cache()
    gc.collect()
    return row_question


def process_test_questions(test_questions):
    """Process the list of test questions."""
    no_prompt_question = []
    count = 0

    for row_question in tqdm(test_questions):
        question = row_question["question"]
        question_type = row_question["question_type"]

        # Special handling for statistics questions
        if question_type == "statistics":
            fout_temp.append(handle_statistics_question(row_question))
            continue

        years_list, company_list = find_company_and_year(question)
        if years_list and company_list:
            # Truncate question
            truncate_index = max(company_list[0][2], question.find(COMPANY_MARKER) + 2)
            cleaned_text = truncate_question(question, truncate_index)
            if question_type == "opening":
                cleaned_text = truncate_opening_question(question, truncate_index)

            # Process content
            content_list, use_excel = process_content_for_years_and_companies(
                question, question_type, years_list, company_list[0][0], cleaned_text
            )

            if not use_excel:
                faiss_vector = _load_vector(
                    TEXT_VECTOR_PATH,
                    company_list[0][0],
                    years_list[0],
                    embeddings,
                    question_type,
                )
                no_prompt_question.append(row_question)
                prompt = load_prompt_vector(
                    faiss_vector,
                    question,
                    question_type,
                    cleaned_text,
                    company_list[0][0],
                    years_list[0],
                )
                fout_temp.append(
                    generate_answer_from_vector_store(
                        row_question, question, prompt, faiss_vector
                    )
                )
            else:
                # Handle Excel logic if applicable
                # Example usage: handle_excel_logic(row_question, ...)
                pass  # Placeholder for handling logic involving Excel usage
            count += 1
        else:
            # Fallback if no year or company found
            similar_fullname = vector_company.similarity_search(question, 1)[
                0
            ].page_content
            short_name = fullname_shortname[similar_fullname]
            cleaned_text = truncate_question(
                question, question.find(COMPANY_MARKER) + 2
            )

            # Process content similarly
            content_list, use_excel = process_content_for_years_and_companies(
                question, question_type, years_list, short_name, cleaned_text
            )
            if not use_excel:
                faiss_vector = _load_vector(
                    TEXT_VECTOR_PATH,
                    short_name,
                    years_list[0],
                    embeddings,
                    question_type,
                )
                no_prompt_question.append(row_question)
                prompt = load_prompt_vector(
                    faiss_vector,
                    question,
                    question_type,
                    cleaned_text,
                    short_name,
                    years_list[0],
                )
                fout_temp.append(
                    generate_answer_from_vector_store(
                        row_question, question, prompt, faiss_vector
                    )
                )
            else:
                # Handle Excel logic if applicable
                pass  # Placeholder for Excel handling logic

    return fout_temp, count


def post_process_answers(fout_temp):
    """Post-process answers to remove commas from numbers and write results to file."""
    pattern = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    with open(OUTPUT_DIR, "w", encoding="utf8") as fout_final:
        for result in fout_temp:
            new_answer = result["answer"]
            matches = re.finditer(pattern, new_answer)
            for single_match in matches:
                sub_str = single_match.group()
                new_answer = new_answer.replace(sub_str, sub_str.replace(",", ""))
            result["answer"] = new_answer
            fout_final.write(json.dumps(result, ensure_ascii=False) + "\n")


# Main Execution
logger.info("Starting prediction process.")
test_questions = load_test_questions(TEST_PATH)
fout_temp, count = process_test_questions(test_questions)
post_process_answers(fout_temp)
logger.info(f"Processed {len(fout_temp)} questions, {count} using Excel.")
