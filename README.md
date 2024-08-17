# Project Focus

This project focuses on answering questions using large language models (LLMs) by extracting insights from financial reports. Given a question library and a set of financial reports, the approach is to target the reports relevant to the question, extract the most relevant content, and then use an LLM to generate an accurate response. The pipeline involves several key steps, from data preprocessing to response generation.

# Methodology

## Step 1: Data Cleaning and Preprocessing

    •	Script: company_product.py, calculate_dict_product.py, excel_preprocess.py
    •	Description: This step involves preparing the data from raw financial reports and question sets. The scripts clean and preprocess the financial data and build mappings for company names, product information, and additional necessary dictionaries.

## Step 2: PDF Title Embedding and Similarity Search

    •	Description: This step involves preparing the data from raw financial reports and question sets. The scripts clean and preprocess the financial data and build mappings for company names, product information, and additional necessary dictionaries.

## Step 3: Text Embedding and Similarity Search

    •	Script: predict.py
    •	Description: This step involves embedding the text of financial reports into vector space. The embeddings are then used to search for content relevant to the target question within the most related PDFs.

## Step 4: Response Generation

    •	Script: predict.py
    •	Description: Once the relevant content is identified, it is used as input to a large language model (LLM). The LLM is prompted with the content and the question to generate a precise response.

# File Descriptions

## company_product.py

This script extracts and prepares company and product information, building mappings that are used later for text embedding and question answering.

## calculate_dict_product.py

Handles dictionary calculations and preprocessing necessary for parsing the financial data.

## excel_preprocess.py

Preprocesses Excel files which are part of the financial reports, structuring the data into a format that can be easily processed by the pipeline.

## data_split.py

Splits and processes financial data for vector embedding. This step is crucial for embedding titles of PDFs and performing similarity searches on those titles to identify relevant documents for answering questions.

## predict.py

The core script for generating responses. It embeds the financial reports into a vector space, performs similarity searches based on the question and content, and then uses an LLM to generate the final answer.
