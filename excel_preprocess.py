import os
import re
import json
import config
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

all_text_path = config.BASE_PATH
all_text_file = os.listdir(all_text_path)
citis = json.load(open("output/cities.json"))


# Helper function to check if a string is numeric
def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# Extract start page from the text
def extract_start_page(content, threshold=10):
    for row in content[:threshold]:
        row_dict = eval(row)
        if not row_dict:
            continue
        inside_text = row_dict["inside"]
        start_page = re.findall(r"第.+节财务报告\.+(\d+)", inside_text)
        if start_page:
            return int(start_page[0])
    return 1


# Process file content to extract specific tables
def extract_table_data(file_name, table_names):
    with open(file_name, "r", encoding="utf-8") as f:
        content = f.readlines()
        result_dict = defaultdict(list)
        start_page = extract_start_page(content)
        address = ""

        mark = False
        count = 0

        for row in content:
            row_dict = eval(row)
            if not row_dict:
                continue

            inside_text = row_dict["inside"]
            text_type = row_dict["type"]
            page_num = row_dict["page"]

            if page_num < start_page:
                continue

            # Extract company address
            if "注册地址" in inside_text and page_num < 20 and not address:
                for city in citis:
                    if city in inside_text:
                        address = city

            # Skip unwanted content
            if not inside_text or text_type in ["页眉", "页脚"]:
                continue

            # Detect table headers
            for table_name in table_names:
                if (
                    text_type == "text"
                    and (table_name in inside_text)
                    and (
                        len(inside_text) <= 25
                        or inside_text.startswith(table_name)
                        or inside_text.endswith(table_name)
                    )
                ):
                    mark = table_name
                    break

            if mark:
                count += 1
                if count > 6 and not result_dict[mark]:
                    mark = False
                elif text_type == "excel":
                    result_dict[mark].append(inside_text)
                else:
                    count = 0

    return result_dict, address


# Convert extracted table data to a more usable format
def transform_table_data(result_dict, year_value):
    new_dict = defaultdict(list)

    for table_name, rows in result_dict.items():
        delete_index = False
        header_length = 0

        for idx, row in enumerate(rows):
            temp_row = []
            data = eval(row)

            if idx == 0:
                header_length = len(data)
                for cell_idx, cell in enumerate(data):
                    if "附注" in cell:
                        delete_index = cell_idx
                    temp_row.append(parse_header(cell, year_value))
                new_dict[table_name].append(temp_row)
            else:
                if not re.search("\d", row):
                    continue

                for cell_idx, cell in enumerate(data):
                    if (
                        delete_index
                        and cell_idx == delete_index
                        and len(data) == header_length
                    ):
                        continue
                    if cell_idx == 0:
                        cell = clean_special_characters(cell)
                    temp_row.append(cell)

                if len(temp_row) > 1:
                    if any(is_numeric(c.replace(",", "")) for c in temp_row[1:]):
                        new_dict[table_name].append(temp_row)

    return new_dict


# Parse header row, handling specific year and period cases
def parse_header(cell, year_value):
    cell = re.sub(r"（.+）|\(.+\)", "", cell).strip()
    year_match = re.findall(r"20(?:\d{2})", cell)
    if "年末余额" == cell or "期末余额" == cell:
        return int(year_value)
    elif "上年年末" in cell or "上期期末" in cell:
        return int(year_value) - 1
    elif year_match:
        return int(year_match[0])
    return cell


# Clean up special characters from text
def clean_special_characters(text):
    text = re.sub(r"、|：|\.|\．", "", text)
    text = re.sub(r"（.+）|\(.+\)", "", text)
    return text.strip()


# Process company basic information and employee data
def process_company_info(txt_files, all_text_files, final_dict_base):
    for file in tqdm(txt_files, desc="Processing Company Info"):
        if file not in all_text_files:
            continue

        file_name = os.path.join(all_text_path, file)
        year_value = file_name.split("__")[-2][:4]

        with open(file_name, "r", encoding="utf-8") as f:
            content = f.readlines()

        address, employee_data = "", {}

        for row in content:
            row_dict = eval(row)
            if not row_dict:
                continue

            inside_text = row_dict["inside"]
            text_type = row_dict["type"]
            page_num = row_dict["page"]

            # Extract company address from early pages
            if "注册地址" in inside_text and page_num < 20 and not address:
                for city in citis:
                    if city in inside_text:
                        address = city
                        break

            # Extract employee data, based on keyword or context clues
            if "员工人数" in inside_text or "员工构成" in inside_text:
                employee_data["employees"] = inside_text

        if address or employee_data:
            output_name = "__".join(file.split("__")[-3:-1])
            final_dict_base[output_name] = {
                "year": year_value,
                "name": file,
                "address": address,
                "employee_data": employee_data,
            }

    return final_dict_base


# Extract and process asset-liability, profit, and cash flow tables
def process_financial_reports(txt_files, all_text_files, final_dict, table_names):
    for file in tqdm(txt_files, desc=f"Processing {table_names[0]}"):
        if file not in all_text_files:
            continue

        file_name = os.path.join(all_text_path, file)
        year_value = file_name.split("__")[-2][:4]
        result_dict, address = extract_table_data(file_name, table_names)

        if not result_dict:
            result_dict, address = extract_table_data(
                file_name, [t.replace("合并", "") for t in table_names]
            )

        transformed_data = transform_table_data(result_dict, year_value)

        if transformed_data:
            output_name = "__".join(file.split("__")[-3:-1])
            transformed_data["year"] = year_value
            transformed_data["name"] = file
            transformed_data["address"] = address
            final_dict[output_name] = transformed_data

    return final_dict


# Main function to handle all processing
def main():
    citis = json.load(open("output/cities.json"))
    all_text_files = os.listdir(config.BASE_PATH)
    txt_files = [
        i.replace(".pdf", ".txt")
        for i in open(config.PDF_LIST, "r", encoding="utf-8").readlines()
    ]

    final_dict_zichan = {}
    final_dict_mugongsi = {}
    final_dict_base = {}

    # Process asset-liability, profit, and cash flow reports for both merged and parent company
    final_dict_zichan = process_financial_reports(
        txt_files,
        all_text_files,
        final_dict_zichan,
        ["合并资产负债表", "合并利润表", "合并现金流量表"],
    )
    json.dump(
        final_dict_zichan,
        open(f"output/zichan.json", "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )

    final_dict_mugongsi = process_financial_reports(
        txt_files,
        all_text_files,
        final_dict_mugongsi,
        ["母公司资产负债表", "母公司利润表", "母公司现金流量表"],
    )
    json.dump(
        final_dict_mugongsi,
        open(f"output/mugongsi.json", "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )

    # Process company basic information and employee data
    final_dict_base = process_company_info(txt_files, all_text_files, final_dict_base)
    json.dump(
        final_dict_base,
        open(f"output/base.json", "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    main()
