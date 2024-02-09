"""
Created on Sat Jul 29 10:29:53 2023
@author: acwilbur
This script runs a labelling process using Sagemaker and a HuggingFace model.
"""

import os
import csv
import json
import boto3
import pandas as pd
import numpy as np
from tqdm import tqdm
from sagemaker import Session
from sagemaker.s3 import S3Uploader, S3Downloader, s3_path_join
from sagemaker.huggingface.model import HuggingFaceModel

# AWS session and role setup
sagemaker_session = Session()
role = sagemaker_session.get_caller_identity_arn()
aws_region = boto3.Session().region_name

# S3 bucket setup
sagemaker_session_bucket = "d3-data-bucket"

# Printing session details
print(f"sagemaker role arn: {role}")
print(f"sagemaker bucket: {sagemaker_session.default_bucket()}")
print(f"sagemaker session region: {sagemaker_session.boto_region_name}")

# HuggingFace Model configuration
hub_ai = {
    'HF_MODEL_ID':'wilburchen42/fbert-aiclass',
    'HF_TASK':'text-classification'
}

# Creating HuggingFace Model Class
huggingface_model_ai = HuggingFaceModel(
   env=hub_ai,
   role=role,
   transformers_version="4.26",
   pytorch_version="1.13",
   py_version='py39',
)

# Transformer for batch job
output_s3_path = 'labs/digital-value/project-ai-transformation-classifier/data/conference_call_data/results'
s3_out_path = f's3://' + sagemaker_session_bucket + '/' + output_s3_path
batch_job_ai = huggingface_model_ai.transformer(
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    output_path=s3_out_path,
    strategy='SingleRecord')

# Creating an S3 client
s3 = boto3.client('s3')

# Processing files by year and type
input_s3_path = 'labs/digital-value/project-ai-transformation-classifier/data/conference_call_data'
for year in range(2016, 2022):
    #for Type in ['questions', 'response', 'presentation']:
    for Type in ['presentation']:
        # Checking if the file exists in S3 bucket
        file_key = input_s3_path + f"/transcript_{year}_{Type}.csv"
        try:
            s3.head_object(Bucket=sagemaker_session_bucket, Key=file_key)
        except Exception as e:
            print(f"The file {file_key} does not exist in the S3 bucket.")
            continue

        # If file exists, download and process it
        dataset_csv_file = f"transcript_data/transcript_{year}_{Type}.csv"
        s3.download_file(sagemaker_session_bucket, file_key, dataset_csv_file)
        export_csv_file = dataset_csv_file.replace(".csv","_labelled.csv")
        dataset_jsonl_file = f"transcript_data/transcript_{year}_{Type}.jsonl"
        sentence_cols = 'question' if Type == 'questions' else Type

        # Checking for NaN values
        def nan_check(x):
            return '' if pd.isnull(x) else x

        # Processing CSV file and creating JSONL file
        with open(dataset_csv_file, "r+") as infile, open(dataset_jsonl_file, "w+") as outfile:
            reader = csv.DictReader(infile)
            for row in reader:
                input_sentence = {"inputs": nan_check(row[sentence_cols])}
                json.dump(input_sentence, outfile)
                outfile.write('\n')

        # Uploading JSONL file to S3 and starting batch transform job
        jsonl_s3_path = f's3://' + sagemaker_session_bucket + '/' + input_s3_path + "/data"
        s3_file_uri = S3Uploader.upload(dataset_jsonl_file, jsonl_s3_path)
        print(f"{dataset_jsonl_file} uploaded to {s3_file_uri}")
        batch_job_ai.transform(
            data=s3_file_uri,
            content_type='application/json',    
            split_type='Line')

        # Creating S3 URI for result file
        s3_dataset_jsonl_file = f"transcript_{year}_{Type}.jsonl"
        output_path = s3_path_join(s3_out_path, s3_dataset_jsonl_file) + '.out'
        local_path = f"transcript_data/{s3_dataset_jsonl_file}.out"
        print(output_path)
        print(local_path)
        try:
            s3.download_file(sagemaker_session_bucket, output_path, local_path)
        except Exception as e:
            print(f"Failed to download the output file from S3. Error: {e}")
            continue

        # Processing output and exporting results
        batch_transform_result = []
        with open(local_path) as f:
            for line in f:
                line = "[" + line.replace("[", "").replace("]", ",") + "]"
                batch_transform_result = literal_eval(line)

        batch_transform_label = [x['label'] for x in batch_transform_result]
        batch_transform_score = [x['score'] for x in batch_transform_result]
        export = pd.read_csv(dataset_csv_file)
        export['ai_labels'] = batch_transform_label
        export['ai_score'] = batch_transform_score
        export.to_csv(export_csv_file)
