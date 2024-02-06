# Import necessary libraries
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.metrics import precision_recall_curve, auc
from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore, SelfCheckNgram, SelfCheckNLI

# Install required packages (uncomment these lines if you're running this script outside Jupyter)
# !pip install selfcheckgpt datasets sentencepiece spacy
# !python -m spacy download en_core_web_sm

# Set the device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data
data = pd.read_csv('s3://d3-data-bucket/labs/trustworthy_ai/data/test.csv')
amazon_dataset = Dataset.from_pandas(data)
amazon_dataset_iter = amazon_dataset.to_iterable_dataset()

# Initialize the SelfCheckNLI model
selfcheck_nli = SelfCheckNLI()

# Define label mapping
label_mapping = {
    'accurate': 0.0,
    'minor_inaccurate': 1.0,
    'major_inaccurate': 1.0,
}

# Process the dataset and calculate scores
scores = []
for sample in amazon_dataset_iter:
    sent_scores_nli = selfcheck_nli.predict(
        sentences=sample['generated_description'],
        sampled_passages=sample['sample_generated_description'],
    )
    scores.extend(sent_scores_nli)

# Output the scores
print(scores)

# Save the scores along with their rows in another CSV locally
# Assuming you want to save the scores in a new column 'scores'
data['scores'] = scores
data.to_csv('data/test_scores_basemodel.csv', index=False)
