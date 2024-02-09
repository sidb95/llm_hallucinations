from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import io
import boto3

# Connect to S3 client

s3 = boto3.client('s3',
                  aws_access_key_id='AKIAUAIWP5FGWDIAHCFJ',
                  aws_secret_access_key='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')

# Define path location from S3 and local file path to access files
bucket_name = 'd3-generative-ai'
file_key = 'data/processed/curated_data/Noble B case.pdf'
local_file_path = '/root/DemoNotebooks/example.pdf'

s3.download_file(bucket_name, file_key, local_file_path)

# Open the PDF file
pdf_path = 'example.pdf'
with open(pdf_path, 'rb') as file:
    parser = PDFParser(file)
    document = PDFDocument(parser)

    # Extract metadata
    metadata = document.info
    print("Metadata:", metadata)

    # Create a PDF resource manager
    rsrcmgr = PDFResourceManager()

    # Set up a text converter
    output = io.StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, output, laparams=laparams)

    # Create a PDF page interpreter
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    # Iterate over the PDF pages
    for page in PDFPage.create_pages(document):
        interpreter.process_page(page)

    # Get the text content
    text_content = output.getvalue()

    # Close the files
    file.close()
    device.close()
    output.close()

    # Save the text content to a .txt file
    with open('output.txt', 'w') as text_file:
        text_file.write(text_content)

    print("Text content saved to 'output.txt'")
