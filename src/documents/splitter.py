import os
import re

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader, CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# load environment variables like user agent etc.
load_dotenv()


def WebsiteLoader(urls_to_be_scrapped):
    loader = WebBaseLoader(urls_to_be_scrapped)
    return loader.load()


def CSVFileLoader(csv_file_paths):
    docs = []
    for csv_file_path in csv_file_paths:
        loader = CSVLoader(file_path=csv_file_path)  # Load each CSV file individually
        docs.extend(loader.load())  # Append loaded data to the list
    return docs


def PDFLoader(pdf_file_paths):
    docs = []
    for pdf_file_path in pdf_file_paths:
        loader = PyPDFLoader(file_path=pdf_file_path)  # Load each PDF file individually
        docs.extend(loader.load())  # Append loaded data to the list
    return docs


def sanitize_filename(filename: str) -> str:
    # Define a regex pattern that matches invalid characters in file names
    invalid_chars = r'[<>:"/\\|?*.]'

    # Replace any invalid characters with an underscore (or any other character you prefer)
    sanitized_name = re.sub(invalid_chars, '', filename)

    # Optionally, truncate the filename to avoid exceeding the OS's file name length limit
    # For example, on Windows the max length is 255 characters
    max_length = 255
    if len(sanitized_name) > max_length:
        sanitized_name = sanitized_name[:max_length]

    return sanitized_name


def saveSplits(document, document_splits, output_directory):
    # Create directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    split_id = 0
    for document_split in document_splits:
        with open(os.path.join(
                output_directory, f"{sanitize_filename(document.metadata.get('source', ''))}_split_{split_id}.txt"),
                "w", encoding="utf-8") as f:
            f.write(document_split)  # Write each split to the file
            f.write("\n")
            split_id += 1

    print(f"Documents saved to {output_directory}")


def splitWebDocument(urls):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=25,
        length_function=len,
        is_separator_regex=False
    )

    # text_splitter = CharacterTextSplitter(
    #     chunk_size=1,
    #     chunk_overlap=1,
    #     length_function=len,
    #     is_separator_regex=False
    # )

    # web urls to be scrapped
    web_documents = WebsiteLoader(urls)

    for web_document in web_documents:
        # Preprocess content to remove blank lines and any extra whitespace
        web_document_text = " ".join([line for line in web_document.page_content.splitlines() if line.strip()])

        # split web documents into chunks
        web_document_splits = text_splitter.split_text(web_document_text)

        # save splits of the document
        saveSplits(web_document, web_document_splits, "../web")


if __name__ == '__main__':
    web_urls = ["https://spellingscan.com"]
    splitWebDocument(web_urls)
