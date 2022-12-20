"""
Example script to create elasticsearch documents.
"""
import argparse
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer
import os
from PyPDF2 import PdfFileReader 

MODEL_NAME = "ricardo-filho/bert-base-portuguese-cased-nli-assin-2"

def get_emb(inputs_list,model_name,max_length=512):
    model = SentenceTransformer(model_name, device='cuda')
    return model.encode(inputs_list)

def create_document(doc, emb, index_name):
    return {
        '_op_type': 'index',
        '_index': index_name,
        'text': doc['text'],
        'title': doc['title'],
        'text_vector': emb
    }

def load_dataset_from_pdf_files(file_path = 'pdf_files'):
    docs = []
    for file in os.listdir(file_path):
        pdfFileObj = open(file_path + '/' + file, 'rb')
        pdfReader = PdfFileReader(pdfFileObj)
        num_pages = pdfReader.numPages
        for count in range(num_pages):
            pageObj = pdfReader.getPage(count)
            text = pageObj.extractText()
            doc = {
                'title': file.replace('.pdf',''),
                'text': text
            }
            docs.append(doc)
    return docs


def bulk_predict(docs, model_name,batch_size=256):
    """Predict bert embeddings."""
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i: i+batch_size]
        embeddings = get_emb(inputs_list=[doc['text'] for doc in batch_docs], model_name = model_name, max_length=512)
        for emb in embeddings:
            yield emb


def load_dataset(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main(args):
    # create docoments
    print("loading data from mysql....")
    docs = load_dataset_from_pdf_files()
    print("creating documents...")
    with open(args.data, 'w') as f:
        for doc, emb in zip(docs, bulk_predict(docs,model_name=MODEL_NAME)):
            d = create_document(doc, emb, args.index_name)
            f.write(json.dumps(d) + '\n')

    # create index
    print("creating index in elasticsearch...")
    client = Elasticsearch()
    client.indices.delete(index=args.index_name, ignore=[404])
    with open(args.index_file) as index_file:
        source = index_file.read().strip()
        client.indices.create(index=args.index_name, body=source)

    #index documents
    print("index documents...")
    client = Elasticsearch()
    docs = load_dataset(args.data)
    bulk(client, docs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating elasticsearch documents.')
    # parser.add_argument('--model_name', default='t5-small', help='model name, could be '
    # '"t5-small","t5-base","t5-large","t5-3b" and "t5-11b" for t5')
    parser.add_argument('--index_file', default='index.json', help='Elasticsearch index file.')
    parser.add_argument('--index_name', default='docsearch', help='Elasticsearch index name.')
    parser.add_argument('--data', default='documents.jsonl', help='Elasticsearch documents.')
    args = parser.parse_args()
    main(args)
