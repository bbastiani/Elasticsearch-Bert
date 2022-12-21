from pprint import pprint
from flask import Flask, render_template, jsonify, request
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import os
SEARCH_SIZE = 10
MODEL_NAME = "ricardo-filho/bert-base-portuguese-cased-nli-assin-2"

INDEX_NAME = "docsearch"
app = Flask(__name__)


def get_emb(inputs_list,model_name,max_length=512):
    model = SentenceTransformer(model_name, device='cpu')
    return model.encode(inputs_list)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search')
def analyzer():
    client = Elasticsearch('http://elasticsearch:9200')

    query = request.args.get('q')
    query_vector = get_emb(inputs_list=[query],model_name=MODEL_NAME,max_length=768)[0]

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'text_vector') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }

    response = client.search(
        index=INDEX_NAME,
        body={
            "size": SEARCH_SIZE,
            "query": script_query,
            "_source": {"includes": ["title", "text"]}
        }
    )
    print(query)
    pprint(response)
    return jsonify(response.body)


if __name__ == '__main__':
    print("Starting Flask Server")
    app.run(host='0.0.0.0', port=5000)
