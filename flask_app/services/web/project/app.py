import torch
from flask import Flask, jsonify
from flask_restful import Api, Resource
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import os

app = Flask(__name__)
api = Api(app)

'''
Read the model from local for testing
In production, consider uploading the model to S3
and have the app query and fetch the model.
Use docker secrets for safe authentication
'''
file_name = os.getenv("APP_HOME") + "/project/model"
model = BertForSequenceClassification.from_pretrained(file_name)
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

'''
#for debuggingg purposes
@app.route("/")
def hello_world():
    return jsonify(hello='world')
'''

class ModelOutput(Resource):
    def get(self, text):
        with torch.inference_mode():

            X = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding='max_length')

            input_ids = X['input_ids'].reshape(
                    (X['input_ids'].shape[0], X['input_ids'].shape[-1]))
            attention_mask = X['attention_mask'].reshape(
                    (X['attention_mask'].shape[0], X['attention_mask'].shape[-1]))

            predictions = model(input_ids=input_ids, token_type_ids=None,
                                attention_mask=attention_mask, return_dict=False)
            
            predicted_class = torch.argmax(predictions[0], axis=1)


        return {"model_class_prediction": str(predicted_class.item())}

api.add_resource(ModelOutput, "/model_score/<string:text>")
