import os
from transformers import BertTokenizer
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import random

NUM_LABELS = 8
DATA_FILE_NAME = "./data/file.txt"
TRAIN_FILE_NAME = "./data/train.txt"
VAL_FILE_NAME = "./data/val.txt"
TEST_FILE_NAME = "./data/test.txt"

seed_val = 42

random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


#define class for dataLoader
class DocumentData(Dataset):

    def __init__(self, filename, max_length,  tokenizer):
        self.filename = filename
        self.max_length = max_length

        self.X, self.y = parse_document_content(self.filename, zero_index=False)
        self.n_samples = len(self.X)
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        sample_x, sample_y = self.X[index], self.y[index]
        token_x = self.tokenizer(sample_x, return_tensors="pt", truncation=True, max_length=self.max_length, padding='max_length')
        return token_x, sample_y

    def __len__(self):
        return self.n_samples



def parse_document_content(filename, zero_index=True):
    x, y = [], []
    
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            
            doc_class = line[0]
            doc_content = line[2:-2]
            x.append([doc_content])
            if zero_index:
                y.append([int(doc_class)-1])
            else:
                y.append([int(doc_class)])
            
    return x, y

def write_train_val_test(data_x, data_y, filename):

    assert len(data_x) == len(data_y)

    with open(filename, "w") as f:
        for i in range(len(data_x)):
            y = data_y[i][0]
            x = data_x[i][0]
            
            f.write(str(y) + " " + x + "\n")

def stratified_sampling():

    X, y = parse_document_content(DATA_FILE_NAME)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.15)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size = 0.20)

    #roughly, with these numbers, split works out to 70% train, 15% val, 15% test

    write_train_val_test(X_train, y_train, TRAIN_FILE_NAME)
    write_train_val_test(X_val, y_val, VAL_FILE_NAME)
    write_train_val_test(X_test, y_test, TEST_FILE_NAME)

def process_data(filename, batch_size, max_length):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset  = DocumentData(filename,max_length,tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return dataloader


"""
When importing this file, the following lines
will run stratified sampling and create new txt
files in the dir ./data . files val.txt, train.txt
and test.txt will be created (if these paths do not exists)
"""


if (not os.path.exists(TRAIN_FILE_NAME) or not
        os.path.exists(VAL_FILE_NAME) or not
        os.path.exists(TRAIN_FILE_NAME)):
        print(f"*****RUNNING STRATIFIED SAMPLING******")
        stratified_sampling()
        


if __name__ == "__main__":
    #if running this function as a main function, run the process data
    #for debugging purposes!
    process_data(TRAIN_FILE_NAME, batch_size=16, max_length=256)


    

    



