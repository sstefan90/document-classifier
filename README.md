# Fine Tuned Bert Sequence Classifier For Document Classification
## project structure
### data_exploration.ipybn
Data analysis and investigation of the dataset
### bert.py
Main pytorch and huggingface training loop file. This file instantiates *bert-uncased-small model* from huggingface and finetunes a multiclass clasification model
### utils.py
helper methods to split the data into train, val, test. Methods to load training data into a pytorch DataLoader for mini-batch training
### requirements.txt
file that contains the packages required to run this project
### evaluation.ipybn
final evaluation of the trained model, loaded from a checkpoint

## Model Training

This model was trained using an AWS EC2 instance equipt with a Tesla T4 GPU. On local, training takes approximately 2 hours. On the GPU, training takes minutes
