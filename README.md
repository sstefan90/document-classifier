# Fine Tuned Bert Sequence Classifier For Document Classification (HOSTED ON AWS)


## Docker and Model Deployment
The trained Model was hosted on AWS with an exposed API that anyone could query. A Flask app was created to run model inference on a text provided in the body of a 'get' request. To host the model, a Docker Image was created to (1) containerize the flask app, and (2) containerize the reverse proxy (Ngnix).
All relevant files for model deployment (with the exception of the trained model itself, as the filesize is too big for github) are available under the *flask_app* directory. 

The docker image was moved to a remote host machine on AWS. Docker was installed on the host machine, and the port 1337 was also configured to be exposed, to make sure clients can reach the Nginx reverse proxy.

If using flask_app to start you project, include your model in a folder directory under flask_app/services/web/project, or better yet, upload your model to S3 and query the file storage to get your model on docker build and setup. Please remember if doing so to inject secrets into your docker container, and to exclude any files that contain sensitive information from github by including said file paths in a .gitignore. Please take security seriously! While this setup makes sure that the docker container is not running as the root user, the docker deamon certainly is. This is okay for quick experimentation, however in a production environment, consider taking the additional step!

##Model Training
This model was trained using an AWS EC2 instance equipped with a Tesla T4 GPU. On local, training takes approximately 2 hours. On the GPU, training takes minutes
### project structure
#### data_exploration.ipybn
Data analysis and investigation of the dataset
#### bert.py
Main pytorch and huggingface training loop file. This file instantiates *bert-uncased-small model* from huggingface and finetunes a multiclass clasification model
#### utils.py
Helper methods to split the data into train, val, test. Methods to load training data into a pytorch DataLoader for mini-batch training
####requirements.txt
File that contains the packages required to run this project
#### evaluation.ipybn
final evaluation of the trained model, loaded from a checkpoint





