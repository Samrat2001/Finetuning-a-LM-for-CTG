# Finetuning-a-LM-for-CTG

In this work, we propose a model capable of generating affect-driven and topic-focused sentences without losing grammatical correctness by incorporating emotion as a prior for the probabilistic state-of-the-art text generation model such as GPT-2. This model allows users to control the category and intensity of emotion and the topic of the generated text.
The code was from https://github.com/ishikasingh/Affective-text-gen repository. Further modifications are done using their code as base.

To run the code first use this Docker command to start the docker container 
sudo docker run --name jup --rm -it -v "$(pwd):/pack/wrkdir" -p 1234:8888 --gpus all samrat29042001/samrat-containers:summer1 bash

To install the dependencies
pip install -r requirements.txt

Then to run the model
python run.py
