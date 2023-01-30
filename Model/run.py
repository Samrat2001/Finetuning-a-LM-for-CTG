from score_model import run_pplm_example
from transformers.modeling_gpt2 import GPT2LMHeadModel
from transformers import GPT2Tokenizer
import torch
import sys


topics = ["legal",'military','monsters', 'science','space','politics', 'religion','technology','positive_words']
affects = ['anticipation', 'disgust', 'surprise', 'trust']#['joy', 'anger', 'sadness','fear']
prompts = [ 'The book', 'The issue focused on', 'The robots', 'The relationship','The road']
knob_vals = [0.1,0.5,1]
outputs = []




  for topic in topics:
    for affect in affects:
      for knob in knob_vals:
        print("topic:", topic, ", affect:", affect, ", knob is:", knob)
        
        
        output = run_pplm_example(
            affect_weight=1,  
            knob = knob, # 0-1,
            cond_text=prompt,
            num_samples=1,
            bag_of_words=topic,
            bag_of_words_affect=affect,
            length=50,
            stepsize=8e-4, 
            sample=True,
            num_iterations=40,
            window_length=6,
            gamma=1.5,
            gm_scale=0.95,
            score_scale=1,
            verbosity='REGULAR',
            end_lr = 1e-4,
            N = 10,
            power = 2
        )
        print(output)