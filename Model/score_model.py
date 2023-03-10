from __future__ import print_function
import argparse
import json
from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange
from transformers import GPT2Tokenizer
from transformers.file_utils import cached_path
from transformers.modeling_gpt2 import GPT2LMHeadModel

from ipywidgets import interact, interactive, fixed, interact_manual
# from flask_socketio import SocketIO, join_room, emit, send
import ipywidgets as widgets

resultContainer = {
    "text":[]
}

class ClassificationHead(torch.nn.Module):
    """Classification Head for  transformer encoders"""

    def __init__(self, class_size, embed_size):
        super(ClassificationHead, self).__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        # self.mlp1 = torch.nn.Linear(embed_size, embed_size)
        # self.mlp2 = (torch.nn.Linear(embed_size, class_size))
        self.mlp = torch.nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        # hidden_state = F.relu(self.mlp1(hidden_state))
        # hidden_state = self.mlp2(hidden_state)
        logits = self.mlp(hidden_state)
        return logits

PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
BOW_AFFECT = 4
SMALL_CONST = 1e-15
BIG_CONST = 1e10

QUIET = 0
REGULAR = 1
VERBOSE = 2
VERY_VERBOSE = 3
VERBOSITY_LEVELS = {
    'quiet': QUIET,
    'regular': REGULAR,
    'verbose': VERBOSE,
    'very_verbose': VERY_VERBOSE,
}

BAG_OF_WORDS_ARCHIVE_MAP = {
    'legal': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/legal.txt",
    'military': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/military.txt",
    'monsters': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/monsters.txt",
    'politics': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/politics.txt",
    'positive_words': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/positive_words.txt",
    'religion': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/religion.txt",
    'science': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/science.txt",
    'space': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/space.txt",
    'technology': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/technology.txt",
}

DISCRIMINATOR_MODELS_PARAMS = {
    "clickbait": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/clickbait_classifier_head.pt",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"non_clickbait": 0, "clickbait": 1},
        "default_class": 1,
        "pretrained_model": "gpt2-medium",
    },
    "sentiment": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/SST_classifier_head.pt",
        "class_size": 5,
        "embed_size": 1024,
        "class_vocab": {"very_positive": 2, "very_negative": 3},
        "default_class": 3,
        "pretrained_model": "gpt2-medium",
    },
}

def to_var(x, requires_grad=False, volatile=False, device='cuda'):
    if torch.cuda.is_available() and device == 'cuda':
        x = x.cuda()
    elif device != 'cuda':
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins,
                           torch.ones_like(logits) * -BIG_CONST,
                           logits)

def gaussian(x, mu, sig):
  x = np.array(x)
  return list(np.exp(-0.5*((x-mu)/sig)**2)/(sig*(2*np.pi)**0.5))

def perturb_past(
        past,
        model,
        last,
        affect_weight=0.2,
        unpert_past=None,
        unpert_logits=None,
        accumulated_hidden=None,
        grad_norms=None,
        stepsize=0.01,
        one_hot_bows_vectors=None,
        one_hot_bows_affect=None,
        affect_int = None,
        knob = None,
        classifier=None,
        class_label=None,
        loss_type=0,
        num_iterations=3,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        score_scale=0.01,
        device='cuda',
        verbosity_level=REGULAR,
        beta1=0.6,
        end_lr = 0.5,
        N = 15,
        power = 2
):
    # Generate inital perturbed past
#     unpart = past + tuple()
#     grad_accumulator = [
#         (np.zeros(p.shape).astype("float32"))
#         for p in past
#     ]

    if accumulated_hidden is None:
        accumulated_hidden = 0
    if decay:
        decay_mask = torch.arange(
            0.,
            1.0 + SMALL_CONST,
            1.0 / (window_length)
        )[1:]
    else:
        decay_mask = 1.0
    # Generate a mask is gradient perturbated is based on a past window
    _, _, _, curr_length, _ = past[0].shape

    if curr_length > window_length and window_length > 0:
        ones_key_val_shape = (
                tuple(past[0].shape[:-2])
                + tuple([window_length])
                + tuple(past[0].shape[-1:])
        )

        zeros_key_val_shape = (
                tuple(past[0].shape[:-2])
                + tuple([curr_length - window_length])
                + tuple(past[0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat(
            (ones_mask, torch.zeros(zeros_key_val_shape)),
            dim=-2
        ).to(device)
    else:
        window_mask = torch.ones_like(past[0]).to(device)

    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    for i in range(1,num_iterations+1):
        if verbosity_level >= VERBOSE:
            print("Iteration ", i + 1)
#         unpart = past + tuple()
        past = tuple([
            to_var(p_, requires_grad=True, device=device)
            for p_ in past
        ])
        # Compute hidden using perturbed past 
        perturbed_past = past
        _, _, _, curr_length, _ = perturbed_past[0].shape
        all_logits, _, all_hidden = model(last, past_key_values=perturbed_past)
        hidden = all_hidden[-1]
        new_accumulated_hidden = accumulated_hidden + torch.sum(
            hidden,
            dim=1
        ).detach()
        
        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        loss = 0.0
        loss_list = []
        if loss_type == PPLM_BOW or loss_type == BOW_AFFECT:
            for one_hot_bow in one_hot_bows_vectors:
                bow_logits = torch.mm(probs, torch.t(one_hot_bow))
                #print(type(bow_logits))
                bow_loss = -torch.log(torch.sum(bow_logits))
                #print(bow_loss)
                loss +=  bow_loss
                loss_list.append(bow_loss)
            if loss_type == BOW_AFFECT:
              for one_hot_bow in one_hot_bows_affect:
                  bow_logits = torch.mm(probs, torch.t(one_hot_bow))
                 # print(bow_logits.size(), torch.FloatTensor(affect_int).size())
                  bow_loss = -torch.log(torch.matmul(bow_logits, torch.t(torch.FloatTensor(gaussian(affect_int, knob, .1)).to(device))))#-torch.log(torch.sum(bow_logits))#
                  # print(bow_loss)

                  loss += affect_weight * bow_loss[0]
                  loss_list.append(bow_loss)
            if verbosity_level >= VERY_VERBOSE:
                print(" pplm_bow_loss:", loss.data.cpu().numpy())

        score_loss = 0.0
        # To Calculate the KL Loss every iteration we need the unpert prob every iteration but it is no calculated
        
        if score_scale > 0.0 :
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            score = torch.sum(torch.mul(unpert_probs,probs)/(torch.norm(probs)*torch.norm(unpert_probs)))
            score_loss = -score_scale *  score
            loss += score_loss
            if verbosity_level >= VERY_VERBOSE:
                print(' Score_loss', score_loss.data.cpu().numpy())
        
        
        loss_per_iter.append(loss.data.cpu().numpy())
        if verbosity_level >= VERBOSE:
            print(' Total_loss', (loss).data.cpu().numpy())
        
        # compute gradients
        loss.backward()
#         print(past[0].grad),curr_perturbation[0].grad)

#         # calculate gradient norms
        if grad_norms is not None and loss_type == PPLM_BOW:
            grad_norms = [
                torch.max(grad_norms[index], torch.norm(p_.grad * window_mask))
                for index, p_ in enumerate(perturbed_past)
            ]
        else:
            grad_norms = [
                (torch.norm(p_.grad * window_mask) + SMALL_CONST)
                for index, p_ in enumerate(perturbed_past)
            ]

        # Initialise m_0
        m_0 = tuple([ torch.tensor(np.zeros(p.shape).astype("float32")) for p in perturbed_past ])
        grad = tuple([
            (p_.grad * window_mask/ grad_norms[index] ** gamma ).data.cpu().numpy()
            for index, p_ in enumerate(perturbed_past)
        ])
#         print(perturbed_past)
        # beta1 = 0.6
#         beta2 = 0.99
#         print(len(m_0))
        # end_lr = 0.5
        # N = 15
        # alpha= end_lr
        initial_lr = stepsize - end_lr
        lr = initial_lr * ((num_iterations - i)/(num_iterations - N)) ** power # Polynomial Decay
        # lr = stepsize * (alpha**np.floor(i/N)) # Exponential Decay
        r_t = beta1/(1 - (beta1)**i)
        r_t_1 = (1 - beta1)/(1 - (beta1)**i)
        if i ==1:
            m_t = [(r_t*momentum) + (r_t_1*grad[index]) for index , momentum in enumerate(m_0)] 
        else:
            m_t = [(r_t*momentum) + (r_t_1*grad[index]) for index , momentum in enumerate(m_t)]
        
#         print(grad[0][0][0].shape)
        # perturbing the past
        past = [torch.tensor(perturbed_past[index].detach() - lr * m_t[index].to(device)) for index , p_ in enumerate(perturbed_past)]
#         print(perturbed_past[0],grad[0],type(grad[0]),type(perturbed_past[0]))

        # reset gradients, just to make sure
        for p_ in perturbed_past:
            p_.grad.data.zero_()
        
        
        
        

    pert_past = perturbed_past
    return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter


def get_classifier(
        name: Optional[str],
        class_label: Union[str, int],
        device: str,
        verbosity_level: int = REGULAR
) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(
        class_size=params['class_size'],
        embed_size=params['embed_size']
    ).to(device)
    if "url" in params:
        resolved_archive_file = cached_path(params["url"])
    elif "path" in params:
        resolved_archive_file = params["path"]
    else:
        raise ValueError("Either url or path have to be specified "
                         "in the discriminator model parameters")
    classifier.load_state_dict(
        torch.load(resolved_archive_file, map_location=device))
    classifier.eval()

    if isinstance(class_label, str):
        if class_label in params["class_vocab"]:
            label_id = params["class_vocab"][class_label]
        else:
            label_id = params["default_class"]
            if verbosity_level >= REGULAR:
                print("class_label {} not in class_vocab".format(class_label))
                print("available values are: {}".format(params["class_vocab"]))
                print("using default class {}".format(label_id))

    elif isinstance(class_label, int):
        if class_label in set(params["class_vocab"].values()):
            label_id = class_label
        else:
            label_id = params["default_class"]
            if verbosity_level >= REGULAR:
                print("class_label {} not in class_vocab".format(class_label))
                print("available values are: {}".format(params["class_vocab"]))
                print("using default class {}".format(label_id))

    else:
        label_id = params["default_class"]

    return classifier, label_id


def get_bag_of_words_indices(bag_of_words_ids_or_paths: List[str], tokenizer) -> \
        List[List[List[int]]]:
    bow_indices = []
    for id_or_path in bag_of_words_ids_or_paths:
        if id_or_path in BAG_OF_WORDS_ARCHIVE_MAP:
            filepath = cached_path(BAG_OF_WORDS_ARCHIVE_MAP[id_or_path])
        else:
            filepath = id_or_path
        with open(filepath, "r") as f:
            words = f.read().strip().split("\n")
        bow_indices.append(
            [tokenizer.encode(word.strip(),
                              add_prefix_space=True,
                              add_special_tokens=False)
             for word in words])
    return bow_indices

def get_affect_words_and_int (affect_class):
  emotions = "https://raw.githubusercontent.com/ishikasingh/Affective-text-gen/master/NRC-Emotion-Intensity-Lexicon-v1.txt"
  filepath = cached_path(emotions)
  with open(filepath, "r") as f:
      words = f.read().strip().split("\n")[1:]
  words = [w.split("\t") for w in words]
  return [w[0] for w in words if w[1] == affect_class], [float(w[-1]) for w in words if w[1] == affect_class]

def build_bows_one_hot_vectors(bow_indices, tokenizer, device='cuda'):
    if bow_indices is None:
        return None

    one_hot_bows_vectors = []
    for single_bow in bow_indices:
        single_bow = list(filter(lambda x: len(x) <= 1, single_bow))

        single_bow = torch.tensor(single_bow).to(device)
        num_words = single_bow.shape[0]
        # print(num_words)
        one_hot_bow = torch.zeros(num_words, tokenizer.vocab_size).to(device)
        one_hot_bow.scatter_(1, single_bow, 1)
        one_hot_bows_vectors.append(one_hot_bow)
    return one_hot_bows_vectors

def build_bows_one_hot_vectors_aff(bow_indices,affect_int, tokenizer, device='cuda'):
    if bow_indices is None or affect_int is None:
        return None, None

    one_hot_bows_vectors = []
    # print(np.array(bow_indices).shape)
    for single_bow in bow_indices:
        zipped = [[single_bow[i], affect_int[i]] for i in range(len(single_bow))]
        single_bow_int = list(filter(lambda x: len(x[0]) <= 1, zipped))
        single_bow = [single_bow_int[i][0] for i in range(len(single_bow_int)) ]
        affect_ints = [single_bow_int[i][1] for i in range(len(single_bow_int)) ]
        # print(single_bow, affect_ints)
        # print(len(single_bow), len(affect_ints))
        single_bow = torch.tensor(single_bow).to(device)
        num_words = single_bow.shape[0]
        # print(num_words)
        one_hot_bow = torch.zeros(num_words, tokenizer.vocab_size).to(device)
        one_hot_bow.scatter_(1, single_bow, 1)
        one_hot_bows_vectors.append(one_hot_bow)
    return one_hot_bows_vectors, affect_ints



def full_text_generation(
        model,
        tokenizer,
        affect_weight=0.2,
        knob = None,
        context=None,
        num_samples=1,
        device="cuda",
        bag_of_words=None,
        bag_of_words_affect=None,
        discrim=None,
        class_label=None,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        score_scale=0.01,
        verbosity_level=REGULAR,
        beta1=0.6,
        end_lr = 0.5,
        N = 15,
        power = 2,
        **kwargs
):
    classifier, class_id = get_classifier(discrim, class_label, device)
    # print("bog is here", bag_of_words)
    # print("affect is: ", bag_of_words_affect)
    bow_indices = []
    bow_indices_affect = []
    if bag_of_words:
      bow_indices = get_bag_of_words_indices(bag_of_words.split(";"), tokenizer)
    if bag_of_words_affect: 
      affect_words, affect_int = get_affect_words_and_int(bag_of_words_affect)
      bow_indices_affect.append([tokenizer.encode(word.strip(),add_prefix_space=True, add_special_tokens=False)for word in affect_words])
    loss_type = PPLM_BOW
    if bag_of_words_affect:
      loss_type = BOW_AFFECT

    unpert_gen_tok_text, _, _ = generate_text_pplm(
        model=model,
        tokenizer=tokenizer,
        context=context,
        device=device,
        length=length,
        sample=sample,
        perturb=False,
        verbosity_level=verbosity_level
    )

    if device == 'cuda':
        torch.cuda.empty_cache()

    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []
    print("After Perturbation")
    for i in range(num_samples):
        pert_gen_tok_text, discrim_loss, loss_in_time = generate_text_pplm(
            model=model,
            tokenizer=tokenizer,
            affect_weight=affect_weight,
            context=context,
            device=device,
            perturb=True,
            bow_indices=bow_indices,
            bow_indices_affect=bow_indices_affect,
            affect_int = affect_int,
            knob = knob,
            classifier=classifier,
            class_label=class_id,
            loss_type=loss_type,
            length=length,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            sample=sample,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            score_scale=score_scale,
            verbosity_level=verbosity_level,
            beta1=beta1,
            end_lr = end_lr,
            N = N,
            power = power
        )
        pert_gen_tok_texts.append(pert_gen_tok_text)
        if classifier is not None:
            discrim_losses.append(discrim_loss.data.cpu().numpy())
        losses_in_time.append(loss_in_time)

    if device == 'cuda':
        torch.cuda.empty_cache()

    return unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time


def get_affect_words_and_int1 (affect_class):
  emotions = "https://raw.githubusercontent.com/ishikasingh/Affective-text-gen/master/NRC-AffectIntensity-Lexicon.txt"
  filepath = cached_path(emotions)
  with open(filepath, "r") as f:
      words = f.read().strip().split("\n")[37:]
  words = [w.split("\t") for w in words]
  return [w[0] for w in words if w[-1] == affect_class], [float(w[1]) for w in words if w[-1] == affect_class]

def run_pplm_example(
        pretrained_model="gpt2-medium",
        cond_text="",
        affect_weight=0.2,
        knob = None,
        uncond=False,
        num_samples=1,
        bag_of_words=None,
        bag_of_words_affect=None,
        discrim=None,
        discrim_weights=None,
        discrim_meta=None,
        class_label=-1,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        score_scale=0.01,
        seed=0,
        no_cuda=False,
        colorama=False,
        verbosity='regular',
        beta1=0.6,
        end_lr = 0.5,
        N = 15,
        power = 2
        ):
    # set Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set verbosiry
    verbosity_level = VERBOSITY_LEVELS.get(verbosity.lower(), REGULAR)

    # # set the device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
    # device = "cpu"

    if discrim == 'generic':
        set_generic_model_params(discrim_weights, discrim_meta)

    if discrim is not None:
        discriminator_pretrained_model = DISCRIMINATOR_MODELS_PARAMS[discrim][
            "pretrained_model"
        ]
        if pretrained_model != discriminator_pretrained_model:
            pretrained_model = discriminator_pretrained_model
            if verbosity_level >= REGULAR:
                print("discrim = {}, pretrained_model set "
                "to discriminator's = {}".format(discrim, pretrained_model))

    # load pretrained model
    model = GPT2LMHeadModel.from_pretrained(
        pretrained_model,
        output_hidden_states=True
    )
    model.to(device)
    model.eval()

    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    # figure out conditioning text
    if uncond:
        tokenized_cond_text = tokenizer.encode([tokenizer.bos_token],add_special_tokens=False)
    else:
        raw_text = cond_text
        while not raw_text:
            print("Did you forget to add `--cond_text`? ")
            raw_text = input("Model prompt >>> ")
        tokenized_cond_text = tokenizer.encode(tokenizer.bos_token + raw_text,add_special_tokens=False)
    print("= Prefix of sentence =")
    # generate unperturbed and perturbed texts

    # full_text_generation returns:
    # unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time
    unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation(
        model=model,
        tokenizer=tokenizer,
        affect_weight=affect_weight,
        knob = knob,
        context=tokenized_cond_text,
        device=device,
        num_samples=num_samples,
        bag_of_words=bag_of_words,
        bag_of_words_affect=bag_of_words_affect,
        discrim=discrim,
        class_label=class_label,
        length=length,
        stepsize=stepsize,
        temperature=temperature,
        top_k=top_k,
        sample=sample,
        num_iterations=num_iterations,
        grad_length=grad_length,
        horizon_length=horizon_length,
        window_length=window_length,
        decay=decay,
        gamma=gamma,
        gm_scale=gm_scale,
        score_scale=score_scale,
        verbosity_level=verbosity_level,
        beta1=beta1,
        end_lr = end_lr,
        N = N,
        power = power
    )

    # untokenize unperturbed text
    unpert_gen_text = tokenizer.decode(unpert_gen_tok_text.tolist()[0])

    if verbosity_level >= REGULAR:
        print("=" * 80)
    print("= Unperturbed generated text =")
    print(unpert_gen_text)
    print()

    generated_texts = []

    # iterate through the perturbed texts
    for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
        try:
            # untokenize unperturbed text
            pert_gen_text = tokenizer.decode(pert_gen_tok_text.tolist()[0])
            # print("= Perturbed generated text {} =".format(i + 1))
            # print(pert_gen_text)
            # print()
        except:
            pass

        # keep the prefix, perturbed seq, original seq for each index
        generated_texts.append(
            (tokenized_cond_text, pert_gen_tok_text, unpert_gen_tok_text)
        )

    return pert_gen_text

def generate_text_pplm(
        model,
        tokenizer,
        affect_weight=0.2,
        context=None,
        past=None,
        device="cuda",
        perturb=True,
        bow_indices=None,
        bow_indices_affect=None,
        affect_int = None,
        knob = None,
        classifier=None,
        class_label=None,
        loss_type=0,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        score_scale=0.01,
        verbosity_level=REGULAR,
        beta1=0.6,
        end_lr = 0.5,
        N = 15,
        power = 2
):
    output_so_far = None
    if context:
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        output_so_far = context_t

    # collect one hot vectors for bags of words
    one_hot_bows_vectors = build_bows_one_hot_vectors(bow_indices, tokenizer, device)
    affect_int_orig = affect_int
    one_hot_bows_affect, affect_int = build_bows_one_hot_vectors_aff(bow_indices_affect, affect_int, tokenizer, device)
#    print(torch.FloatTensor(one_hot_bows_affect).size())
    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []

    if verbosity_level >= VERBOSE:
        range_func = trange(length, ascii=True)
    else:
        range_func = range(length)
    count = 0
    int_score = 0
    for i in range_func:
        if count == 2:
          break
        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current_token

        # run model forward to obtain unperturbed
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            if output_so_far.shape[1] > 1:
                _, past, _ = model(output_so_far[:, :-1])

        unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
        unpert_last_hidden = unpert_all_hidden[-1]

        # check if we are abowe grad max length
        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        # modify the past if necessary
        if not perturb or num_iterations == 0:
            pert_past = past

        else:
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            if past is not None:
                pert_past, _, grad_norms, loss_this_iter = perturb_past(
                    past,
                    model,
                    last,
                    affect_weight = affect_weight,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=current_stepsize,
                    one_hot_bows_vectors=one_hot_bows_vectors,
                    one_hot_bows_affect=one_hot_bows_affect,
                    affect_int = affect_int,
                    knob = knob,
                    classifier=classifier,
                    class_label=class_label,
                    loss_type=loss_type,
                    num_iterations=num_iterations,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    decay=decay,
                    gamma=gamma,
                    score_scale=score_scale,
                    device=device,
                    verbosity_level=verbosity_level,
                    beta1=beta1,
                    end_lr = end_lr,
                    N = N,
                    power = power
                )
                loss_in_time.append(loss_this_iter)
            else:
                pert_past = past

        pert_logits, past, pert_all_hidden = model(last, past=pert_past)
        pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST
        pert_probs = F.softmax(pert_logits, dim=-1)

        if classifier is not None:
            ce_loss = torch.nn.CrossEntropyLoss()
            prediction = classifier(torch.mean(unpert_last_hidden, dim=1))
            label = torch.tensor([class_label], device=device,
                                 dtype=torch.long)
            unpert_discrim_loss = ce_loss(prediction, label)
            if verbosity_level >= VERBOSE:
                print(
                    "unperturbed discrim loss",
                    unpert_discrim_loss.data.cpu().numpy()
                )
        else:
            unpert_discrim_loss = 0

        # Fuse the modified model and original model
        if perturb:

            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

            pert_probs = ((pert_probs ** gm_scale) * (
                    unpert_probs ** (1 - gm_scale)))  # + SMALL_CONST
            pert_probs = top_k_filter(pert_probs, k=top_k,
                                      probs=True)  # + SMALL_CONST

            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)

        else:
            pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
            pert_probs = F.softmax(pert_logits, dim=-1)

        # sample or greedy
        if sample:
            last = torch.multinomial(pert_probs, num_samples=1)
            # print('pert_prob, last ', pert_probs, last)

        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)

        # update context/output_so_far appending the new token
        output_so_far = (
            last if output_so_far is None
            else torch.cat((output_so_far, last), dim=1)
        )

        resultContainer["text"].append(tokenizer.decode(output_so_far.tolist()[0])[-1])
        # toemit = tokenizer.decode(output_so_far.tolist()[0])
        # toemit = toemit.split("<|endoftext|>")[1]
        # if perturb:
            # emit('word', {"value": toemit}, broadcast=True)
        if verbosity_level >= REGULAR:
            print(tokenizer.decode(output_so_far.tolist()[0]))
        if(tokenizer.decode(output_so_far.tolist()[0])[-1] == '.' ):
          count = count+1
        if bow_indices_affect is not None and [output_so_far.tolist()[0][-1]] in bow_indices_affect[0]:
          int_word = affect_int_orig[bow_indices_affect[0].index([output_so_far.tolist()[0][-1]])]
          print(tokenizer.decode(output_so_far.tolist()[0][-1]), int_word)
          int_score = int_score + int_word
    print("int_score: ", int_score)
    # print("int.. " , output_so_far.tolist()[0][-1])
    return output_so_far, unpert_discrim_loss, loss_in_time






def set_generic_model_params(discrim_weights, discrim_meta):
    if discrim_weights is None:
        raise ValueError('When using a generic discriminator, '
                         'discrim_weights need to be specified')
    if discrim_meta is None:
        raise ValueError('When using a generic discriminator, '
                         'discrim_meta need to be specified')

    with open(discrim_meta, 'r') as discrim_meta_file:
        meta = json.load(discrim_meta_file)
    meta['path'] = discrim_weights
    DISCRIMINATOR_MODELS_PARAMS['generic'] = meta

# # set the device
# device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

