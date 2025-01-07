from openai import OpenAI
import streamlit as st
import gc
import os
import re
import nltk
import torch
import pickle
import joblib

import numpy as np
import pandas as pd
import sklearn
import json
from pathlib import Path
from streamlit_lottie import st_lottie

from tqdm.auto import tqdm
from collections import Counter
from nltk.data import load as nltk_load
from nltk.tokenize import PunktSentenceTokenizer
from typing import Callable, List, Tuple
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM
import torch
from transformers import PreTrainedTokenizerBase
from torch.nn.functional import cross_entropy
from datetime import datetime

session_logs = []

def gptneo_features(text, tokenizer: PreTrainedTokenizerBase, model, sent_cut):
    # Tokenize
    CROSS_ENTROPY = torch.nn.CrossEntropyLoss(reduction='none')
    NLTK          = PunktSentenceTokenizer()
    DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(
        text,
        max_length=1024,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).to(DEVICE)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_target = input_ids[..., 1:].contiguous()

    loss = cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_target.view(-1),
        reduction="none"
    ).view(shift_target.size())

    # Sentence-level perplexity calculations
    sentences = sent_cut(text)
    offsets = []
    token_ids = []
    for sent in sentences:
        tokens = tokenizer(sent, truncation=True, return_tensors="pt").input_ids.squeeze(0)
        offsets.append((len(token_ids), len(token_ids) + len(tokens)))
        token_ids.extend(tokens.tolist())

    sent_ppl = []
    for start, end in offsets:
        nll = loss[0, start:end].sum() / (end - start)
        sent_ppl.append(nll.exp().item())

    max_sent_ppl = max(sent_ppl)
    sent_ppl_avg = sum(sent_ppl) / len(sent_ppl)
    sent_ppl_std = torch.std(torch.tensor(sent_ppl)).item() if len(sent_ppl) > 1 else 0

    # Step-level perplexity calculations
    mask = torch.ones_like(loss[0]).to(DEVICE)
    step_ppl = loss[0].cumsum(dim=0) / mask.cumsum(dim=0)
    step_ppl = step_ppl.exp()
    max_step_ppl = step_ppl.max().item()
    step_ppl_avg = step_ppl.mean().item()
    step_ppl_std = step_ppl.std().item() if step_ppl.size(0) > 1 else 0

    # Overall perplexity
    text_ppl = loss.mean().exp().item()

    # Rank computation (GLTR metrics)
    all_probs = torch.softmax(shift_logits, dim=-1)
    rank_counts = [0, 0, 0, 0]
    for i in range(shift_target.size(1)):  # Iterate along sequence length
        token_probs = all_probs[0, i]
        target_token = shift_target[0, i]
        token_rank = (token_probs > token_probs[target_token]).sum().item()

        if token_rank < 10:
            rank_counts[0] += 1
        elif token_rank < 100:
            rank_counts[1] += 1
        elif token_rank < 1000:
            rank_counts[2] += 1
        else:
            rank_counts[3] += 1

    # Compile perplexity metrics
    ppls = [
        text_ppl, max_sent_ppl, sent_ppl_avg, sent_ppl_std,
        max_step_ppl, step_ppl_avg, step_ppl_std
    ]

    return rank_counts, ppls

def gpt2_features(text, tokenizer, model, sent_cut):
    # Tokenize
    CROSS_ENTROPY = torch.nn.CrossEntropyLoss(reduction='none')
    NLTK          = PunktSentenceTokenizer()
    DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_max_length = tokenizer.model_max_length - 2
    token_ids, offsets = list(), list()
    sentences = sent_cut(text)
    
    for s in sentences:
        tokens = tokenizer.tokenize(s)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        difference = len(token_ids) + len(ids) - input_max_length
        if difference > 0:
            ids = ids[:-difference]
        offsets.append((len(token_ids), len(token_ids) + len(ids)))
        token_ids.extend(ids)
        if difference >= 0:
            break

    input_ids = torch.tensor([tokenizer.bos_token_id] + token_ids).to(DEVICE)
    logits = model(input_ids).logits
    
    # Shift so that n-1 predict n
    shift_logits = logits[:-1].contiguous()
    shift_target = input_ids[1:].contiguous()
    loss = CROSS_ENTROPY(shift_logits, shift_target)

    all_probs = torch.softmax(shift_logits, dim=-1)
    sorted_ids = torch.argsort(all_probs, dim=-1, descending=True)  # stable=True
    expanded_tokens = shift_target.unsqueeze(-1).expand_as(sorted_ids)
    indices = torch.where(sorted_ids == expanded_tokens)
    rank = indices[-1]
    counter = [
        rank < 10,
        (rank >= 10) & (rank < 100),
        (rank >= 100) & (rank < 1000),
        rank >= 1000
    ]
    counter = [c.long().sum(-1).item() for c in counter]


    # compute different-level ppl
    text_ppl = loss.mean().exp().item()
    sent_ppl = list()
    for start, end in offsets:
        nll = loss[start: end].sum() / (end - start)
        sent_ppl.append(nll.exp().item())
        
    max_sent_ppl = max(sent_ppl)
    sent_ppl_avg = sum(sent_ppl) / len(sent_ppl)
    if len(sent_ppl) > 1:
        sent_ppl_std = torch.std(torch.tensor(sent_ppl)).item()
    else:
        sent_ppl_std = 0

    mask = torch.tensor([1] * loss.size(0)).to(DEVICE)
    step_ppl = loss.cumsum(dim=-1).div(mask.cumsum(dim=-1)).exp()
    max_step_ppl = step_ppl.max(dim=-1)[0].item()
    step_ppl_avg = step_ppl.sum(dim=-1).div(loss.size(0)).item()
    if step_ppl.size(0) > 1:
        step_ppl_std = step_ppl.std().item()
    else:
        step_ppl_std = 0
    ppls = [
        text_ppl, max_sent_ppl, sent_ppl_avg, sent_ppl_std,
        max_step_ppl, step_ppl_avg, step_ppl_std
    ]
    return counter, ppls  # type: ignore

def get_result(prompt):
        
    train_feats_gpt2_large = []
    train_feats_gpt_neo = []
    
    for model in tqdm(['gpt-neo-125m']):
        NLTK        = PunktSentenceTokenizer()
        DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sent_cut_en = NLTK.tokenize
    
        cols = [
        'text_ppl', 'max_sent_ppl', 'sent_ppl_avg', 'sent_ppl_std', 'max_step_ppl', 
        'step_ppl_avg', 'step_ppl_std', 'rank_0', 'rank_10', 'rank_100', 'rank_1000'
        ]
    
        TOKENIZER_EN = AutoTokenizer.from_pretrained("gpt-neo-125m/tokenizer")
        MODEL_EN = AutoModelForCausalLM.from_pretrained("gpt-neo-125m/model").to(DEVICE)
    
        train_ppl_feats  = []
        train_gltr_feats = []
        with torch.no_grad():
            gltr, ppl = gptneo_features(prompt, TOKENIZER_EN, MODEL_EN, sent_cut_en)
            train_ppl_feats.append(ppl)
            train_gltr_feats.append(gltr)
    
        X_train = pd.DataFrame(
            np.concatenate((train_ppl_feats, train_gltr_feats), axis=1), 
            columns=[f'gpt-neo-125m-{col}' for col in cols]
        )
        train_feats_gpt_neo.append(X_train)

    for model in tqdm(['gpt2-large']):
        TOKENIZER_EN = AutoTokenizer.from_pretrained(f'{model}/tokenizer')
        MODEL_EN = AutoModelForCausalLM.from_pretrained(f'{model}/model').to(DEVICE)
        
        train_ppl_feats = []
        train_gltr_feats = []
        with torch.no_grad():
            gltr, ppl = gpt2_features(prompt, TOKENIZER_EN, MODEL_EN, sent_cut_en)
            train_ppl_feats.append(ppl)
            train_gltr_feats.append(gltr)
        
        # Create features DataFrame
        X_train = pd.DataFrame(
            np.concatenate((train_ppl_feats, train_gltr_feats), axis=1), 
            columns=[f'{model}-{col}' for col in cols]
        )
        
        # Append to the list
        train_feats_gpt2_large.append(X_train)

    # Concatenate all collected features
    if train_feats_gpt2_large:  # Ensure the list is not empty
        train_feats_gpt2_large = pd.concat(train_feats_gpt2_large , axis=1)
        train_feats_gpt_neo = pd.concat(train_feats_gpt_neo, axis = 1)

        train_feats = train_feats_gpt_neo.merge(train_feats_gpt2_large, left_index=True, right_index=True, how='left')
    else:
        raise ValueError("No features were extracted, check the models and input.")

    # Load the saved model and predict
    with open('FinalModel.pkl', 'rb') as fp:
        model = pickle.load(fp)

    result = model.predict_proba(train_feats)

    # Assuming binary classification and you want the probability of the positive class
    percentage = int(result[0][1] * 100)  # Extract the positive class probability and convert to percentage
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    session_logs.append({
        "text": prompt,
        "prediction": f"{percentage}%",
        "timestamp": timestamp
    })

    return f"The percentage of your text to be generated by AI is {percentage}%"

