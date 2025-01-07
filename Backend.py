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

def extract_gptneo_features(text, tokenizer: PreTrainedTokenizerBase, model, sent_cut):
    # Tokenize
    CROSS_ENTROPY = torch.nn.CrossEntropyLoss(reduction='none')
    NLTK          = PunktSentenceTokenizer()
    DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_max_length = 1022
    token_ids, offsets = [], []
    sentences = sent_cut(text)
    
    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        truncation_limit = len(token_ids) + len(ids) - input_max_length
        
        if truncation_limit > 0:
            ids = ids[:-truncation_limit]
        
        offsets.append((len(token_ids), len(token_ids) + len(ids)))
        token_ids.extend(ids)
        
        if truncation_limit >= 0:
            break
    
    input_ids = torch.tensor([tokenizer.bos_token_id] + token_ids).to(DEVICE)
    # Ensure input_ids is 2D
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # Convert to (1, seq_length)

    # Ensure attention mask is correctly shaped
    attention_mask = torch.ones_like(input_ids)

    # Run model safely
    logits = model(input_ids, attention_mask=attention_mask).logits

    # Ensure logits has correct dimensions
    if logits.dim() == 2:  # If logits is (seq_length, vocab_size), add batch dim
        logits = logits.unsqueeze(0)  # Convert to (1, seq_length, vocab_size)


    # Shift logits to align with targets
    shift_logits = logits[:, :-1, :].contiguous()
    shift_target = input_ids[:, 1:].contiguous()
    loss = CROSS_ENTROPY(shift_logits.view(-1, shift_logits.size(-1)), shift_target.view(-1))
    
    all_probs = torch.softmax(shift_logits, dim=-1)
    sorted_ids = torch.argsort(all_probs, dim=-1, descending=True)
    expanded_tokens = shift_target.unsqueeze(-1).expand_as(sorted_ids)
    indices = torch.where(sorted_ids == expanded_tokens)
    rank = indices[-1]
    
    # Rank distribution counters
    rank_counters = [
        (rank < 10).long().sum().item(),
        ((rank >= 10) & (rank < 100)).long().sum().item(),
        ((rank >= 100) & (rank < 1000)).long().sum().item(),
        (rank >= 1000).long().sum().item()
    ]
    
    # Compute different levels of perplexity (ppl)
    text_ppl = loss.mean().exp().item()
    sent_ppl = [(loss[start:end].sum() / (end - start)).exp().item() for start, end in offsets]
    
    max_sent_ppl = max(sent_ppl)
    sent_ppl_avg = sum(sent_ppl) / len(sent_ppl)
    sent_ppl_std = torch.std(torch.tensor(sent_ppl)).item() if len(sent_ppl) > 1 else 0
    
    mask = torch.ones(loss.size(0), device=DEVICE)
    step_ppl = loss.cumsum(dim=-1).div(mask.cumsum(dim=-1)).exp()
    max_step_ppl = step_ppl.max().item()
    step_ppl_avg = step_ppl.mean().item()
    step_ppl_std = step_ppl.std().item() if step_ppl.size(0) > 1 else 0
    
    ppl_metrics = [
        text_ppl, max_sent_ppl, sent_ppl_avg, sent_ppl_std,
        max_step_ppl, step_ppl_avg, step_ppl_std
    ]
    
    return rank_counters, ppl_metrics

def extract_gpt2_features(text, tokenizer, model, sent_cut):
    # Tokenize
    CROSS_ENTROPY = torch.nn.CrossEntropyLoss(reduction='none')
    NLTK          = PunktSentenceTokenizer()
    DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_max_length = tokenizer.model_max_length - 2
    token_ids, offsets = [], []
    sentences = sent_cut(text)
    
    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        truncation_limit = len(token_ids) + len(ids) - input_max_length
        
        if truncation_limit > 0:
            ids = ids[:-truncation_limit]
        
        offsets.append((len(token_ids), len(token_ids) + len(ids)))
        token_ids.extend(ids)
        
        if truncation_limit >= 0:
            break
    
    input_ids = torch.tensor([tokenizer.bos_token_id] + token_ids).to(DEVICE)
    logits = model(input_ids).logits
    
    # Shift logits to align with targets
    shift_logits = logits[:-1].contiguous()
    shift_target = input_ids[1:].contiguous()
    loss = CROSS_ENTROPY(shift_logits, shift_target)
    
    all_probs = torch.softmax(shift_logits, dim=-1)
    sorted_ids = torch.argsort(all_probs, dim=-1, descending=True)
    expanded_tokens = shift_target.unsqueeze(-1).expand_as(sorted_ids)
    indices = torch.where(sorted_ids == expanded_tokens)
    rank = indices[-1]
    
    # Rank distribution counters
    rank_counters = [
        (rank < 10).long().sum().item(),
        ((rank >= 10) & (rank < 100)).long().sum().item(),
        ((rank >= 100) & (rank < 1000)).long().sum().item(),
        (rank >= 1000).long().sum().item()
    ]
    
    # Compute different levels of perplexity (ppl)
    text_ppl = loss.mean().exp().item()
    sent_ppl = [(loss[start:end].sum() / (end - start)).exp().item() for start, end in offsets]
    
    max_sent_ppl = max(sent_ppl)
    sent_ppl_avg = sum(sent_ppl) / len(sent_ppl)
    sent_ppl_std = torch.std(torch.tensor(sent_ppl)).item() if len(sent_ppl) > 1 else 0
    
    mask = torch.ones(loss.size(0), device=DEVICE)
    step_ppl = loss.cumsum(dim=-1).div(mask.cumsum(dim=-1)).exp()
    max_step_ppl = step_ppl.max().item()
    step_ppl_avg = step_ppl.mean().item()
    step_ppl_std = step_ppl.std().item() if step_ppl.size(0) > 1 else 0
    
    ppl_metrics = [
        text_ppl, max_sent_ppl, sent_ppl_avg, sent_ppl_std,
        max_step_ppl, step_ppl_avg, step_ppl_std
    ]
    
    return rank_counters, ppl_metrics


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
            gltr, ppl = extract_gptneo_features(prompt, TOKENIZER_EN, MODEL_EN, sent_cut_en)
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
            gltr, ppl = extract_gpt2_features(prompt, TOKENIZER_EN, MODEL_EN, sent_cut_en)
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

