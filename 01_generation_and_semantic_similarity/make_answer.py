
import torch
import pandas as pd
import os
import random
import pickle
import json
import time
from tqdm import tqdm
import re
import numpy as np
import argparse
import glob
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

import unicodedata


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", DEVICE)

# Set the following in your environment before running:
#   export OPENAI_API_KEY=...     (only needed for OpenAI API models)
#   export HF_TOKEN=...           (gated HuggingFace models)
#   export LANGCHAIN_API_KEY=...  (optional)
os.environ['DEVICE'] = DEVICE


### Answer generation

class LLM_model:
    def __init__(self, model_id: str, pipe, tokenizer, temperature: float = 0.7, repetition_penalty: float = 1.2):
        self.model_name = model_id
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        
        # Dispatch on the model id. Only OpenAI API models start with "gpt";
        # every other id (Llama, HuatuoGPT-o1, ...) is a local HuggingFace model.
        if model_id.lower().startswith("gpt"):
            self.llm_type = "gpt"
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=temperature,
            )
        else:
            self.pipe = pipe
            self.llm_type = "local"
            self.llm = HuggingFacePipeline(pipeline=pipe)
