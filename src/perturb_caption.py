import datasets
import torch
from datasets import load_dataset, load_from_disk
import numpy as np
from PIL import Image
from io import BytesIO
import time
from nltk.tokenize import  sent_tokenize

from tqdm import tqdm
import json
import pickle

import argparse

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-hf','--hf_file', help='HuggingFace data of image preference alignment, such as from our Pick-double', default='./pickdouble_customed_s6_v2.hf')
parser.add_argument('-out','--out_folder', help='Output folder', default='./perturbed_caption/')
args = parser.parse_args()

data_file = args.hf_file
out_folder = args.out_folder
data = load_from_disk(data_file)

"""
data must be a list() of image preference ground truth meta data. 
Each meta data shall be consisted of at leastthe the following fields:
  caption_0 and caption_1 - the generated captions for image_0 and image_1
  label_0 and label_1 - the gt preference scores where label_0 != label_1
"""

from transformers import T5Tokenizer, T5ForConditionalGeneration

class DipperParaphraser(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        if verbose:
            print(f"{model} model loaded in {time.time() - time1}")
        self.model.cuda()
        self.model.eval()

    def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text
        
dp = DipperParaphraser()
prompt = "Paraphrase and perturb the following image caption."
lex_div_list = [40, 60, 80] # weak, medium, strong perturbation values
output_texts = dict()

for i in tqdm(range(len(data))):
    current_data = data['train'][i]
    if current_data['label_1'] < current_data['label_0']:
        caption_preferred_ind = 0
        caption_preferred = current_data['caption_0']
        caption_lesspreferred = current_data['caption_1']
    else:
        caption_preferred_ind = 1
        caption_preferred = current_data['caption_1']
        caption_lesspreferred = current_data['caption_0']
        
    current_out = dict()
    current_out['caption_preferred_ind'] = caption_preferred_ind
    current_out['caption_preferred'] = caption_preferred
    current_out['caption_lesspreferred'] = caption_lesspreferred
    
    for lex_div in lex_div_list:
        # caption_preferred_perturbed_40 as weak perturbation, and so on
        current_out[f'caption_preferred_perturbed_{lex_div}'] = dp.paraphrase(caption_preferred, lex_diversity=lex_div, order_diversity=0, prefix=prompt, do_sample=True, top_p=0.75, top_k=None, max_length=256)

    for lex_div in lex_div_list:
        current_out[f'caption_lesspreferred_perturbed_{lex_div}'] = dp.paraphrase(caption_lesspreferred, lex_diversity=lex_div, order_diversity=0, prefix=prompt, do_sample=True, top_p=0.75, top_k=None, max_length=256)
        
    output_texts[i] = current_out

with open(f'{out_folder}/{data_file.replace('.', '-')}_dipperxxl_full.pickle', 'wb') as handle:
    pickle.dump(output_texts, handle, protocol=pickle.HIGHEST_PROTOCOL)