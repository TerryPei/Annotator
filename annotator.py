import openai
import json, os, re, sys
from evaluation.eval import SequenceEvaluator
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import random
import argparse
import logging
import time
from cryptography.fernet import Fernet

with open("prepare/key_data.txt", "r") as file:
    key, encrypted_api_key = file.read().strip().split("\n")

cipher_suite = Fernet(key.encode())
decrypted_api_key = cipher_suite.decrypt(encrypted_api_key.encode())

# Set the decrypted API key in the openai library
openai.api_key = decrypted_api_key.decode()

# encoding_instruction = """
# You are an expert in the field of neural architecture search. Your task is to summarise the neural architectures based on the computation operators list. The summary will be used to recover the original operators list and your objective is to provide the summary that could maximize the ability to recover the operators. 
# Graph edges is an edge list representation of a directed graph. 
# The graph represents a neural network, where the nodes are operations, and the directed edges represents information flows with `'->'`.  
# The candidate operators of the nodes are ```
#     `'INPUT'`: input of network. 
#     `'OUTPUT'` : output of network.
#     `'CONCAT'`: concatenate 
#     `'bn'`: batch normalization.
#     `'avgpool'`: average pooling on layer.
#     `'maxpool'`: max pooling.
#     `'sepconv'`: separable convolution.
#     `'dilconv'`: dilated convolution.
#     `'skipconnect'`: skip connection.
#     ```
# `'18maxpool312'` means max pooling operation with 3x3 filter 1 stride applied in layer 19.
 
#  The summary of the neural network architecture in natural language should be in 100 words, including: 
#     1. Each block composed with what operators, and identify the block name with resnet, bottleneck, or other blocks.
#     2. Identify the depth number of each block in 1. and width parallel blocks.
#     3. Pros/cons of the design based on the block structure.
# Your return format is json dict {Summary:  }
# """


# decoding_instruction = """
#     You are tasked with recovering the cells for a neural network architecture given a summary of the network's key features. 
#     The architecture is represented as an edge list where nodes are operations and edges represent information flow with '->'. 
#     The candidate operators for the nodes are INPUT, OUTPUT, CONCAT, bn, avgpool, maxpool, sepconv, dilconv, skipconnect, and linear. 
#     Separable convolution operator with 3x3 filter means `'sepconv33'`
#     Layer 2, max pooling operator with 3x3 filter 1 stride means `'2maxpool331'`
#     Your objective is to extract the key information from the provided summary to recover the cells that maximize the distance between the recovered cells and the original cells list, 
#     with the output format being a pure dict containing only the cells key and its corresponding value. 
#     For example, the output for the first architecture could be: {'cells': 'INPUT->conv33, conv33->bn, conv33->avgpool331, avgpool331->OUTPUT, INPUT->OUTPUT'}.
# """

class Annotator:

    def __init__(self, 
                 model_name="gpt-3.5-turbo", 
                 support_set="dataset/darts-json-100", 
                 valid_set="dataset/darts-json-valid",
                 generate_set="darts-json-sample", 
                 encoding_max_tokens=350,
                 decoding_max_tokens=500,
                 max_iteration=10,
                 instruction_path=""):
        
        self.model_name = model_name
        self.support_set = support_set
        self.valid_set = valid_set
        self.generate_set = generate_set
        self.encoding_max_tokens = encoding_max_tokens
        self.decoding_max_tokens = decoding_max_tokens
        self.best_support_sim = 0.
        self.val_sim = 0.
        self.current_sim_score = 0.
        self.evaluator = SequenceEvaluator()
        # Define initial few-shot examples
        self.one_shot_templates = [{
            "cells": "INPUT->conv33,conv33->bn,conv33->avgpool331,avgpool331->OUTPUT,INPUT->OUTPUT",
            "summary": "The neural network architecture consists of 1 block with 2 depth. The block is composed of convolutional operators with 3*3 filters, and pooling operators. And there is also a skip connection operator directly from input to output, which allows for effective feature extraction and transfer learning, and reduce the computational resources."
        }]
        self.max_iteration = max_iteration
        self.instruction_path = instruction_path
        self.encoding_instruction, self.decoding_instruction = self.get_instruction()

    def encoding(self, cells):
        # # Clean the input by removing certain characters
        # output = [[s.replace(".", "").replace("-", "").replace("_", "").replace("layer", "").replace("x", "").replace("stem", "").replace("head", "") for s in edge] for edge in cells]
        # # Convert the cleaned input back into a string
        # cleaned_cells = ",".join("->".join(edge) for edge in output)
        # Prepare the input messages for the API call
        messages = [{"role": "system", "content": self.encoding_instruction}]
        # Loop through the few-shot templates and add them to the input messages
        for i, one_shot_template in enumerate(self.one_shot_templates):
            one_shot_cells, one_shot_summary = one_shot_template["cells"], one_shot_template["summary"]
            messages.append({"role": "user", "content": f"Sample {i}:\nCells: {one_shot_cells}"})
            messages.append({"role": "assistant", "content": f"Sample {i}:\nSummary: {one_shot_summary}"})
        # Add the cleaned input to the input messages
        messages.append({"role": "user", "content": cells})
        # Call the OpenAI API to generate a summary
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.encoding_max_tokens,
            n=1,
            stop=None,
            temperature=0.8,
        )
        # Extract the generated summary from the API response
        answer = response['choices'][0]['message']['content']
        # Add the generated summary to the input messages for context in the next API call
        messages.append({"role": "assistant", "content": answer})
        # Return the generated summary
        return answer
    
    # @staticmethod
    def decoding(self, summary):
        # Prepare the input messages for the API call
        messages = [{"role": "system", "content": self.decoding_instruction}]

        for i, one_shot_template in enumerate(self.one_shot_templates):
            one_shot_cells, one_shot_summary = one_shot_template["cells"], one_shot_template["summary"]
            messages.append({"role": "user", "content": f"Sample {i}:\nSummary: {one_shot_summary}"})
            messages.append({"role": "assistant", "content": f"Sample {i}:\nCells: {one_shot_cells}"})
        # Add the summary as the user input
        messages.append({"role": "user", "content": summary})
        # Call the OpenAI API to generate recovered cells
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.decoding_max_tokens,
            n=1,
            stop=None,
            temperature=0.8,
        )
        # Extract the recovered cells from the API response
        recovered_cells = response['choices'][0]['message']['content']
        # Return the recovered cells
        return recovered_cells

    def one_shot(self):
        support_set_files = os.listdir(self.support_set)
        json_files = [file for file in support_set_files if file.endswith(".json")]
        random_files = random.sample(json_files, self.max_iteration)

        for count, sample in enumerate(random_files):

            self.iterative_update(sample)
        
        return self.one_shot_templates

    def iterative_update(self, file_name):
        infile_path = os.path.join(self.support_set, file_name)
        with open(infile_path, "r") as infile:
            data = json.load(infile)
            cells = data["cells"]
        
        # print("Original cells: ", cells)
        cells = [[re.sub(r"->|:|Cells|cells|\s+|\n|adaptive|-|layer|stem|head|x|_|0|1|2|3|4|\.", "", s) for s in edge] for edge in cells]
        cells = ",".join("->".join(edge) for edge in cells)
        # print("Original cells: ", cells)
        
        summary = self.encoding(cells)
        data["natural_language_summary"] = summary

        recovered_cells = self.decoding(summary)
        recovered_cells = re.sub(r":|Cells|cells|\s+|\n|_|0|1|2|3|4|\.|\[|\]", "", recovered_cells)
        # recovered_cells = ",".join("->".join(edge) for edge in recovered_cells)
        data["recovered_cells"] = recovered_cells

        # sim_score = self.evaluator.sts_similarity(cells, recovered_cells)
        sim_score = self.evaluator.rouge_score(cells, recovered_cells)['rougeL'] # for ablation study

        if sim_score > self.best_support_sim:
            
            self.best_support_sim = sim_score
            
            val_sim = self.val_sim_score()

            if  val_sim > self.val_sim:

                self.val_sim = val_sim
                
                #### Replace current template for the next round annotation ####
                self.one_shot_templates.pop()
                self.one_shot_templates.append({"cells": cells, "summary": summary})
                #### Replace current template for the next round annotation ####
                
                log_message = f"Iteratively update few shot template with by recovered sim score: {sim_score}"
                log_message = f"Iteratively update few shot template with by recovered sim score: {val_sim}"
                logging.info(log_message)
                log_message = f"Iteratively update few shot template with cells: {cells}"
                logging.info(log_message)
                log_message = f"Iteratively update few shot template with summary: {summary}"
                logging.info(log_message)

        outfile_path = os.path.join(self.generate_set, file_name)

        with open(outfile_path, "w") as outfile:
            json.dump(data, outfile, indent=4)

    def val_sim_score(self):
        valid_set_files = os.listdir(self.valid_set)
        json_files = [file for file in valid_set_files if file.endswith(".json")]
        val_sim = 0.
        for count, sample in enumerate(json_files):

            infile_path = os.path.join(self.valid_set, sample)
            with open(infile_path, "r") as infile:
                data = json.load(infile)
                cells = data["cells"]
            
            # print("Original cells: ", cells)
            cells = [[re.sub(r"->|:|Cells|cells|\s+|\n|adaptive|-|layer|stem|head|x|_|0|1|2|3|4|\.", "", s) for s in edge] for edge in cells]
            cells = ",".join("->".join(edge) for edge in cells)
            # print("Original cells: ", cells)
            
            summary = self.encoding(cells)
            data["natural_language_summary"] = summary

            recovered_cells = self.decoding(summary)
            recovered_cells = re.sub(r":|Cells|cells|\s+|\n|_|0|1|2|3|4|\.|\[|\]", "", recovered_cells)
            # recovered_cells = ",".join("->".join(edge) for edge in recovered_cells)
            data["recovered_cells"] = recovered_cells

            # sim_score = self.evaluator.sts_similarity(cells, recovered_cells)
            sim_score = self.evaluator.rouge_score(cells, recovered_cells)['rougeL'] # for ablation study
            val_sim += sim_score
        
        return val_sim / len(json_files)
    
    def get_instruction(self):
        dataset_name = self.instruction_path.split("/")[-1].split("-")[0]
        
        with open(self.prompt_folder_path, 'r', encoding='utf-8') as file:
            prompts = json.load(file)
        prompt = prompts[dataset_name.lower()]
        self.encoding_instruction = prompt["encoding_instruction"]
        self.decoding_instruction = prompt["decoding_instruciton"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments for the Annotator.")
    parser.add_argument("--model_name", default="gpt-3.5-turbo", help="Model name to use.")
    parser.add_argument("--support_set", default="dataset/darts-json-100", help="Folder path containing support set.")
    parser.add_argument("--valid_set", default="dataset/darts-json-valid", help="Folder path containing valid set.")
    parser.add_argument("--generate_set", default="dataset/darts-json-100-summary", help="Folder path of output generated set.")
    parser.add_argument("--max_iteration", type=int, default=10, help="Number of samples to process.")
    parser.add_argument("--log_file", default="logs/one_shot_log.log", help="Log file path.")
    parser.add_argument("--prompt_folder_path", default="prompts/prompts.json", help="Folder path to store prompts.")

    args = parser.parse_args()

    log_file_path = args.log_file
    log_file_exists = os.path.isfile(log_file_path)

    if log_file_exists:
        log_file_directory, log_file_name = os.path.split(log_file_path)
        log_file_base_name, log_file_extension = os.path.splitext(log_file_name)
        log_file_id = 0

        while log_file_id <= 10000:
            new_log_file_name = f"{log_file_base_name}_{log_file_id:02d}{log_file_extension}"
            new_log_file_path = os.path.join(log_file_directory, new_log_file_name)
            if not os.path.isfile(new_log_file_path):
                log_file_path = new_log_file_path
                break
            log_file_id += 1

    logging.basicConfig(filename=log_file_path, level=logging.INFO)

    annotator = Annotator(model_name=args.model_name, 
                           support_set=args.support_set, 
                           generate_set=args.generate_set, 
                           max_iteration=args.max_iteration)
    
    best_template = annotator.one_shot()

    if best_template is not None:
        log_message = f"After iteratively updating few shot template, now the best sim score: {annotator.best_support_sim}"
        logging.info(log_message)