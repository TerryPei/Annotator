import openai
import json, os, re
# openai.api_key = ""
from evaluation.eval import SequenceEvaluator
import random
import argparse
import logging
import time

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
# The return format is {Summary:  }
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

class FewShotGPT:

    def __init__(self, model_name="gpt-3.5-turbo", 
                 folder_path="darts-json-1", 
                 sample_folder_path="darts-json-sample", 
                 sample_num=10):
        
        self.model_name = model_name
        self.folder_path = folder_path
        self.sample_folder_path = sample_folder_path
        self.best_sim_score = 0.
        self.current_sim_score = 0.
        self.evaluator = SequenceEvaluator()
        # Define initial few-shot examples
        self.few_shot_templates = [{
            # "cells": "CN1CCC2=CC(=C(C3=C2[C@@H]1CC4=CC=C(C=C4)OC5=C(C=CC(=C5)C[C@@H]6C7=CC(=C(C=C7CCN6)OC)O3)O)O)OC",
            # "summary": "Within this complex ring system, there are various functional groups present. The compound contains a nitrogen atom (N) within the fused ring system and a nitrogen-containing group (CCN6) attached to one of the rings. There are also several oxygen atoms (O) present in the structure."
            "cells": "INPUT->conv,conv->bn,bn->sepconv,bn->skipconnect,bn->skipconnect,sepconv->ADD,skipconnect->ADD,ADD->mapool,ADD->CONCAT,mapool->ADD,skipconnect->ADD,ADD->CONCAT,bn->sepconv,bn->dilconv,CONCAT->mapool,CONCAT->dilconv,sepconv->ADD,mapool->ADD,ADD->CONCAT,dilconv->ADD,dilconv->ADD,ADD->CONCAT,CONCAT->sepconv,CONCAT->dilconv,CONCAT->mapool,CONCAT->dilconv,sepconv->ADD,mapool->ADD,ADD->CONCAT,dilconv->ADD,dilconv->ADD,ADD->CONCAT,CONCAT->sepconv,CONCAT->skipconnect,CONCAT->skipconnect,sepconv->ADD,skipconnect->ADD,ADD->mapool,ADD->CONCAT,mapool->ADD,skipconnect->ADD,ADD->CONCAT,CONCAT->avgpool,avgpool->linear",
            "summary": "The neural network architecture consists of 3 blocks with 4 depth each. The blocks are composed of various operators such as convolutional operators, batch normalization, pooling operators, skip connection, separable convolution, and dilated convolution. The first block uses a combination of convolutional and separable convolutional operators. The second block uses skip connections to combine the feature maps from different levels, which helps to improve the gradient flow and reduce the vanishing gradient problem. The third block uses concatenation to combine the feature maps from different levels and then applies pooling to reduce the dimensionality of the feature maps. The final block uses average pooling to obtain the final output. The use of skip connections and concatenation helps to improve the accuracy of the model by allowing the model to capture both low-level and high-level features. However, the use of these operators can also increase the complexity of the model and require more computational resources."
        }]
        self.sample_num = sample_num

    def generate_summary(self, encoding_instruction):
        all_files = os.listdir(self.folder_path)
        json_files = [file for file in all_files if file.endswith(".json")]

        for sample in json_files:
            # sample = f"graph-0000000{i:05d}-0000000{i:05d}.json"
            infile_path = os.path.join(self.folder_path, sample)
            outfile_path = os.path.join(self.sample_folder_path, sample)

            ##### penai.error.RateLimitError: That model is currently overloaded with other requests. #######

            if os.path.exists(outfile_path): # May interupt caused by restriction of API.
                continue
            
            print(sample)
            with open(infile_path, "r") as infile:
                data = json.load(infile)
                orginal_data = data["Isomeric SMILES"]        
            # print("Original cells: ", cells)

            summary = self.encoding(orginal_data, encoding_instruction)
            data["natural_language_summary"] = summary
            # print("Summary: ", summary)

            with open(outfile_path, "w") as outfile:
                json.dump(data, outfile)

    def encoding(self, orginal_data, encoding_instruction):
        messages = [{"role": "system", "content": encoding_instruction}]
        # Loop through the few-shot templates and add them to the input messages
        # for i, few_shot_template in enumerate(self.few_shot_templates):
        #     few_shot_cells, few_shot_summary = few_shot_template["cells"], few_shot_template["summary"]
        #     messages.append({"role": "user", "content": f"Sample {i}:\nCells: {few_shot_cells}"})
        #     messages.append({"role": "assistant", "content": f"Sample {i}:\nSummary: {few_shot_summary}"})
        # Add the cleaned input to the input messages
        # messages.append({"role": "user", "content": cells})
        # Add the batch of cleaned input cells to the input messages
        # for cells in cell_batch:
        messages.append({"role": "user", "content": orginal_data})
        # Call the OpenAI API to generate a summary
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            max_tokens=350,
            n=1,
            stop=None,
            temperature=0.5,
        )
        # Extract the generated summary from the API response
        answer = response['choices'][0]['message']['content']
        return answer             
    
    # # @staticmethod
    def encoding_batch(self, cell_batch, encoding_instruction):
        messages = [{"role": "system", "content": encoding_instruction}]
        # Loop through the few-shot templates and add them to the input messages
        for i, few_shot_template in enumerate(self.few_shot_templates):
            few_shot_cells, few_shot_summary = few_shot_template["cells"], few_shot_template["summary"]
            messages.append({"role": "user", "content": f"Sample {i}:\nCells: {few_shot_cells}"})
            messages.append({"role": "assistant", "content": f"Sample {i}:\nSummary: {few_shot_summary}"})
        # Add the cleaned input to the input messages
        # messages.append({"role": "user", "content": cells})
        # Add the batch of cleaned input cells to the input messages
        promptsArray = cell_batch
        stringifiedPromptsArray = json.dumps(promptsArray)

        batchInstruction = {
            "role": "system",
            "content": "Complete every element of the array. Reply with an array of all completions."
        }

        # for cells in cell_batch:
        #     messages.append({"role": "user", "content": cells})
        messages.append({"role": "user", "content": stringifiedPromptsArray})
        messages.append(batchInstruction)
        # Call the OpenAI API to generate a summary
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            max_tokens=350,
            # n=len(cell_batch),
            stop=None,
            temperature=0,
        )

        # Extract the answers from the response
        answers = []
        for choice in response['choices']:
            # Extract the assistant's message from the choice
            assistant_message = choice['message']
            print(assistant_message['content'])
            exit(0)
            # answer = json.loads(assistant_message['content'])
            # answers.append(answer)

        return answers
    
    def generate_summary_batch(self, batch_size=10):
        all_files = os.listdir(self.folder_path)
        json_files = [file for file in all_files if file.endswith(".json")]

        for i in range(0, len(json_files), batch_size):
            batch_files = json_files[i:i + batch_size]
            cell_batch = []
            processed_files = []

            for sample in batch_files:
                infile_path = os.path.join(self.folder_path, sample)
                outfile_path = os.path.join(self.sample_folder_path, sample)

                if os.path.exists(outfile_path):
                    continue

                with open(infile_path, "r") as infile:
                    data = json.load(infile)
                    cells = data["cells"]

                cells = [[re.sub(r"->|:|Cells|cells|\s+|\n|adaptive|-|layer|stem|head|x|_|0|1|2|3|4|\.", "", s) for s in edge] for edge in cells]
                cells = ",".join("->".join(edge) for edge in cells)
                cell_batch.append(cells)
                processed_files.append(sample)

            if cell_batch:  # Check if cell_batch is not empty
                print(len(cell_batch), processed_files)
                
                start_time = time.time()  # Record the start time
                summaries = self.encoding_batch(cell_batch)
                end_time = time.time()  # Record the end time

                elapsed_time = end_time - start_time  # Calculate the elapsed time
                print(f"Time cost for encoding_batch: {elapsed_time:.2f} seconds")  # Print the elapsed time


                for summary, sample in zip(summaries, processed_files): 
                    infile_path = os.path.join(self.folder_path, sample)
                    outfile_path = os.path.join(self.sample_folder_path, sample)

                    with open(infile_path, "r") as infile:
                        data = json.load(infile)

                    data["natural_language_summary"] = summary

                    with open(outfile_path, "w") as outfile:
                        json.dump(data, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments for the FewShotGPT.")
    parser.add_argument("--model_name", default="gpt-3.5-turbo", help="Model name to use.")
    # parser.add_argument("--model_name", default="davinci", help="Model name to use.")
    parser.add_argument("--folder_path", default="pubmed", help="Folder path containing input files.")
    parser.add_argument("--sample_folder_path", default="pubmed-3.5", help="Folder path to store output files.")
    parser.add_argument("--prompt_folder_path", default="prompts/prompts.json", help="Folder path to store prompts.")

    parser.add_argument("--sample_num", type=int, default=5, help="Number of samples to process.")
    parser.add_argument("--log_file", default="logs/pubmed.log", help="Log file path.")
    
    args = parser.parse_args()

    dataset_name = args.folder_path.split("-")[0]

    print(dataset_name)

    with open(args.prompt_folder_path, 'r', encoding='utf-8') as file:
        prompts = json.load(file)
    prompt = prompts[dataset_name.lower()]

    encoding_instruction = prompt["encoding_instruction"]
    decoding_instruction = prompt["decoding_instruction"]

    logging.basicConfig(filename=args.log_file, level=logging.INFO)

    processor = FewShotGPT(model_name=args.model_name, 
                           folder_path=args.folder_path, 
                           sample_folder_path=args.sample_folder_path, 
                           sample_num=args.sample_num)

    processor.generate_summary(encoding_instruction)
