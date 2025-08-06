import re
import os
import sys
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import json
from typing import List
from transformers import GPT2Tokenizer, GPT2Model
from scipy.spatial.distance import cosine
import torch
from torch import nn
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import SmoothingFunction
import argparse
import logging

class SequenceEvaluator:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2Model.from_pretrained("gpt2")
        self.tfidf_vectorizer = TfidfVectorizer()
        # self.loss_function = nn.CrossEntropyLoss(reduction="mean")
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        self.sts_model = SentenceTransformer('paraphrase-distilroberta-base-v2')

    @staticmethod
    def preprocess_sequence(sequence: str) -> List[str]:
        sequence = sequence.replace("->", " ").replace(",", " ")
        tokens = sequence.split()
        return tokens

    def get_embeddings(self, tokens: List[str]) -> List:
        input_ids = self.tokenizer.encode(tokens, add_special_tokens=False, is_split_into_words=True)
        with torch.no_grad():
            outputs = self.model(torch.tensor([input_ids]))
            embeddings = outputs[0][0].numpy().tolist()
        return embeddings

    def similarity_score(self, cells: str, recovered_cells: str) -> float:
        cells_tokens = self.preprocess_sequence(cells)
        recovered_cells_tokens = self.preprocess_sequence(recovered_cells)

        cells_embeddings = self.get_embeddings(cells_tokens)
        recovered_cells_embeddings = self.get_embeddings(recovered_cells_tokens)

        total_similarity = 0
        count = 0

        for embedding1 in cells_embeddings:
            for embedding2 in recovered_cells_embeddings:
                total_similarity += 1 - cosine(embedding1, embedding2)
                count += 1

        average_similarity = total_similarity / count
        return average_similarity

    
    def bleu_k_score(self, cells: str, recovered_cells: str, k: int = 1) -> float:
        cells_tokens = SequenceEvaluator.preprocess_sequence(cells)
        recovered_cells_tokens = SequenceEvaluator.preprocess_sequence(recovered_cells)

        weights = [1 / k] * k  # Equal weights for all n-grams up to k

        smoothie = SmoothingFunction().method1  # You can try other smoothing methods if needed (e.g., method2, method3, etc.)
        bleu_score = sentence_bleu([cells_tokens], recovered_cells_tokens, weights=weights, smoothing_function=smoothie)
        return bleu_score

    def tfidf_similarity(self, cells: str, recovered_cells: str) -> float:
        sequences = [cells, recovered_cells]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(sequences)
        similarity = 1 - cosine(tfidf_matrix[0].toarray().flatten(), tfidf_matrix[1].toarray().flatten())
        return similarity
    
    def rouge_score(self, cells: str, recovered_cells: str) -> dict:
        scores = self.rouge.score(cells, recovered_cells)
        return {key: value.fmeasure for key, value in scores.items()}

    # def meteor_score(self, cells: str, recovered_cells: str) -> float:
    #     cells_tokens = self.preprocess_sequence(cells)
    #     recovered_cells_tokens = self.preprocess_sequence(recovered_cells)

    #     # Convert the tokens back to strings separated by spaces
    #     cells_str = ' '.join(cells_tokens)
    #     recovered_cells_str = ' '.join(recovered_cells_tokens)

    #     return meteor_score([cells_str], recovered_cells_str)

    def sts_similarity(self, cells: str, recovered_cells: str) -> float:
        embeddings = self.sts_model.encode([cells, recovered_cells])
        similarity = 1 - cosine(embeddings[0], embeddings[1])
        return similarity


def filter_edges(edges, substring):
    pattern = re.compile(substring)
    filtered_edges = [edge for edge in edges if not pattern.search(edge)]
    return filtered_edges

def evaluate_samples(folder_path: str) -> None:
    evaluator = SequenceEvaluator()
    total_similarity = 0
    total_bleu_1 = 0
    total_bleu_4 = 0
    total_rouge1 = 0
    total_rougeL = 0
    total_tfidf_similarity = 0
    total_sts_similarity = 0
    total_samples = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename)) as file:
                sample = json.load(file)
                cells = sample["cells"]
                # print(cells)
                # exit(0)
                # cells = filter_edges(cells, 'ADD')
                cells = [[re.sub(r"->|:|Cells|cells|\s+|\n|adaptive|-|layer|stem|head|x|_|0|1|2|3|4|5|\.", "", s) for s in edge] for edge in cells]
                cells = [[re.sub(r"CONCAT", "concat", s) for s in edge] for edge in cells]
                # cells = [[re.sub(r"CONCAT", "concat", s) for s in edge] for edge in cells]
                cells = [edge for edge in cells if not re.search(r'ADD', str(edge))]
                # print(cells)
                # exit(0)
                cells = ",".join("->".join(edge) for edge in cells) 
                # cells = [[s.replace(".", "").replace("-", "").replace("_", "").replace("layer", "").replace("x", "").replace("stem", "").replace("head", "").replace("1", "") for s in edge] for edge in cells]
                recovered_cells = sample["recovered_cells"]
                recovered_cells = re.sub(r":|Cells|cells|\s+|\n|_|0|1|2|3|4|5|\.|branch", "", recovered_cells)
                recovered_cells = re.sub(r",INPUT->OUTPUT", "", recovered_cells)

                # similarity = evaluator.similarity_score(cells, recovered_cells)
                bleu_1 = evaluator.bleu_k_score(cells, recovered_cells, k=1)
                bleu_4 = evaluator.bleu_k_score(cells, recovered_cells, k=4)
                tfidf_similarity = evaluator.tfidf_similarity(cells, recovered_cells)
                rouge_scores = evaluator.rouge_score(cells, recovered_cells)
                # meteor = evaluator.meteor_score(cells, recovered_cells)
                sts_similarity = evaluator.sts_similarity(cells, recovered_cells)


                # total_similarity += similarity
                total_bleu_1 += bleu_1
                total_bleu_4 += bleu_4
                total_tfidf_similarity += tfidf_similarity
                total_rouge1 += rouge_scores['rouge1']
                total_rougeL += rouge_scores['rougeL']
                # total_meteor += meteor
                total_sts_similarity += sts_similarity

                total_samples += 1

    # average_similarity = total_similarity / total_samples
    average_bleu_1 = total_bleu_1 / total_samples
    average_bleu_4 = total_bleu_4 / total_samples
    average_tfidf_similarity = total_tfidf_similarity / total_samples
    average_rouge1 = total_rouge1 / total_samples
    average_rougeL = total_rougeL / total_samples
    # average_meteor = total_meteor / total_samples
    average_sts_similarity = total_sts_similarity / total_samples

    # print(f"Average similarity score: {average_similarity}")
    logging.info(f"Average BLEU-1 score: {average_bleu_1}")
    logging.info(f"Average BLEU-4 score: {average_bleu_4}")
    logging.info(f"Average TF-IDF similarity: {average_tfidf_similarity}")
    logging.info(f"Average ROUGE-1 score: {average_rouge1}")
    logging.info(f"Average ROUGE-L score: {average_rougeL}")
    # print(f"Average METEOR score: {average_meteor}")
    logging.info(f"Average STS similarity: {average_sts_similarity}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process arguments for the FewShotGPT.")
    parser.add_argument("--folder_path", type=str, default="darts-15552-3.5", help="Path to the folder containing the samples.")
    parser.add_argument("--log_file", default="logs/eval_by_recoverd_cells.log", help="Log file path.")
    
    args = parser.parse_args()


    logging.basicConfig(filename=args.log_file, level=logging.INFO)
    logging.info("evaluating by recovered cells on dataset: darts-15552-3.5")
    
    evaluate_samples(args.folder_path)
    


    # cells = "INPUT0->conv33,conv33->bn,bn->0avgpool331,bn->0avgpool332,bn->0mapool331"
    # recovered_cells = "INPUT0->conv33,conv33->->0avgpool331,bn->"

    # evaluator = SequenceEvaluator()

    # similarity = evaluator.similarity_score(cells, recovered_cells)
    # print(f"Similarity score: {similarity}")

    # bleu_1 = evaluator.bleu_k_score(cells, recovered_cells, k=1)
    # print(f"BLEU-1 score: {bleu_1}")

    # bleu_2 = evaluator.bleu_k_score(cells, recovered_cells, k=2)
    # print(f"BLEU-2 score: {bleu_2}")

    # bleu_3 = evaluator.bleu_k_score(cells, recovered_cells, k=3)
    # print(f"BLEU-3 score: {bleu_3}")

    # bleu_4 = evaluator.bleu_k_score(cells, recovered_cells, k=4)
    # print(f"BLEU-4 score: {bleu_4}")

    # tfidf_similarity = evaluator.tfidf_similarity(cells, recovered_cells)
    # print(f"TF-IDF similarity: {tfidf_similarity}")

    # rouge_scores = evaluator.rouge_score(cells, recovered_cells)
    # print(f"ROUGE-1 score: {rouge_scores['rouge1']}")
    # print(f"ROUGE-L score: {rouge_scores['rougeL']}")

    # # meteor = evaluator.meteor_score(cells, recovered_cells)
    # # print(f"METEOR score: {meteor}")

    # sts_similarity = evaluator.sts_similarity(cells, recovered_cells)
    # print(f"STS similarity: {sts_similarity}")
