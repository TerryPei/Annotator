import argparse
import logging
import torch
import json
import os
import sys
from tqdm import tqdm
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import KFold

class RewardTrainer:
    def __init__(self, model_name, ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters())

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def loss_fn(reward1, reward2, c):
        return torch.log(1 + torch.exp(-c * (reward1 - reward2))).sum()

    def prepare_dataset(self, folder1, folder2):
        data_pairs = []
        for file_name in tqdm(os.listdir(folder1), desc="Reading data"):  # Use tqdm here
            with open(os.path.join(folder1, file_name)) as file1, open(os.path.join(folder2, file_name)) as file2:
                summary1 = json.load(file1)["natural_language_summary"]
                summary2 = json.load(file2)["natural_language_summary"]
                data_pairs.append((summary1, summary2))
        return data_pairs

    def tokenize_pairs(self, text_pairs):
        inputs = self.tokenizer(text_pairs, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        return inputs

    def train(self, folder1, folder2, num_epochs, batch_size, save_path, n_splits=5):
        self.model.train()
        text_pairs = self.prepare_dataset(folder1, folder2)

        # K-Fold Cross Validation
        kf = KFold(n_splits=n_splits)
        
        for fold, (train_index, val_index) in enumerate(kf.split(text_pairs)):

            train_text_pairs = [text_pairs[i] for i in train_index]
            val_text_pairs = [text_pairs[i] for i in val_index]
            

            train_dataloader = DataLoader(train_text_pairs, batch_size, shuffle=True)
            val_dataloader = DataLoader(val_text_pairs, batch_size, shuffle=False)

            for epoch in range(num_epochs):
                # Training
                for batch in train_dataloader:

                    inputs1 = self.tokenize_pairs([pair[0] for pair in batch])
                    inputs2 = self.tokenize_pairs([pair[1] for pair in batch])

                    # Move inputs to the same device as the model
                    inputs1 = {name: tensor.to(self.device) for name, tensor in inputs1.items()}
                    inputs2 = {name: tensor.to(self.device) for name, tensor in inputs2.items()}

                    # Forward pass
                    outputs1 = self.model(**inputs1)
                    outputs2 = self.model(**inputs2)

                    reward1, reward2 = outputs1.logits, outputs2.logits

                    # Use a comparison value of 1 as an example
                    c = torch.tensor(1.0).to(self.device)

                    # Calculate loss
                    loss = self.loss_fn(reward1, reward2, c)
                    
                    # Backward pass and optimization
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Validation on each fold
                with torch.no_grad():
                    total_loss = 0
                    total_reward1 = 0
                    total_reward2 = 0
                    for batch in val_dataloader:

                        inputs1 = self.tokenize_pairs([pair[0] for pair in batch])
                        inputs2 = self.tokenize_pairs([pair[1] for pair in batch])
                        inputs1 = {name: tensor.to(self.device) for name, tensor in inputs1.items()}
                        inputs2 = {name: tensor.to(self.device) for name, tensor in inputs2.items()}

                        outputs1 = self.model(**inputs1)
                        outputs2 = self.model(**inputs2)
                        reward1, reward2 = outputs1.logits, outputs2.logits

                        loss = self.loss_fn(reward1, reward2, c)

                        total_loss += loss.item()
                        total_reward1 += reward1.mean().item()
                        total_reward2 += reward2.mean().item()
                    
                    average_reward1 = total_reward1 / len(val_dataloader)
                    average_reward2 = total_reward2 / len(val_dataloader)

                    print(f"Fold {fold+1}/{n_splits}, Epoch {epoch+1}/{num_epochs}, Average Reward1: {average_reward1}, Average Reward2: {average_reward2}")
                    logging.info(f"Fold {fold+1}/{n_splits}, Epoch {epoch+1}/{num_epochs}, Average Reward1: {average_reward1}, Average Reward2: {average_reward2}")
                    
            # Save model after each fold
            self.save_model(f"{save_path}_fold_{fold+1}")

    # def train(self, folder1, folder2, num_epochs, batch_size, save_path):
    #     self.model.train()
    #     text_pairs = self.prepare_dataset(folder1, folder2)
    #     dataloader = DataLoader(text_pairs, batch_size, shuffle=True)
    #     for epoch in range(num_epochs):
    #         for batch in dataloader:
    #             inputs1 = self.tokenize_pairs([pair[0] for pair in batch])
    #             inputs2 = self.tokenize_pairs([pair[1] for pair in batch])

    #             # Move inputs to the same device as the model
    #             inputs1 = {name: tensor.to(self.device) for name, tensor in inputs1.items()}
    #             inputs2 = {name: tensor.to(self.device) for name, tensor in inputs2.items()}
    #             # print(inputs1.keys(), inputs2.keys())
    #             # print(inputs1['input_ids'].shape, inputs2['input_ids'].shape)
    #             # exit(0)
    #             # Forward pass
    #             outputs1 = self.model(**inputs1)
    #             outputs2 = self.model(**inputs2)

    #             reward1, reward2 = outputs1.logits, outputs2.logits
    #             # print(reward1.shape, reward2.shape)
    #             # exit(0)
    #             # Use a comparison value of 1 as an example
    #             c = torch.tensor(1.0).to(self.device)

    #             # Calculate loss
    #             loss = self.loss_fn(reward1, reward2, c)
                
    #             # Backward pass and optimization
    #             loss.backward()
    #             self.optimizer.step()
    #             self.optimizer.zero_grad()

    #         logging.info(f"Epoch {epoch+1}/{num_epochs} Loss: {loss.item()}")

    #     self.save_model(save_path)

    def evaluate(self, folder1, folder2, batch_size):
        self.model.eval()
        text_pairs = self.prepare_dataset(folder1, folder2)
        dataloader = DataLoader(text_pairs, batch_size, shuffle=False)

        total_loss = 0
        total_reward = 0
        n_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs1 = self.tokenize_pairs([pair[0] for pair in batch])
                inputs2 = self.tokenize_pairs([pair[1] for pair in batch])

                inputs1 = {name: tensor.to(self.device) for name, tensor in inputs1.items()}
                inputs2 = {name: tensor.to(self.device) for name, tensor in inputs2.items()}

                outputs1 = self.model(**inputs1)
                outputs2 = self.model(**inputs2)
                reward1, reward2 = outputs1.logits, outputs2.logits

                c = torch.tensor(1.0).to(self.device)
                loss = self.loss_fn(reward1, reward2, c)

                total_loss += loss.item()
                total_reward += (reward1.sum() + reward2.sum()).item()
                n_samples += reward1.numel() + reward2.numel()

        avg_loss = total_loss / len(dataloader)
        avg_reward = total_reward / n_samples

        logging.info(f"Test Loss: {avg_loss}, Average Reward: {avg_reward}")
        return avg_loss, avg_reward

    def save_model(self, path):
        self.model.save_pretrained(path)
        logging.info(f'Model saved to {path}')

    def load_model(self, path):
        self.model = AutoModel.from_pretrained(path)
        logging.info(f'Model loaded from {path}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="AdamG012/chat-opt-350m-reward-deepspeed")
    parser.add_argument('--folder1', type=str, default='darts-json')
    parser.add_argument('--folder2', type=str, default='darts-json')
    parser.add_argument('--save_path', type=str, default='rlfh/models')
    # parser.add_argument('--load_path', type=str, default='rlfh/models')

    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--n_split', type=int, default=5)

    parser.add_argument("--log_file", default="logs/reward_train.log", help="Log file path.")
    
    args = parser.parse_args()
    logging.basicConfig(filename=args.log_file, format='%(asctime)s - %(message)s', level=logging.INFO)

    trainer = RewardTrainer(args.model_name)

    trainer.train(args.folder1, args.folder2, args.num_epochs, args.batch_size, args.save_path, args.n_split)
