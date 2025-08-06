import openai
from cryptography.fernet import Fernet

def main():
    # Read the encryption key and encrypted API key from the file
    with open("key_data.txt", "r") as file:
        key, encrypted_api_key = file.read().strip().split("\n")
    
    cipher_suite = Fernet(key.encode())
    decrypted_api_key = cipher_suite.decrypt(encrypted_api_key.encode())

    openai.api_key = decrypted_api_key.decode()

    models = openai.Model.list()

if __name__ == "__main__":
    main()

