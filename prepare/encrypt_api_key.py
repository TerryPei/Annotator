from cryptography.fernet import Fernet

def main():
    # Generate a key
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)

    # Encrypt the API key
    api_key = b"your_api_id"
    encrypted_api_key = cipher_suite.encrypt(api_key)

    # Write the encryption key and encrypted API key to a file
    with open("key_data.txt", "w") as file:
        file.write(key.decode() + "\n" + encrypted_api_key.decode())

    print("API key encrypted and saved to file.")

if __name__ == "__main__":
    main()


