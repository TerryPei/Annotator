import tiktoken
import json, os

encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')

folder_path = "darts-json-50" # 1226
# folder_path = "darts-json-16384" # 1655

max_length = 0
max_cells = None

for filename in os.listdir(folder_path):
    # Check if the file is a JSON file
    if filename.endswith(".json"):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r") as f:
            data = json.load(f)

        length = len(data["cells"])

        # Update max_length if the current length is greater
        if length > max_length:
            max_cells = data["cells"]
            max_length = length

print("Max length of cells in all json:", max_length)

cells = max_cells
tokens = encoding.encode(str(cells))




