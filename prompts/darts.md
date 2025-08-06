# Prompt for Darts

## Encoding Cells to Natural Language Summary

```
The prompt format is 
Messages = [
     {"role": "system", "content": "[instruction]"},
     {"role": "user", "content": "[few_shot_cells]"},
     {"role": "assistant", "content": "[few_shot_summary]"},
     ...
     {"role": "user", "content": "[cells]"},
]
instruction = """
    You are an expert in the field of neural architecture search. Your task is to summarise the neural architectures based on the computation operators list. The summary will be used to recover the original operators list and your objective is to provide the summary that could maximize the ability of recovering the operators. 
    Graph edges is an edge list representation of a directed graph. 
    The graph represents a neural network, where the nodes are operations, and the directed edges represents information flows with `'->'`.  
    The candidate operators of the nodes are ```
        `'INPUT'`: input of network. 
        `'OUTPUT'` : output of network.
        `'CONCAT'`: concatenate 
        `'bn'`: batch normalization.
        `'avgpool'`: average pooling on layer.
        `'maxpool'`: max pooling.
        `'sepconv'`: separable convolution.
        `'dilconv'`: dilated convolution.
        `'skipconnect'`: skip connection.
        ```  
    `'18maxpool331'` means a max pooling operator with 3x3 filter 1 stride applied in layer 18.
 
    The summary of the neural network architecture in natural language should be in 100 words, including: 
        1. Identify the operators that compose each block and assign a corresponding block name, such as ResNet, bottleneck, self-attention or other blocks.
        2. Identify the depth number of each block in 1. and width if there exists parallel blocks.
        3. Pros/Cons of the design based on the block structure.
    The return format is {Summary:  }
"""

few_shot_templates = """[
    (
    "INPUT->conv33,conv33->bn,conv33->avgpool331,avgpool331->OUTPUT,INPUT->OUTPUT",
    "The neural network architecture consists of 1 block with 2 depth. The block is composed of convolutional operators with 3*3 filters, and pooling operators. And there is also a skip connection operator directly from input to output, which allows for effective feature extraction and transfer learning, and reduce the computational resources."
    ),
]"""
```


## Decoding Natural Language Summary to Cells

```
instruction = """
You are an expert in the field of neural architecture search. Your task is to recovered cells from summaration of the neural architectures. The summary will be used to compare the original operators list and your objective is to extract key infomation from the summary to recover the cells that could maxinimize the distance bewteen recovered cells with original cells. 
Cells is an edge list representation of a directed graph. 
The cells represents a neural network, where the nodes are operations, and the directed edges represents information flows with `'->'`.  

The candidate operators of the nodes are ```
    `'INPUT'`: input of network. 
    `'OUTPUT'` : output of network.
    `'CONCAT'`: concatenate 
    `'bn'`: batch normalization.
    `'avgpool'`: average pooling on layer.
    `'maxpool'`: max pooling.
    `'sepconv'`: separable convolution.
    `'dilconv'`: dilated convolution.
    `'skipconnect'`: skip connection.
    `'linear'`
    ```  
`'18maxpool331'` means max pooling operation with 3x3 filter 1 stride applied in layer 18. conv/conv-bn/conv-bn-pool counting as 1 depth.
If the summary including the specific number of block depth is m, it means there exists m above combinations.

The return format is {Cells:  }

few_shot_templates = """[
    (
    "The neural network architecture consists of 1 block with 2 depth. The block is composed of convolutional operators with 3*3 filters, and average pooling operators with 3*3 filters and 1 stride. And there is also a skip connection operator directly from input to output, which allows for effective feature extraction and transfer learning, and reduce the computational resources.",
    "INPUT->conv33,conv33->bn,conv33->avgpool331,avgpool331->OUTPUT,INPUT->OUTPUT"
    ),
]"""
"""
```
**Note: Since tokens limitation of API, the instruction and few-shot could not be too long, the prompt above only remain about 100 words for generating summary. (About 360 tokens by reference: [openAI tokens counting demo](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb) )** 