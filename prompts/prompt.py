import json
import os
import sys
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def save_string_to_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def load_string_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def main():

    encoding_instruction = """
You are an expert in the field of neural architecture search. Your task is to summarise the neural architectures based on the computation operators list. The summary will be used to recover the original operators list and your objective is to provide the summary that could maximize the ability to recover the operators. 
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
`'18maxpool312'` means max pooling operation with 3x3 filter 1 stride applied in layer 19.
 
 The summary of the neural network architecture in natural language should be in 100 words, including: 
    1. Each block composed with what operators, and identify the block name with resnet, bottleneck, or other blocks.
    2. Identify the depth number of each block in 1. and width parallel blocks.
    3. Pros/cons of the design based on the block structure.
Your return format is json dict {Summary:  }
"""

    decoding_instruction = """
    You are tasked with recovering the cells for a neural network architecture given a summary of the network's key features. 
    The architecture is represented as an edge list where nodes are operations and edges represent information flow with '->'. 
    The candidate operators for the nodes are INPUT, OUTPUT, CONCAT, bn, avgpool, maxpool, sepconv, dilconv, skipconnect, and linear. 
    Separable convolution operator with 3x3 filter means `'sepconv33'`
    Layer 2, max pooling operator with 3x3 filter 1 stride means `'2maxpool331'`
    Your objective is to extract the key information from the provided summary to recover the cells that maximize the distance between the recovered cells and the original cells list, 
    with the output format being a pure dict containing only the cells key and its corresponding value. 
    For example, the output for the first architecture could be: {'cells': 'INPUT->conv33, conv33->bn, conv33->avgpool331, avgpool331->OUTPUT, INPUT->OUTPUT'}.
    """

    darts_encoding_instruction = """
    You are an expert in the field of neural architecture search. Your task is to summarise the neural architectures based on the computation operators list. The summary will be used to recover the original operators list and your objective is to provide the summary that could maximize the ability to recover the operators. 
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
    `'18maxpool312'` means max pooling operation with 3x3 filter 1 stride applied in layer 19.
     
     The summary of the neural network architecture in natural language should be in 100 words, including: 
        1. Each block composed with what operators, and identify the block name with resnet, bottleneck, or other blocks.
        2. Identify the depth number of each block in 1. and width parallel blocks.
        3. Pros/cons of the design based on the block structure.
    Your return format is json dict {'Summary':  }.
    """

    darts_decoding_instruction = """
    You are tasked with recovering the cells for a neural network architecture given a summary of the network's key features. 
    The architecture is represented as an edge list where nodes are operations and edges represent information flow with '->'. 
    The candidate operators for the nodes are INPUT, OUTPUT, CONCAT, bn, avgpool, maxpool, sepconv, dilconv, skipconnect, and linear. 
    Separable convolution operator with 3x3 filter means `'sepconv33'`
    Layer 2, max pooling operator with 3x3 filter 1 stride means `'2maxpool331'`
    Your objective is to extract the key information from the provided summary to recover the cells that maximize the distance between the recovered cells and the original cells list, 
    with the output format being a pure dict containing only the cells key and its corresponding value. 
    For example, the output for the first architecture could be: {'Cells': 'INPUT->conv33, conv33->bn, conv33->avgpool331, avgpool331->OUTPUT, INPUT->OUTPUT'}.
    Your return format is json dict {'Cells':  }.
    """

    pubmed_encoding_instruction = """
        You are a professional annotator of Isomeric SMILES based on the organic compound description. 
        In SMILES, atoms are represented by their atomic symbols. The second letter of two-character atomic symbols must be entered in lower case. Each non-hydrogen atom is specified independently by its atomic symbol enclosed in square brackets, [ ] (for example, [Au] or [Fe]). Square brackets may be omitted for elements in the “organic subset” (B, C, N, O, P, S, F, Cl, Br, and I) if the proper number of “implicit” hydrogen atoms is assumed. “Explicitly” attached hydrogens and formal charges are always specified inside brackets. A formal charge is represented by one of the symbols + or -. Single, double, triple, and aromatic bonds are represented by the symbols, -, =, #, and :, respectively. Single and aromatic bonds may be, and usually are, omitted. Here are some examples of SMILES strings.

        C Methane (CH4)
        CC Ethane (CH3CH3)
        C=C Ethene (CH2CH2)
        C#C Ethyne (CHCH)
        COC Dimethyl ether (CH3OCH3)
        CCO Ethanol (CH3CH2OH)
        CC=O Acetaldehyde (CH3-CH=O)
        C#N Hydrogen Cyanide (HCN)
        [C-]#N Cyanide anion
        Branches are specified by enclosures in parentheses and can be nested or stacked, as shown in these examples.

        CC(C)CO Isobutyl alcohol (CH3-CH(CH3)-CH2-OH)
        CC(CCC(=O)N)CN 5-amino-4-methylpentanamide
        Rings are represented by breaking one single or aromatic bond in each ring, and designating this ring-closure point with a digit immediately following the atoms connected through the broken bond. Atoms in aromatic rings are specified by lower cases letters. Therefore, cyclohexane and benzene can be represented by the following SMILES.

        C1CCCCC1 Cyclohexane (C6H12)
        c1ccccc1 Benzene (C6H6)
        Although the carbon-carbon bonds in these two SMILES are omitted, it is possible to deduce that the omitted bonds are single bonds (for cyclohexane) and aromatic bonds (for benzene). One can also represent an aromatic compound as a non-aromatic, KeKulé structure. For example, the following is a valid SMILES string for benzene.

        C1=CC=CC=C1 Benzene (C6H6)

        I will provide a Isomeric SMILES you return a summary. The return format is json dict with {Summary: }
    """

    pubmed_decoding_instruction = """
        You are a professional annotator of Isomeric SMILES based on the organic compound description. 
        In SMILES, atoms are represented by their atomic symbols. The second letter of two-character atomic symbols must be entered in lower case. Each non-hydrogen atom is specified independently by its atomic symbol enclosed in square brackets, [ ] (for example, [Au] or [Fe]). Square brackets may be omitted for elements in the “organic subset” (B, C, N, O, P, S, F, Cl, Br, and I) if the proper number of “implicit” hydrogen atoms is assumed. “Explicitly” attached hydrogens and formal charges are always specified inside brackets. A formal charge is represented by one of the symbols + or -. Single, double, triple, and aromatic bonds are represented by the symbols, -, =, #, and :, respectively. Single and aromatic bonds may be, and usually are, omitted. Here are some examples of SMILES strings.

        C Methane (CH4)
        CC Ethane (CH3CH3)
        C=C Ethene (CH2CH2)
        C#C Ethyne (CHCH)
        COC Dimethyl ether (CH3OCH3)
        CCO Ethanol (CH3CH2OH)
        CC=O Acetaldehyde (CH3-CH=O)
        C#N Hydrogen Cyanide (HCN)
        [C-]#N Cyanide anion
        Branches are specified by enclosures in parentheses and can be nested or stacked, as shown in these examples.

        CC(C)CO Isobutyl alcohol (CH3-CH(CH3)-CH2-OH)
        CC(CCC(=O)N)CN 5-amino-4-methylpentanamide
        Rings are represented by breaking one single or aromatic bond in each ring, and designating this ring-closure point with a digit immediately following the atoms connected through the broken bond. Atoms in aromatic rings are specified by lower cases letters. Therefore, cyclohexane and benzene can be represented by the following SMILES.

        C1CCCCC1 Cyclohexane (C6H12)
        c1ccccc1 Benzene (C6H6)
        Although the carbon-carbon bonds in these two SMILES are omitted, it is possible to deduce that the omitted bonds are single bonds (for cyclohexane) and aromatic bonds (for benzene). One can also represent an aromatic compound as a non-aromatic, KeKulé structure. For example, the following is a valid SMILES string for benzene.

        C1=CC=CC=C1 Benzene (C6H6)

        I will provide a description, you return a Isomeric SMILES. The return format is json dict with {Isomeric SMILES: }
    """
    sentimental_encoding_instruction = """Analyze the following product review and determine if the sentiment is: positive or negative. 
                                        Return answer is the description of query sentence and a single word as either positive or negative: {description} {text}"""
    
    sentimental_decoding_instruction = """Recover the sentence without the label of positive or negtive, given by an answer of sentimental analysis."""
    


    prompts = {
            "darts": {"encoding_instruction": darts_encoding_instruction, 
                      "decoding_instruction": darts_decoding_instruction},
            "pubmed": {"encoding_instruction": pubmed_encoding_instruction, 
                       "decoding_instruction": pubmed_decoding_instruction},
            "sentimental": {"encoding_instruction": sentimental_encoding_instruction, 
                            "decoding_instruction": sentimental_decoding_instruction}
    }
    # Save to JSON file
    save_string_to_json("prompts/prompts.json", prompts)
    # Load from JSON file
    prompts = load_string_from_json("prompts/prompts.json")
    print(prompts['pubmed']['encoding_instruction'])

if __name__ == "__main__":
    main()
