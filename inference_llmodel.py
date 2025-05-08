"""
This script evaluates a LLM model fine-tuned using LoRA adapters on a test dataset.
It generates predictions based on the model's responses to prompts and compares them to ground truth values.
It uses RDKit for SMILES validation and canonicalization.
"""

import os
import re
import json 
import torch
import argparse
from rdkit import Chem
from peft import PeftModel
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_from_disk

N_EXPERIMENTS = 3
SAVE_PATH_ACCURACY = "Accuracy_counts_"
SAVE_PATH_PREDICTIONS = "Pred_target_pairs_"

PATTERNS_ZERO_SHOT = {
    'ec': re.compile(r'\d+\.\d+\.\d+\.\d+'),
    'substrate': re.compile(r'([A-Za-z0-9@+\-\[\]\(\)\\/=#$]+)'),
    'product': re.compile(r'([A-Za-z0-9@+\-\[\]\(\)\\/=#$]+)')
}

PATTERNS_WITH_TAGS = {
    'ec': re.compile(r'<EC>\s*(\d+\.\d+\.\d+\.\d+)\s*</EC>'),
    'substrate': re.compile(r'<SMILES>\s*(.*?)\s*</SMILES>'),
    'product': re.compile(r'<SMILES>\s*(.*?)\s*</SMILES>')
}



def get_response(prompt, text_generator):
    """Generate model response given a prompt."""
  
    sequences = text_generator(prompt)
    gen_text = sequences[0]["generated_text"][-1]
    return gen_text



def evaluate_smiles_prediction(prediction, ground_truth, results_dict, other_outputs = None):
    """Evaluate SMILES prediction vs ground truth with RDKit canonicalization."""

    try:
        # Convert ground truth and prediction to RDKit molecules
        mol_gt = Chem.MolFromSmiles(ground_truth)
        mol_pred = Chem.MolFromSmiles(prediction)

        # Check if the prediction is invalid
        if mol_pred is None:
            results_dict["invalid"] += 1
            return results_dict

        # Convert both to canonical SMILES
        canonical_gt = Chem.MolToSmiles(mol_gt, canonical=True)
        canonical_pred = Chem.MolToSmiles(mol_pred, canonical=True)

        # Case 1: Exact canonical match
        if prediction == ground_truth:
            results_dict["canonical_match"] += 1

        # Case 2: Exact solution but non-canonical
        elif canonical_pred == canonical_gt and prediction != ground_truth:
            results_dict["noncanonical_match"] += 1

        # Case 3: Check against other outputs if provided and not empty
        elif other_outputs:
            match_found = False
            for alt_truth in other_outputs:
                mol_alt = Chem.MolFromSmiles(alt_truth)
                if mol_alt is None:
                    continue  # Skip invalid alternate truths

                canonical_alt = Chem.MolToSmiles(mol_alt, canonical=True)
                
                # Exact canonical match with alternate truth
                if prediction == canonical_alt:
                    results_dict["canonical_match"] += 1
                    match_found = True
                    print('canonical match found in other truths')
                    break

                # Exact solution but non-canonical with alternate truth
                if canonical_pred == canonical_alt:
                    results_dict["noncanonical_match"] += 1
                    match_found = True
                    print('non-canonical match found in other truths')
                    break
            
            # If a match was found, skip further checks
            if match_found:
                return results_dict

        # Case 3: Wrong answer but chemically valid and canonical
        if canonical_pred != canonical_gt and prediction == canonical_pred:
            results_dict["canonical_valid"] += 1

        # Case 4: Wrong answer but chemically valid and non-canonical
        elif canonical_pred != canonical_gt:
            results_dict["noncanonical_valid"] += 1

    except Exception as e:
        # If there is any error in processing, consider it as invalid
        print(f"Error processing SMILES '{prediction}': {e}")
        results_dict["invalid"] += 1

    return results_dict



def test(test_dataset, text_generator, task, check_other_outputs=False, zero_shot = False, generalize = False):
    """Evaluate the model on the test dataset and return accuracy counts."""

    pattern = PATTERNS_ZERO_SHOT.get(task) if zero_shot else PATTERNS_WITH_TAGS.get(task)
    pattern_other_truths = PATTERNS_WITH_TAGS.get(task)

    results_dict = (
        {class_label: 0 for class_label in range(1, 8)}
        if task == 'ec'
        else {"canonical_match": 0, "noncanonical_match": 0, "canonical_valid": 0, "noncanonical_valid": 0, "invalid": 0}
    )

    all_pred_target_pairs = []
    print('\n############### Entering for loop for experiment ###############\n\n')
    for data in test_dataset:
        
        prompt = data['message'][:-1]
        target = data['message'][-1]['content']
        other_truths = data['other_truths'] if check_other_outputs else []
        pred_raw = get_response(prompt, text_generator)
        prompt = prompt[0]['content']
        pred = prompt+' '+pred_raw['content']
        
        
        matches = re.findall(pattern, pred[(len(prompt)+1):])
        target = re.findall(pattern_other_truths, target)[0]
        if not matches or not target:
            print(f"Warning: Missing match or target in sample.")
            all_pred_target_pairs.append(pred[len(prompt) + 1:])
            continue

        if task == 'ec':            
            if matches[0][0] == target[0]:
                results_dict[int(target[0])] += 1
        elif task == 'substrate' or task == 'product':
            
            extracted_other_truths = [
                re.findall(pattern_other_truths, truth)[0]  # Extract the first match
                for truth in other_truths if re.findall(pattern_other_truths, truth)
            ]
            results_dict = evaluate_smiles_prediction(matches[0], target, results_dict, 
                                                      other_outputs = extracted_other_truths if check_other_outputs else None)    
            print('results_dict', results_dict)
        all_pred_target_pairs.append(pred[len(prompt) + 1:])
    
    print(results_dict)
    
    return results_dict, all_pred_target_pairs



def main():

    if "HF_HOME" in os.environ:
        hf_home_path = os.environ["HF_HOME"]
        print(f"HF_HOME is set to: {hf_home_path}")

    # Parsing arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="Base model for finetuning",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        choices=["meta-llama/Meta-Llama-3.1-70B-Instruct", 
                 "meta-llama/Meta-Llama-3.1-8B-Instruct"],
    )
    parser.add_argument(
        "--dataset_path",
        default='./dataset.hf',
    )
    parser.add_argument(
        "--lora_directory",
        default="./Finetuning_default",
    )
    parser.add_argument(
        "--save_path",
        default="prova",
    )
    parser.add_argument(
        "--task",
        default="",
    )
    parser.add_argument(
        "--quantized",
        help="Loading quantized model in 4-bit",
        action='store_true',
    )
    parser.add_argument(
        "--check_other_truths",
        type=str,  
        default="false",  
        choices=["true", "false"], 
        help="Set to 'true' to enable checking other (branching) truths, or 'false' to disable"
    )
    parser.add_argument(
        "--zero_shot",
        help="Run in zero-shot mode (skip LoRA adapter)",
        action='store_true',
    )
    parser.add_argument(
        "--generalize",
        help="Generalize single-task model over other tasks in inference",
        action='store_true',
    )

    args = parser.parse_args()
    model_id = args.model
    dataset_path = args.dataset_path
    lora_directory = args.lora_directory
    model_name = args.save_path
    prefix = args.task
    check_other_outputs = args.check_other_truths
    zero_shot = args.zero_shot
    generalize = args.generalize
    
    check_other_outputs = args.check_other_truths.lower() == "true"
    base_model_name = model_id

    if zero_shot:
        path1 = SAVE_PATH_ACCURACY + prefix + '_' + base_model_name[11:]
        path2 = SAVE_PATH_PREDICTIONS + prefix + '_' + base_model_name[11:]
    
    else:
        path1 = SAVE_PATH_ACCURACY + prefix + '_' + model_name
        path2 = SAVE_PATH_PREDICTIONS + prefix + '_' + model_name

    test_dataset = load_from_disk(dataset_path)['test']
    test_dataset

    config_data = json.load(open("config.json"))
    HF_TOKEN = config_data["HF_TOKEN"]
    
    # TOKENIZER
    # Load tokenizer (optional for running on test set)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name,
                                            token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    # MODEL
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        )

    base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            token=HF_TOKEN,
            low_cpu_mem_usage=True
        )
    base_model.eval()

    if not zero_shot:
        # LORA ADAPTER
        adapter_save_path =  lora_directory # Path to saved LoRA adapter
        print('\n###### lora_directory:', adapter_save_path, lora_directory)
        peft_model = PeftModel.from_pretrained(base_model, 
                                            adapter_save_path,
                                            # is_trainable=True
                                            )
        peft_model.eval()
        print(peft_model.print_trainable_parameters())
        print("\n###### Model name is:", model_id)
    else:
        print("\n### Running in Zero-Shot Mode (Base Model Only) ###\n")
        peft_model = base_model
    
    text_generator = pipeline(
        "text-generation",
        model=peft_model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.6,    # default for generate is 1.0
        top_p=0.9,          # default for generate is 1.0
    )
    
    class_accuracy_counts = []
    pred_target_pairs = []
    for i in range(N_EXPERIMENTS):

        print(f"\n\nExperiment {i+1}\n")
        dict_accuracy_counts, all_pred_target_pairs = (
            test(test_dataset, text_generator = text_generator, task = prefix, 
                 check_other_outputs = check_other_outputs, 
                 zero_shot = zero_shot, generalize = generalize)
        )
        class_accuracy_counts.append(dict_accuracy_counts)
        pred_target_pairs.append(all_pred_target_pairs)

    if True:
        with open(path1, 'w') as f:
            json.dump(class_accuracy_counts, f, indent=2) 

        with open(path2, 'w') as f:
            json.dump(pred_target_pairs, f, indent=2)

if __name__ == "__main__":
    main()