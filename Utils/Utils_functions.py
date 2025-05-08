import os
import shutil
import json
import re
import random
from random import Random
import numpy as np
from datasets import load_dataset
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


from System_prompts import (
        ADJ,
        EC,
        SUBSTRATE,
        PRODUCT,
        CHEM,
)



def load_data(file_path):
    """
    Load data from a file.
    """
    try:
        with open(file_path, 'r') as file:
            return file.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise



def parse_reactions(data):
    """
    Parse reactions from the dataset.
    """
    pattern = re.compile(r"^(.*?)\|([^>]+)>>([^,]+)")
    substrate_ec_pairs = []
    product_ec_pairs = []
    substrates = []
    ecs = []
    products = []
    row_indices_substrate_ec = defaultdict(list)
    row_indices_product_ec = defaultdict(list)
    products_per_substrate_ec = defaultdict(set)
    substrates_per_product_ec = defaultdict(set)
    

    for i, line in enumerate(data):
        match = pattern.match(line)
        if match:
            substrate = match.group(1)
            ec = match.group(2)
            product = match.group(3)

            substrates.append(substrate)
            ecs.append(ec)
            products.append(product.strip())  # Strip any leading/trailing whitespace
            substrate_ec_pairs.append((substrate, ec))
            product_ec_pairs.append((product, ec))
            row_indices_substrate_ec[(substrate, ec)].append(i + 1)
            row_indices_product_ec[(product, ec)].append(i + 1)
            products_per_substrate_ec[(substrate, ec)].add(product)
            substrates_per_product_ec[(product, ec)].add(substrate)

    return (substrate_ec_pairs, product_ec_pairs, row_indices_substrate_ec,
            row_indices_product_ec, products_per_substrate_ec, substrates_per_product_ec,
            substrates, ecs, products)



def find_duplicates(ec_pairs, row_indices):
    """
    Find duplicates in EC pairs and return subgroups.
    """
    ec_count = Counter(ec_pairs)
    duplicates = {pair: count for pair, count in ec_count.items() if count > 1}
    subgroups = []

    for pair, _ in duplicates.items():
        rows = row_indices[pair]
        subgroups.append(rows)

    return duplicates, subgroups



def merge_subgroups(subgroups):
    """
    Merge overlapping subgroups.
    """
    merged = []
    for subgroup in subgroups:
        added = False
        for m in merged:
            if set(subgroup) & set(m):  # Check for overlap
                m.update(subgroup)  # Merge the subgroups
                added = True
                break
        if not added:
            merged.append(set(subgroup))
    return [list(m) for m in merged]



def prepare_and_split_subgroups(
    substrate_subgroups, 
    product_subgroups, 
    dataset_size, 
    mode="BOTH", 
    train_ratio=0.7, 
    seed=0
):
    """
    Mix, merge, and split subgroups into train/test sets with multiplicity handling.
    """
    print('##### Taking care of multiplicity #####')

    data_points = list(range(dataset_size))
    substrate_subgroups = substrate_subgroups.copy()
    product_subgroups = product_subgroups.copy()

    all_substrate_elements = {elem for group in substrate_subgroups for elem in group}
    all_product_elements = {elem for group in product_subgroups for elem in group}

    non_substrate_elements = [i for i in data_points if i not in all_substrate_elements]
    non_product_elements = [i for i in data_points if i not in all_product_elements]

    print("Number of 'repeated product' (branching substrate) and 'repeated substrate' (branching product) subgroups:",
          len(product_subgroups), len(substrate_subgroups))
    print("Respectively containing #reactions:", len(all_product_elements), len(all_substrate_elements))

    # Choose mixing strategy
    if mode == "BOTH":
        all_subgroups = substrate_subgroups + product_subgroups
    elif mode == "PRODUCT":
        all_subgroups = product_subgroups
    elif mode == "SUBSTRATE":
        all_subgroups = substrate_subgroups
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'BOTH', 'PRODUCT', or 'SUBSTRATE'.")

    # Add unique elements as singleton groups
    all_subgroup_elements = {elem for group in all_subgroups for elem in group}
    non_mixed_elements = [i for i in data_points if i not in all_subgroup_elements]
    all_subgroups.extend([[i] for i in non_mixed_elements])

    substrate_subgroups.extend([[i] for i in non_substrate_elements])
    product_subgroups.extend([[i] for i in non_product_elements])

    product_total = sum(len(g) for g in product_subgroups)
    substrate_total = sum(len(g) for g in substrate_subgroups)
    mixed_total = sum(len(g) for g in all_subgroups)

    print('Number of product and substrate subgroups after adding unique reactions (as groups of size 1), and the total #reactions:',
          len(product_subgroups), len(substrate_subgroups), ', ', product_total, substrate_total)
    print('Number of mixed subgroups + unique reactions (not accounting for overlapping reactions):',
          len(all_subgroups), mixed_total)

    # Detect overlapping reactions
    reaction_counts = Counter(i for group in all_subgroups if len(group) > 1 for i in group)
    overlapping_reactions = [i for i, c in reaction_counts.items() if c > 1]

    print()
    print(f'Total reaction number is exceeding by {mixed_total - product_total} reactions')
    print('Double check: Reactions that are branching both in a substrate and a product subgroup (overlapping reactions):',
          len(overlapping_reactions))
    print(f'Test passed: {(mixed_total - product_total) == len(overlapping_reactions)}')

    # Merge overlapping subgroups
    final_subgroups = merge_subgroups(all_subgroups)
    while sum(len(g) for g in final_subgroups) != dataset_size:
        final_subgroups = merge_subgroups(final_subgroups)

    print()
    print("Total number of reactions after merging:", sum(len(g) for g in final_subgroups))
    print("Number of subgroups in merged list:", len(final_subgroups))
    print("Number of subgroups with size >1:", len([g for g in final_subgroups if len(g) > 1]),
          " accounting for:", sum(len(g) for g in final_subgroups if len(g) > 1))

    print(f"Double check: Total #reactions appearing in (substrate+product)-overlap = "
          f"{len(all_substrate_elements)} + {len(all_product_elements)} - {len(overlapping_reactions)} = "
          f"{len(all_substrate_elements) + len(all_product_elements) - len(overlapping_reactions)}")
    print(f"Double check: Total #reactions in merged list = "
          f"{len(final_subgroups)} - {len([g for g in final_subgroups if len(g) > 1])} + "
          f"{sum(len(g) for g in final_subgroups if len(g) > 1)} = "
          f"{len(final_subgroups) - len([g for g in final_subgroups if len(g) > 1]) + sum(len(g) for g in final_subgroups if len(g) > 1)}")
    print(f"Test passed: {(len(all_substrate_elements) + len(all_product_elements) - len(overlapping_reactions)) == sum(len(g) for g in final_subgroups if len(g) > 1)}")

    # Shuffle and split
    Random(seed).shuffle(final_subgroups)
    total_samples = sum(len(g) for g in final_subgroups)
    desired_train_size = int(total_samples * train_ratio)

    train_subgroups, test_subgroups = [], []
    current_train_size = 0
    for group in final_subgroups:
        if current_train_size + len(group) <= desired_train_size:
            train_subgroups.append(group)
            current_train_size += len(group)
        else:
            test_subgroups.append(group)

    train_set = [i for group in train_subgroups for i in group]
    test_set = [i for group in test_subgroups for i in group]

    print()
    print('##### Splitting #####')
    print()
    print(f"Train-test {train_ratio}-{1-train_ratio:.1f} (ideal): {total_samples*train_ratio:.0f}, {total_samples*(1-train_ratio):.0f}")
    print(f"Actual split (with branching): {len(train_set)}, {len(test_set)}")

    train_lengths = [len(g) for g in train_subgroups]
    test_lengths = [len(g) for g in test_subgroups]
    print()
    print(f"Number of subgroups in train/test: {len(train_lengths)} + {len(test_lengths)} = {len(train_lengths) + len(test_lengths)}")
    print(f"Corresponding to {len(train_lengths)*100/len(final_subgroups):.4f}% and {len(test_lengths)*100/len(final_subgroups):.4f}% of all groups respectively")

    return train_set, test_set, final_subgroups, train_subgroups, test_subgroups, substrate_subgroups, product_subgroups



def assign_subgroups_by_task(subgroups):
    
    ec, prod, sub = [], [], []
    for i, group in enumerate(subgroups):
        (ec, prod, sub)[i % 3].append(group)
    return ec, prod, sub

def flatten(grouped):
    
    return [i for group in grouped for i in group]

def subgroup_finder(list1, list2):
    
    matching_list = []
    # Iterate through each sublist in list1
    for index in list1:
        # Find the sublist in list2 where the index appears
        corresponding_sublist = next(s for s in list2 if index in s)
        matching_list.append(corresponding_sublist)

    print('Index list:', list1)
    print("Matching list:", matching_list)
    print()
    
    return matching_list



def prompt_formatter(input_a, input_b, output, sep):
    
    raw_input = f"{input_a}{sep}{input_b}"
    raw_output = output
    return raw_input, raw_output



# Function to generate a random prompt and update the dictionary
def generate_and_update_prompt(prompts, sample_dict, i, idx, size, raw_input, raw_output, mode = 'train', ground_truths = None):
    """
    Generate a random prompt as a data point
    """

    # Randomly select a prompt
    prompt = random.choice(prompts)
    # Replace placeholders with random selections from the corresponding lists
    prompt = prompt.replace('[ADJ]', random.choice(ADJ))
    prompt = prompt.replace('[EC]', random.choice(EC))
    prompt = prompt.replace('[SUBSTRATE]', random.choice(SUBSTRATE))
    prompt = prompt.replace('[PRODUCT]', random.choice(PRODUCT))
    prompt = prompt.replace('[CHEM]', random.choice(CHEM))
    
    sample_dict['raw_input'] = raw_input
    sample_dict['raw_output'] = raw_output

    if mode=='train':
        thresh = int(size*0.9)
        if idx < thresh:
            sample_dict['split'] = 'train'
        else:
            idx = idx % thresh
            sample_dict['split'] = 'validation'
    else:
        sample_dict['split'] = 'test'

    # Replace [...] with <tag> raw_input </tag>
    raw_input_tagged = f"{sample_dict['input_core_tag_left']} {sample_dict['raw_input']} {sample_dict['input_core_tag_right']}"
    prompt = prompt.replace('[...]', raw_input_tagged)
    
    # Prepare <tag> raw_output </tag> for output
    raw_output_tagged = f"{sample_dict['output_core_tag_left']} {sample_dict['raw_output']} {sample_dict['output_core_tag_right']}"
    
    sample_dict['other_raw_outputs'] = ground_truths
    new_samples = []
    for sample in ground_truths:
        new_samples.append(f"{sample_dict['output_core_tag_left']} {sample} {sample_dict['output_core_tag_right']}")
    sample_dict['other_outputs'] = new_samples

    # Update the 'input' and 'output' field in the dictionary
    sample_dict['input'] = prompt
    sample_dict['output'] = raw_output_tagged
    sample_dict['id'] = i
    sample_dict['idx'] = idx
            
    # Update the 'sample_id' with the actual iterator value
    sample_dict['sample_id'] = f"{sample_dict['task']}.{sample_dict['split']}.{i}"

    return sample_dict



def format_prompt_and_ground_truth(count, idx, i, substrates, products, ecs, other_gt):
    """
    Helper function to format input, output, and ground truths.
    """
    converted_gt = []
    
    if count == 0:
        raw_input, raw_output = prompt_formatter(substrates[i], products[i], ecs[i], sep='>>')
    elif count == 1:
        raw_input, raw_output = prompt_formatter(products[i], ecs[i], substrates[i], sep='|')
        if len(other_gt[idx]) > 1:
            converted_gt = [substrates[s - 1] for s in other_gt[idx]]
    elif count == 2:
        raw_input, raw_output = prompt_formatter(substrates[i], ecs[i], products[i], sep='|')
        if len(other_gt[idx]) > 1:
            converted_gt = [products[s - 1] for s in other_gt[idx]]

    return raw_input, raw_output, converted_gt



def process_samples(count, indexes, prompt, data_dict, substrates, products, ecs, other_gt, mode, size):
    """
    Helper function to process train/test samples.
    """
    samples = []
    for idx, i in enumerate(indexes):
        i -= 1
        raw_input, raw_output, converted_gt = format_prompt_and_ground_truth(count, idx, i, substrates, products, ecs, other_gt)
        sample = generate_and_update_prompt(
            prompt, data_dict, i + 1, idx, size, raw_input, raw_output, mode=mode, ground_truths=converted_gt
        )
        samples.append(sample.copy())
    return samples



def save_smart_json_subsets(prompts, data_dicts, file_paths_train, file_paths_test, train_indexes, test_indexes, train_other_truths, test_other_truths, ecs, products, substrates):
    """ 
    Save training and testing samples to JSON files.
    """
    for count, (prompt, data_dict, file_path_train, file_path_test, other_gt, test_other_gt) in enumerate(zip(
        prompts, data_dicts, file_paths_train, file_paths_test, train_other_truths, test_other_truths
    )):
        size = len(train_indexes[count])

        # Process training samples
        train_samples = process_samples(
            count, train_indexes[count], prompt, data_dict, substrates, products, ecs, other_gt, mode='train', size=size
        )
        with open(file_path_train, 'w') as json_file:
            json.dump(train_samples, json_file, indent=4)

        # Process testing samples
        test_samples = process_samples(
            count, test_indexes[count], prompt, data_dict, substrates, products, ecs, test_other_gt, mode='test', size=size
        )
        with open(file_path_test, 'w') as json_file:
            json.dump(test_samples, json_file, indent=4)



def split_conversations(multiple_conversations, test_multiple_conversations):
    """
    Splits conversations into training, validation, and test sets.
    """
    train_convs, validation_convs, test_convs = [], [], []

    for convs, test_convs_task in zip(multiple_conversations, test_multiple_conversations):
        train_conv, validation_conv, test_conv = [], [], []
        
        for conv in convs:
            if conv['split'] == 'train':
                train_conv.append(conv)
            elif conv['split'] == 'validation':
                validation_conv.append(conv)

        test_conv.extend(test_convs_task)

        train_convs.append(train_conv)
        validation_convs.append(validation_conv)
        test_convs.append(test_conv)

    return train_convs, validation_convs, test_convs



def process_conversations(file_paths, prompts, split_type):
    """
    Process conversations from input files into a structured format.
    """
    all_conversations = []
    grouped_conversations = []

    for file_path, prompt in zip(file_paths, prompts):
        with open(file_path, 'r') as file:
            samples = json.load(file)

        conversations = []
        for sample in samples:
            messages = {
                'split': sample['split'],
                'idx': sample['idx'],
                'id': sample['id'],
                'message': [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": sample['input']},
                    {"role": "assistant", "content": sample['output']},
                ],
                'other_truths': sample['other_outputs'],
            }
            conversations.append(messages)

        grouped_conversations.append(conversations)
        all_conversations.extend(conversations)

    print(f"{split_type.capitalize()} data: {len(all_conversations)} examples processed.")
    return all_conversations, grouped_conversations



def save_single_task_datasets_for_HF_loader(train_convs, validation_convs, test_convs=None):
    """
    Saves datasets for Hugging Face loaders, organizes train/validation datasets,
    and optionally saves test datasets as separate Hugging Face-compatible files.
    """
    names = ['ec', 'substrate', 'product']
    train_val_folders = []
    test_folders = []

    for train_conv, validation_conv, name in zip(train_convs, validation_convs, names):
        # Define the base folder and file paths for train/validation
        base_folder = f"{name}_dataset"
        os.makedirs(base_folder, exist_ok=True)

        train_file_path = os.path.join(base_folder, 'train.jsonl')
        val_file_path = os.path.join(base_folder, 'validation.jsonl')
        dataset_dict_path = os.path.join(base_folder, 'dataset_dict.json')

        # Save dataset metadata for train/validation
        dataset_dict = {"splits": ["train", "validation"]}
        with open(dataset_dict_path, 'w') as f:
            json.dump(dataset_dict, f)
        print(f"Created dataset_dict.json at {dataset_dict_path}")

        # Save train conversations in 'train.jsonl'
        with open(train_file_path, 'w') as train_file:
            train_file.writelines(json.dumps(conv) + '\n' for conv in train_conv)
        print(f"Train conversations saved to {train_file_path}")

        # Save validation conversations in 'validation.jsonl'
        with open(val_file_path, 'w') as val_file:
            val_file.writelines(json.dumps(conv) + '\n' for conv in validation_conv)
        print(f"Validation conversations saved to {val_file_path}")

        # Convert to Hugging Face dataset and save train/validation
        dataset = load_dataset(path=base_folder)
        new_folder = f"{base_folder}.hf"
        dataset.save_to_disk(new_folder)
        train_val_folders.append(new_folder)

        # Organize JSONL files inside the new folder
        os.makedirs(os.path.join(new_folder, 'train'), exist_ok=True)
        os.makedirs(os.path.join(new_folder, 'validation'), exist_ok=True)
        shutil.move(train_file_path, os.path.join(new_folder, 'train/train.jsonl'))
        shutil.move(val_file_path, os.path.join(new_folder, 'validation/validation.jsonl'))

        # Clean up temporary folder
        shutil.rmtree(base_folder)

    if test_convs:
        for test_conv, name in zip(test_convs, names):
            # Define the base folder and file paths for test
            test_base_folder = f"{name}_test_dataset"
            os.makedirs(test_base_folder, exist_ok=True)

            test_file_path = os.path.join(test_base_folder, 'test.jsonl')
            test_dataset_dict_path = os.path.join(test_base_folder, 'dataset_dict.json')

            # Save dataset metadata for test
            test_dataset_dict = {"splits": ["test"]}
            with open(test_dataset_dict_path, 'w') as f:
                json.dump(test_dataset_dict, f)
            print(f"Created dataset_dict.json at {test_dataset_dict_path}")

            # Save test conversations in 'test.jsonl'
            with open(test_file_path, 'w') as test_file:
                test_file.writelines(json.dumps(conv) + '\n' for conv in test_conv)
            print(f"Test conversations saved to {test_file_path}")

            # Convert to Hugging Face dataset and save test
            test_dataset = load_dataset(path=test_base_folder)
            test_folder = f"{test_base_folder}.hf"
            test_dataset.save_to_disk(test_folder)
            test_folders.append(test_folder)

            # Organize JSONL files inside the new test folder
            os.makedirs(os.path.join(test_folder, 'test'), exist_ok=True)
            shutil.move(test_file_path, os.path.join(test_folder, 'test/test.jsonl'))

            # Clean up temporary folder
            shutil.rmtree(test_base_folder)

    return {"train_val_folders": train_val_folders, "test_folders": test_folders}



def save_multi_task_datasets_for_HF_loader(multitask_train_convs, multitask_validation_convs, multitask_test_convs=None):
    """
    Saves multi-task datasets for Hugging Face loaders, organizes train/validation datasets,
    and optionally saves test datasets as separate Hugging Face-compatible files.
    """
    train_val_folders = []
    test_folders = []

    # Define the base folder and file paths for train/validation
    base_folder = 'multitask_train_dataset'
    os.makedirs(base_folder, exist_ok=True)

    train_file_path = os.path.join(base_folder, 'train.jsonl')
    val_file_path = os.path.join(base_folder, 'validation.jsonl')
    dataset_dict_path = os.path.join(base_folder, 'dataset_dict.json')

    # Save dataset metadata for train/validation
    dataset_dict = {"splits": ["train", "validation"]}
    with open(dataset_dict_path, 'w') as f:
        json.dump(dataset_dict, f)
    print(f"Created dataset_dict.json at {dataset_dict_path}")

    # Save train conversations in 'train.jsonl'
    with open(train_file_path, 'w') as train_file:
        train_file.writelines(json.dumps(conv) + '\n' for conv in multitask_train_convs)
    print(f"Train conversations saved to {train_file_path}")

    # Save validation conversations in 'validation.jsonl'
    with open(val_file_path, 'w') as val_file:
        val_file.writelines(json.dumps(conv) + '\n' for conv in multitask_validation_convs)
    print(f"Validation conversations saved to {val_file_path}")

    # Convert to Hugging Face dataset and save train/validation
    dataset = load_dataset(path=base_folder)
    new_folder = f"{base_folder}.hf"
    dataset.save_to_disk(new_folder)
    train_val_folders.append(new_folder)

    # Organize JSONL files inside the new folder
    os.makedirs(os.path.join(new_folder, 'train'), exist_ok=True)
    os.makedirs(os.path.join(new_folder, 'validation'), exist_ok=True)
    shutil.move(train_file_path, os.path.join(new_folder, 'train/train.jsonl'))
    shutil.move(val_file_path, os.path.join(new_folder, 'validation/validation.jsonl'))

    # Clean up temporary folder
    shutil.rmtree(base_folder)

    if multitask_test_convs:
        # Define the base folder and file paths for test
        test_base_folder = 'multitask_test_dataset'
        os.makedirs(test_base_folder, exist_ok=True)

        test_file_path = os.path.join(test_base_folder, 'test.jsonl')
        test_dataset_dict_path = os.path.join(test_base_folder, 'dataset_dict.json')

        # Save dataset metadata for test
        test_dataset_dict = {"splits": ["test"]}
        with open(test_dataset_dict_path, 'w') as f:
            json.dump(test_dataset_dict, f)
        print(f"Created dataset_dict.json at {test_dataset_dict_path}")

        # Save test conversations in 'test.jsonl'
        with open(test_file_path, 'w') as test_file:
            test_file.writelines(json.dumps(conv) + '\n' for conv in multitask_test_convs)
        print(f"Test conversations saved to {test_file_path}")

        # Convert to Hugging Face dataset and save test
        test_dataset = load_dataset(path=test_base_folder)
        test_folder = f"{test_base_folder}.hf"
        test_dataset.save_to_disk(test_folder)
        test_folders.append(test_folder)

        # Organize JSONL files inside the new test folder
        os.makedirs(os.path.join(test_folder, 'test'), exist_ok=True)
        shutil.move(test_file_path, os.path.join(test_folder, 'test/test.jsonl'))

        # Clean up temporary folder
        shutil.rmtree(test_base_folder)

    return {"train_val_folders": train_val_folders, "test_folders": test_folders}



def remove_between_symbols(lst):
    return [re.sub(r"(\|.*?)(>>)", r"\2", s) for s in lst]



def read_and_preprocess_dataset(file_path = './BRENDA_ECREACT.txt', seed = 0, subset_size=50, split=0.5):

    train_size = int(subset_size * split)
    test_size = int(subset_size * (1 - split))

    dataset = []
    targets = []
    origin = []

    print('File_path:', file_path,
          '\nSeed:', seed,
          '\nSubset_size:', subset_size,
          '\nSplit:', split)
    
    # Open the file
    with open(file_path, "r") as file:   

        for line in file:
            # Split the line into values using comma as separator
            values = line.strip().split(",")

            # Extract and append values to respective lists
            dataset.append(values[0])
            targets.append(values[1])
            origin.append(values[2])

    dataset = remove_between_symbols(dataset)

    hist = {}

    for EC in targets:
        # Get the first character (assuming it's a number)
        EC_class = EC[0]
        # Update the histogram for the corresponding class
        hist[EC_class] = hist.get(EC_class, 0) + 1

    hist = dict(sorted(hist.items()))

    data = list(zip(dataset, targets))
    Random(seed).shuffle(data)  # deterministic random split

    subset_data = data[:subset_size]  # reducing dataset size
    train_data, test_data = subset_data[:train_size], subset_data[-test_size:]
    train_set, train_targets = zip(*train_data)
    test_set, test_targets = zip(*test_data)

    print()
    print("Size of training and test sets:", len(train_set), len(test_set))

    return (
        data,
        subset_data,
        train_data,
        test_data,
        train_set,
        train_targets,
        test_set,
        test_targets,
        train_size,
        test_size,
        origin,
    )



def create_histogram_and_classes(data):
    hist = {}
    hist_classes = {}

    # Loop through the strings
    for EC in data:
        # Get the first character (assuming it's a number)
        EC_class = EC[0] if isinstance(EC, str) else EC[1][0]
        
        # Update the histogram for the corresponding first letter
        hist[EC_class] = hist.get(EC_class, 0) + 1

        # Initialize or increment the count in hist_classes
        EC_class_int = int(EC_class)
        hist_classes[EC_class_int] = hist_classes.get(EC_class_int, 0) + 1

    # Sort the dictionaries
    hist = dict(sorted(hist.items()))
    hist_classes = dict(sorted(hist_classes.items()))

    return hist, hist_classes



def compute_confusion_matrix(accuracies, exp=0, level=1, norm = True):

    actual_classes = []
    predicted_classes = []

    list_level = 'list_level_' + str(level)
    dict_level = 'dict_test_' + str(level)
    
    for pair in accuracies[list_level][exp]:
        actual_classes.append(pair[0])
        predicted_classes.append(pair[1])

    # Get sorted list of unique class labels
    all_classes = sorted(set(actual_classes).union(predicted_classes))    
    ordered_actual_classes = sorted(set(actual_classes))

    # Create an empty matrix for the heatmap
    conf_matrix = np.zeros((len(all_classes), len(all_classes)))
    reduced_conf_matrix = np.zeros((len(ordered_actual_classes), len(all_classes))) # Maybe this is the right version?
    
    # Create a mapping from class labels to indices for rows (actual) and columns (predicted)
    actual_to_index = {label: index for index, label in enumerate(ordered_actual_classes)}
    # # Create a mapping from class labels to indices
    class_to_index = {label: index for index, label in enumerate(all_classes)}

    # Populate the matrix with values from the lists
    for actual, predicted in zip(actual_classes, predicted_classes):
        j = class_to_index[actual]
        j2 = actual_to_index[actual]
        k = class_to_index[predicted]
        # conf_matrix[i, j] += 1 / accuracies[dict_level][str(i + 1)]
        if norm:
            conf_matrix[j, k] += 1 / accuracies[dict_level][actual]
            reduced_conf_matrix[j2, k] += 1 / accuracies[dict_level][actual]
        else:
            conf_matrix[j, k] += 1
            reduced_conf_matrix[j2, k] += 1
    return conf_matrix, reduced_conf_matrix, all_classes, ordered_actual_classes




# Define a function to format the ticks
def to_percentage(x, _):
    return f"{x * 100:.0f}"



def compute_accuracies_from_confusion_matrices(accuracies, N_experiments, end=4, plot_stratified_per_experiment=False, stratified=False, up_to_class = 7, trimmed = False):
    mean_per_level, std_per_level = [], []
    mean_per_class, std_per_class = [], []
    mean_per_sublevel2, std_per_sublevel2 = [], []
    mean_per_sublevel3, std_per_sublevel3 = [], []

    # Prepare data structures for EC levels and class-specific accuracy
    up_to_class = up_to_class
    class_accuracies_per_level = {cls: [] for cls in range(1, up_to_class)}
    for level in range(1, end):
        all_values = []
        all_values_per_EC_level = []
        # all_class_accuracies_per_experiment = {exp: {cls: [] for cls in range(1, 7)} for exp in range(N_experiments)} ### attention here
        all_class_accuracies_per_experiment = {exp: {cls: [] for cls in range(1, up_to_class+1)} for exp in range(N_experiments)}

        for i in range(N_experiments):
            confusion_matrix, _, all_classes, _ = compute_confusion_matrix(accuracies, exp=i, level=level)
            # print(all_classes)
            diag_values = np.diag(confusion_matrix)
            non_zero_row_indices = [j for j in range(confusion_matrix.shape[0]) if np.any(confusion_matrix[j] != 0)]
            diag = diag_values[non_zero_row_indices]
            
            diag_mean = np.mean(diag)
            all_values.append(diag_mean)
            all_values_per_EC_level.append(diag)
            

            if level in [2, 3]:
                for idx, main_class in enumerate(all_classes):
                    row_sum = np.sum(confusion_matrix[idx])
                    if row_sum > 0:
                        accuracy = confusion_matrix[idx, idx] / row_sum
                        main_class_top_level = int(main_class.split('.')[0])
                        all_class_accuracies_per_experiment[i][main_class_top_level].append(accuracy)

        mean_per_level.append(np.mean(all_values))
        std_per_level.append(np.std(all_values))
        if level==1:
            mean_per_class.append(np.mean(all_values_per_EC_level, axis=0))
            std_per_class.append(np.std(all_values_per_EC_level, axis=0))
            
        print(f"\nAverage accuracies per EC sub-level {level}: {all_values}")
        print(f"Average global accuracy over {N_experiments} experiments: {mean_per_level[-1]:.3f} ± {std_per_level[-1]:.3f}")

        per_class_averages = {cls: [] for cls in range(1, up_to_class)}

        if level==1:
            for main_class, _ in per_class_averages.items():        
                class_accuracies_per_level[main_class].append((mean_per_class[0][int(main_class)-1], std_per_class[0][int(main_class)-1]))

        if level in [2, 3]:
            mean_list = mean_per_sublevel2 if level == 2 else mean_per_sublevel3
            std_list = std_per_sublevel2 if level == 2 else std_per_sublevel3

            for cls in range(1, up_to_class):
                class_experiment_accuracies = [
                    np.mean(all_class_accuracies_per_experiment[exp][cls]) 
                    for exp in range(N_experiments) 
                    if all_class_accuracies_per_experiment[exp][cls]
                ]
                if class_experiment_accuracies:
                    per_class_averages[cls] = class_experiment_accuracies

            print(f"\nSub-level {level} accuracies per main EC class:")
            for main_class, acc_list in per_class_averages.items():
                if acc_list:
                    mean_class_accuracy = np.mean(acc_list)                    
                    std_class_accuracy = np.std(acc_list)
                    mean_list.append(mean_class_accuracy)
                    std_list.append(std_class_accuracy)
                    class_accuracies_per_level[main_class].append((mean_class_accuracy, std_class_accuracy))
                    print(f"  Class {main_class}: Mean accuracy = {mean_class_accuracy:.3f}, Std = {std_class_accuracy:.3f}")

            recovered_global_mean = np.mean([np.mean(acc) for acc in per_class_averages.values() if acc])
            print(f"Recovered global average accuracy for sub-level {level} from class means: {recovered_global_mean:.3f}")

    if plot_stratified_per_experiment:
        plt.figure(figsize=(12, 8))
        # colors = plt.cm.tab10.colors
        colors = plt.cm.tab20b.colors
        # shared_colors = plt.cm.tab20b(np.linspace(0, 1, num_classes))
        colors = [[0.22, 0.23, 0.47, 1.  ],
                [0.61, 0.62, 0.87, 1.  ],
                [0.71, 0.81, 0.42, 1.  ],
                [0.91, 0.73, 0.32, 1.  ],
                [0.68, 0.29, 0.29, 1.  ],
                [0.48, 0.25, 0.45, 1.  ],
                [0.87, 0.62, 0.84, 1.  ]]
        # plt.cm.tab20b(np.linspace(0, 1, len(innermost_labels)))
        ec_levels = [1, 2, 3]  # EC levels for the x-axis

        # Small offset for each class to avoid perfect alignment on the x-axis
        offset = 0.025

        for cls in range(1, up_to_class):
            # Extract means and stds for available levels only
            means = [mean for mean, std in class_accuracies_per_level[cls]]
            stds = [std for mean, std in class_accuracies_per_level[cls]]
            
            # Offset each class slightly on the x-axis
            available_levels = [x + (cls - 3) * offset for x in ec_levels[:len(means)]]
            
            # Plot with larger dots and offset x-points
            plt.errorbar(
                available_levels, means, yerr=stds, linestyle='-.', fmt='-o', color=colors[(cls - 1) % len(colors)],
                label=f'Class {cls}', markersize=10, capsize=8  # Larger dots and error caps
            )

        # Customize the plot
        plt.grid(True, linestyle='--', alpha=0.4)
        # Apply the formatter to the y-axis
        formatter = mticker.FuncFormatter(to_percentage)
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlabel('EC Level', fontsize=30)
        plt.ylabel('Accuracy (%)', fontsize=30)
        # plt.title('Accuracy (%) vs EC Level for Each Main Class', fontsize=20)
        plt.xticks(ec_levels, fontsize=25)
        plt.yticks(fontsize=25)
        plt.legend(title="EC Class", fontsize=20, title_fontsize=22)
        plt.tight_layout()

            # Make plot borders thicker
        ax = plt.gca()  # Get current axis
        ax.spines['top'].set_linewidth(1.5)    # Top border
        ax.spines['bottom'].set_linewidth(1.5) # Bottom border
        ax.spines['left'].set_linewidth(1.5)   # Left border
        ax.spines['right'].set_linewidth(1.5)  # Right border
        plt.show()


    if not stratified:
        return mean_per_level, std_per_level
    else:
        return mean_per_level, std_per_level, mean_per_sublevel2, std_per_sublevel2, mean_per_sublevel3, std_per_sublevel3



def compute_accuracies_at_all_levels( 
    test_targets, 
    N_experiments_all_pred_target_pairs,
    tags=False,
    trimmed = False
):

    # These lists store the target and predicted labels on different sublevels 
    # (EC level 1, 2 or 3), for the calculation of the confusion matrices
    list_level_1 = []
    list_level_2 = []
    list_level_3 = []

    # These lists store the occurrences of target and predicted labels on different sublevels 
    # (EC level 1, 2 or 3), for the calculation of the confusion matrices
    list_dict_1_pred = []
    list_dict_2_pred = []
    list_dict_3_pred = []

    # These dictionaries store the test distribution used to prompt the models
    dict_1_test = {str(class_label): 0 for class_label in range(1, 8)}
    dict_2_test = {}
    dict_3_test = {}

    # Iterate through each experiment
    for ex,experiment in enumerate(N_experiments_all_pred_target_pairs):
        
        dict_1_pred = {str(class_label): 0 for class_label in range(1, 8)}   
        dict_2_pred = {}
        dict_3_pred = {}
    
        dict_1_pred['?'] = 0
        dict_2_pred['?'] = 0
        dict_3_pred['?'] = 0
                
        level_1 = []
        level_2 = []
        level_3 = []

        pattern = re.compile(r'^\d+\.\d+\.\d+\.\d+$')
        pattern_level_2 = re.compile(r'^(\d+\.\d+)')
        pattern_level_3 = re.compile(r'^(\d+\.\d+\.\d+)')        


        for i, target in enumerate(test_targets):
            # print(target)
            if trimmed:
                if target[0] == '7':
                    print('skipping entry n.',i,' because target[0] is ==', target[0])
                    continue

            # Iterate through each predicted pair in the current experiment
            pred = experiment[i]

            # Split the target and prediction into their components
            target_components = target.split('.')
            
            # Reading each test label for increasing EC sub-levels
            level1_target = pattern_level_2.search(target).group(1)[0]
            level2_target = pattern_level_2.search(target).group(1)
            level3_target = pattern_level_3.search(target).group(1)

            # Initializing new occurrences in dictionary for target EC classes
            if level2_target[0] not in dict_1_test:
                dict_1_test[level2_target[0]] = 0

            # Initializing new occurrences in dictionary for target EC sub-level 2
            if level2_target not in dict_2_test:
                dict_2_test[level2_target] = 0

            # Initializing new occurrences in dictionary for target EC sub-level 3
            if level3_target not in dict_3_test:
                dict_3_test[level3_target] = 0

            if tags == True:
                # pattern_tag = re.compile(r'<EC>\s*(\d+\.\d+\.\d+\.\d+)\s*</EC>')
                pattern_tag = re.compile(r'<EC>\s*(.*?)\s*</EC>')
                # print(pattern_tag, pred)
                try:
                    pred = re.findall(pattern_tag, pred)[0]
                except:
                    pass

            #Check if the answer doesn't follow the required format
            if not pattern.fullmatch(pred):
                # print(pred)  
                dict_1_pred['?'] += 1
                dict_2_pred['?'] += 1
                dict_3_pred['?'] += 1

                level_1.append([level1_target,'?'])
                level_2.append([level2_target,'?'])
                level_3.append([level3_target,'?'])

                # Storing test set distribution (all three experiments have same test set distribution and shuffle)
                if ex == (len(N_experiments_all_pred_target_pairs)-1):
                    dict_1_test[level2_target[0]] += 1
                    dict_2_test[level2_target] += 1
                    dict_3_test[level3_target] += 1
            
            else:
                # Reading each predicted label for increasing EC sub-levels
                level1_pred = pattern_level_2.search(pred).group(1)[0]
                level2_pred = pattern_level_2.search(pred).group(1)
                level3_pred = pattern_level_3.search(pred).group(1)

                # Storing [target,prediction] pair for each sample in the experiment
                level_1.append([level1_target,level1_pred])
                level_2.append([level2_target,level2_pred])
                level_3.append([level3_target,level3_pred])

                if ex == (len(N_experiments_all_pred_target_pairs)-1):
                    dict_1_test[level2_target[0]] += 1
                    dict_2_test[level2_target] += 1
                    dict_3_test[level3_target] += 1

                # Check if the prediction is correct across multiple EC levels
                
                # EC level 1 (class)
                pred_components = pred.split('.')
                if target_components[0] == pred_components[0]:
                    # level_1_matches += 1
                    # hist_level_1[int(target_components[0])] += 1
                    dict_1_pred[str(target_components[0])] += 1
                    
                    # print('level1 matching!')

                    # EC level sub-level 2
                    if target_components[1] == pred_components[1]:
                        
                        if pattern_level_2.search(target).group(1) not in dict_2_pred:
                            dict_2_pred[pattern_level_2.search(target).group(1)] = 0
                        
                        # level_2_matches += 1
                        dict_2_pred[pattern_level_2.search(target).group(1)] += 1
                            
                        # print('level2 matching!')

                        # EC level sub-level 3
                        if target_components[2] == pred_components[2]:
                                
                            if pattern_level_3.search(target).group(1) not in dict_3_pred:
                                dict_3_pred[pattern_level_3.search(target).group(1)] = 0
        
                            # level_3_matches += 1
                            dict_3_pred[pattern_level_3.search(target).group(1)] += 1
                            
                            # print('level3 matching!')

                            # if target_components[3] == pred_components[3]:
                            #     level_4_matches += 1



        list_level_1.append(level_1)
        list_level_2.append(level_2)
        list_level_3.append(level_3)
    
        list_dict_1_pred.append(dict(sorted(dict_1_pred.items())))
        list_dict_2_pred.append(dict(sorted(dict_2_pred.items())))
        list_dict_3_pred.append(dict(sorted(dict_3_pred.items())))

        # print(f'\n\n############### End experiment {ex+1} ###############')
        # break
        
    dict_1_test = dict(sorted(dict_1_test.items()))
    dict_2_test = dict(sorted(dict_2_test.items()))
    dict_3_test = dict(sorted(dict_3_test.items()))

    # returning:
    # - three lists of all target and predicted labels pairs, across EC levels 1,2 and 3 (one list per each EC level)
    # - three dictionaries of test labels distributions across EC levels 1,2 and 3
    # - three lists of dictionaries, of predicted labels distributions across EC levels 1,2 and 3 (one per each EC level)
    return {
        "list_level_1": list_level_1,
        "list_level_2": list_level_2,
        "list_level_3": list_level_3,
        "dict_test_1": dict_1_test,
        "dict_test_2": dict_2_test,
        "dict_test_3": dict_3_test,
        "list_dict_pred_1": list_dict_1_pred,
        "list_dict_pred_2": list_dict_2_pred,
        "list_dict_pred_3": list_dict_3_pred,

    }
