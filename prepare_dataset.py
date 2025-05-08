"""
This script is used to prepare the data for training. Datasets are saved both for single-task and multitask training in Hugging Face format.
Reactions that present the same {substrate, EC} or {product, EC} pairs are grouped so that they end up in only one task/split
"""

import random
from collections import Counter

from Utils.System_prompts import (
        EC_finetuning_prompt,
        EC_prompts,
        EC_prediction_data_dict,
        substrate_finetuning_prompt,
        substrate_prompts,
        substrate_prediction_data_dict,
        product_finetuning_prompt,
        product_prompts,
        product_prediction_data_dict
)

from Utils.Utils_functions import (
    load_data,
    parse_reactions,
    find_duplicates,
    prepare_and_split_subgroups,
    assign_subgroups_by_task,
    flatten,
    subgroup_finder,
    split_conversations,
    save_smart_json_subsets,
    process_conversations,
    save_single_task_datasets_for_HF_loader,
    save_multi_task_datasets_for_HF_loader
)

RANDOM_SEED = 0
FILE_PATH = "./Data_raw/BRENDA_ECREACT_canonical.txt"
DATASET_SIZE = 8496

if __name__ == "__main__":

    random.seed(RANDOM_SEED)    
    # Load data
    data = load_data(FILE_PATH)

    # Parse reactions
    (substrate_ec_pairs, product_ec_pairs, row_indices_substrate_ec,
     row_indices_product_ec, products_per_substrate_ec, substrates_per_product_ec,
    substrates, ecs, products) = parse_reactions(data)

    # Compute EC counts (needed later for histogram analysis)
    substrate_ec_count = Counter(substrate_ec_pairs)
    product_ec_count = Counter(product_ec_pairs)

    # Find duplicates
    substrate_ec_duplicates, substrate_subgroups = find_duplicates(
        substrate_ec_pairs, row_indices_substrate_ec)
    product_ec_duplicates, product_subgroups = find_duplicates(
        product_ec_pairs, row_indices_product_ec)

    # Print summary
    print("\nSummary:")
    print(f"Total (substrate, EC) duplicates: {len(substrate_ec_duplicates)}")
    print(f"Total (product, EC) duplicates: {len(product_ec_duplicates)}")
    print()

    # Prepare and split the dataset
    train_set, test_set, final_subgroups, train_subgroups, test_subgroups, sub_subgroups, prod_subgroups = prepare_and_split_subgroups(
        substrate_subgroups,
        product_subgroups,
        dataset_size=DATASET_SIZE,
        mode="BOTH",  # or "PRODUCT", "SUBSTRATE"
        train_ratio=0.7,
        seed=RANDOM_SEED
    )

    train_group_ec, train_group_prod, train_group_sub = assign_subgroups_by_task(train_subgroups)
    test_group_ec, test_group_prod, test_group_sub = assign_subgroups_by_task(test_subgroups)

    train_ec_smart_idx = flatten(train_group_ec)
    train_prod_smart_idx = flatten(train_group_prod)
    train_sub_smart_idx = flatten(train_group_sub)
    test_ec_smart_idx = flatten(test_group_ec)
    test_prod_smart_idx = flatten(test_group_prod)
    test_sub_smart_idx = flatten(test_group_sub)

    # For each reaction in the product (substrate) prediction set, we check if they belong to a subset 
    # of branching products (substrates), and assing the subset to that corresponding index
    prod_other_truths = subgroup_finder(train_prod_smart_idx, sub_subgroups)
    sub_other_truths = subgroup_finder(train_sub_smart_idx, prod_subgroups)
    test_prod_other_truths = subgroup_finder(test_prod_smart_idx, sub_subgroups)
    test_sub_other_truths = subgroup_finder(test_sub_smart_idx, prod_subgroups)

    prompts = [EC_prompts, substrate_prompts, product_prompts]
    finetuning_prompts = [EC_finetuning_prompt, substrate_finetuning_prompt, product_finetuning_prompt]
    data_dicts = [EC_prediction_data_dict, substrate_prediction_data_dict, product_prediction_data_dict]

    file_paths_train = ['ec_train_set.json', 'substrate_train_set.json', 'product_train_set.json']
    file_paths_test = ['ec_test_set.json', 'substrate_test_set.json', 'product_test_set.json']

    train_other_truths = [[], sub_other_truths, prod_other_truths] 
    test_other_truths = [[], test_sub_other_truths, test_prod_other_truths] 

    other_truths = [train_other_truths, test_other_truths]
    train_indexes = [train_ec_smart_idx, train_sub_smart_idx, train_prod_smart_idx]
    test_indexes = [test_ec_smart_idx, test_sub_smart_idx, test_prod_smart_idx]

    save_smart_json_subsets(prompts, data_dicts, file_paths_train, file_paths_test, train_indexes, test_indexes, train_other_truths, test_other_truths, ecs, products, substrates)

    # Train/Validation processing
    train_file_paths = file_paths_train  # List of train file paths
    train_prompts = finetuning_prompts  # List of prompts for training
    multitask_train_convs, train_grouped_convs = process_conversations(train_file_paths, train_prompts, split_type='train')

    # Test processing
    test_file_paths = file_paths_test  # List of test file paths
    test_prompts = finetuning_prompts  # List of prompts for testing
    multitask_test_convs, test_grouped_convs = process_conversations(test_file_paths, test_prompts, split_type='test')

    # Output for debugging
    print(f"Train grouped: {len(train_grouped_convs)} files, Test grouped: {len(test_grouped_convs)} files.")

    # Splitting training and validation data in two lists to be saved as Hugging Face dataset
    multitask_train_messages = []
    multitask_val_messages = []
    for conv in multitask_train_convs:
        if conv['split'] == 'train':        
            multitask_train_messages.append(conv)
        elif conv['split'] == 'validation':    
            multitask_val_messages.append(conv) 
    print(len(multitask_train_messages), len(multitask_val_messages))

    train_convs, validation_convs, test_convs = split_conversations(
        train_grouped_convs,  # Training/validation conversations grouped by task
        test_grouped_convs    # Test conversations grouped by task
    )

    print(f"Train: {sum(len(tc) for tc in train_convs)} examples")
    print(f"Validation: {sum(len(vc) for vc in validation_convs)} examples")
    print(f"Test: {sum(len(tc) for tc in test_convs)} examples")


    folders = save_single_task_datasets_for_HF_loader(train_convs, validation_convs, test_convs)
    multitask_folder = save_multi_task_datasets_for_HF_loader(multitask_train_messages, multitask_val_messages, multitask_test_convs)
