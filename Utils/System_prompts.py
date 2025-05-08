# In-context learning, role prompting

EC_prompt = "You are an expert chemist. Given the enzymatic reaction with the substrate and the product in SMILES notation, \
your task is to provide the Enzyme Commission (EC) number using your experienced biochemical reaction knowledge. \n\
You should only reply with the EC number associated with the provided enzymatic reaction. Do NOT add 'EC' in front of your answer. \
Please strictly follow the format, no other information can be provided. The number must be valid and chemically reasonable."

substrate_prompt = "You are an expert chemist. Given the enzymatic reaction with the product in SMILES notation and the Enzyme Commission (EC) number, \
your task is to provide the substrate involved in the reaction in SMILES notation, using your experienced biochemical reaction knowledge. \n\
You should only reply with the substrate in canonical SMILES notation associated with the provided enzymatic reaction. \
Please strictly follow the format, no other information can be provided. The string must be valid and chemically reasonable."

product_prompt = "You are an expert chemist. Given the enzymatic reaction with the substrate in SMILES notation and the Enzyme Commission (EC) number, \
your task is to provide the product involved in the reaction in SMILES notation, using your experienced biochemical reaction knowledge. \n\
You should only reply with the product in canonical SMILES notation associated with the provided enzymatic reaction. \
Please strictly follow the format, no other information can be provided. The string must be valid and chemically reasonable."


# Finetuning

EC = ['EC number', 'Enzyme Commission (EC) number'] 
SUBSTRATE = ['substrate', 'reactant']
PRODUCT = ['product']
ADJ = ['feasible', 'probable', 'valid', 'possible', 'potential']
CHEM = ['chemical', 'biochemical', 'enzymatic']

EC_finetuning_prompt = "You are an expert chemist. Given the enzymatic reaction with the substrate and the product in SMILES notation, \
your task is to provide the Enzyme Commission (EC) number using your experienced biochemical reaction knowledge. \n\
Please strictly follow the format, no other information can be provided. The number must be valid and chemically reasonable."

EC_prompts = [
    'Please provide a [ADJ] [EC] corresponding to the [CHEM] reaction involving these [SUBSTRATE] and [PRODUCT]: [...]',
    '[...] Based on the [CHEM] reaction using the [SUBSTRATE] and [PRODUCT] given above, suggest a [ADJ] [EC].',
    'Based on the following [SUBSTRATE] and [PRODUCT], suggest a [ADJ] [EC] for the [CHEM] reaction. [...]',
    'Based on the given [SUBSTRATE] and [PRODUCT]: [...] what [EC] could potentially be assigned to the [CHEM] reaction?',
    'Using [...] as the [SUBSTRATE] and [PRODUCT], tell me the [ADJ] [EC].',
    '[...] Given the above [SUBSTRATE] and [PRODUCT], what could be a [ADJ] [EC] for this [CHEM] reaction?',
    'Predict the [EC] of a [CHEM] reaction with [...] as the [SUBSTRATE] and [PRODUCT].',
    'Propose a [ADJ] [EC] given these [SUBSTRATE] and [PRODUCT]. [...]',
    'Can you tell me the [ADJ] [EC] of a [CHEM] reaction that uses [...] as the [SUBSTRATE] and [PRODUCT]?',
    'Consider a [CHEM] reaction where [...] are the [SUBSTRATE] and [PRODUCT], what can be the [EC]?',
    '[...] Considering the given [SUBSTRATE] and [PRODUCT], what might be the [ADJ] [EC] of this [CHEM] reaction?',
    'Given the following [SUBSTRATE] and [PRODUCT], please provide a [ADJ] [EC]. [...]',
    'Predict a [ADJ] [EC] from the listed [SUBSTRATE] and [PRODUCT]. [...]',
    'Can you tell me the [ADJ] [EC] of a [CHEM] reaction that has [...] as the [SUBSTRATE] and [PRODUCT]?',
]

EC_prediction_data_dict = {
    'sample_id': '',
    'id': '',
    'idx': '',
    'input': '',
    'output': '<EC> </EC>',
    'raw_input': '',
    'raw_output': '',
    'other_raw_outputs': '',
    'other_outputs': '',
    'split': '',
    'task': 'ec_prediction',
    'input_core_tag_left': '<SMILES>',
    'input_core_tag_right': '</SMILES>',
    'output_core_tag_left': '<EC>',
    'output_core_tag_right': '</EC>',
}

substrate_finetuning_prompt = "You are an expert chemist. Given the enzymatic reaction with the product in SMILES notation and the Enzyme Commission (EC) number, \
your task is to provide the substrate involved in the reaction in SMILES notation using your experienced biochemical reaction knowledge.\n\
Please strictly follow the format and provide the canonical SMILES string for the substrate. The string must be valid and chemically reasonable."

substrate_prompts = [
    'Please provide a [ADJ] [SUBSTRATE] corresponding to the [CHEM] reaction involving these [PRODUCT] and [EC]: [...]',
    '[...] Based on the [CHEM] reaction using the [PRODUCT] and [EC] given above, suggest a [ADJ] [SUBSTRATE].',
    'Based on the following [PRODUCT] and [EC], suggest a [ADJ] [SUBSTRATE] for the [CHEM] reaction. [...]',
    'Based on the given [PRODUCT] and [EC]: [...] what [SUBSTRATE] could potentially be assigned to the [CHEM] reaction?',
    'Using [...] as the [PRODUCT] and [EC], tell me the [ADJ] [SUBSTRATE].',
    '[...] Given the above [PRODUCT] and [EC], what could be a [ADJ] [SUBSTRATE] for this [CHEM] reaction?',
    'Predict the [SUBSTRATE] of a [CHEM] reaction with [...] as the [PRODUCT] and [EC].',
    'Propose a [ADJ] [SUBSTRATE] given these [PRODUCT] and [EC]. [...]',
    'Can you tell me the [ADJ] [SUBSTRATE] of a [CHEM] reaction that uses [...] as the [PRODUCT] and [EC]?',
    'Consider a [CHEM] reaction where [...] are the [PRODUCT] and [EC], what can be the [SUBSTRATE]?',
    '[...] Considering the given [PRODUCT] and [EC], what might be the [ADJ] [SUBSTRATE] of this [CHEM] reaction?',
    'Given the following [PRODUCT] and [EC], please provide a [ADJ] [SUBSTRATE]. [...]',
    'Predict a [ADJ] [SUBSTRATE] from the listed [PRODUCT] and [EC]. [...]',
    'Can you tell me the [ADJ] [SUBSTRATE] of a [CHEM] reaction that has [...] as the [PRODUCT] and [EC]?',
]

substrate_prediction_data_dict = {
    'sample_id': '',
    'id': '',
    'idx': '',
    'input': '',
    'output': '<SMILES>  </SMILES>',
    'raw_input': '',
    'raw_output': '',
    'other_raw_outputs': '',
    'other_outputs': '',
    'split': '',
    'task': 'retrosynthesis',
    'input_core_tag_left': '<SMILES|EC>',
    'input_core_tag_right': '</SMILES|EC>',
    'output_core_tag_left': '<SMILES>',
    'output_core_tag_right': '</SMILES>',
}

product_finetuning_prompt = "You are an expert chemist. Given the enzymatic reaction with the substrate in SMILES notation and the Enzyme Commission (EC) number, \
your task is to provide the product involved in the reaction in SMILES notation, using your experienced biochemical reaction knowledge. \n\
Please strictly follow the format and provide the canonical SMILES string for the product. The string must be valid and chemically reasonable."

product_prompts = [
    'Please provide a [ADJ] [PRODUCT] corresponding to the [CHEM] reaction involving these [SUBSTRATE] and [EC]: [...]',
    '[...] Based on the [CHEM] reaction using the [SUBSTRATE] and [EC] given above, suggest a [ADJ] [PRODUCT].',
    'Based on the following [SUBSTRATE] and [EC], suggest a [ADJ] [PRODUCT] for the [CHEM] reaction. [...]',
    'Based on the given [SUBSTRATE] and [EC]: [...] what [PRODUCT] could potentially come out from the [CHEM] reaction?',
    'Using [...] as the [SUBSTRATE] and [EC], tell me the [ADJ] [PRODUCT].',
    '[...] Given the above [SUBSTRATE] and [EC], what could be a [ADJ] [PRODUCT] for this [CHEM] reaction?',
    'Predict the [PRODUCT] of a [CHEM] reaction with [...] as the [SUBSTRATE] and [EC].',
    'Propose a [ADJ] [PRODUCT] given these [SUBSTRATE] and [EC]. [...]',
    'Can you tell me the [ADJ] [PRODUCT] of a [CHEM] reaction that uses [...] as the [SUBSTRATE] and [EC]?',
    'Consider a [CHEM] reaction where [...] are the [SUBSTRATE] and [EC] associated to it, what can be the [PRODUCT]?',
    '[...] Considering the given [SUBSTRATE] and [EC], what might be the [ADJ] [PRODUCT] of this [CHEM] reaction?',
    'Given the following [SUBSTRATE] and [EC], please provide a [ADJ] [PRODUCT]. [...]',
    'Predict a [ADJ] [PRODUCT] from the listed [SUBSTRATE] and [EC]. [...]',
    'Can you tell me the [ADJ] [PRODUCT] of a [CHEM] reaction that has [...] as the [SUBSTRATE] and [EC]?',
]

product_prediction_data_dict = {
    'sample_id': '',
    'id': '',
    'idx': '',
    'input': '',
    'output': '<SMILES>  </SMILES>',
    'raw_input': '',
    'raw_output': '',
    'other_raw_outputs': '',
    'other_outputs': '',
    'split': '',
    'task': 'forward_synthesis',
    'input_core_tag_left': '<SMILES|EC>',
    'input_core_tag_right': '</SMILES|EC>',
    'output_core_tag_left': '<SMILES>',
    'output_core_tag_right': '</SMILES>',
}    