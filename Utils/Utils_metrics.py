import numpy as np
from rdkit import Chem
import re

def compute_accuracy(confusion_matrix):
    """
    Computes the accuracy from a confusion matrix.

    Parameters:
    confusion_matrix (np.array): A square confusion matrix where rows represent actual classes 
                                 and columns represent predicted classes.

    Returns:
    float: The accuracy of the predictions.
    """
    # Sum of diagonal elements represents correct predictions
    correct_predictions = np.trace(confusion_matrix)
    
    # Total predictions are the sum of all elements in the matrix
    total_predictions = np.sum(confusion_matrix)
    
    # Compute accuracy
    accuracy = correct_predictions / total_predictions
    
    return accuracy



def compute_f1_from_confusion_matrix(conf_matrix):
    N = conf_matrix.shape[0]  # Number of classes
    f1_scores = []
    precisions = []
    recalls = []
    

    for i in range(N):
        TP = conf_matrix[i, i]
        FP = np.sum(conf_matrix[:, i]) - TP
        FN = np.sum(conf_matrix[i, :]) - TP
        
        # Avoid division by zero
        if (TP + FP) == 0 or (TP + FN) == 0:
            precision, recall = 0, 0
        else:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
        
        if (precision + recall) == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)

    # Macro F1 score (unweighted average of individual F1 scores)
    f1_macro = np.mean(f1_scores)

    return f1_macro, f1_scores, precisions, recalls


def evaluate_smiles_prediction(prediction, ground_truth, results_dict, other_outputs = None):
    
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
            print('entering other truths')
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



def evaluate(data, zero_shot=True, task= 'ec'):
    if zero_shot:
        patterns = {
            'ec': re.compile(r'\d+\.\d+\.\d+\.\d+'),  # EC number: X.X.X.X
            'substrate': re.compile(r'([A-Za-z0-9@+\-\[\]\(\)\\/=#$]+)'),  # General SMILES regex
            'product': re.compile(r'([A-Za-z0-9@+\-\[\]\(\)\\/=#$]+)')  # General SMILES regex
        }
    else:
        patterns = {
            'ec': re.compile(r'<EC>\s*(\d+\.\d+\.\d+\.\d+)\s*</EC>'),
            'substrate': re.compile(r'<SMILES>\s*(.*?)\s*</SMILES>'),
            'product': re.compile(r'<SMILES>\s*(.*?)\s*</SMILES>')
        }
    
    pattern = patterns.get(task)
    all_invalids = []
    for exp in data:
        evaluation_results = {
            "canonical_match": 0,
            "noncanonical_match": 0,
            "canonical_valid": 0,
            "noncanonical_valid": 0,
            "invalid": 0
        }
        invalids = 0
        for pred in exp:    
            try:
                print(pattern, pred)
                target = re.findall(pattern, pred)[0]
                if task == 'substrate' or task == 'product':
                    
                    mol_pred = Chem.MolFromSmiles(target)
                    print(mol_pred)
                    # Check if the prediction is invalid
                    if mol_pred is None:
                        evaluation_results["invalid"] += 1

                
            except:
                invalids +=1
        if task == 'ec':
            all_invalids.append(invalids/len(exp)*100)
        elif task == 'substrate' or task == 'product':
            all_invalids.append(evaluation_results["invalid"]/len(exp)*100)
    return all_invalids



