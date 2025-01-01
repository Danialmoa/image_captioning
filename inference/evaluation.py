import torch
from torchmetrics.text import BLEUScore
import pandas as pd
import numpy as np
from tqdm import tqdm


def evaluate_captions(true_captions, predicted_captions):
    """
    Evaluate captions using BLEU score from torchmetrics
    """
    # Initialize BLEU score metric
    bleu = BLEUScore(n_gram=1)  # Using up to 4-grams
    
    # Calculate individual BLEU scores for analysis
    individual_scores = []
    
    for true, pred in tqdm(zip(true_captions, predicted_captions), total=len(true_captions)):
        # Convert single caption to list format required by torchmetrics
        true_list = [true]
        pred_list = [pred]
        
        # Calculate individual score
        score = bleu(pred_list, [true_list])
        individual_scores.append(score.item())
    
    # Calculate overall BLEU score
    overall_bleu = bleu(predicted_captions, [[cap] for cap in true_captions])
    
    return individual_scores, overall_bleu.item()

def post_process_caption(caption, type="prediction"):
    """
    Apply post-processing to the generated caption
    """
    # Ensure there's a period at the end
    if not caption.endswith('.'):
        caption = caption + '.'
        
    # Drop <pad> , <end> , <start>
    caption = caption.replace('<pad>', '').replace('<end>', '').replace('<start>', '')
        
    if type == "prediction":
        # Remove duplicate consecutive words
        words = caption.split()
        words = [word for word in words if words.count(word) == 1]
        caption = ' '.join(words)
    
    return caption

def analyze_results(true_captions, predicted_captions, model_name):
    """
    Analyze and display caption generation results
    """
    # Apply post-processing to predictions
    processed_predictions = [post_process_caption(cap, type="prediction") for cap in predicted_captions]
    processed_true_captions = [post_process_caption(cap, type="true") for cap in true_captions]
    
    # Calculate BLEU scores
    individual_scores, overall_bleu = evaluate_captions(processed_true_captions, processed_predictions)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'True Caption': processed_true_captions,
        'Predicted Caption': processed_predictions,
        'BLEU Score': individual_scores
    })
    
    # Sort by BLEU score
    results_df = results_df.sort_values('BLEU Score')
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"Overall BLEU Score: {overall_bleu:.4f}")
    print(f"Average Individual BLEU Score: {np.mean(individual_scores):.4f}")
    print(f"Median BLEU Score: {np.median(individual_scores):.4f}")
    print(f"Standard Deviation: {np.std(individual_scores):.4f}")
    
    results_df.to_csv(f"models/checkpoints/{model_name}/results.csv", index=False)
    
    return results_df

if __name__ == "__main__":
    MODEL_NAME = "new_runs/run_11"
    
    data_path = f"models/checkpoints/{MODEL_NAME}/predictions.csv"
    results_df = pd.read_csv(data_path)
    true_captions = results_df["True Caption"].tolist()
    predicted_captions = results_df["Predicted Caption"].tolist()

    model_name = MODEL_NAME
    analyze_results(true_captions, predicted_captions, model_name)