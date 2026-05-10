import torch
import json
import os
import sys

# Add the project root to sys.path to allow importing CalculusSolver
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from CalculusSolver.training.model_trainer import SLaNgTokenizer, Seq2SeqTransformer

# Fix for joblib unpickling issue
import sys
import CalculusSolver.training.model_trainer
sys.modules['__main__'].SLaNgTokenizer = SLaNgTokenizer
sys.modules['__main__'].Seq2SeqTransformer = Seq2SeqTransformer

from CalculusSolver.inference.inference_engine import CalculusInference
from CalculusSolver.data_pipeline.data_generator import SLaNgConverter
import sympy as sp

def check_accuracy(dataset_path="CalculusSolver/data/dataset.json", num_samples=100):
    """
    Checks the accuracy of the model by comparing its predictions 
    against the ground truth in the dataset.
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    with open(dataset_path, "r") as f:
        data = json.load(f)
    
    # Use a subset for faster evaluation if needed
    samples = data[:num_samples]
    
    try:
        engine = CalculusInference()
    except Exception as e:
        print(f"Error initializing inference engine: {e}")
        return

    correct = 0
    total = len(samples)
    
    print(f"Evaluating {total} samples...\n")
    
    for i, item in enumerate(samples):
        problem = item['input']
        expected = item['output']
        
        try:
            print(f"Solving sample {i+1}: {problem}")
            predicted = engine.solve(problem)
            print(f"Result: {predicted}")
            
            # Simple string comparison (can be improved with symbolic comparison)
            is_correct = predicted.strip() == expected.strip()
            
            if is_correct:
                correct += 1
            
            # Print progress every 10 samples
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{total}...")
                
        except Exception as e:
            print(f"Error processing problem '{problem}': {e}")

    accuracy = (correct / total) * 100
    print(f"\nEvaluation Results:")
    print(f"Total Samples: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    # You can increase num_samples for a more comprehensive check
    check_accuracy(num_samples=50)
