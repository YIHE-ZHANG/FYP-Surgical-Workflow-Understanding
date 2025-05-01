import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import os

def analyze_rarp50_results(split_num=1):
    """
    Analyze the results of RARP50 testing
    
    Args:
        split_num: The split number used for testing
    """
    # Define the class names for RARP50
    class_names = [
        "Other",
        "Picking-up needle",
        "Positioning needle",
        "Pushing needle",
        "Pulling suture",
        "Tying knot",
        "Cutting suture",
        "Returning needle"
    ]
    
    # Load the test results
    results_dir = f'./prompt_test/rarp50/split{split_num}'
    
    try:
        final_act_1 = np.load(f'{results_dir}/final_act_1.npy')
        final_cnt_1 = np.load(f'{results_dir}/final_cnt_1.npy')
        gt_act = np.load(f'{results_dir}/gt_act.npy')
        
        print(f"Loaded data from {results_dir}")
        print(f"Predicted actions shape: {final_act_1.shape}")
        print(f"Predicted counts shape: {final_cnt_1.shape}")
        print(f"Ground truth shape: {gt_act.shape}")
        
        # If the ground truth is 2D (samples, multiple_actions), take the first action for each sample
        if len(gt_act.shape) > 1 and gt_act.shape[1] > 1:
            # Filter out -1 values (no action)
            valid_actions = gt_act >= 0
            first_valid_actions = np.zeros(gt_act.shape[0], dtype=int)
            
            for i in range(gt_act.shape[0]):
                valid_indices = np.where(valid_actions[i])[0]
                if len(valid_indices) > 0:
                    first_valid_actions[i] = gt_act[i, valid_indices[0]]
                else:
                    first_valid_actions[i] = -1
            
            gt_act_single = first_valid_actions
        else:
            gt_act_single = gt_act.flatten()
        
        # Get first predicted action for each sample
        pred_act_single = final_act_1[:, 0]
        
        # Calculate accuracy (only for valid actions, ignoring -1)
        valid_indices = gt_act_single >= 0
        accuracy = accuracy_score(gt_act_single[valid_indices], pred_act_single[valid_indices])
        print(f"Overall accuracy: {accuracy:.4f}")
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            gt_act_single[valid_indices], 
            pred_act_single[valid_indices],
            labels=range(len(class_names)),
            zero_division=0
        )
        
        # Print per-class metrics
        print("\nPer-class metrics:")
        print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
        print("-" * 60)
        for i in range(len(class_names)):
            print(f"{class_names[i]:<20} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}")
        
        # Create confusion matrix
        cm = confusion_matrix(
            gt_act_single[valid_indices], 
            pred_act_single[valid_indices],
            labels=range(len(class_names))
        )
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('RARP50 Confusion Matrix')
        plt.tight_layout()
        
        # Save confusion matrix
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/rarp50_confusion_matrix.png')
        print(f"Saved confusion matrix to results/rarp50_confusion_matrix.png")
        
        # Analyze count prediction accuracy
        predicted_counts = final_cnt_1.flatten()
        
        # Count the actual number of actions in each sample
        true_counts = np.zeros(gt_act.shape[0], dtype=int)
        for i in range(gt_act.shape[0]):
            true_counts[i] = np.sum(gt_act[i] >= 0)
        
        count_accuracy = accuracy_score(true_counts, predicted_counts)
        print(f"\nCount prediction accuracy: {count_accuracy:.4f}")
        
        # Save the results to a text file
        with open('results/rarp50_results.txt', 'w') as f:
            f.write(f"RARP50 Action Recognition Results (Split {split_num})\n")
            f.write(f"Overall accuracy: {accuracy:.4f}\n\n")
            
            f.write("Per-class metrics:\n")
            f.write(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}\n")
            f.write("-" * 60 + "\n")
            for i in range(len(class_names)):
                f.write(f"{class_names[i]:<20} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}\n")
            
            f.write(f"\nCount prediction accuracy: {count_accuracy:.4f}\n")
        
        print(f"Saved detailed results to results/rarp50_results.txt")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure you have run the test.py script first and results are saved in {results_dir}")

if __name__ == "__main__":
    analyze_rarp50_results()