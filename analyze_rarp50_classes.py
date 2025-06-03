#!/usr/bin/env python3
"""
RARP-50 Class Imbalance Analysis - Fixed Visualization
Standalone script to analyze class distribution with improved plot formatting
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

def analyze_rarp50_class_imbalance(data_root='./data'):
    """
    Analyze class distribution in RARP-50 dataset from action_discrete.txt files
    
    Args:
        data_root: Path to data directory containing RARP50_train and RARP50_test folders
    """
    
    print("RARP-50 Class Distribution Analysis")
    print("=" * 60)
    print(f"Data root: {os.path.abspath(data_root)}")
    
    # RARP-50 action class definitions
    action_classes = {
        0: "Other",
        1: "Picking-up the needle",
        2: "Positioning the needle tip", 
        3: "Pushing the needle through the tissue",
        4: "Pulling the needle out of the tissue",
        5: "Tying a knot",
        6: "Cutting the suture",
        7: "Returning/dropping the needle"
    }
    
    # Shortened names for visualization
    short_action_names = {
        0: "Other",
        1: "Pick needle",
        2: "Position needle",
        3: "Push needle",
        4: "Pull needle",
        5: "Tie knot",
        6: "Cut suture",
        7: "Drop needle"
    }
    
    # Look for train and test directories
    possible_train_dirs = ['RARP50_train', 'rarp50_train', 'train']
    possible_test_dirs = ['RARP50_test', 'rarp50_test', 'test']
    
    train_dir = None
    test_dir = None
    
    # Find train directory
    for train_name in possible_train_dirs:
        candidate_path = os.path.join(data_root, train_name)
        if os.path.exists(candidate_path):
            train_dir = candidate_path
            break
    
    # Find test directory  
    for test_name in possible_test_dirs:
        candidate_path = os.path.join(data_root, test_name)
        if os.path.exists(candidate_path):
            test_dir = candidate_path
            break
    
    # Debug: Show what directories are available
    print(f"\nAvailable directories in {data_root}:")
    try:
        for item in os.listdir(data_root):
            item_path = os.path.join(data_root, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}")
                # Show subdirectories for potential train/test folders
                try:
                    subdirs = [x for x in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, x))]
                    if subdirs:
                        print(f"     Contains: {subdirs[:5]}{'...' if len(subdirs) > 5 else ''}")
                except:
                    pass
            else:
                print(f"  📄 {item}")
    except Exception as e:
        print(f"  Error listing directory: {e}")
    
    # Check what we found
    if train_dir:
        print(f"\n✅ Found training data: {train_dir}")
    else:
        print(f"\n❌ No training directory found. Looked for: {possible_train_dirs}")
        
    if test_dir:
        print(f"✅ Found test data: {test_dir}")
    else:
        print(f"❌ No test directory found. Looked for: {possible_test_dirs}")
    
    if not train_dir and not test_dir:
        print("\n❌ ERROR: No valid data directories found!")
        print("Please check your data path and directory structure.")
        return None
    
    # Collect labels from available sets
    all_results = {}
    
    if train_dir:
        print(f"\n🔍 Analyzing training set...")
        train_labels = collect_labels_from_videos(train_dir)
        if train_labels:
            train_analysis = analyze_class_distribution(train_labels, "Training Set", action_classes)
            all_results['train'] = train_analysis
        else:
            print("❌ No training labels found!")
    
    if test_dir:
        print(f"\n🔍 Analyzing test set...")
        test_labels = collect_labels_from_videos(test_dir)
        if test_labels:
            test_analysis = analyze_class_distribution(test_labels, "Test Set", action_classes)
            all_results['test'] = test_analysis
        else:
            print("❌ No test labels found!")
    
    # Combined analysis
    if len(all_results) == 2:  # Both train and test
        combined_labels = train_labels + test_labels
        combined_analysis = analyze_class_distribution(combined_labels, "Combined Dataset", action_classes)
        all_results['combined'] = combined_analysis
    elif len(all_results) == 1:  # Only one dataset available
        combined_analysis = list(all_results.values())[0]
        all_results['combined'] = combined_analysis
    else:
        print("❌ ERROR: No valid datasets found!")
        return None
    
    # Calculate imbalance metrics
    imbalance_metrics = calculate_imbalance_metrics(combined_analysis)
    
    # Create visualizations with improved formatting
    create_comprehensive_plots_fixed(all_results, action_classes, short_action_names)
    
    # Generate detailed report
    generate_analysis_report(all_results, imbalance_metrics, action_classes)
    
    return {
        'results': all_results,
        'imbalance_metrics': imbalance_metrics,
        'action_classes': action_classes
    }

def collect_labels_from_videos(split_dir):
    """
    Collect all action labels from video directories in a split
    
    Args:
        split_dir: Path to train or test split directory (e.g., data/RARP50_train)
    
    Returns:
        List of all action labels found
    """
    all_labels = []
    video_count = 0
    
    print(f"Scanning directory: {split_dir}")
    
    if not os.path.exists(split_dir):
        print(f"  Error: Directory {split_dir} does not exist!")
        return all_labels
    
    # Iterate through all video_xx directories
    for item in os.listdir(split_dir):
        video_path = os.path.join(split_dir, item)
        
        # Check if it's a directory and matches video_xx pattern
        if os.path.isdir(video_path):
            action_file = os.path.join(video_path, 'action_discrete.txt')
            
            if os.path.exists(action_file):
                print(f"  Processing: {item}/action_discrete.txt")
                video_labels = read_action_discrete_file(action_file)
                all_labels.extend(video_labels)
                video_count += 1
                
                if video_count % 5 == 0:  # More frequent updates
                    print(f"    Progress: {video_count} videos processed, {len(all_labels)} labels collected")
            else:
                print(f"  Warning: No action_discrete.txt found in {video_path}")
                # List what files are actually in this directory
                try:
                    files_in_dir = os.listdir(video_path)
                    print(f"    Files found: {files_in_dir}")
                except:
                    print(f"    Could not list contents of {video_path}")
        else:
            print(f"  Skipping non-directory: {item}")
    
    print(f"  Final count: {video_count} videos processed, {len(all_labels)} total labels")
    return all_labels

def read_action_discrete_file(file_path):
    """
    Read action labels from action_discrete.txt file
    
    Args:
        file_path: Path to action_discrete.txt file
    
    Returns:
        List of action labels (integers 0-7)
    """
    labels = []
    
    try:
        with open(file_path, 'r') as f:
            # Read as CSV (comma-separated)
            reader = csv.reader(f)
            
            for row_num, row in enumerate(reader):
                if len(row) >= 2:  # Should have timestamp, action_label
                    try:
                        # Second column is the action label
                        action_label = int(float(row[1]))  # Convert via float in case of decimals
                        
                        # Validate label is in expected range
                        if 0 <= action_label <= 7:
                            labels.append(action_label)
                        else:
                            print(f"    Warning: Invalid label {action_label} in {file_path} row {row_num}")
                    
                    except (ValueError, IndexError) as e:
                        print(f"    Warning: Could not parse row {row_num} in {file_path}: {row}")
                        continue
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    return labels

def analyze_class_distribution(labels, dataset_name, action_classes):
    """
    Analyze the distribution of action labels
    
    Args:
        labels: List of action labels
        dataset_name: Name for this dataset (e.g., "Training Set")
        action_classes: Dictionary mapping class IDs to names
    
    Returns:
        Dictionary with analysis results
    """
    
    if not labels:
        print(f"No labels found for {dataset_name}")
        return None
    
    # Count label occurrences
    label_counts = Counter(labels)
    
    # Ensure all classes 0-7 are represented (even with 0 count)
    for class_id in range(8):
        if class_id not in label_counts:
            label_counts[class_id] = 0
    
    # Calculate statistics
    total_samples = len(labels)
    class_percentages = {k: (v/total_samples)*100 for k, v in label_counts.items()}
    
    # Sort by class ID for consistent output
    sorted_counts = dict(sorted(label_counts.items()))
    sorted_percentages = dict(sorted(class_percentages.items()))
    
    analysis = {
        'dataset_name': dataset_name,
        'total_samples': total_samples,
        'class_counts': sorted_counts,
        'class_percentages': sorted_percentages,
        'most_common': label_counts.most_common(1)[0],
        'least_common': [item for item in label_counts.most_common() if item[1] > 0][-1]
    }
    
    # Print detailed breakdown
    print(f"\n{dataset_name} Analysis:")
    print(f"{'='*50}")
    print(f"Total samples: {total_samples:,}")
    print(f"\nClass Distribution:")
    print(f"{'Class':<6} {'Action Name':<35} {'Count':<10} {'Percentage':<12}")
    print(f"{'-'*70}")
    
    for class_id in sorted(label_counts.keys()):
        class_name = action_classes.get(class_id, f"Unknown-{class_id}")
        count = label_counts[class_id]
        percentage = class_percentages[class_id]
        print(f"{class_id:<6} {class_name:<35} {count:<10,} {percentage:<12.2f}%")
    
    return analysis

def calculate_imbalance_metrics(analysis):
    """
    Calculate various class imbalance metrics
    
    Args:
        analysis: Analysis results from analyze_class_distribution
    
    Returns:
        Dictionary with imbalance metrics
    """
    
    if not analysis:
        return None
    
    counts = list(analysis['class_counts'].values())
    positive_counts = [c for c in counts if c > 0]  # Remove zero counts
    
    if not positive_counts:
        return None
    
    max_count = max(positive_counts)
    min_count = min(positive_counts)
    mean_count = np.mean(positive_counts)
    std_count = np.std(positive_counts)
    
    # Calculate various imbalance metrics
    metrics = {
        'imbalance_ratio': max_count / min_count if min_count > 0 else float('inf'),
        'coefficient_of_variation': std_count / mean_count if mean_count > 0 else 0,
        'gini_coefficient': calculate_gini_coefficient(positive_counts),
        'max_count': max_count,
        'min_count': min_count,
        'mean_count': mean_count,
        'std_count': std_count,
        'zero_classes': sum(1 for c in counts if c == 0)
    }
    
    # Interpret severity
    ratio = metrics['imbalance_ratio']
    if ratio < 2:
        severity = "Minimal"
        color = "🟢"
    elif ratio < 5:
        severity = "Moderate"
        color = "🟡"
    elif ratio < 10:
        severity = "High"
        color = "🟠"
    else:
        severity = "Extreme"
        color = "🔴"
    
    metrics['severity'] = severity
    metrics['severity_color'] = color
    
    print(f"\n{'Class Imbalance Analysis:'}")
    print(f"{'='*40}")
    print(f"Imbalance Ratio (max/min): {ratio:.2f}:1")
    print(f"Coefficient of Variation: {metrics['coefficient_of_variation']:.3f}")
    print(f"Gini Coefficient: {metrics['gini_coefficient']:.3f}")
    print(f"Classes with zero samples: {metrics['zero_classes']}")
    print(f"Imbalance Severity: {color} {severity}")
    
    return metrics

def calculate_gini_coefficient(counts):
    """Calculate Gini coefficient for measuring inequality"""
    sorted_counts = sorted(counts)
    n = len(sorted_counts)
    cumsum = np.cumsum(sorted_counts)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

def create_comprehensive_plots_fixed(results, action_classes, short_action_names):
    """
    Create comprehensive visualizations with improved formatting to prevent overlapping text
    
    Args:
        results: Dictionary with analysis results for different splits
        action_classes: Dictionary mapping class IDs to names
        short_action_names: Dictionary with shortened action names for better display
    """
    
    # Create larger figure with more space
    fig = plt.figure(figsize=(20, 14))
    
    # Create a more flexible grid layout
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 2, 1.5], width_ratios=[1.2, 1, 1], 
                          hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle('RARP-50 Dataset: Class Distribution Analysis', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Get main analysis data
    main_analysis = results.get('combined', list(results.values())[0])
    
    # Plot 1: Main class distribution bar chart (larger, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    plot_class_distribution_bar_fixed(main_analysis, short_action_names, ax1)
    
    # Plot 2: Improved heatmap
    ax2 = fig.add_subplot(gs[0, 2])
    plot_imbalance_heatmap_fixed(main_analysis, short_action_names, ax2)
    
    # Plot 3: Better pie chart with external labels
    ax3 = fig.add_subplot(gs[1, 0])
    plot_distribution_pie_fixed(main_analysis, short_action_names, ax3)
    
    # Plot 4: Train vs Test comparison or cumulative
    ax4 = fig.add_subplot(gs[1, 1:])
    if 'train' in results and 'test' in results:
        plot_train_test_comparison_fixed(results['train'], results['test'], short_action_names, ax4)
    else:
        plot_cumulative_distribution_fixed(main_analysis, short_action_names, ax4)
    
    # Plot 5: Summary statistics table
    ax5 = fig.add_subplot(gs[2, :])
    plot_summary_table(main_analysis, action_classes, ax5)
    
    plt.savefig('rarp50_class_analysis_fixed.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"\nImproved visualization saved as 'rarp50_class_analysis_fixed.png'")
    plt.show()

def plot_class_distribution_bar_fixed(analysis, short_names, ax):
    """Plot class distribution with improved formatting"""
    
    class_ids = list(analysis['class_counts'].keys())
    counts = list(analysis['class_counts'].values())
    
    # Create color gradient
    colors = plt.cm.viridis(np.linspace(0.2, 1.0, len(counts)))
    
    bars = ax.bar(class_ids, counts, color=colors, edgecolor='white', 
                  linewidth=1, alpha=0.8)
    
    ax.set_title(f'{analysis["dataset_name"]} - Class Distribution\n'
                f'Total: {analysis["total_samples"]:,} samples', 
                fontweight='bold', fontsize=14, pad=20)
    ax.set_xlabel('Action Class ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    
    # Set x-axis with class names
    ax.set_xticks(class_ids)
    ax.set_xticklabels([f'C{cid}\n{short_names[cid]}' for cid in class_ids], 
                       fontsize=10, rotation=0)
    
    # Add value labels on bars with better positioning
    max_height = max(counts)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        if height > 0:
            # Position label above bar
            ax.text(bar.get_x() + bar.get_width()/2., height + max_height*0.02,
                    f'{count:,}', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
    
    # Improve grid and styling
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set y-axis to start from 0 and add some padding at top
    ax.set_ylim(0, max_height * 1.15)

def plot_imbalance_heatmap_fixed(analysis, short_names, ax):
    """Plot improved heatmap with better labels"""
    
    counts = list(analysis['class_counts'].values())
    # Reshape into 2x4 grid
    counts_matrix = np.array(counts).reshape(2, 4)
    
    # Create custom labels for heatmap
    labels_matrix = np.array([[f'{short_names[i]}\n({counts[i]:,})' 
                             for i in range(j*4, (j+1)*4)] 
                            for j in range(2)])
    
    # Use a better colormap
    sns.heatmap(counts_matrix, annot=labels_matrix, fmt='', 
                cmap='Reds', square=True,
                xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'], 
                yticklabels=['Classes 0-3', 'Classes 4-7'],
                ax=ax, cbar_kws={'label': 'Sample Count', 'shrink': 0.8},
                annot_kws={'fontsize': 9, 'ha': 'center'})
    
    ax.set_title('Class Frequency Heatmap', fontweight='bold', fontsize=12)
    
    # Rotate x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

def plot_distribution_pie_fixed(analysis, short_names, ax):
    """Plot pie chart with external labels to prevent overlapping"""
    
    counts = list(analysis['class_counts'].values())
    class_ids = list(analysis['class_counts'].keys())
    
    # Only show classes with non-zero counts
    non_zero_data = [(cid, count) for cid, count in zip(class_ids, counts) if count > 0]
    non_zero_ids, non_zero_counts = zip(*non_zero_data)
    
    # Create labels with class info and percentages
    total = sum(non_zero_counts)
    labels = [f'C{cid}: {short_names[cid]}' for cid in non_zero_ids]
    
    # Use distinct colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(non_zero_counts)))
    
    # Create pie chart with external labels
    wedges, texts, autotexts = ax.pie(non_zero_counts, 
                                     labels=None,  # Don't use labels parameter
                                     autopct='%1.1f%%',
                                     colors=colors, 
                                     startangle=90,
                                     pctdistance=0.85,
                                     explode=[0.05]*len(non_zero_counts))  # Small explosion for separation
    
    # Add legend outside the pie
    ax.legend(wedges, labels, title="Action Classes", 
              loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=9)
    
    ax.set_title('Class Distribution\n(Non-zero classes only)', 
                 fontweight='bold', fontsize=12, pad=20)
    
    # Improve autotext appearance
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)

def plot_train_test_comparison_fixed(train_analysis, test_analysis, short_names, ax):
    """Compare train and test distributions with better formatting"""
    
    class_ids = list(train_analysis['class_counts'].keys())
    train_percentages = [train_analysis['class_percentages'][cid] for cid in class_ids]
    test_percentages = [test_analysis['class_percentages'][cid] for cid in class_ids]
    
    x = np.arange(len(class_ids))
    width = 0.35
    
    # Create grouped bar chart
    bars1 = ax.bar(x - width/2, train_percentages, width, 
                   label='Train', alpha=0.8, color='steelblue', edgecolor='white')
    bars2 = ax.bar(x + width/2, test_percentages, width, 
                   label='Test', alpha=0.8, color='coral', edgecolor='white')
    
    ax.set_title('Train vs Test Distribution Comparison', 
                 fontweight='bold', fontsize=12, pad=15)
    ax.set_xlabel('Action Class ID', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{cid}\n{short_names[cid][:8]}...' if len(short_names[cid]) > 8 
                       else f'C{cid}\n{short_names[cid]}' for cid in class_ids], 
                       fontsize=9)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0.5:  # Only label if percentage is significant
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', 
                        fontsize=8, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_cumulative_distribution_fixed(analysis, short_names, ax):
    """Plot cumulative distribution with better formatting"""
    
    counts = sorted(analysis['class_counts'].values(), reverse=True)
    cumulative = np.cumsum(counts) / sum(counts) * 100
    
    ax.plot(range(1, len(counts)+1), cumulative, 'o-', 
            linewidth=3, markersize=8, color='darkblue', 
            markerfacecolor='lightblue', markeredgecolor='darkblue')
    
    ax.set_title('Cumulative Class Distribution', fontweight='bold', fontsize=12)
    ax.set_xlabel('Class Rank (by frequency)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=11, fontweight='bold')
    
    # Add reference lines
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.7, 
               label='80% threshold', linewidth=2)
    ax.axhline(y=95, color='orange', linestyle='--', alpha=0.7, 
               label='95% threshold', linewidth=2)
    
    # Add grid and styling
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_summary_table(analysis, action_classes, ax):
    """Create a summary statistics table"""
    
    # Calculate key statistics
    counts = list(analysis['class_counts'].values())
    max_count = max(counts)
    min_count_nonzero = min([c for c in counts if c > 0])
    imbalance_ratio = max_count / min_count_nonzero if min_count_nonzero > 0 else float('inf')
    
    # Create table data
    table_data = [
        ['Total Samples', f"{analysis['total_samples']:,}"],
        ['Number of Classes', '8'],
        ['Most Common Class', f"Class {analysis['most_common'][0]} ({analysis['most_common'][1]:,} samples)"],
        ['Least Common Class', f"Class {analysis['least_common'][0]} ({analysis['least_common'][1]:,} samples)"],
        ['Imbalance Ratio', f"{imbalance_ratio:.2f}:1"],
        ['Classes with Zero Samples', f"{sum(1 for c in counts if c == 0)}"]
    ]
    
    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=['Metric', 'Value'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.4, 0.6])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows with alternating colors
    for i in range(1, len(table_data) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    ax.set_title('Dataset Summary Statistics', fontweight='bold', fontsize=12, pad=20)
    ax.axis('off')

def generate_analysis_report(results, imbalance_metrics, action_classes):
    """
    Generate a detailed analysis report for the MEng thesis
    
    Args:
        results: Dictionary with analysis results
        imbalance_metrics: Imbalance metrics
        action_classes: Class definitions
    """
    
    report_filename = 'rarp50_class_imbalance_report.txt'
    
    with open(report_filename, 'w') as f:
        f.write("RARP-50 SURGICAL ACTION DATASET - CLASS IMBALANCE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 20 + "\n")
        if imbalance_metrics:
            f.write(f"Class Imbalance Severity: {imbalance_metrics['severity']}\n")
            f.write(f"Imbalance Ratio: {imbalance_metrics['imbalance_ratio']:.2f}:1\n")
            f.write(f"Most vs Least Common Class Ratio: {imbalance_metrics['max_count']:,} vs {imbalance_metrics['min_count']:,}\n\n")
        
        # Dataset overview
        if 'combined' in results:
            analysis = results['combined']
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 15 + "\n")
            f.write(f"Total Samples: {analysis['total_samples']:,}\n")
            f.write(f"Number of Classes: {len(action_classes)}\n")
            f.write(f"Most Common Action: Class {analysis['most_common'][0]} - {action_classes[analysis['most_common'][0]]} ({analysis['most_common'][1]:,} samples)\n")
            f.write(f"Least Common Action: Class {analysis['least_common'][0]} - {action_classes[analysis['least_common'][0]]} ({analysis['least_common'][1]:,} samples)\n\n")
        
        # Detailed breakdown for each split
        for split_name, analysis in results.items():
            if analysis:
                f.write(f"{analysis['dataset_name'].upper()} DETAILED BREAKDOWN\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'Class':<5} {'Action Name':<35} {'Count':<10} {'Percentage':<10}\n")
                f.write("-" * 70 + "\n")
                
                for class_id, count in analysis['class_counts'].items():
                    class_name = action_classes[class_id]
                    percentage = analysis['class_percentages'][class_id]
                    f.write(f"{class_id:<5} {class_name:<35} {count:<10,} {percentage:<10.2f}%\n")
                f.write("\n")
        
        # Imbalance metrics
        if imbalance_metrics:
            f.write("IMBALANCE METRICS\n")
            f.write("-" * 17 + "\n")
            f.write(f"Imbalance Ratio (max/min): {imbalance_metrics['imbalance_ratio']:.2f}\n")
            f.write(f"Coefficient of Variation: {imbalance_metrics['coefficient_of_variation']:.3f}\n")
            f.write(f"Gini Coefficient: {imbalance_metrics['gini_coefficient']:.3f}\n")
            f.write(f"Standard Deviation: {imbalance_metrics['std_count']:.1f}\n")
            f.write(f"Classes with Zero Samples: {imbalance_metrics['zero_classes']}\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS FOR MENG PROJECT\n")
        f.write("-" * 35 + "\n")
        
        if imbalance_metrics and imbalance_metrics['imbalance_ratio'] > 10:
            f.write("1. CRITICAL IMBALANCE DETECTED - Use aggressive balancing techniques:\n")
            f.write("   - Implement focal loss or class-balanced loss\n")
            f.write("   - Use oversampling for minority classes (SMOTE)\n")
            f.write("   - Consider class-weighted training\n")
            f.write("   - Monitor per-class metrics (precision, recall, F1)\n\n")
        elif imbalance_metrics and imbalance_metrics['imbalance_ratio'] > 5:
            f.write("1. SIGNIFICANT IMBALANCE - Implement moderate balancing:\n")
            f.write("   - Use class-weighted loss function\n")
            f.write("   - Consider stratified sampling\n")
            f.write("   - Monitor minority class performance\n\n")
        else:
            f.write("1. BALANCED DATASET - Standard training approaches suitable:\n")
            f.write("   - Regular cross-entropy loss should work well\n")
            f.write("   - Focus on other aspects like temporal modeling\n\n")
        
        f.write("2. EVALUATION STRATEGY:\n")
        f.write("   - Use macro-averaged F1 score for overall performance\n")
        f.write("   - Report per-class precision, recall, and F1 scores\n")
        f.write("   - Create confusion matrix for detailed analysis\n")
        f.write("   - Consider surgical-specific metrics (action transition accuracy)\n\n")
        
        f.write("3. TECHNICAL IMPLEMENTATION:\n")
        f.write("   - Log per-class metrics during training\n")
        f.write("   - Implement early stopping based on minority class performance\n")
        f.write("   - Use stratified validation splits\n")
        f.write("   - Consider ensemble methods for robust predictions\n\n")
    
    print(f"\nDetailed analysis report saved as '{report_filename}'")


# Main execution
if __name__ == "__main__":
    import sys
    
    # Allow command line argument for data path
    data_path = sys.argv[1] if len(sys.argv) > 1 else './data'
    
    print("Starting RARP-50 Class Imbalance Analysis...")
    print(f"Data path: {data_path}")
    
    # Run the analysis
    results = analyze_rarp50_class_imbalance(data_path)
    
    if results:
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("\nFiles generated:")
        print("1. rarp50_class_analysis_fixed.png - Improved visualization plots")
        print("2. rarp50_class_imbalance_report.txt - Detailed report")
        print("\nUse these results in your MEng report to justify your technical choices!")