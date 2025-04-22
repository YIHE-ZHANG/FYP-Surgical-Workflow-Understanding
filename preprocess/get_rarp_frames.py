import os
import argparse
import shutil
import csv
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='rarp50')
parser.add_argument('--train_path', default='./data/RARP50_train/')
parser.add_argument('--test_path', default='./data/RARP50_test/')
parser.add_argument('--output_base', default='./data/rarp50/')
args = parser.parse_args()

def read_timestamps(action_file):
    """Read timestamps from action_discrete.txt file"""
    timestamps = []
    with open(action_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            timestamps.append(row[0])  # First column contains the timestamp
    return timestamps

# Extract frames from videos in either train or test set
def extract_frames(input_path, output_base, is_train):
    # Create output directory if it doesn't exist
    subset_type = "train" if is_train else "test"
    output_path = os.path.join(output_base, subset_type)
    os.makedirs(output_path, exist_ok=True)
    
    # Find all video directories in the input path
    video_dirs = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d)) and d.startswith('video')]
    
    for video_dir in video_dirs:
        video_path = os.path.join(input_path, video_dir)
        video_file = os.path.join(video_path, 'video_left.avi')
        action_file = os.path.join(video_path, 'action_discrete.txt')
        
        if not os.path.exists(video_file):
            print(f"Video file not found in {video_path}")
            continue
            
        if not os.path.exists(action_file):
            print(f"Action file not found in {video_path}")
            continue
        
        # Create output directory for this video
        video_output_dir = os.path.join(output_path, video_dir)
        rgb_dir = os.path.join(video_output_dir, 'rgb')
        os.makedirs(rgb_dir, exist_ok=True)
        
        # Copy the action_discrete.txt file
        shutil.copy2(action_file, os.path.join(video_output_dir, 'action_discrete.txt'))
        
        # Read timestamps from action_discrete.txt
        timestamps = read_timestamps(action_file)
        print(f"Found {len(timestamps)} timestamps in {action_file}")
        
        # First approach: Try using ffmpeg with mapped frame numbers
        # This is a quick two-step approach that should work well
        
        # Step 1: Create a temporary folder for sequential frames
        temp_dir = os.path.join(video_output_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Extract all frames sequentially
        video_file_fixed = video_file.replace('\\', '/')
        temp_dir_fixed = temp_dir.replace('\\', '/')
        
        # Extract every 6th frame
        cmd = f'ffmpeg -i "{video_file_fixed}" -vsync 0 -vf "select=\'not(mod(n,6))\',setpts=N/TB" "{temp_dir_fixed}/%09d.jpg"'
        print(cmd)
        os.system(cmd)
        
        # Step 2: Rename the frames to match the timestamps
        # The timestamps in action_discrete.txt are 000000000, 000000006, 000000012, etc.
        # The extracted frames are named 000000001, 000000002, 000000003, etc.
        
        # Create a mapping between sequential numbers and timestamps
        for i, timestamp in enumerate(timestamps, 1):
            try:
                # Source file (sequential number)
                src_file = os.path.join(temp_dir, f"{i:09d}.jpg")
                # Destination file (timestamp)
                dst_file = os.path.join(rgb_dir, f"{timestamp}.jpg")
                
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dst_file)
                else:
                    print(f"Warning: Missing frame {src_file}")
            except Exception as e:
                print(f"Error renaming {i:09d} to {timestamp}: {e}")
        
        # Remove temporary directory
        shutil.rmtree(temp_dir)
        
        print(f"Processed {video_file}: renamed {len(timestamps)} frames to match timestamps")

# Process both train and test sets
extract_frames(args.train_path, args.output_base, True)
extract_frames(args.test_path, args.output_base, False)

# Generate split files
train_output = os.path.join(args.output_base, "splits")
os.makedirs(train_output, exist_ok=True)

# Create train split
train_dir = os.path.join(args.output_base, "train")
train_videos = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
with open(os.path.join(train_output, "train_split1.txt"), "w") as f:
    for video in train_videos:
        f.write(f"train/{video}\n")

# Create test split
test_dir = os.path.join(args.output_base, "test")
test_videos = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
with open(os.path.join(train_output, "test_split1.txt"), "w") as f:
    for video in test_videos:
        f.write(f"test/{video}\n")

print("Frame extraction and split file generation complete!")