import os
import clip
import torch.nn as nn
from datasets import Breakfast, GTEA, SALADS, RARP50  # Add RARP50 import
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
import numpy as np  # Changed from 'numpy' to the standard 'np'
from modules.fusion_module import fusion_earlyhyp
from utils.Augmentation import get_augmentation
import torch
from utils.text_prompt import *


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)


def validate(epoch, val_loader, device, model, fusion_model, config,
             text_dict_cnts, text_dict_acts, text_dict_posemb, num_aug, cnt_max, dataset_name):
    model.eval()
    fusion_model.eval()

    final_act_1 = []
    final_act_5 = []
    final_cnt = []
    gt_act = []

    with torch.no_grad():
        text_inputs_cnts = text_dict_cnts.to(device)
        text_dict_posemb = text_dict_posemb.to(device)
        text_dict_acts = text_dict_acts.to(device)
        text_features_cnts = model.encode_text(text_inputs_cnts)
        text_dict_posemb = model.encode_text(text_dict_posemb)
        
        # ---------- DEBUG INFORMATION START ----------
        print("\n------ DEBUG: TEXT FEATURES INFORMATION ------")
        print(f"Text dict acts shape: {text_dict_acts.shape}")
        text_features_raw = model.encode_text(text_dict_acts)
        print(f"Text features acts size before reshape: {text_features_raw.size()}")
        print(f"cnt_max = {cnt_max}, embedding_dim = {text_dict_posemb.shape[-1]}")
        total_elements = text_features_raw.numel()
        print(f"Total elements in text features: {total_elements}")
        
        # Check if tensor is divisible as expected for first reshape
        expected_middle_dim = total_elements // (cnt_max * text_dict_posemb.shape[-1])
        remainder = total_elements % (cnt_max * text_dict_posemb.shape[-1])
        print(f"Expected middle dimension: {expected_middle_dim}")
        print(f"Divisibility check: {total_elements} / ({cnt_max} * {text_dict_posemb.shape[-1]}) = {expected_middle_dim} with remainder {remainder}")
        
        # We need to ensure that the middle dimension is also divisible by num_aug
        # This is critical for the subsequent view operation
        adjusted_middle_dim = (expected_middle_dim // num_aug) * num_aug
        print(f"Adjusted middle dimension for num_aug={num_aug}: {adjusted_middle_dim}")
        
        # Calculate how many elements we need for exact division by both cnt_max and num_aug
        elements_needed = cnt_max * adjusted_middle_dim * text_dict_posemb.shape[-1]
        print(f"Elements needed for clean division: {elements_needed}")
        
        if elements_needed != total_elements:
            print(f"WARNING: Truncating tensor from {total_elements} to {elements_needed} elements for exact division")
            # Truncate the tensor to make it divisible by both cnt_max and num_aug
            elements_to_keep = elements_needed
            # Reshape to 2D, truncate, and reshape back
            text_features_raw = text_features_raw.view(-1, text_dict_posemb.shape[-1])[:elements_to_keep // text_dict_posemb.shape[-1]]
            total_elements = elements_to_keep
        else:
            print("OK: Tensor shape is compatible with desired reshape")
            
        # ---------- DEBUG INFORMATION END ----------
        
        # Safe reshape approach - now we're sure it's divisible by both cnt_max and num_aug
        safe_middle_dim = total_elements // (cnt_max * text_dict_posemb.shape[-1])
        print(f"Using middle dimension: {safe_middle_dim}")
        
        # Safe reshape - this should now work cleanly
        text_features_acts = text_features_raw.view(cnt_max, safe_middle_dim, text_dict_posemb.shape[-1])
        print(f"Reshaped text features acts to: {text_features_acts.shape}")
        
        # Verify that safe_middle_dim is divisible by num_aug (critical for the next reshape)
        print(f"Verification: {safe_middle_dim} divisible by {num_aug}? {safe_middle_dim % num_aug == 0}")
        
        for iii, (image, class_id) in enumerate(tqdm(val_loader)):
            image = image.view((-1, config.data.num_frames, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            text_features_posemb = text_dict_posemb.repeat(b, 1, 1)
            gt_act.append(class_id.numpy())
            image_input = image.to(device).view(-1, c, h, w)
            image_features = model.encode_image(image_input).view(b, t, -1)
            image_features = image_features.unsqueeze(1).repeat(1, cnt_max, 1, 1)
            cnt_emb, image_features = fusion_model(image_features, text_features_posemb)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            cnt_emb /= cnt_emb.norm(dim=-1, keepdim=True)
            text_features_cnts /= text_features_cnts.norm(dim=-1, keepdim=True)
            text_features_acts /= text_features_acts.norm(dim=-1, keepdim=True)
            similarity_cnts = (100.0 * cnt_emb @ text_features_cnts.T)
            similarity_cnts = similarity_cnts.view(b, -1).softmax(dim=-1)
            _, indices_cnts = similarity_cnts.topk(1, dim=-1)
            final_ind_1 = torch.zeros(b, cnt_max).long().to(device)
            final_ind_5 = torch.zeros(b, cnt_max, 5).long().to(device)
            
            # Adjust the max_class_id based on dataset
            max_class_id = 19  # Default for other datasets
            if dataset_name == 'rarp50':
                max_class_id = len(val_loader.dataset.classes) - 1  # Use actual number of classes
                
            for i in range(text_features_acts.shape[0]):
                similarity_acts = (100.0 * image_features[:, i, :] @ text_features_acts[i, :].T)
                
                # More robust reshaping that handles edge cases
                if similarity_acts.numel() % num_aug != 0:
                    # This should not happen now with our preprocessing, but just in case:
                    print(f"WARNING: Similarity acts size {similarity_acts.numel()} not divisible by num_aug={num_aug}")
                    # Adjust similarity_acts to be divisible by num_aug
                    new_size = (similarity_acts.numel() // num_aug) * num_aug
                    similarity_acts = similarity_acts.view(-1)[:new_size].view(b, -1)
                
                # Now reshape safely
                similarity_acts = similarity_acts.view(b, num_aug, -1).softmax(dim=-1)
                similarity_acts = similarity_acts.mean(dim=1, keepdim=False)
                values_1, indices_1 = similarity_acts.topk(1, dim=-1)
                values_5, indices_5 = similarity_acts.topk(5, dim=-1)
                indices_1 = torch.where(indices_1 == max_class_id, -1, indices_1)
                indices_5 = torch.where(indices_5 == max_class_id, -1, indices_5)
                final_ind_1[:, i] = indices_1.squeeze()
                final_ind_5[:, i, :] = indices_5
            
            final_act_1.append(final_ind_1.cpu().numpy())
            final_act_5.append(final_ind_5.cpu().numpy())
            final_cnt.append(indices_cnts.cpu().numpy())

        # Create output directory for test results
        output_dir = f'./prompt_test/{dataset_name}/split{config.data.n_split}'
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n------ Saving test results ------")
        print(f"final_act_1 shape: {np.vstack(final_act_1).shape}")
        print(f"final_act_5 shape: {np.vstack(final_act_5).shape}")
        print(f"final_cnt_1 shape: {np.vstack(final_cnt).shape}")
        print(f"gt_act shape: {np.vstack(gt_act).shape}")
        
        np.save(f'{output_dir}/final_act_1.npy', np.vstack(final_act_1))
        np.save(f'{output_dir}/final_act_5.npy', np.vstack(final_act_5))
        np.save(f'{output_dir}/final_cnt_1.npy', np.vstack(final_cnt))
        np.save(f'{output_dir}/gt_act.npy', np.vstack(gt_act))
        
        print(f"Saved test results to {output_dir}")


def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='./configs/rarp50/rarp50_test.yaml')
    parser.add_argument('--log_time', default='')
    parser.add_argument('--dataset', default='rarp50')  # Changed default to rarp50
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)  # Fixed to use safe_load
    
    # Handle special characters in directory path
    arch_dir = config['network']['arch'].replace('/', '-')
    if args.log_time.strip():
        working_dir = os.path.join('./exp', config['network']['type'], arch_dir, config['data']['dataset'],
                                args.log_time)
    else:
        # Use current timestamp if not provided
        import time
        current_time = time.strftime("%Y%m%d_%H%M%S")
        working_dir = os.path.join('./exp', config['network']['type'], arch_dir, config['data']['dataset'],
                                current_time)
        args.log_time = current_time
        
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('test.py', working_dir)

    print("\n------ Testing setup ------")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    base_model, model_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                             T=config.data.num_segments, dropout=config.network.drop_out,
                                             emb_dropout=config.network.emb_dropout)

    transform_val = get_augmentation(False, config)

    fusion_model = fusion_earlyhyp(config.network.sim_header, model_state_dict, config.data.num_frames)

    model_text = TextCLIP(base_model)
    model_image = ImageCLIP(base_model)

    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()

    print("\n------ Loading dataset ------")
    if args.dataset == 'breakfast':
        val_data = Breakfast(transform=transform_val, mode='val', num_frames=config.data.num_frames,
                             ds=config.data.ds, ol=config.data.ol)
    elif args.dataset == 'gtea':
        val_data = GTEA(transform=transform_val, mode='val', num_frames=config.data.num_frames,
                        n_split=config.data.n_split)
    elif args.dataset == 'salads':
        val_data = SALADS(transform=transform_val, mode='val', num_frames=config.data.num_frames,
                          n_split=config.data.n_split)
    elif args.dataset == 'rarp50':
        # Added RARP50 dataset
        val_data = RARP50(transform=transform_val, mode='test', num_frames=config.data.num_frames,
                          ds=config.data.ds, ol=config.data.ol, n_split=config.data.n_split)
        print(f"Loaded RARP50 test dataset with {len(val_data)} samples")
        
        # Debug class information
        print("RARP50 class mapping:")
        for class_id, class_name in val_data.classes.items():
            print(f"  {class_id}: {class_name}")

    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False,
                            pin_memory=True, drop_last=False)

    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        clip.model.convert_weights(model_text)
        clip.model.convert_weights(model_image)

    start_epoch = config.solver.start_epoch

    print("\n------ Loading model checkpoint ------")
    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            base_model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            print("=> loaded checkpoint successfully")
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))
            print("WARNING: Testing without a trained model may not give meaningful results.")

    # Use the same cnt_max as in the dataset class (8 for RARP50)
    cnt_max = config.data.max_act if hasattr(config.data, 'max_act') else 8
    print(f"\nUsing maximum action count (cnt_max): {cnt_max}")
    
    text_dict_posemb = text_prompt_ord_emb(cnt_max=cnt_max)

    # Pass the dataset name to get appropriate prompts
    print("\n------ Generating text prompts ------")
    text_dict_cnts, text_dict_acts, num_aug = text_prompt_slide_val_all(val_data.classes, cnt_max=cnt_max, dataset=args.dataset)
    print(f"Text counts shape: {text_dict_cnts.shape}")
    print(f"Text acts shape: {text_dict_acts.shape}")
    print(f"Number of augmentations: {num_aug}")

    print("\n------ Starting validation ------")
    best_prec1 = 0.0
    validate(start_epoch, val_loader, device, base_model, fusion_model, config,
             text_dict_cnts, text_dict_acts, text_dict_posemb, num_aug, cnt_max, args.dataset)
    
    print("\n------ Evaluation completed successfully! ------")


if __name__ == '__main__':
    main()