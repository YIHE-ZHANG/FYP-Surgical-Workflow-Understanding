# import os
# import torch
# import torch.nn as nn
# from datasets import Breakfast, GTEA, SALADS, RARP50
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import wandb
# import argparse
# import shutil
# import time
# from pathlib import Path
# import yaml
# from dotmap import DotMap
# import pprint
# from modules.fusion_module import fusion_earlyhyp
# from utils.KLLoss import KLLoss
# from utils.Augmentation import *
# from utils.solver import _optimizer, _lr_scheduler
# from utils.tools import *
# from utils.text_prompt import *
# from utils.saving import *

# from collections import Counter
# import csv

# def compute_class_weights(train_data):
#     """
#     Compute class weights by directly accessing the dataset's label files,
#     bypassing the need to load and transform images.
#     This avoids triggering the RandAugment error.
#     """
#     print("Computing class weights for balanced training (without loading images)...")
    
#     # For RARP-50, we can access the ground truth labels directly
#     all_ids = []
    
#     # Get the number of samples in a safe way
#     num_samples = len(train_data.train_split)
#     print(f"Processing {num_samples} samples...")
    
#     # Process each video in the split
#     for idx in range(num_samples):
#         try:
#             # Get video information without loading images
#             videoname = train_data.train_split[idx]
#             video_path = videoname[0]
            
#             # Construct label file path - following the same pattern as in __getitem__
#             label_file = os.path.join(train_data.frame_dir, video_path, 'action_discrete.txt')
            
#             # Read labels directly from file
#             labels = []
#             if os.path.exists(label_file):
#                 with open(label_file, 'r') as f:
#                     reader = csv.reader(f)
#                     for row in reader:
#                         if len(row) == 2:  # Ensure row has timestamp and label
#                             timestamp, label = row
#                             labels.append(int(label))
                
#                 # Sample frames based on the dataset parameters (like in frame_sampler)
#                 start_idx = int(videoname[1])
#                 ds = int(videoname[2]) if hasattr(videoname, '__len__') and len(videoname) > 2 else train_data.ds
#                 frame_indices = range(start_idx, start_idx + train_data.num_frames * ds, ds)
                
#                 # Get labels for sampled frames
#                 for frame_idx in frame_indices:
#                     if 0 <= frame_idx < len(labels):
#                         label = labels[frame_idx]
#                         if label >= 0:  # Valid action
#                             all_ids.append(label)
#             else:
#                 print(f"Warning: Label file not found at {label_file}")
                
#         except Exception as e:
#             print(f"Warning: Error processing sample {idx}: {e}")
#             continue
    
#     # Count occurrences of each class
#     from collections import Counter
#     class_counts = Counter(all_ids)
    
#     # Ensure all class IDs from 0-7 are present
#     for class_id in range(8):  # RARP-50 has 8 classes
#         if class_id not in class_counts:
#             class_counts[class_id] = 1  # Assign minimum count for rare classes
    
#     # Compute class weights (inverse frequency)
#     total_samples = sum(class_counts.values())
#     weights = {}
#     for class_id, count in class_counts.items():
#         # Use inverse frequency to balance classes
#         weights[class_id] = total_samples / (8 * count)
    
#     # Optionally apply smoothing to prevent extreme weights
#     max_weight = max(weights.values())
#     if max_weight > 10:  # If weights are too extreme
#         smoothing_factor = 0.5
#         for class_id in weights:
#             weights[class_id] = weights[class_id] ** smoothing_factor
    
#     print("Class distribution:", dict(class_counts))
#     print("Computed weights:", weights)
    
#     return weights

# class TextCLIP(nn.Module):
#     def __init__(self, model):
#         super(TextCLIP, self).__init__()
#         self.model = model

#     def forward(self, text):
#         return self.model.encode_text(text)


# class ImageCLIP(nn.Module):
#     def __init__(self, model):
#         super(ImageCLIP, self).__init__()
#         self.model = model

#     def forward(self, image):
#         return self.model.encode_image(image)


# # def get_clip_loss(image_embedding, text_embedding, logit_scale, loss_img, loss_txt, device, labels):
# #     logits_per_image, logits_per_text = create_logits(image_embedding, text_embedding, logit_scale)
# #     ground_truth = torch.tensor(labels, dtype=image_embedding.dtype, device=device)
# #     loss_imgs = loss_img(logits_per_image, ground_truth)
# #     loss_texts = loss_txt(logits_per_text, ground_truth)
# #     total_loss = (loss_imgs + loss_texts) / 2
# #     return total_loss

# def get_clip_loss(image_embedding, text_embedding, logit_scale, loss_img, loss_txt, device, labels, class_weights=None):
#     logits_per_image, logits_per_text = create_logits(image_embedding, text_embedding, logit_scale)
#     ground_truth = torch.tensor(labels, dtype=image_embedding.dtype, device=device)
    
#     # Apply class weights if provided
#     if class_weights is not None:
#         # Create weight tensor based on ground truth labels
#         weights = torch.ones(len(ground_truth), device=device)
#         for i, label in enumerate(labels):
#             # Handle different label types: scalars, numpy arrays, tensors
#             if hasattr(label, 'item'):
#                 # This handles both numpy scalars and tensors with single elements
#                 try:
#                     label_key = label.item()
#                 except ValueError:
#                     # If .item() fails, it means the array has multiple elements
#                     # Take the first element for multi-element arrays
#                     if hasattr(label, '__len__') and len(label) > 0:
#                         label_key = int(label[0])
#                     else:
#                         label_key = int(label)
#             else:
#                 # This handles regular Python integers and floats
#                 label_key = int(label)
                
#             if label_key in class_weights and label_key >= 0:  # Valid label check
#                 weights[i] = class_weights[label_key]
        
#         # Apply weights to loss
#         loss_imgs = loss_img(logits_per_image, ground_truth) * weights
#         loss_texts = loss_txt(logits_per_text, ground_truth) * weights
        
#         # Take mean of weighted losses
#         loss_imgs = loss_imgs.mean()
#         loss_texts = loss_texts.mean()
#     else:
#         loss_imgs = loss_img(logits_per_image, ground_truth)
#         loss_texts = loss_txt(logits_per_text, ground_truth)
    
#     total_loss = (loss_imgs + loss_texts) / 2
#     return total_loss


# def validate(base_model, fusion_model, model_text, model_image, val_loader, device, config):
#     """
#     Validate the model on the validation set
#     """
#     model_image.eval()
#     model_text.eval()
#     fusion_model.eval()
    
#     total_correct = 0
#     total_samples = 0
#     class_correct = {i: 0 for i in range(config.data.num_classes)}
#     class_total = {i: 0 for i in range(config.data.num_classes)}
    
#     print('Validating...')
#     with torch.no_grad():
#         for i, (images, list_id) in enumerate(tqdm(val_loader)):
#             # Process images
#             images = images.view((-1, config.data.num_frames, 3) + images.size()[-2:])
#             b, t, c, h, w = images.size()
#             images = images.to(device, non_blocking=True).view(-1, c, h, w)
            
#             # Process text prompts
#             text_cnt, text_acts, text_all, label_cnt = text_prompt_slide_val_all(
#                 val_loader.dataset.classes, config.data.max_act, dataset=config.data.dataset)
            
#             # Move to device
#             text_cnt = text_cnt.to(device, non_blocking=True)
#             text_acts = text_acts.to(device, non_blocking=True)
            
#             # Get image embeddings
#             image_embedding = model_image(images)
#             image_embedding = image_embedding.view(b, t, -1)
            
#             # Get text embeddings
#             text_cnt_embedding = model_text(text_cnt)
#             text_acts_embedding = model_text(text_acts)
            
#             # Reshape text embeddings
#             num_temp = text_acts_embedding.shape[0] // (config.data.num_classes + 1)
#             text_acts_embedding = text_acts_embedding.view(-1, num_temp, text_acts_embedding.shape[-1])
            
#             # Get logit scale
#             logit_scale = base_model.logit_scale.exp()
            
#             # Prepare position embeddings
#             text_dict_posemb = text_prompt_ord_emb(config.data.max_act).to(device)
#             text_dict_pos = text_dict_posemb.unsqueeze(0).repeat(b, 1, 1)
            
#             # Create image prompt features
#             image_embedding = image_embedding.unsqueeze(1).repeat(1, config.data.max_act, 1, 1)
#             cnt_emb, image_embedding = fusion_model(image_embedding, text_dict_pos)
            
#             # Predict action count
#             img_cnt_logits = logit_scale * cnt_emb @ text_cnt_embedding.t()
#             predicted_counts = torch.argmax(img_cnt_logits, dim=1)
            
#             # Predict actions for each position
#             correct_actions = 0
#             total_actions = 0
            
#             for batch_idx in range(b):
#                 for pos_idx in range(config.data.max_act):
#                     if pos_idx < predicted_counts[batch_idx]:
#                         # Get true label for this position
#                         true_label = list_id[batch_idx, pos_idx].item()
#                         if true_label < 0:
#                             continue
                        
#                         # Get image embedding for this position
#                         pos_img_emb = image_embedding[batch_idx, pos_idx]
                        
#                         # Get action logits
#                         action_logits = []
#                         for temp_idx in range(num_temp):
#                             temp_logits = logit_scale * pos_img_emb @ text_acts_embedding[:, temp_idx].t()
#                             action_logits.append(temp_logits)
                        
#                         # Average across templates
#                         action_logits = torch.stack(action_logits).mean(0)
                        
#                         # Get predicted action
#                         predicted_action = torch.argmax(action_logits).item()
                        
#                         # Count correct predictions
#                         if predicted_action == true_label:
#                             correct_actions += 1
#                             class_correct[true_label] += 1
                        
#                         total_actions += 1
#                         class_total[true_label] += 1
            
#             total_correct += correct_actions
#             total_samples += total_actions
    
#     # Calculate accuracy
#     overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
#     # Calculate per-class accuracy
#     class_accuracy = {}
#     for class_id in class_total:
#         if class_total[class_id] > 0:
#             class_accuracy[class_id] = class_correct[class_id] / class_total[class_id]
#         else:
#             class_accuracy[class_id] = 0
    
#     print(f"Validation - Overall Accuracy: {overall_accuracy:.4f}")
#     print("Per-class accuracies:")
#     for class_id, acc in class_accuracy.items():
#         class_name = val_loader.dataset.classes.get(class_id, f"Class {class_id}")
#         print(f"  {class_name}: {acc:.4f} ({class_correct[class_id]}/{class_total[class_id]})")
    
#     return overall_accuracy, class_accuracy


# def main():
#     # Add debugging for GPU availability
#     print(f"PyTorch version: {torch.__version__}")
#     print(f"CUDA available: {torch.cuda.is_available()}")
#     if torch.cuda.is_available():
#         print(f"CUDA device count: {torch.cuda.device_count()}")
#         print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
#     else:
#         print("No CUDA devices available!")

#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

#     wandb_on = True
#     global args, best_prec1
#     global global_step
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', '-cfg', default='./configs/breakfast/breakfast_ft.yaml')
#     parser.add_argument('--log_time', default='')
#     args = parser.parse_args()
#     with open(args.config, 'r') as f:
#         config = yaml.safe_load(f)
#     args.dataset = config['data']['dataset']
#     working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'],
#                                args.log_time)
#     if wandb_on:
#         wandb.init(project=config['network']['type'],
#                    name='{}_{}_{}_{}'.format(args.log_time, config['network']['type'], config['network']['arch'],
#                                              config['data']['dataset']))
#     print('-' * 80)
#     print(' ' * 20, "working dir: {}".format(working_dir))
#     print('-' * 80)

#     print('-' * 80)
#     print(' ' * 30, "Config")
#     pp = pprint.PrettyPrinter(indent=4)
#     pp.pprint(config)
#     print('-' * 80)

#     config = DotMap(config)

#     Path(working_dir).mkdir(parents=True, exist_ok=True)
#     shutil.copy(args.config, working_dir)
#     shutil.copy('train.py', working_dir)
#     shutil.copy('modules/fusion_module.py', working_dir)

#     device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

#     base_model, model_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
#                                              T=config.data.num_segments, dropout=config.network.drop_out,
#                                              emb_dropout=config.network.emb_dropout, pretrain=config.network.init,
#                                              joint=config.network.joint)  # Must set jit=False for training  ViT-B/32

#     transform_train = get_augmentation(True, config)
#     transform_val = get_augmentation(False, config)

#     if config.data.randaug.N > 0:
#         transform_train = randAugment(transform_train, config)

#     print('train transforms: {}'.format(transform_train.transforms))
#     print('val transforms: {}'.format(transform_val.transforms))

#     fusion_model = fusion_earlyhyp(config.network.sim_header, model_state_dict, config.data.num_frames)
#     model_text = TextCLIP(base_model)
#     model_image = ImageCLIP(base_model)
#     model_text = torch.nn.DataParallel(model_text).cuda()
#     model_image = torch.nn.DataParallel(model_image).cuda()
#     fusion_model = torch.nn.DataParallel(fusion_model).cuda()
#     if wandb_on:
#         wandb.watch(base_model)
#         wandb.watch(fusion_model)

#     if args.dataset == 'breakfast':
#         train_data = Breakfast(transform=transform_train, mode='train', num_frames=config.data.num_frames,
#                                ds=config.data.ds, ol=config.data.ol, n_split=config.data.n_split)

#     elif args.dataset == 'gtea':
#         train_data = GTEA(transform=transform_train, mode='train', num_frames=config.data.num_frames,
#                           n_split=config.data.n_split)

#     elif args.dataset == 'salads':
#         train_data = SALADS(transform=transform_train, mode='train', num_frames=config.data.num_frames,
#                             n_split=config.data.n_split)
    
#     elif args.dataset == 'rarp50':
#         train_data = RARP50(transform=transform_train, mode='train', num_frames=config.data.num_frames,
#                        ds=config.data.ds, ol=config.data.ol, n_split=config.data.n_split)

#     train_loader = DataLoader(train_data, batch_size=config.data.batch_size, num_workers=config.data.workers,
#                               shuffle=True, pin_memory=True, drop_last=True)
    
#     print(f"Initial data loading completed at {time.strftime('%H:%M:%S')}")

#     # Compute class weights to address imbalance
#     print("Computing class weights for balanced training...")
#     class_weights = compute_class_weights(train_data)


#         # Create validation dataloader
#     if args.dataset == 'breakfast':
#         val_data = Breakfast(transform=transform_val, mode='val', num_frames=config.data.num_frames,
#                         ds=config.data.ds, ol=config.data.ol, n_split=config.data.n_split)
#     elif args.dataset == 'gtea':
#         val_data = GTEA(transform=transform_val, mode='val', num_frames=config.data.num_frames,
#                     n_split=config.data.n_split)
#     elif args.dataset == 'salads':
#         val_data = SALADS(transform=transform_val, mode='val', num_frames=config.data.num_frames,
#                         n_split=config.data.n_split)
#     elif args.dataset == 'rarp50':
#         val_data = RARP50(transform=transform_val, mode='val', num_frames=config.data.num_frames,
#                         ds=config.data.ds, ol=config.data.ol, n_split=config.data.n_split)

#     val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers,
#                         shuffle=False, pin_memory=True, drop_last=False)

#     if device == "cpu":
#         model_text.float()
#         model_image.float()
#     else:
#         clip.model.convert_weights(model_text)
#         clip.model.convert_weights(model_image)

#     loss_img = KLLoss()
#     loss_txt = KLLoss()

#     start_epoch = config.solver.start_epoch

#     if config.pretrain:
#         if os.path.isfile(config.pretrain):
#             print(("=> loading checkpoint '{}'".format(config.pretrain)))
#             checkpoint = torch.load(config.pretrain)
#             base_model.load_state_dict(checkpoint['model_state_dict'])
#             del checkpoint
#         else:
#             print(("=> no checkpoint found at '{}'".format(config.resume)))

#     if config.resume:
#         if os.path.isfile(config.resume):
#             print(("=> loading checkpoint '{}'".format(config.resume)))
#             checkpoint = torch.load(config.resume)
#             base_model.load_state_dict(checkpoint['model_state_dict'])
#             fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
#             start_epoch = checkpoint['epoch'] + 1
#             print(("=> loaded checkpoint '{}' (epoch {})"
#                    .format(config.evaluate, start_epoch)))
#             del checkpoint
#         else:
#             print(("=> no checkpoint found at '{}'".format(config.pretrain)))

#     text_dict_posemb = text_prompt_ord_emb(config.data.max_act)

#     optimizer = _optimizer(config, base_model, fusion_model)
#     lr_scheduler = _lr_scheduler(config, optimizer)

#     best_prec1 = 0.0

#     for k, v in base_model.named_parameters():
#         print('{}: {}'.format(k, v.requires_grad))
#     for epoch in range(start_epoch, config.solver.epochs):
#         model_image.train()
#         model_text.train()
#         fusion_model.train()
#         text_dict_posemb = text_dict_posemb.to(device, non_blocking=True)
#         text_dict_pos = text_dict_posemb.repeat(config.data.batch_size, 1, 1)
#         text_dict_pos = text_dict_pos.view(-1, text_dict_pos.shape[-1])

#         # Right before the DataLoader iteration begins
#         print(f"Epoch {epoch}: Starting batch fetching at {time.strftime('%H:%M:%S')}")

#         for kkk, (images, list_id) in enumerate(tqdm(train_loader)):
#             # Add these lines to check device location
#             if kkk == 0:  # Only print on first batch to avoid cluttering output
#                 print(f"Device of images: {images.device}")
#                 print(f"Device of model: {next(model_image.parameters()).device}")

#             if config.solver.type != 'monitor':
#                 if (kkk + 1) == 1 or (kkk + 1) % 10 == 0:
#                     lr_scheduler.step(epoch + kkk / len(train_loader))
#             optimizer.zero_grad()

#             images = images.view((-1, config.data.num_frames, 3) + images.size()[-2:])
#             b, t, c, h, w = images.size()

#             text_cnt, text_acts, text_all, label_cnt = text_prompt_slide(train_data.classes,
#                                                                          list_id, args.dataset,
#                                                                          config.data.max_act)

#             images = images.to(device, non_blocking=True).view(-1, c, h, w)  # omit the Image.fromarray if the images
#             text_cnt = text_cnt.to(device, non_blocking=True)
#             text_acts = text_acts.to(device, non_blocking=True)

#             text_all = text_all.to(device, non_blocking=True)

#             image_embedding = model_image(images)
#             image_embedding = image_embedding.view(b, t, -1)

#             text_acts = text_acts.view(-1, text_acts.shape[-1])

#             text_all_embedding = model_text(text_all)
#             text_cnt_embedding = model_text(text_cnt)
#             text_acts_embedding = model_text(text_acts)
#             text_pos_embedding = model_text(text_dict_pos)

#             text_acts_embedding = text_acts_embedding.view(b, -1, text_acts_embedding.shape[-1])
#             text_pos_embedding = text_pos_embedding.view(b, -1, text_pos_embedding.shape[-1])

#             image_embedding = image_embedding.unsqueeze(1).repeat(1, config.data.max_act, 1, 1)
#             cnt_emb, image_embedding = fusion_model(image_embedding, text_pos_embedding)

#             if config.network.fix_text:
#                 text_cnt_embedding.detach_()
#                 text_acts_embedding.detach_()
#                 text_all_embedding.detach_()
#                 text_pos_embedding.detach_()

#             logit_scale = base_model.logit_scale.exp()

#             act_loss = 0
#             image_embedding_mean = image_embedding.mean(dim=1, keepdim=False)
#             # all_loss = get_clip_loss(image_embedding_mean, text_all_embedding, logit_scale, loss_img, loss_txt, device,
#             #                          gen_label_4list(list_id))
#             # cnt_loss = get_clip_loss(cnt_emb, text_cnt_embedding, logit_scale, loss_img, loss_txt, device,
#             #                          gen_label(label_cnt))
#             # for dd in range(text_acts_embedding.shape[1]):
#             #     act_loss += get_clip_loss(image_embedding[:, dd, :], text_acts_embedding[:, dd, :], logit_scale,
#             #                               loss_img,
#             #                               loss_txt, device, gen_label(list_id[:, dd]))

#             all_loss = get_clip_loss(image_embedding_mean, text_all_embedding, logit_scale, loss_img, loss_txt, device,
#                          gen_label_4list(list_id), class_weights)
#             cnt_loss = get_clip_loss(cnt_emb, text_cnt_embedding, logit_scale, loss_img, loss_txt, device,
#                                     gen_label(label_cnt))
#             for dd in range(text_acts_embedding.shape[1]):
#                 act_loss += get_clip_loss(image_embedding[:, dd, :], text_acts_embedding[:, dd, :], logit_scale,
#                                         loss_img,
#                                         loss_txt, device, gen_label(list_id[:, dd]), class_weights)

#             total_loss = all_loss + act_loss + cnt_loss
#             if wandb_on:
#                 wandb.log({"train_total_loss": total_loss})
#                 wandb.log({"train_loss_all": all_loss})
#                 wandb.log({"train_loss_acts": act_loss})
#                 wandb.log({"train_loss_cnt": cnt_loss})
#                 wandb.log({"lr": optimizer.param_groups[0]['lr']})
#             total_loss.backward()

#             if device == "cpu":
#                 optimizer.step()
#             else:
#                 convert_models_to_fp32(base_model)
#                 optimizer.step()
#                 clip.model.convert_weights(base_model)
        
#         # Validate the model after each epoch
#         if epoch % config.logging.eval_freq == 0:
#             print(f"Validating after epoch {epoch}")
#             val_acc, class_acc = validate(base_model, fusion_model, model_text, model_image, val_loader, device, config)
            
#             # Log validation results
#             if wandb_on:
#                 wandb.log({"val_accuracy": val_acc})
#                 for class_id, acc in class_acc.items():
#                     class_name = val_data.classes.get(class_id, f"Class {class_id}")
#                     wandb.log({f"val_acc_{class_name}": acc})
            
#             # Save best model
#             if val_acc > best_prec1:
#                 best_prec1 = val_acc
#                 epoch_saving(epoch, base_model, fusion_model, optimizer, "{}/best_model.pt".format(working_dir))
#                 print(f"New best model saved with accuracy: {best_prec1:.4f}")

#         epoch_saving(epoch, base_model, fusion_model, optimizer, "{}/".format(working_dir) + str(epoch) + "_epoch.pt")
#         epoch_saving(epoch, base_model, fusion_model, optimizer, "{}/last_model.pt".format(working_dir))


# if __name__ == '__main__':
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True
#     main()


# Updated Train
import os
import torch
import torch.nn as nn
from datasets import Breakfast, GTEA, SALADS, RARP50
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import shutil
import time
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
from modules.fusion_module import fusion_earlyhyp
from utils.KLLoss import KLLoss
from utils.Augmentation import *
from utils.solver import _optimizer, _lr_scheduler
from utils.tools import *
from utils.text_prompt import *
from utils.saving import *

# from memory_optimizer import create_memory_optimizer, GPUMemoryOptimizer

from collections import Counter
import csv

def compute_class_weights(train_data):
    """Compute class weights for balanced training"""
    print("Computing class weights for balanced training...")
    
    all_ids = []
    num_samples = len(train_data.train_split)
    print(f"Processing {num_samples} samples...")
    
    for idx in range(num_samples):
        try:
            videoname = train_data.train_split[idx]
            video_path = videoname[0]
            label_file = os.path.join(train_data.frame_dir, video_path, 'action_discrete.txt')
            
            labels = []
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) == 2:
                            timestamp, label = row
                            labels.append(int(label))
                
                start_idx = int(videoname[1])
                ds = int(videoname[2]) if hasattr(videoname, '__len__') and len(videoname) > 2 else train_data.ds
                frame_indices = range(start_idx, start_idx + train_data.num_frames * ds, ds)
                
                for frame_idx in frame_indices:
                    if 0 <= frame_idx < len(labels):
                        label = labels[frame_idx]
                        if label >= 0:
                            all_ids.append(label)
            else:
                print(f"Warning: Label file not found at {label_file}")
                
        except Exception as e:
            print(f"Warning: Error processing sample {idx}: {e}")
            continue
    
    class_counts = Counter(all_ids)
    
    # Ensure all class IDs are present
    for class_id in range(8):
        if class_id not in class_counts:
            class_counts[class_id] = 1
    
    # Compute weights with smoothing
    total_samples = sum(class_counts.values())
    weights = {}
    for class_id, count in class_counts.items():
        weights[class_id] = total_samples / (8 * count)
    
    # Apply smoothing to prevent extreme weights
    max_weight = max(weights.values())
    if max_weight > 10:
        smoothing_factor = 0.6  # More aggressive smoothing
        for class_id in weights:
            weights[class_id] = weights[class_id] ** smoothing_factor
    
    print("Class distribution:", dict(class_counts))
    print("Computed weights:", weights)
    return weights

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

def get_clip_loss(image_embedding, text_embedding, logit_scale, loss_img, loss_txt, device, labels, class_weights=None):
    logits_per_image, logits_per_text = create_logits(image_embedding, text_embedding, logit_scale)
    ground_truth = torch.tensor(labels, dtype=image_embedding.dtype, device=device)
    
    if class_weights is not None:
        weights = torch.ones(len(ground_truth), device=device)
        for i, label in enumerate(labels):
            if hasattr(label, 'item'):
                try:
                    label_key = label.item()
                except (ValueError, RuntimeError):
                    if hasattr(label, '__len__') and len(label) > 0:
                        label_key = int(label[0])
                    else:
                        label_key = int(label)
            else:
                label_key = int(label)
                
            if label_key in class_weights and label_key >= 0:
                weights[i] = class_weights[label_key]
        
        loss_imgs = loss_img(logits_per_image, ground_truth) * weights
        loss_texts = loss_txt(logits_per_text, ground_truth) * weights
        loss_imgs = loss_imgs.mean()
        loss_texts = loss_texts.mean()
    else:
        loss_imgs = loss_img(logits_per_image, ground_truth)
        loss_texts = loss_txt(logits_per_text, ground_truth)
    
    total_loss = (loss_imgs + loss_texts) / 2
    return total_loss

def save_checkpoint(epoch, base_model, fusion_model, optimizer, lr_scheduler, loss, working_dir):
    """Optimized checkpoint saving with automatic cleanup"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': base_model.state_dict(),
        'fusion_model_state_dict': fusion_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'loss': loss,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Always save latest.pt
    latest_path = os.path.join(working_dir, "latest.pt")
    torch.save(checkpoint, latest_path)
    print(f"Checkpoint saved: {latest_path}")
    
    # Save milestone every 3 epochs for quick training, 5 for quality
    milestone_freq = 3  # Adjust based on your config
    if epoch % milestone_freq == 0:
        milestone_path = os.path.join(working_dir, f"epoch_{epoch}.pt")
        torch.save(checkpoint, milestone_path)
        print(f"Milestone saved: {milestone_path}")
    
    # Cleanup old files
    try:
        for file in os.listdir(working_dir):
            if file.startswith("epoch_") and file.endswith(".pt"):
                epoch_num = int(file.split("_")[1].split(".")[0])
                if epoch_num % milestone_freq != 0 and epoch_num != epoch:
                    old_file = os.path.join(working_dir, file)
                    if os.path.exists(old_file):
                        os.remove(old_file)
    except Exception as e:
        print(f"Warning: Cleanup failed: {e}")

def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Argument parsing with resume support
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='./configs/rarp50/rarp50_ft.yaml')
    parser.add_argument('--log_time', default='')
    parser.add_argument('--resume', default='', help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config = DotMap(config)
    args.dataset = config.data.dataset
    
    # Generate timestamp if not provided
    if not args.log_time:
        args.log_time = time.strftime('%Y%m%d_%H%M')
    
    working_dir = os.path.join('./exp', config.network.type, config.network.arch.replace('/', '-'), 
                               config.data.dataset, args.log_time)
    
    # Add missing config defaults
    if not hasattr(config, 'logging'):
        config.logging = DotMap()
    config.logging.print_freq = getattr(config.logging, 'print_freq', 50)
    config.logging.eval_freq = getattr(config.logging, 'eval_freq', 1)
    
    print('-' * 80)
    print(f"Working directory: {working_dir}")
    print('-' * 80)
    
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)

    # Initialize W&B
    wandb.init(
        project=config.network.type,
        name=f'{args.log_time}_{config.network.type}_{config.network.arch.replace("/", "-")}_{config.data.dataset}',
        config=config
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    base_model, model_state_dict = clip.load(
        config.network.arch, device=device, jit=False,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init
    )

    # Setup transforms
    transform_train = get_augmentation(True, config)
    if config.data.randaug.N > 0:
        transform_train = randAugment(transform_train, config)

    # Initialize models
    fusion_model = fusion_earlyhyp(config.network.sim_header, model_state_dict, config.data.num_frames)
    model_text = TextCLIP(base_model)
    model_image = ImageCLIP(base_model)
    
    if device == "cuda":
        model_text = torch.nn.DataParallel(model_text).cuda()
        model_image = torch.nn.DataParallel(model_image).cuda()
        fusion_model = torch.nn.DataParallel(fusion_model).cuda()
        clip.model.convert_weights(model_text)
        clip.model.convert_weights(model_image)

    # Create dataset
    train_data = RARP50(
        transform=transform_train, mode='train',
        num_frames=config.data.num_frames,
        ds=config.data.ds, ol=config.data.ol,
        n_split=config.data.n_split
    )

    train_loader = DataLoader(
        train_data, batch_size=config.data.batch_size,
        num_workers=config.data.workers, shuffle=True,
        pin_memory=True, drop_last=True
    )

    print(f"Data loading completed at {time.strftime('%H:%M:%S')}")
    
    # Compute class weights
    class_weights = compute_class_weights(train_data)

    # Initialize training components
    loss_img = KLLoss()
    loss_txt = KLLoss()
    optimizer = _optimizer(config, base_model, fusion_model)
    lr_scheduler = _lr_scheduler(config, optimizer)
    
    start_epoch = 0
    
    # Handle resume
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            
            base_model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            
            print(f"=> Resumed from epoch {start_epoch}")
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print(f"=> No checkpoint found at '{args.resume}'")
    elif config.pretrain:
        if os.path.isfile(config.pretrain):
            print(f"=> Loading pretrained model '{config.pretrain}'")
            checkpoint = torch.load(config.pretrain, map_location=device)
            base_model.load_state_dict(checkpoint['model_state_dict'])
            del checkpoint
            torch.cuda.empty_cache()

    text_dict_posemb = text_prompt_ord_emb(config.data.max_act)

    # Training loop
    for epoch in range(start_epoch, config.solver.epochs):
        print(f"\n{'='*20} EPOCH {epoch}/{config.solver.epochs} {'='*20}")
        start_time = time.time()
        
        model_image.train()
        model_text.train()
        fusion_model.train()
        
        text_dict_posemb = text_dict_posemb.to(device, non_blocking=True)
        text_dict_pos = text_dict_posemb.repeat(config.data.batch_size, 1, 1)
        text_dict_pos = text_dict_pos.view(-1, text_dict_pos.shape[-1])

        epoch_loss = 0.0
        num_batches = 0
        
        try:
            for kkk, (images, list_id) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
                optimizer.zero_grad()
                
                images = images.to(device, non_blocking=True)
                
                # Learning rate scheduling
                if config.solver.type != 'monitor' and (kkk + 1) % 10 == 0:
                    lr_scheduler.step(epoch + kkk / len(train_loader))

                images = images.view((-1, config.data.num_frames, 3) + images.size()[-2:])
                b, t, c, h, w = images.size()

                # Generate text prompts
                text_cnt, text_acts, text_all, label_cnt = text_prompt_slide(
                    train_data.classes, list_id, args.dataset, config.data.max_act
                )

                images = images.view(-1, c, h, w)
                text_cnt = text_cnt.to(device, non_blocking=True)
                text_acts = text_acts.to(device, non_blocking=True)
                text_all = text_all.to(device, non_blocking=True)

                # Forward pass
                image_embedding = model_image(images)
                image_embedding = image_embedding.view(b, t, -1)

                text_acts = text_acts.view(-1, text_acts.shape[-1])

                text_all_embedding = model_text(text_all)
                text_cnt_embedding = model_text(text_cnt)
                text_acts_embedding = model_text(text_acts)
                text_pos_embedding = model_text(text_dict_pos)

                text_acts_embedding = text_acts_embedding.view(b, -1, text_acts_embedding.shape[-1])
                text_pos_embedding = text_pos_embedding.view(b, -1, text_pos_embedding.shape[-1])

                image_embedding = image_embedding.unsqueeze(1).repeat(1, config.data.max_act, 1, 1)
                cnt_emb, image_embedding = fusion_model(image_embedding, text_pos_embedding)

                if config.network.fix_text:
                    text_cnt_embedding.detach_()
                    text_acts_embedding.detach_()
                    text_all_embedding.detach_()
                    text_pos_embedding.detach_()

                logit_scale = base_model.logit_scale.exp()

                # Compute losses
                image_embedding_mean = image_embedding.mean(dim=1, keepdim=False)
                
                all_loss = get_clip_loss(
                    image_embedding_mean, text_all_embedding, logit_scale,
                    loss_img, loss_txt, device, gen_label_4list(list_id), class_weights
                )
                cnt_loss = get_clip_loss(
                    cnt_emb, text_cnt_embedding, logit_scale,
                    loss_img, loss_txt, device, gen_label(label_cnt)
                )
                
                act_loss = 0
                for dd in range(text_acts_embedding.shape[1]):
                    act_loss += get_clip_loss(
                        image_embedding[:, dd, :], text_acts_embedding[:, dd, :], logit_scale,
                        loss_img, loss_txt, device, gen_label(list_id[:, dd]), class_weights
                    )

                total_loss = all_loss + act_loss + cnt_loss
                
                # Logging
                if kkk % config.logging.print_freq == 0:
                    wandb.log({
                        "epoch": epoch,
                        "batch": kkk,
                        "train_total_loss": total_loss.item(),
                        "train_loss_all": all_loss.item(),
                        "train_loss_acts": act_loss.item(),
                        "train_loss_cnt": cnt_loss.item(),
                        "lr": optimizer.param_groups[0]['lr']
                    })
                
                # Backward pass
                total_loss.backward()

                if device == "cpu":
                    optimizer.step()
                else:
                    convert_models_to_fp32(base_model)
                    optimizer.step()
                    clip.model.convert_weights(base_model)
                
                epoch_loss += total_loss.item()
                num_batches += 1
                
                # Memory management
                if (kkk + 1) % 100 == 0:
                    torch.cuda.empty_cache()
                    # memory_optimizer.monitor_and_optimize(kkk)
        
        except Exception as e:
            print(f"ERROR during epoch {epoch}: {e}")
            # Emergency save
            emergency_path = os.path.join(working_dir, f"emergency_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': base_model.state_dict(),
                'fusion_model_state_dict': fusion_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'error': str(e)
            }, emergency_path)
            raise e
        
        # Epoch completion
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch} completed in {epoch_time/60:.2f} minutes")
        print(f"Average loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        save_checkpoint(epoch, base_model, fusion_model, optimizer, lr_scheduler, avg_epoch_loss, working_dir)
        
        wandb.log({
            "epoch_completed": epoch,
            "epoch_avg_loss": avg_epoch_loss,
            "epoch_duration_minutes": epoch_time/60
        })

    print("Training completed successfully!")
    wandb.finish()

if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    main()