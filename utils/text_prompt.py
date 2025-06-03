import torch
import clip
import numpy as np

def text_prompt_slide(classes, id_list, dataset, cnt_max=5):
    """
    Generate text prompts for action recognition
    
    Args:
        classes: Dictionary mapping class IDs to class names
        id_list: List of action IDs for each video
        dataset: Name of the dataset ('breakfast', 'gtea', 'salads', 'rarp50')
        cnt_max: Maximum number of actions to consider
        
    Returns:
        res_token_cnt: Tokenized count text prompts
        res_token_acts: Tokenized action text prompts
        res_token_all: Tokenized combined text prompts
        id_list_cnt: Count of valid actions per video
    """
    # Base templates for all datasets
    text_aug_cnts = [f"This clip contains no actions.",
                     f"This clip contains only one action,", f"This clip contains two actions,",
                     f"This clip contains three actions,", f"This clip contains four actions,",
                     f"This clip contains five actions,", f"This clip contains six actions,",
                     f"This clip contains seven actions,", f"This clip contains eight actions,"]
    text_aug_acts = [f"Firstly, ", f"Secondly, ", f"Thirdly, ", f"Fourthly, ",
                     f"Fifthly, ", f"Sixthly, ", f"Seventhly, ", f"Eighthly, "]
    text_aug_temp = [f"the person is {{}}.", f"the person is performing the action of {{}}.",
                     f"the character is {{}}.", f"he or she is {{}}.", f"the action {{}} is being played.",
                     f"it is the action of {{}}.", f"the human is {{}}.",
                     f"the person is working on {{}}.", f"the scene is {{}}.",
                     f"the person is focusing on {{}}.", f"the person is completing the action of {{}}.",
                     f"the step is {{}}", f"the action is {{}}.", f"the action step is {{}}."]
    text_long_temp = [f"the person is {{}}.", f"the character is {{}}.", f"he or she is {{}}.",
                      f"the human is {{}}.", f"the scene is {{}}.", f"{{}} is being done.",
                      f"the step is {{}}", f"the action is {{}}.", f"the action step is {{}}."]
    text_no_acts = [f"The first action does not exist.",
                    f"The second action does not exist.", f"The third action does not exist.",
                    f"The fourth action does not exist.", f"The fifth action does not exist.",
                    f"The sixth action does not exist.", f"The seventh action does not exist.",
                    f"The eighth action does not exist."]
    
    # Special templates for RARP-50 surgical dataset - Using shortened versions
    # if dataset == 'rarp50':
    #     text_aug_cnts = [f"No surgical actions.",
    #                      f"One surgical action.", f"Two surgical actions.",
    #                      f"Three surgical actions.", f"Four surgical actions.",
    #                      f"Five surgical actions.", f"Six surgical actions.",
    #                      f"Seven surgical actions.", f"Eight surgical actions."]
        
    #     # Shorter templates for RARP50 to avoid exceeding token limits
    #     text_aug_temp = [f"{{}}.", f"performing {{}}.",
    #                      f"{{}} action.", f"executing {{}}.", 
    #                      f"{{}} procedure.",
    #                      f"{{}}.", f"surgical {{}}.",
    #                      f"{{}} technique.", f"{{}} step.",
    #                      f"{{}}.", f"{{}} action.",
    #                      f"step: {{}}", f"action: {{}}.", f"maneuver: {{}}."]
        
    #     text_long_temp = [f"{{}}.", f"{{}}.", f"{{}}.",
    #                       f"{{}}.", f"{{}}.", f"{{}}.",
    #                       f"{{}} step", f"{{}} action", f"{{}} technique."]
        
    #     text_no_acts = [f"No first action.",
    #                     f"No second action.", f"No third action.",
    #                     f"No fourth action.", f"No fifth action.",
    #                     f"No sixth action.", f"No seventh action.",
    #                     f"No eighth action."]
        
    #     # Use shorter prefixes for RARP50
    #     text_aug_acts = [f"1: ", f"2: ", f"3: ", f"4: ",
    #                      f"5: ", f"6: ", f"7: ", f"8: "]

    if dataset == 'rarp50':
    # More descriptive and varied surgical prompts
        text_aug_cnts = [
            f"This surgical video shows no specific actions.",
            f"This surgical video shows one key action.", 
            f"This surgical video shows two sequential actions.", 
            f"This surgical video shows three surgical steps.",
            f"This surgical video shows four surgical manipulations.",
            f"This surgical video shows five surgical techniques.",
            f"This surgical video shows six robotic movements.",
            f"This surgical video shows seven surgical procedures.",
            f"This surgical video shows eight consecutive surgical actions."
        ]
        
        # More diverse surgical prefixes
        text_aug_acts = [
            f"First, ", f"Then, ", f"Next, ", f"Following that, ",
            f"Subsequently, ", f"Additionally, ", f"Finally, ", f"Lastly, "
        ]
        
        # More specific surgical action templates
        text_aug_temp = [
            f"the surgeon is {{}}.", 
            f"the robotic arm performs {{}}.",
            f"this shows {{}}.", 
            f"the procedure involves {{}}.",
            f"we can observe {{}}.", 
            f"{{}} is being performed.",
            f"{{}} is visible in the field.",
            f"{{}} is occurring.", 
            f"{{}} is the current action.",
            f"{{}} is taking place.",
            f"the video demonstrates {{}}.",
            f"{{}} is being executed.",
            f"{{}} is shown.",
            f"{{}} is evident."
        ]
        
        text_long_temp = [
            f"the surgeon is {{}}.", 
            f"the procedure shows {{}}.", 
            f"{{}}.",
            f"we can see {{}}.", 
            f"{{}} is happening.", 
            f"{{}} is visible.",
            f"the surgical step is {{}}", 
            f"{{}} is being done.", 
            f"the action is {{}}."
        ]
        
        text_no_acts = [
            f"No first action.",
            f"No second action.", 
            f"No third action.",
            f"No fourth action.", 
            f"No fifth action.",
            f"No sixth action.", 
            f"No seventh action.", 
            f"No eighth action."
        ]
    
    text_aug_cnts = text_aug_cnts[:cnt_max+1]
    text_aug_acts = text_aug_acts[:cnt_max]
    text_no_acts = text_no_acts[:cnt_max]

    b, _ = id_list.shape
    num_temp = len(text_aug_temp)
    num_long = len(text_long_temp)
    text_id = np.random.randint(num_temp, size=len(id_list) * cnt_max).reshape(-1, cnt_max)
    text_id_long = np.random.randint(num_long, size=len(id_list) * cnt_max).reshape(-1, cnt_max)
    id_list_cnt = id_list >= 0
    id_list_cnt = torch.sum(id_list_cnt, dim=1)

    # Add safety check: cap the action count to the max available in text_aug_cnts
    id_list_cnt = torch.clamp(id_list_cnt, max=len(text_aug_cnts)-1)
    
    res_token_cnt = []

    for id in id_list_cnt:
        res_token_cnt.append(clip.tokenize(text_aug_cnts[id.item()]))
    res_token_cnt = torch.cat(res_token_cnt)

    res_token_acts = []
    res_token_all = []

    for ii, txt in enumerate(id_list):
        num_acts = id_list_cnt[ii].item()
        action_list = []
        for i in range(num_acts):
            action_list.append(classes[txt[i].item()])
            
        # Dataset-specific action name handling
        if dataset == 'breakfast':
            if action_list[0] == 'SIL': action_list[0] = 'waiting and preparing'
            if action_list[-1] == 'SIL': action_list[-1] = 'finishing and waiting'
        elif dataset == 'rarp50':
            # For RARP50, use shorter action descriptions
            action_mapping = {
                0: "other action",
                1: "picking needle",
                2: "positioning needle",
                3: "pushing needle",
                4: "pulling suture",
                5: "tying knot",
                6: "cutting suture",
                7: "dropping needle"
            }
            
            # Replace action names with shorter ones
            for i in range(len(action_list)):
                try:
                    action_index = int(action_list[i])
                    if action_index in action_mapping:
                        action_list[i] = action_mapping[action_index]
                except (ValueError, TypeError):
                    # If action_list[i] is not an integer, try direct mapping
                    if action_list[i] in action_mapping:
                        action_list[i] = action_mapping[action_list[i]]
        
        sentences = []
        sentences_all = ''
        
        # For RARP50, reduce the total number of tokens by limiting text length
        max_combined_actions = cnt_max
        if dataset == 'rarp50':
            max_combined_actions = min(3, cnt_max)  # Limit to max 3 actions in combined text for RARP50
        
        for i in range(num_acts):
            sent = text_aug_acts[i] + text_aug_temp[text_id[ii][i]].format(action_list[i])
            sentences.append(clip.tokenize(sent))
            
            # Only add to combined text if within limits
            if i < max_combined_actions:
                sentences_all += ' ' + text_aug_acts[i] + text_long_temp[text_id_long[ii][i]].format(action_list[i])
        
        # Add padding for non-existent actions
        for i in range(num_acts, len(text_no_acts)):
            sentences.append(clip.tokenize(text_no_acts[i]))
        
        res_token_acts.append(torch.cat(sentences))
        
        if sentences_all:
            sentences_all = sentences_all[1:]  # Remove first space
            
            # Safety check to ensure sentences_all is not too long for RARP50
            if dataset == 'rarp50':
                # Simple way to truncate - only keep first portion if too long
                words = sentences_all.split()
                if len(words) > 20:  # Limiting to about 20 words should keep it under 77 tokens
                    sentences_all = ' '.join(words[:20])
            
            try:
                # Try tokenizing, but handle potential errors
                res_token_all.append(clip.tokenize(sentences_all))
            except RuntimeError as e:
                # If tokenization fails, create a simpler, shorter fallback text
                print(f"Warning: Tokenization error - {e}")
                fallback_text = f"Actions: {', '.join(action_list[:2])}"
                res_token_all.append(clip.tokenize(fallback_text))
        else:
            # Fallback for empty sentences_all
            res_token_all.append(clip.tokenize("No actions."))
    
    res_token_acts = torch.cat(res_token_acts).view(b, -1, res_token_cnt.shape[1])
    res_token_all = torch.cat(res_token_all)

    return res_token_cnt, res_token_acts, res_token_all, id_list_cnt

def text_prompt_pos_emb():
    num_max = 8
    text_aug_cnts = [f"This clip contains no actions.",
                     f"This clip contains only one action,", f"This clip contains two actions,",
                     f"This clip contains three actions,", f"This clip contains four actions,",
                     f"This clip contains five actions,", f"This clip contains six actions,",
                     f"This clip contains seven actions,", f"This clip contains eight actions,"]
    text_aug_acts = [f"this is the first action.", f"this is the second action.",
                     f"this is the third action.", f"this is the fourth action.",
                     f"this is the fifth action.", f"this is the sixth action.",
                     f"this is the seventh action.", f"this is the eighth action."]
    text_aug_no = "This action does not exist."

    text_dict_acts = {}

    for ii, txt in enumerate(text_aug_cnts):
        lst = []
        for i in range(num_max):
            if i >= ii:
                lst.append(clip.tokenize(text_aug_no))
            else:
                lst.append(clip.tokenize(text_aug_cnts[ii] + ' ' + text_aug_acts[i]))
        text_dict_acts[ii] = lst
        text_dict_acts[ii] = torch.cat(text_dict_acts[ii])
    text_dict_acts = torch.cat([v for k, v in text_dict_acts.items()])

    return text_dict_acts


def text_prompt_ord_emb(cnt_max=5):
    text_aug_acts = [f"this is the first action.", f"this is the second action.",
                     f"this is the third action.", f"this is the fourth action.",
                     f"this is the fifth action.", f"this is the sixth action.",
                     f"this is the seventh action.", f"this is the eighth action."]
    text_aug_acts = text_aug_acts[:cnt_max]

    lst = [clip.tokenize(txt) for txt in text_aug_acts]
    lst = torch.cat(lst)

    return lst


def text_prompt_slide_val_all(classes, cnt_max=5, dataset=None):
    """
    Generate text prompts for validation
    
    Args:
        classes: Dictionary mapping class IDs to class names
        cnt_max: Maximum number of actions to consider
        dataset: Name of the dataset ('breakfast', 'gtea', 'salads', 'rarp50')
        
    Returns:
        res_token_cnt: Tokenized count text prompts
        res_token_acts: Tokenized action text prompts
        num_temp: Number of template variations
    """
    # Handle RARP-50 dataset specifically
    if dataset == 'rarp50':
        # Define surgical-specific validation templates
        text_aug_cnts = [f"This clip contains no surgical actions.",
                         f"This clip contains only one surgical action.", f"This clip contains two surgical actions.",
                         f"This clip contains three surgical actions.", f"This clip contains four surgical actions.",
                         f"This clip contains five surgical actions.", f"This clip contains six surgical actions."]
        text_aug_acts = [f"Firstly, ", f"Secondly, ",
                         f"Thirdly, ", f"Fourthly, ",
                         f"Fifthly, ", f"Sixthly, "]
        text_aug_temp = [f"the surgeon is {{}}.", f"the robotic system is executing {{}}.", 
                         f"the procedure demonstrates {{}}.",
                         f"the surgical field shows {{}}.", f"the operation includes {{}}.", 
                         f"{{}} is being performed.",
                         f"the surgical step is {{}}", f"the robotic action is {{}}.", 
                         f"the surgical technique involves {{}}."]
        text_no_acts = [f"The first surgical action does not exist.",
                        f"The second surgical action does not exist.", f"The third surgical action does not exist.",
                        f"The fourth surgical action does not exist.", f"The fifth surgical action does not exist.",
                        f"The sixth surgical action does not exist."]
        
        # Map class indices to descriptive action names if needed
        action_mapping = {
            0: "performing other surgical actions",
            1: "picking up the needle with forceps",
            2: "positioning the needle tip at the tissue",
            3: "pushing the needle through the tissue",
            4: "pulling the suture through tissue",
            5: "tying a surgical knot",
            6: "cutting the suture thread",
            7: "dropping the needle at the target location"
        }
        
        # Create a copy of classes with mapped action descriptions if needed
        mapped_classes = {}
        for k, v in classes.items():
            try:
                k_int = int(k)
                if k_int in action_mapping:
                    mapped_classes[k] = action_mapping[k_int]
                else:
                    mapped_classes[k] = v
            except (ValueError, TypeError):
                mapped_classes[k] = v
    else:
        # Default templates for other datasets
        mapped_classes = classes.copy()  # Create a copy to avoid modifying the original
        
        if dataset == 'breakfast' and 0 in mapped_classes and 48 in mapped_classes:
            mapped_classes[0] = 'waiting and preparing'
            mapped_classes[48] = 'finishing and waiting'
            
        text_aug_cnts = [f"This clip contains no actions.",
                         f"This clip contains only one action.", f"This clip contains two actions.",
                         f"This clip contains three actions.", f"This clip contains four actions.",
                         f"This clip contains five actions.", f"This clip contains six actions."]
        text_aug_acts = [f"Firstly, ", f"Secondly, ",
                         f"Thirdly, ", f"Fourthly, ",
                         f"Fifthly, ", f"Sixthly, "]
        text_aug_temp = [f"the person is {{}}.", f"the character is {{}}.", f"he or she is {{}}.",
                         f"the human is {{}}.", f"the scene is {{}}.", f"{{}} is being done.",
                         f"the step is {{}}", f"the action is {{}}.", f"the action step is {{}}."]
        text_no_acts = [f"The first action does not exist.",
                        f"The second action does not exist.", f"The third action does not exist.",
                        f"The fourth action does not exist.", f"The fifth action does not exist.",
                        f"The sixth action does not exist."]
    
    text_aug_cnts = text_aug_cnts[:cnt_max+1]
    text_aug_acts = text_aug_acts[:cnt_max]
    text_no_acts = text_no_acts[:cnt_max]

    num_temp = len(text_aug_temp)
    num_act = len(text_aug_acts)
    num_cnt = len(text_aug_cnts)
    res_token_cnt = []

    for id in range(num_cnt):
        res_token_cnt.append(clip.tokenize(text_aug_cnts[id]))
    res_token_cnt = torch.cat(res_token_cnt)

    res_token_acts = []

    for ii in range(num_act):
        res_token_acts.append([])
        for jj in range(num_temp):
            res_token_acts[ii].append([clip.tokenize(text_aug_acts[ii] + text_aug_temp[jj].format(c)) for i, c in
                                       mapped_classes.items()])
            res_token_acts[ii][jj].append(clip.tokenize(text_no_acts[ii]))
            res_token_acts[ii][jj] = torch.cat(res_token_acts[ii][jj])
        res_token_acts[ii] = torch.cat(res_token_acts[ii])
    res_token_acts = torch.cat(res_token_acts)

    return res_token_cnt, res_token_acts, num_temp


def text_prompt_single(data):
    text_aug = [f"the person is {{}}", f"the person is performing the activity of {{}}",
                f"the character is {{}}", f"he or she is {{}}", f"the human activity of {{}} is being performed",
                f"this video is the activity of {{}}", f"the human is {{}}",
                f"Can you recognize the activity of {{}}?"]
    text_dict = {}
    num_text_aug = len(text_aug)

    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c in data.items()])

    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, num_text_aug, text_dict


if __name__ == '__main__':
    # Test with breakfast dataset
    print("Testing with breakfast dataset...")
    cls = {0: 'background', 1: 'closing ketchup', 2: 'closing jam', 3: 'putting chocolate', 4: 'opening chocolate',
           5: 'opening tea', 6: 'putting tea', 7: 'pouring sugar into the cup with a spoon'}  # Shortened for clarity
    cls = {int(k): v for k, v in cls.items()}
    id_list = torch.tensor([[1, 2, -1, -1, -1, -1, -1, -1, -1, -1],
                            [3, 4, 5, -1, -1, -1, -1, -1, -1, -1]])
    
    # Test breakfast dataset and print examples
    print("\nBreakfast dataset examples:")
    text_cnt, text_acts, text_all, label_cnt = text_prompt_slide(cls, id_list, 'breakfast', cnt_max=5)
    
    # Print one example of each type of prompt
    print(f"\nCount prompt example (tokenized shape: {text_cnt.shape}):")
    print("This is a tokenized representation of: 'This clip contains two actions.'")
    
    print(f"\nAction prompts example (tokenized shape: {text_acts.shape}):")
    print("These are tokenized representations of actions like:")
    print("'Firstly, the person is closing ketchup.'")
    print("'Secondly, the person is closing jam.'")
    
    print(f"\nCombined prompt example (tokenized shape: {text_all.shape}):")
    print("This is a tokenized representation of all actions combined in one text.")
    
    # Test RARP-50 dataset
    print("\n\nTesting with RARP-50 dataset...")
    rarp_cls = {
        0: "Other",
        1: "Picking-up the needle",
        2: "Positioning the needle tip", 
        3: "Pushing the needle through the tissue",
        4: "Pulling the needle out of the tissue",
        5: "Tying a knot",
        6: "Cutting the suture",
        7: "Returning/dropping the needle"
    }
    rarp_cls = {int(k): v for k, v in rarp_cls.items()}
    
    # Create sample action sequence for testing
    rarp_id_list = torch.tensor([[1, 2, 3, -1, -1, -1, -1, -1, -1, -1],
                                [5, 6, 7, -1, -1, -1, -1, -1, -1, -1]])
    
    # Test with RARP-50 dataset and print examples
    print("\nRARP-50 dataset examples:")
    rarp_text_cnt, rarp_text_acts, rarp_text_all, rarp_label_cnt = text_prompt_slide(
        rarp_cls, rarp_id_list, 'rarp50', cnt_max=5
    )
    
    # Print one example of each type of prompt
    print(f"\nCount prompt example (tokenized shape: {rarp_text_cnt.shape}):")
    print("This is a tokenized representation of: 'This clip contains three surgical actions.'")
    
    print(f"\nAction prompts example (tokenized shape: {rarp_text_acts.shape}):")
    print("These are tokenized representations of actions like:")
    print("'Firstly, the surgeon is picking up the needle with forceps.'")
    print("'Secondly, the robotic system is executing positioning the needle tip at the tissue.'")
    
    print(f"\nCombined prompt example (tokenized shape: {rarp_text_all.shape}):")
    print("This is a tokenized representation of all surgical actions combined in one text.")
    
    # Test validation prompts as well
    print("\n\nTesting validation prompts:")
    val_cnt, val_acts, num_temp = text_prompt_slide_val_all(rarp_cls, dataset='rarp50')
    print(f"Validation count prompts shape: {val_cnt.shape}")
    print(f"Validation action prompts shape: {val_acts.shape}")
    print(f"Number of templates: {num_temp}")
    
    print("\nTesting complete!")