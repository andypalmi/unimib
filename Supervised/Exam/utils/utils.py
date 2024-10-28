import os
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import random
import torch
from torch import nn
import numpy as np
import torch
from collections import OrderedDict
import numpy as np
from torch.autograd import Variable

def load_pretrained_model(model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> nn.Module:
    """
    Loads a pretrained model and prepares it for classification fine-tuning.
    
    Args:
        model_path (str): Path to the saved model checkpoint
        device (str): Device to load the model on
    
    Returns:
        nn.Module: Modified model ready for classification
    """
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create a new model instance
    model = checkpoint['model_architecture']

    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    model = model.to(device)

    # blocks_to_freeze = ['enc1', 'enc2', 'enc3', 'enc4', 'enc5']
    # for block in blocks_to_freeze:
    #     for param in getattr(model, block).parameters():
    #         param.requires_grad = False
    
    # # Freeze all parameters
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # Only keep the first projection head for feature extraction
    # and create a new classifier
    model.projection2 = None

    # # Modify classifier head
    # model.classifier = nn.Linear(in_features=128, out_features=251)

    # Unfreeze the classifier parameters
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model

def load_data(split=0.2, random_state=69420, verbose=True):
    """
    Load data and return dictionaries for training, validation, and test sets and their respective transform counts.
    Duplicates images based on class distribution to balance the dataset.
    
    Returns:
        train_dict (dict): Dictionary where keys are image paths and values are labels for the training set.
        val_dict (dict): Dictionary where keys are image paths and values are labels for the validation set.
        test_dict (dict): Dictionary where keys are image paths and values are labels for the test set.
        train_transform_dict (dict): Dictionary where keys are labels and values are the number of augmentations for the training set.
        val_transform_dict (dict): Dictionary where keys are labels and values are the number of augmentations for the validation set.
        test_transform_dict (dict): Dictionary where keys are labels and values are the number of augmentations for the test set.
    """
    # Load the data
    columns = ['image', 'label']
    labels_path = 'data/labels'
    train_df = pd.read_csv(os.path.join(labels_path, 'train_info.csv'), names=columns, header=None)
    val_df = pd.read_csv(os.path.join(labels_path, 'val_info.csv'), names=columns, header=None)

    # Add full paths to images
    images_path = 'data/images'
    train_path = 'train_set'
    val_path = 'val_set'
    train_df['image'] = images_path + '/' + train_path + '/' + train_df['image']
    val_df['image'] = images_path + '/' + val_path + '/' + val_df['image']

    # Split the training data
    train_df, val_split_df = train_test_split(
        train_df, 
        test_size=split, 
        random_state=random_state, 
        stratify=train_df['label']
    )

    # Function to duplicate images based on class counts
    def duplicate_images(df):
        """
        Duplicate images in each class to match the size of the largest class.
        """
        class_counts = Counter(df['label'])
        max_count = max(class_counts.values())
        new_data = []
        
        for label in class_counts.keys():
            # Get all images for this class
            class_images = df[df['label'] == label]['image'].tolist()
            if not class_images:  # Skip if no images for this class
                continue
                
            current_count = len(class_images)
            
            # Add all original images
            new_data.extend([{'image': img, 'label': label} for img in class_images])
            
            # Calculate how many additional copies we need
            needed_copies = max_count - current_count
            
            # Duplicate images randomly until we reach max_count
            for _ in range(needed_copies):
                img = random.choice(class_images)
                new_data.append({'image': img, 'label': label})
        return pd.DataFrame(new_data)

    # Create balanced datasets by duplicating images
    train_df_balanced = duplicate_images(train_df)
    val_df_balanced = duplicate_images(val_split_df)
    test_df_balanced = duplicate_images(val_df)

    # Create dictionaries
    train_dict = dict(zip(train_df_balanced['image'], train_df_balanced['label']))
    val_dict = dict(zip(val_df_balanced['image'], val_df_balanced['label']))
    test_dict = dict(zip(test_df_balanced['image'], test_df_balanced['label']))

    if verbose:
        print(f'Starting Training samples: {len(train_df)} | Validation samples: {len(val_split_df)} | Test samples: {len(val_df)}')
        print(f'Actual   Training samples: {len(train_df_balanced)} | Validation samples: {len(val_df_balanced)} | Test samples: {len(test_df_balanced)}')

    return (train_dict, val_dict, test_dict)

def summary(model, input_size, batch_size=-1, print_network=True, device="cuda"):
    """
    Generates a summary of the given PyTorch model, including the input and output shapes of each layer,
    the number of parameters, and whether the parameters are trainable.

    Args:
        model (torch.nn.Module): The PyTorch model to summarize.
        input_size (tuple or list of tuples): The size of the input tensor(s). If the model has multiple inputs,
                                              provide a list of tuples.
        batch_size (int, optional): The batch size to use for the input tensor(s). Default is -1.
        print_network (bool, optional): Whether to print the network summary. Default is True.
        device (str, optional): The device to use for the model ('cuda' or 'cpu'). Default is 'cuda'.

    Returns:
        None

    Raises:
        AssertionError: If the specified device is not 'cuda' or 'cpu'.

    Example:
        summary(model, (3, 224, 224))
    """

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor # type: ignore
    else:
        dtype = torch.FloatTensor

    model.to(device)

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size] # type: ignore
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    if print_network:
        print("----------------------------------------------------------------")
        line_new = "{:>20} {:>25} {:>25} {:>15}".format("Layer (type)", "Input Shape", "Output Shape", "Param #")
        print(line_new)
        print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20} {:>25} {:>25} {:>15}".format(
            layer,
            str(summary[layer]["input_shape"]),
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        if print_network:
            print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.)) # type: ignore
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params)) # type: ignore
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary
