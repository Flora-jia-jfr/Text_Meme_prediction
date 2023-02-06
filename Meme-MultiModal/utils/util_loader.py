from .dataset.memotion import *
from .dataset.reddit import *
from .dataset.multi_category_memotion import *

from torch.utils.data import DataLoader
from torchvision import transforms
import torch


def data_loader(args, tokenizer, split, debugflag=False, multicategory_dataframe=None):

    if args.dataset == 'memotion':
        dataset = MemotionDataset(
            args.root_path,
            split,
            args.bert_model,
            args.max_token_length,
            tokenizer,
            transform=transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                ]))
    elif args.dataset == 'reddit':
        dataset = RedditDataset(
            args.root_path,
            split,
            args.bert_model,
            args.max_token_length,
            tokenizer,
            )
    elif args.dataset == 'multi_category_memotion_dataset':
        dataset = MultiCategoryeMemotionDataset(
            args.root_path,
            multicategory_dataframe,
            args.max_token_length,
            tokenizer,
            transform=transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        )
    else:
        raise Exception("No matching dataset.")

    if debugflag:
        dataset_loader = DataLoader(
            dataset,
            batch_size=8,
            num_workers=args.num_workers,
            generator=torch.Generator(device='cuda')
        )
    else:
        dataset_loader = DataLoader(dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    generator=torch.Generator(device='cuda'))
    return dataset_loader, len(dataset)

def get_data(args, split='train'):
    if split == 'val':
        return None
    else:
        data = MemotionDataset(args.root_path,
                               split,
                               args.bert_model,
                               args.max_token_length,
                               transform=transforms.Compose([
                                   transforms.Resize((256, 256)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                   ]))
    return data
