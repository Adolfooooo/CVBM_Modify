import time
import random

import torch
import torch.backends.cudnn as cudnn

import numpy as np

def reproducible_random_training(logging=None):
    """可复现的随机训练"""
    
    # 生成基于时间的种子
    seed = int(time.time() * 1000000) % (2**32)
    
    # 保存种子以便复现
    seed_info = {
        'timestamp': time.time(),
        'seed': seed,
        'pytorch_version': torch.__version__
    }
    
    # # 保存种子信息
    # with open('training_seed.json', 'w') as f:
    #     json.dump(seed_info, f, indent=2)
    
    # 设置确定性环境
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    logging.info(f"使用种子: {seed}")
    logging.info("successful setting seed!")
    
    return seed
