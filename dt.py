import torch
import torch.distributed as dist
import os 
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'
rank = dist.get_rank()
worldsize = dist.get_world_size()
torch.distributed.init_process_group(backend='gloo', rank=rank, world_size=worldsize, init_method=f"tcp://127.0.0.1:12356")
print(f"rank = {torch.distributed.get_rank()}")
