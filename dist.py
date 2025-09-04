import torch
import torch.distributed as dist

# https://claude.ai/share/2739ac44-dfcc-4cb8-80bd-5f5d82a761f3

def check_all_ranks_alive(timeout=10) -> bool:
    """Check if all ranks are responsive; return False is any rank is dead"""
    try:
        # Create a tensor to reduce across all ranks
        tensor = torch.tensor(1.0, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set timeout for the operation
        work = dist.all_reduce(tensor, async_op=True)
        work.wait(timeout=timeout)
        
        # If we reach here, all ranks participated successfully
        return True
    except Exception as e:
        print(f"!!! Rank {dist.get_rank()}: Health check failed - {e}")
        return False
    

