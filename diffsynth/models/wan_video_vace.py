import torch
from .wan_video_dit import DiTBlock
from .utils import hash_state_dict_keys
import torch.nn as nn

class VaceWanAttentionBlock(DiTBlock):
    def __init__(self, has_image_input, dim, num_heads, ffn_dim, eps=1e-6, block_id=0, use_film=False, film_emb_dim=256, use_cfa=False):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps, use_film=use_film, film_emb_dim=film_emb_dim, use_cfa=use_cfa)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = torch.nn.Linear(self.dim, self.dim)
        self.after_proj = torch.nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, context, t_mod, freqs, 
                task_embedding=None, 
                context_over_exposed=None, 
                context_under_exposed=None, 
                token_mask_over=None, 
                token_mask_under=None):
                
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = super().forward(c, context, t_mod, freqs, 
                            task_embedding=task_embedding, 
                            context_over_exposed=context_over_exposed, 
                            context_under_exposed=context_under_exposed, 
                            token_mask_over=token_mask_over, 
                            token_mask_under=token_mask_under)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


class VaceWanModel(torch.nn.Module):
    def __init__(
        self,
        vace_layers=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
        vace_in_dim=96,
        patch_size=(1, 2, 2),
        has_image_input=False,
        dim=1536,
        num_heads=12,
        ffn_dim=8960,
        eps=1e-6,
        use_film=False,
        film_emb_dim=256,
        use_cfa=False
    ):
        super().__init__()
        self.vace_layers = vace_layers
        self.vace_in_dim = vace_in_dim
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        self.use_film = use_film                               # [TASK-EMB]
        self.film_emb_dim = film_emb_dim                       # [TASK-EMB]

        if self.use_film:                      # [TASK-EMB]
            self.task_table = torch.nn.Embedding(1, film_emb_dim)  # [TASK-EMB]
            torch.nn.init.zeros_(self.task_table.weight)       # [TASK-EMB] start as identity

        # vace blocks
        self.vace_blocks = torch.nn.ModuleList([
            VaceWanAttentionBlock(has_image_input, dim, num_heads, ffn_dim, eps, block_id=i, use_film=use_film, film_emb_dim=film_emb_dim, use_cfa=use_cfa)
            for i in self.vace_layers
        ])

        # vace patch embeddings
        self.vace_patch_embedding = torch.nn.Conv3d(vace_in_dim, dim, kernel_size=patch_size, stride=patch_size)

    def forward(
        self, x, vace_context, context, t_mod, freqs,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        context_over_exposed=None,
        context_under_exposed=None,
        token_mask_over=None,
        token_mask_under=None,
    ):
        f'''
        x: [1, 32400, 5120] -> torch.Tensor
        vace_context: [1, 96, 9, 90, 160] -> torch.Tensor
        context: [1, 512, 5120] -> torch.Tensor
        c: ([1, 5120, 9, 45, 80]) -> list
        tmod: [1, 6, 5120] -> torch.Tensor
        freqs: [32400, 1, 64] -> torch.Tensor
        '''

        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context] #[1, 96, 9, 90, 160] -> ([1, 5120, 9, 45, 80])
        c = [u.flatten(2).transpose(1, 2) for u in c] #[1, 5120, 9, 45, 80] -> ([1, 32400, 5120])
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, x.shape[1] - u.size(1), u.size(2))],
                      dim=1) for u in c
        ]) # [1, 32400, 5120]
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        if self.use_film:
            task_id = torch.tensor([0], device=x.device)            # e.g., “HDR-indoor”
            task_embedding = self.task_table(task_id).expand(1, -1)       # (B,256)
        else:
            task_embedding = None

        for block in self.vace_blocks:
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    c = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        c, x, context, t_mod, freqs, task_embedding,
                        context_over_exposed,
                        context_under_exposed,
                        token_mask_over,
                        token_mask_under,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                c = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    c, x, context, t_mod, freqs, task_embedding,
                    context_over_exposed,
                    context_under_exposed,
                    token_mask_over,
                    token_mask_under,
                    use_reentrant=False,
                )
            else:
                c = block(c, x, context, t_mod, freqs,
                task_embedding=task_embedding,
                context_over_exposed=context_over_exposed,
                context_under_exposed=context_under_exposed,
                token_mask_over=token_mask_over,
                token_mask_under=token_mask_under)

        hints = torch.unbind(c)[:-1]
        return hints
    
    @staticmethod
    def state_dict_converter():
        return VaceWanModelDictConverter()
    
    
class VaceWanModelDictConverter:
    def __init__(self):
        pass
    
    def from_civitai(self, state_dict):
        state_dict_ = {name: param for name, param in state_dict.items() if name.startswith("vace")}
        if hash_state_dict_keys(state_dict_) == '3b2726384e4f64837bdf216eea3f310d': # vace 14B
            config = {
                "vace_layers": (0, 5, 10, 15, 20, 25, 30, 35),
                "vace_in_dim": 96,
                "patch_size": (1, 2, 2),
                "has_image_input": False,
                "dim": 5120,
                "num_heads": 40,
                "ffn_dim": 13824,
                "eps": 1e-06,                
            }
        else:
            config = {}
        return state_dict_, config
