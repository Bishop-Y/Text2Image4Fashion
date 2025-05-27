import torch
import torch.nn.utils.prune as prune_utils
import hydra
from omegaconf import DictConfig
from fashion_generator.modules.gan_lit_module import GANLitModule
import builtins

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def prune_model(cfg: DictConfig):
    GANLitModule.print = builtins.print
    ckpt = cfg.model.net_g
    amount = cfg.prune.amount
    out_path = cfg.prune.out

    lit = GANLitModule(cfg=cfg, output_dir='./tmp_prune')
    lit.eval()

    gen = lit.netG
    parameters = [(m, 'weight') for m in gen.modules()
                  if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    prune_utils.global_unstructured(
        parameters=parameters,
        pruning_method=prune_utils.L1Unstructured,
        amount=amount
    )
    for module, _ in parameters:
        prune_utils.remove(module, 'weight')

    torch.save(gen.state_dict(), out_path)
    print(f"Pruned model saved to {out_path}")

if __name__ == '__main__':
    prune_model()