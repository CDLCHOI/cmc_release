import options.option_transformer as option_trans
args = option_trans.get_args_parser()
from dataset.dataset_critic import CriticDataset
    from utils.model_util import create_gaussian_diffusion_simple

if __name__ == '__main__':
    diffusion = create_gaussian_diffusion_simple(args, None, args.modeltype, clip_model)
    dataset = CriticDataset(args, )