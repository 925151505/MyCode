import argparse
import torch
class Config_path(object):
    """
    Model path
    """
    def __init__(self):
        self.config_dict = {
            "model_path": {
                "xlnet-base-cased": './pretrain_model/xlnet-base-cased',
                "bert-large-uncased": './pretrain_model/bert-large-uncased',
                "bert-base-uncased": './pretrain_model/bert-base-uncased',
                "xlnet-base-cased-mosi": './pretrain_model/xlnet-base-cased_mosi',
                'roberta-base':'./pretrain_model/roberta-base'
            }
        }

    def get(self, section, name):
        return self.config_dict[section][name]


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    choices=["mosi", "mosei"], default="mosei")
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=64)

parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=50)
parser.add_argument("--beta_shift", type=float, default=1.0)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument(
    "--model",
    type=str,
    choices=["bert-large-uncased", "xlnet-base-cased", "xlnet-base-cased-mosi", "bert-base-uncased", "roberta-base", 'xlnet-large-cased'],
    default="xlnet-base-cased",
)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument('--num_heads', type=int, default=1,
                    help='number of heads for the transformer network (MOSEI:1、MOSI:12)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')
parser.add_argument('--embed_dropout', type=float, default=0.5,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.5,
                    help='residual block dropout')
parser.add_argument('--attn_dropout', type=float, default=0.5,
                    help='attention dropout')
parser.add_argument('--relu_dropout', type=float, default=0.5,
                    help='relu dropout')
parser.add_argument('--nlevels', type=int, default=12,
                    help='number of layers in the network (default: 12)')
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')
parser.add_argument('--attn_dropout_a', type=float, default=0.5,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.5,
                    help='attention dropout (for visual)')


DEVICE = torch.device("cuda:2")
INJECTION_INDEX = 0
args = parser.parse_args()
# MOSI SETTING
# ACOUSTIC_DIM = 74
# VISUAL_DIM = 47
#MOSEI SETTING
ACOUSTIC_DIM = 74
VISUAL_DIM = 35

if 'base' in args.model:
    TEXT_DIM = 768
elif 'large' in args.model:
    TEXT_DIM = 1024