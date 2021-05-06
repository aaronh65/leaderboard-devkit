
from dqn.src.offline.split_dataset
import matplotlib as mpl
mpl.use('TkAgg')

parser.add_argument('--dataset_dir', type=Path,
        default='/data/aaronhua/leaderboard/data/lbc/autopilot/autopilot_devtest_toy')
#parser.add_argument('--dataset_dir', type=Path, 
#        default='/data/aaronhua/leaderboard/data/lbc/autopilot/autopilot_devtest')
#parser.add_argument('--weights_path', type=str,
#        default='/data/aaronhua/leaderboard/training/lbc/20210405_225046/epoch=22.ckpt')
parser.add_argument('--n', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--save_visuals', action='store_true')
parser.add_argument('--throttle_mode', type=str, default='speed')
parser.add_argument('--max_speed', type=int, default=10)
parser.add_argument('--hard_prop', type=float, default=0.5)
parser.add_argument('--max_epochs', type=int, default=10)

args = parser.parse_args()
loader = get_dataloader(args, args.dataset_dir, is_train=True)
