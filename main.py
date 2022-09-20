import pandas as pd
from log import *
from eval import *
from utils import *
from train import *
#import numba
from module import CAWN
from graph import NeighborFinder
import resource
import sys
import pickle

args, sys_argv = get_args() #sys_argv is list

print(f"This is the type of args {type(args)}.")
print(args)

with open("args.pickle", mode="wb") as f:
    pickle.dump(args,f)
    print("Saving args as pickle.")

with open("args.pickle", mode="rb") as f:
    sys_argv=pickle.load(f)
    print(sys_argv)

sys.exit()

print(f"This is the type of sys_argv {type(sys_argv)}.")
print(sys_argv)

sys_argv=['main.py', '-d', 'uci', '--pos_dim', '100', '--bs', '32', '--n_degree', '32', '--n_layer', '1', '--mode', 't', '--bias', '1e-6', '--pos_enc', 'lp', '--walk_pool', 'attn', '--seed', '123']

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
ATTN_NUM_HEADS = args.attn_n_head
DROP_OUT = args.drop_out
GPU = args.gpu
USE_TIME = args.time
ATTN_AGG_METHOD = args.attn_agg_method
ATTN_MODE = args.attn_mode
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
POS_ENC = args.pos_enc
POS_DIM = args.pos_dim
WALK_POOL = args.walk_pool
WALK_N_HEAD = args.walk_n_head
WALK_MUTUAL = args.walk_mutual if WALK_POOL == 'attn' else False
TOLERANCE = args.tolerance
CPU_CORES = args.cpu_cores
NGH_CACHE = args.ngh_cache
VERBOSITY = args.verbosity
AGG = args.agg
SEED = args.seed


# BATCH_SIZE = 64
# NUM_NEIGHBORS = ['64','1']
# NUM_EPOCH = 50
# ATTN_NUM_HEADS = 2
# DROP_OUT = 0.1
# GPU = 0
# USE_TIME = 'time'
# ATTN_AGG_METHOD = 'attn' #attn, lstm, mean
# ATTN_MODE = 'prod' #prod or map
# DATA = 'uci'
# NUM_LAYER = 2
# LEARNING_RATE = 1e-4
# POS_ENC = 'lp' #spd, lp saw
# POS_DIM = 172
# WALK_POOL = 'attn' #attn or sum
# WALK_N_HEAD = 8
# WALK_MUTUAL = True if WALK_POOL == 'attn' else False #walk_mutual が与えられたらTrue
# TOLERANCE = 0
# CPU_CORES = 1 #numbers of cpu_cores used for position encoding
# NGH_CACHE = True #store_true
# VERBOSITY = 1
# AGG = 'walk' #walk or tree
# SEED = 123

# #Add selfly
# DATA_USAGE=1.0 #args.data_usage
# MODE='t' #args.mode
# BIAS=0.0
# SAMPLE_METHOD='binary' #binary or multinomial
# WALK_LINEAR_OUT=True #与えられればTrue
# args.datausage=DATA_USAGE
# args.mode=MODE
# args.bias=BIAS
# args.sample_method=SAMPLE_METHOD
# args.walk_linear_out=WALK_LINEAR_OUT

assert(CPU_CORES >= -1)
set_random_seed(SEED)
logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys_argv)

# Load data and sanity check
g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))
if args.data_usage < 1:
    g_df = g_df.iloc[:int(args.data_usage*g_df.shape[0])]
    logger.info('use partial data, ratio: {}'.format(args.data_usage))
e_feat = np.load('./processed/ml_{}.npy'.format(DATA))
n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA))
# u, iはカラム名を表す。つまり特定の列を取り出している。ここでuはsourcd node idx, iはdstination node idx
src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values
#max_idxはノードidの最大値を保持している。つまり、最大のインデックスである
max_idx = max(src_l.max(), dst_l.max())
assert(np.unique(np.stack([src_l, dst_l])).shape[0] == max_idx or ~math.isclose(1, args.data_usage))  # all nodes except node 0 should appear and be compactly indexed
assert(n_feat.shape[0] == max_idx + 1 or ~math.isclose(1, args.data_usage))  # the nodes need to map one-to-one to the node feat matrix

# split and pack the data by generating valid train/val/test mask according to the "mode"
# quantileは四分位数のことである。つまり、全体でちょうど70%までがtrain set. 70~85までがvalidation set.　残りがtest set
val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))
if args.mode == 't':
    logger.info('Transductive training...')
    valid_train_flag = (ts_l <= val_time)
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
    valid_test_flag = ts_l > test_time

else:
    assert(args.mode == 'i')
    logger.info('Inductive training...')
    # pick some nodes to mask (i.e. reserved for testing) for inductive setting
    total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values]))) # np.stackで配列を結合し、np.uiqueでユニークな要素に限る
    num_total_unique_nodes = len(total_node_set) # src, dst nodeの組み合わせ
    # np.unionで和集合を取る。random.sample(リスト, 整数)で重複なし整数個の要素を取得したリストを返す
    mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])), int(0.1 * num_total_unique_nodes)))
    # map 関数で配列の各要素に関数を適用する。今回はg_dfの各インデックスがmask_node_setに含まれていればTrueに変更している。
    mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
    mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
    none_mask_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)
    valid_train_flag = (ts_l <= val_time) * (none_mask_node_flag > 0.5)
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time) * (none_mask_node_flag > 0.5)  # both train and val edges can not contain any masked nodes
    valid_test_flag = (ts_l > test_time) * (none_mask_node_flag < 0.5)  # test edges must contain at least one masked node
    valid_test_new_new_flag = (ts_l > test_time) * mask_src_flag * mask_dst_flag
    valid_test_new_old_flag = (valid_test_flag.astype(int) - valid_test_new_new_flag.astype(int)).astype(bool)
    logger.info('Sampled {} nodes (10 %) which are masked in training and reserved for testing...'.format(len(mask_node_set)))

# split data according to the mask
train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l = src_l[valid_train_flag], dst_l[valid_train_flag], ts_l[valid_train_flag], e_idx_l[valid_train_flag], label_l[valid_train_flag]
val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l = src_l[valid_val_flag], dst_l[valid_val_flag], ts_l[valid_val_flag], e_idx_l[valid_val_flag], label_l[valid_val_flag]
test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l = src_l[valid_test_flag], dst_l[valid_test_flag], ts_l[valid_test_flag], e_idx_l[valid_test_flag], label_l[valid_test_flag]
if args.mode == 'i':
    test_src_new_new_l, test_dst_new_new_l, test_ts_new_new_l, test_e_idx_new_new_l, test_label_new_new_l = src_l[valid_test_new_new_flag], dst_l[valid_test_new_new_flag], ts_l[valid_test_new_new_flag], e_idx_l[valid_test_new_new_flag], label_l[valid_test_new_new_flag]
    test_src_new_old_l, test_dst_new_old_l, test_ts_new_old_l, test_e_idx_new_old_l, test_label_new_old_l = src_l[valid_test_new_old_flag], dst_l[valid_test_new_old_flag], ts_l[valid_test_new_old_flag], e_idx_l[valid_test_new_old_flag], label_l[valid_test_new_old_flag]
train_data = train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l
val_data = val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l
train_val_data = (train_data, val_data)

# create two neighbor finders to handle graph extraction.
# for transductive mode all phases use full_ngh_finder, for inductive node train/val phases use the partial one
# while test phase still always uses the full one
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, bias=args.bias, use_cache=NGH_CACHE, sample_method=args.pos_sample)
partial_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    partial_adj_list[src].append((dst, eidx, ts))
    partial_adj_list[dst].append((src, eidx, ts))
for src, dst, eidx, ts in zip(val_src_l, val_dst_l, val_e_idx_l, val_ts_l):
    partial_adj_list[src].append((dst, eidx, ts))
    partial_adj_list[dst].append((src, eidx, ts))
partial_ngh_finder = NeighborFinder(partial_adj_list, bias=args.bias, use_cache=NGH_CACHE, sample_method=args.pos_sample)
ngh_finders = partial_ngh_finder, full_ngh_finder

# create random samplers to generate train/val/test instances
train_rand_sampler = RandEdgeSampler((train_src_l, ), (train_dst_l, ))
val_rand_sampler = RandEdgeSampler((train_src_l, val_src_l), (train_dst_l, val_dst_l))
test_rand_sampler = RandEdgeSampler((train_src_l, val_src_l, test_src_l), (train_dst_l, val_dst_l, test_dst_l))
rand_samplers = train_rand_sampler, val_rand_sampler

# multiprocessing memory setting
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (200*args.bs, rlimit[1]))

# model initialization
device = torch.device('cuda:{}'.format(GPU))
cawn = CAWN(n_feat, e_feat, agg=AGG,
            num_layers=NUM_LAYER, use_time=USE_TIME, attn_agg_method=ATTN_AGG_METHOD, attn_mode=ATTN_MODE,
            n_head=ATTN_NUM_HEADS, drop_out=DROP_OUT, pos_dim=POS_DIM, pos_enc=POS_ENC,
            num_neighbors=NUM_NEIGHBORS, walk_n_head=WALK_N_HEAD, walk_mutual=WALK_MUTUAL, walk_linear_out=args.walk_linear_out, walk_pool=args.walk_pool,
            cpu_cores=CPU_CORES, verbosity=VERBOSITY, get_checkpoint_path=get_checkpoint_path)
cawn.to(device)
optimizer = torch.optim.Adam(cawn.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
early_stopper = EarlyStopMonitor(tolerance=TOLERANCE)

# start train and val phases
# ngh_findersとは接続ノードを探すものでpartialとfullがある
train_val(train_val_data, cawn, args.mode, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, ngh_finders, rand_samplers, logger)

# final testing
cawn.update_ngh_finder(full_ngh_finder)  # remember that testing phase should always use the full neighbor finder
test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for {} nodes'.format(args.mode), cawn, test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l)
logger.info('Test statistics: {} all nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_acc, test_auc, test_ap))
test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc = [-1]*6
if args.mode == 'i':
    test_new_new_acc, test_new_new_ap, test_new_new_f1, test_new_new_auc = eval_one_epoch('test for {} nodes'.format(args.mode), cawn, test_rand_sampler, test_src_new_new_l, test_dst_new_new_l, test_ts_new_new_l, test_label_new_new_l, test_e_idx_new_new_l)
    logger.info('Test statistics: {} new-new nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_new_new_acc, test_new_new_auc,test_new_new_ap ))
    test_new_old_acc, test_new_old_ap, test_new_old_f1, test_new_old_auc = eval_one_epoch('test for {} nodes'.format(args.mode), cawn, test_rand_sampler, test_src_new_old_l, test_dst_new_old_l, test_ts_new_old_l, test_label_new_old_l, test_e_idx_new_old_l)
    logger.info('Test statistics: {} new-old nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_new_old_acc, test_new_old_auc, test_new_old_ap))

# save model
logger.info('Saving CAWN model ...')
torch.save(cawn.state_dict(), best_model_path)
logger.info('CAWN model saved')

# save one line result
save_oneline_result('log/', args, [test_acc, test_auc, test_ap, test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc])
# save walk_encodings_scores
# checkpoint_dir = '/'.join(cawn.get_checkpoint_path(0).split('/')[:-1])
# cawn.save_walk_encodings_scores(checkpoint_dir)
