from exp.exp import EXP

from utils.setseed import set_seed

from exp.exp import EXP

from utils.setseed import set_seed

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # 有关实验模型的参数
    parser.add_argument('--model_name', default='DeepTD_LSP', type=str, help='[Our_model,baseline:Fedformer]')
    parser.add_argument('--train', default=False, type=str, help='if train')
    parser.add_argument('--resume', default=False, type=str, help='resume from checkpoint')
    parser.add_argument('--loss', default='quantile', type=str, help='quantile,normal')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    # 调用不同的实验，有些模型需要特殊的实验操作
    parser.add_argument('--exp', default='deep_learning', type=str, help='[deep_learning]')

    # 实验数据相关参数
    parser.add_argument('--data_name', default='traffic', type=str, help='[data:ETTh1,electricity,exchange,ETTm1]')
    parser.add_argument('--seq_len', default=96, type=int, help='input sequence len')
    parser.add_argument('--label_len', default=48, type=int, help='transfomer decoder input part')
    parser.add_argument('--pred_len', default=96, type=int, help='prediction len')
    parser.add_argument('--d_mark', default=4, type=int, help='date embed dim')
    parser.add_argument('--d_feature', default=862, type=int,
                        help='input data feature dim without date :[Etth1:7 , electricity:321,exchange:8,ETTm1:7,illness:7,traffic:863]')
    parser.add_argument('--c_out', type=int, default=862, help='output size')

    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')

    # 超参
    parser.add_argument('--d_model', default=510, type=int, help='feature dim in model')
    parser.add_argument('--d_ff', default=1024, type=int, help='feature dim2 in model')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_d', default=0.05, type=float, help='initial learning rate for discriminator of gan')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--epoches', default=200, type=int, help='Train Epoches')
    parser.add_argument('--patience', default=5, type=int, help='Early Stop patience')

    parser.add_argument('--quantiles', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], type=int,
                        help='分位数损失的ρ参数')

    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
    parser.add_argument('--n_heads', type=int, default=3, help='num of heads')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    # patch
    parser.add_argument('--patch_len', type=int, default=16, help='patch长度')
    parser.add_argument('--stride', type=int, default=8, help='每一个patch的间隔')

    parser.add_argument('--stronger', type=int, default=2, help='Top-K Fourier bases')
    parser.add_argument('--fourier_decomp_ratio', type=float, default=0.5, help='频率拆解模块参数设置')

    args = parser.parse_args()

    for seed in range(1, 2):
        set_seed(seed)
        args.seed = seed

        if args.exp == 'deep_learning':
            exp = EXP(args)
            if args.train:
                exp.train()
            exp.test()

