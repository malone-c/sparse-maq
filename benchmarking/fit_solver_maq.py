if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--n', type=int)
    parser.add_argument('--k', type=int)
    parser.add_argument('--solver', type=str, choices=['maq', 'sparse_maq'])
    args = parser.parse_args()

    if args.solver == 'sparse_maq':
        import polars as pl
        from sparse_maq import Solver

        treatments = pl.read_parquet('data/{args.n=}_{args.k=}/sparse_maq/treatments.parquet')
        patients = pl.read_parquet('data/{args.n=}_{args.k=}/sparse_maq/patients.parquet')
        df = pl.read_parquet('data/{args.n=}_{args.k=}/sparse_maq/data.parquet')

        solver = Solver(treatments, patients)
        solver.fit(df)
    else:
        import numpy as np
        from maq import MAQ

        reward = np.load(f'data/{args.n=}_{args.k=}/maq/reward.npy')
        cost = np.load(f'data/{args.n=}_{args.k=}/maq/cost.npy')
        solver = MAQ()
        solver.fit(reward, cost, reward)

