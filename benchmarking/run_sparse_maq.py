import polars as pl
from sparse_maq import Solver

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--base-path', type=str)
    args = parser.parse_args()

    treatments = pl.read_parquet(f'{args.base_path}/treatments.parquet')
    patients = pl.read_parquet(f'{args.base_path}/patients.parquet')
    data = pl.read_parquet(f'{args.base_path}/data.parquet')
    
    solver = Solver()
    solver.fit_from_polars(data)

