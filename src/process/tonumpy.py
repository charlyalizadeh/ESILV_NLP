import numpy as np
import pandas as pd
import click


def get_numpy(df):
    df[[f'avis_{i}' for i in range(300)]] = df['avis'].str.split(pat=',', expand=True).astype(float)
    df.drop('avis', axis=1, inplace=True)
    return df.to_numpy()


@click.option('--inp')
@click.option('--out', required=False)
@click.command()
def convert_to_numpy(inp, out):
    if not out:
        out = f'{inp[:-3]}npy'
    df = pd.read_csv(inp, sep=';')
    df_np = get_numpy(df)
    np.save(out, df_np)


if __name__ == '__main__':
    convert_to_numpy()
