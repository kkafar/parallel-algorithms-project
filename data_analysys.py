import polars as pl
import matplotlib.pyplot as plt
from typing import Iterable
from dataclasses import dataclass
import argparse

@dataclass
class Args:
    file: Path

def configure_env():
    pl.Config.set_tbl_cols(100)
    pl.Config.set_tbl_rows(100)
    plt.rcParams['figure.figsize'] = (16, 9)


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('f', 'file', dest='file', type=Path, required=True, help='path to data file')
    return parser


def process_data(data_df: pl.DataFrame):
    path_counts = data_df.get_column('path_count').unique().sort()
    dt_series = data_df.get_column('dt').unique().sort()
    # plot_solution_time_by_dt(data_df, path_counts)
    plot_compute_time(data_df, dt=0.01)


def plot_solution_time_by_dt(data_df: pl.DataFrame, path_counts: Iterable[int]):
    fig_agg, plot_agg = plt.subplots(nrows=1, ncols=1)  # time_by_dt aggregated
    for i, path_count in enumerate(path_counts):
        # fig_pc, plot_pc = plt.subplots(nrows=1, ncols=1)  # time_by_dt per path_count
        pc_df = data_df.filter(pl.col('path_count') == path_count)

        x_data = pc_df['dt']
        y_data = pc_df['avg']
        y_err_data = pc_df['std']

        plot_agg.plot(x_data, y_data, label=f'PC: {path_count}', marker='o', linestyle='--')

        # plot_pc.errorbar(x_data, y_data, yerr=y_err_data, capsize=5)
        # plot_pc.set(
        #     title=f'Rozwiązanie dla {path_count} trajektorii liczonych jednoczesnie',
        #     xlabel='Krok czasowy dt',
        #     ylabel='Czas potrzebny na uzyskanie warunku |x| > 1'
        # )
        # fig_pc.savefig(f'plots/avgtime_by_dt_path_count_{path_count}.png')
        # plt.close(fig_pc)

    plot_agg.set(
        title=f'Rozwiązanie dla {path_count} trajektorii liczonych jednoczesnie',
        xlabel='Krok czasowy dt',
        ylabel='Czas potrzebny na uzyskanie warunku |x| > 1'
    )
    # plot_agg.legend()
    fig_agg.savefig(f'plots/avgtime_by_dt_aggregate.png')
    plt.close(fig_agg)


def plot_compute_time(data_df: pl.DataFrame, dt: float):
    print(f'Plotting for dt: {dt:.3f}')
    fig, plots = plt.subplots(nrows=1, ncols=2)
    data_df = data_df.filter(pl.col('dt') == dt).sort(pl.col('path_count'))

    t0 = data_df.filter(pl.col('path_count') == 1).item(0, 'time')

    data_df = data_df.with_columns([
        (t0 * pl.col('path_count') / pl.col('time')).alias('speedup')
    ])
    x_data = data_df.get_column('path_count')
    y_data = data_df.get_column('time')
    y_sp_data = data_df.get_column('speedup')

    plots[0].plot(x_data, y_data, marker='o', linestyle='--')
    plots[0].set(
        title=f'Czas obliczen od rozmiaru liczby bloków GPU (skalowanie słabe), dt: {dt:.3f}',
        xlabel='Liczba bloków GPU / liczonych ścieżek',
        ylabel='Czas wykonania obliczeń [us]'
    )
    plots[1].plot(x_data, y_sp_data, marker='o', linestyle='--')
    plots[1].set(
        title=f'Przyśpieszenie (słabe), dt: {dt:.3f}',
        ylabel='Speedup',
        xlabel='Liczba bloków GPU / liczonych ścieżek'
    )
    fig.savefig(f'plots/compute_time_by_path_count.png')
    plt.close(fig)


def main():
    args: Args = build_cli().parse_args()
    configure_env()

    # path_count,dt,avg,std,time
    data_df = pl.read_csv(args.file, has_header=True)
    process_data(data_df)




if __name__ == "__main__":
    main()







