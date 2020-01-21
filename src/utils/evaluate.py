import click
import os
import re
import glob
import pandas as pd


@click.command()
@click.option('--history_dir', help='Path to .h5 file.')
def main(history_dir):

    columns = ['name', 'max_val', 'max_train', 'lr', 'w', 'd', 'bs', 'loss']
    lname, lmax_val, lmax_train, llr, ld, lw, lbs, lloss = [], [], [], [], [], [], [], []

    df_master = pd.DataFrame(columns=columns)
    models = glob.glob(f'{history_dir}/*.csv')

    for model in models:
        if os.stat(model).st_size == 0: continue

        df = pd.read_csv(model)

        lname.append(model)
        lmax_val.append(max(df['val_categorical_accuracy']))
        lmax_train.append(max(df['categorical_accuracy']))

        lr = re.search(r'lr-(\d\.\d*)', model)
        lr = lr.group(1) if lr else 0
        llr.append(lr)

        w = re.search(r'w-(\d*)', model)
        w = w.group(1) if w else 0
        lw.append(w)

        d = re.search(r'd-(\d*)', model)
        d = d.group(1) if d else 0
        ld.append(d)

        bs = re.search(r'bs-(\d*)', model)
        bs = bs.group(1) if bs else 0
        lbs.append(bs)

        loss = re.search(r'loss-(\d)', model)
        loss = loss.group(1) if loss else 0
        lloss.append(loss)

    df_master['name'] = pd.Series(lname)
    df_master['max_val'] = pd.Series(lmax_val)
    df_master['max_train'] = pd.Series(lmax_train)
    df_master['lr'] = pd.Series(llr)
    df_master['w'] = pd.Series(lw)
    df_master['d'] = pd.Series(lw)
    df_master['bs'] = pd.Series(lbs)
    df_master['loss'] = pd.Series(lloss)
    df_master.to_csv('./output.csv')


if __name__ == "__main__":
    main()


# Plotting
# import pandas as pd
# import seaborn as sns
# df = pd.read_csv('/Users/beichenberger/Github/fluffy-guide/src/output.csv', index_col=1)
# df = df.drop(['Unnamed: 0'], axis=1)
# sns.heatmap(df.corr(), center=0, cmap='vlag', linewidths=.75)
# plt.show()
