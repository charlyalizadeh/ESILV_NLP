import click
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
import numpy as np
from random_forest import train_random_forest, predict_random_forest
from ffnn import train_ffnn, predict_ffnn
from os.path import basename


train_func = {
    'random_forest': train_random_forest,
    'ffnn': train_ffnn
}
predict_func = {
    'random_forest': predict_random_forest,
    'ffnn': predict_ffnn
}


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.pass_context
@click.argument('datapath')
@click.argument('modelname')
@click.argument('out', type=click.File('w'), required=False)
def train_supervised(ctx, datapath, modelname, out):

    d = dict([item.strip('--').split('=') for item in ctx.args])
    d = dict([(k, int(v)) if v.isdigit() else (k, v) for k, v in d.items()])
    train = np.load(datapath)
    X, y = train[:, 1:], train[:, 0]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    if modelname == 'ffnn':
        d['X_val'] = X_val
        d['y_val'] = y_val
        d['loss_fig_out'] = f'./report/images/loss_{basename(datapath)[6:-4]}.pgf'

    model = train_func[modelname](X_train, y_train, **d)

    y_train_pred = predict_func[modelname](X_train, model)
    y_val_pred = predict_func[modelname](X_val, model)
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    train_rmse = mean_squared_error(y_train, y_train_pred, squared=True)
    val_rmse = mean_squared_error(y_val, y_val_pred, squared=True)
    train_cm = confusion_matrix(y_train, y_train_pred, labels=[0, 1, 2, 3, 4])
    val_cm = confusion_matrix(y_val, y_val_pred, labels=[0, 1, 2, 3, 4])

    if not out:
        click.echo(f'{modelname} on {datapath} \n\
Train Accuracy({train_acc}) RMSE({train_rmse}) \n \
Confusion matrix:\n {train_cm} \n\
Val \n Accuracy({val_acc}) RMSE: {val_rmse} \n \
Confusion matrix:\n {val_cm}\n')
    else:
        click.echo(f'{modelname} on {datapath} \n\
Train Accuracy({train_acc}) RMSE({train_rmse}) \n \
Confusion matrix:\n {train_cm} \n\
Val Accuracy({val_acc}) RMSE({val_rmse}) \n\
Confusion matrix:\n {val_cm}\n',
                   file=out)


if __name__ == '__main__':
    train_supervised()
