# plot code to compare predicted vs actual values
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_predictions(y_pred, y_test, y_ct, datetime, title):
    fig, ax = plt.subplots(1,1, figsize=(20, 10 ))
    ax.plot(y_pred, label='Predicted')
    ax.plot(y_ct, label='GPR')
    ax.plot(y_test, label='Ground Truth')

    # plt.gcf().autofmt_xdate()
    # ax.set_xticklabels(datetime, rotation=45, ha='right')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize = 17)
    # plt.xlabel('Timestamp')
    plt.title(title, fontsize = 20)
    plt.tight_layout()
    plt.savefig(title + '.png')


# main function
if __name__ == '__main__':

    pred_path = "/Users/zimenglyu/Documents/code/git/susi/202303_202105_202209_30_5_30_standard_weightAVG_lab_prediction.csv"
    test_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_lab_results.csv"
    # ct_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_ct.csv"
    gpr_path = "/Users/zimenglyu/Documents/code/git/dataset_toolbox/toolbox/results/GPR/202303_202105_202209_GPR_RBF_minmax.csv"
    
    y_pred = pd.read_csv(pred_path)
    y_test = pd.read_csv(test_path)
    y_gpr = pd.read_csv(gpr_path)
    y_test['DateTime']= pd.to_datetime(y_test['DateTime'])
    datetime_string = y_test['DateTime']
    # print(datetime_string)
    # y_test.set_index('DateTime', inplace=True)
    headers = y_pred.columns
    # print(headers)
    # plot predictions
    n = y_pred.shape[1]
    num_rows = min(y_pred.shape[0], y_test.shape[0])
    
    for i in range(1, n):
        plot_predictions(y_pred.iloc[:num_rows,i], y_test.iloc[:num_rows,i], y_gpr.iloc[:num_rows,i], datetime_string, "202303_202105_202209_GPR_RBF_" + headers[i].split("_")[0])
