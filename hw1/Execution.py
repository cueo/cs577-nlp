import pandas as pd

"""
Execution.py is for evaluating your models on the datasets available to you. You can use 
this program to test the accuracy of your models by calling it in the following way:

    import Execution
    Execution.eval(o_train, p_train, o_test, p_test)

In the sample code, o_train is the observed training labels, p_train is the predicted training labels, o_test is the 
observed test labels, and p_test is the predicted test labels. 
"""


def split_dataset(all_data):
    """
    This function will take in as input the whole dataset and you will have to program how to split the dataset into
    training and test datasets. These are the following requirements:
        -The function must take only one parameter which is all_data as a pandas dataframe of the raw dataset.
        -It must return 2 outputs in the specified order: train and test datasets

    It is up to you how you want to do the splitting of the data.
    """
    train_data, test_data = all_data.iloc[:int(len(all_data) * 0.8), :], all_data.iloc[int(len(all_data) * 0.8):, :]
    return train_data, test_data


"""
This function should not be changed at all.
"""


def eval(o_train, p_train, o_val, p_val, o_test, p_test):
    print('\nTraining Accuracy Result!')
    accuracy(o_train, p_train)
    print('\nTesting Accuracy Result!')
    accuracy(o_val, p_val)
    print('\nUnseen Test Set Accuracy Result!')
    accuracy(o_test, p_test)


"""
This function should not be changed at all.
"""


def accuracy(orig, pred):
    num = len(orig)
    if (num != len(pred)):
        print('Error!! Num of labels are not equal.')
        return
    match = 0
    for i in range(len(orig)):
        o_label = orig[i]
        p_label = pred[i]
        if (o_label == p_label):
            match += 1
    print('***************\nAccuracy: ' + str(float(match) / num) + '\n***************')


if __name__ == '__main__':
    """
    The code below these comments must not be altered in any way. This code is used to evaluate the predicted labels of
    your models against the ground-truth observations.
    """
    from Logistic import Logistic
    from Ensemble import Ensemble

    all_data = pd.read_csv('data.csv', index_col=0)
    train_data, test_data = split_dataset(all_data)

    # placeholder dataset -> when we run your code this will be an unseen test set your model will be evaluated on
    test_data_unseen = pd.read_csv('test_data.csv', index_col=0)

    logistic = Logistic()
    logistic.train(train_data)
    predicted_train_labels_logistic = logistic.predict(train_data)
    predicted_test_labels_logistic = logistic.predict(test_data)
    predicted_test_labels_unseen_logistic = logistic.predict(test_data_unseen)

    ensemble = Ensemble()
    ensemble.train(train_data)
    predicted_train_labels_ensemble = ensemble.predict(train_data)
    predicted_test_labels_ensemble = ensemble.predict(test_data)
    predicted_test_labels_unseen_ensemble = ensemble.predict(test_data_unseen)

    print('\n\n-------------Logistic Function Performance-------------\n')
    # This command also runs the evaluation on the unseen test
    eval(train_data['Label'].tolist(), predicted_train_labels_logistic, test_data['Label'].tolist(),
         predicted_test_labels_logistic, test_data_unseen['Label'].tolist(), predicted_test_labels_unseen_logistic)

    print('\n\n-------------Ensemble Method Performance-------------\n')
    eval(train_data['Label'].tolist(), predicted_train_labels_ensemble, test_data['Label'].tolist(),
         predicted_test_labels_ensemble, test_data_unseen['Label'].tolist(), predicted_test_labels_unseen_ensemble)
