import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix, cohen_kappa_score, precision_score, recall_score, classification_report, f1_score
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.utils import resample

def correlation_matrix(df):
    '''Produces correlation matrix from dataframe'''
    
    corr=np.abs(df.corr()) # corr(x,y) = corr(y, x), corr(x,x) = 1

    #Set up mask for triangle representation
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(16, 16))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask,  vmax=1,square=True, linewidths=.5, cbar_kws={"shrink": .5},annot = corr)
    
    plt.show()

def test_data_knn(df, x=3, y=11, rand_state=0):
    '''Evaluating the dataset for accuracy, kappa, and recall with a KNN ML model at a range of k-values
    Takes a starting dataframe (heart_df), and a range of k-values to test (x: min, y: max))
    Starting dataframe must contain 'HeartDiseaseorAttack' column
    Returns a dataframe of results'''
    
    target = df['HeartDiseaseorAttack']
    features = df.drop('HeartDiseaseorAttack', axis=1)
    
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=rand_state)
    
    results = []

    # Evaluate the model on the test set
    for i in range(x, y+1):
        print(f"\rRunning for n_neighbors: {i}", end=" ", flush=True)
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        accuracy = ((cm[0][0]+cm[1][1])/(sum(cm[0])+sum(cm[1])))

        results.append({
            "k": i,
            "Accuracy": accuracy,
            "Recall": recall,
            "Kappa": kappa
        })
    
    results_df = pd.DataFrame(results)

    return results_df

def test_normalised_data_knn(df, x=3, y=11, rand_state=0):
    '''Evaluating the dataset for accuracy, kappa, and recall with a KNN ML model at a range of k-values
    Takes a starting dataframe (heart_df), and a range of k-values to test (x: min, y: max))
    Normalised all columns for min-max to 0-1
    Starting dataframe must contain 'HeartDiseaseorAttack' column
    Returns a dataframe of results'''
    
    target = df['HeartDiseaseorAttack']
    features = df.drop('HeartDiseaseorAttack', axis=1)
    
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=rand_state)
    
    #Normalise all columns to be 0-1
    normalizer = MinMaxScaler()
    normalizer.fit(x_train)
    
    x_train_norm = normalizer.transform(x_train)
    x_test_norm = normalizer.transform(x_test)
    
    x_train_norm = pd.DataFrame(x_train_norm, columns=x_train.columns, index=x_train.index )
    x_test_norm = pd.DataFrame(x_test_norm, columns=x_test.columns, index=x_test.index)
    results = []

    # Evaluate the model on the test set
    for i in range(x, y+1):
        print(f"\rRunning for n_neighbors: {i}", end=" ", flush=True)
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train_norm, y_train)
        y_pred = knn.predict(x_test_norm)
        cm = confusion_matrix(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        accuracy = ((cm[0][0]+cm[1][1])/(sum(cm[0])+sum(cm[1])))

        results.append({
            "k": i,
            "Accuracy": accuracy,
            "Recall": recall,
            "Kappa": kappa
        })
    results_df = pd.DataFrame(results)

    return results_df

def test_col_pairs(df, k_neighbours=16, rand_state=0):
    '''Tests the knn model using every combination of two columns from the training data
    Input a dataframe, number of neighnboutrs, and random state to use
    Returns a dataframe of the results
    '''
    
    target = df['HeartDiseaseorAttack']
    features = df.drop('HeartDiseaseorAttack', axis=1)
    
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=rand_state)
    
    #Normalise all columns to be 0-1
    normalizer = MinMaxScaler()
    normalizer.fit(x_train)
    
    x_train_norm = normalizer.transform(x_train)
    x_test_norm = normalizer.transform(x_test)
    
    x_train_norm = pd.DataFrame(x_train_norm, columns=x_train.columns, index=x_train.index )
    x_test_norm = pd.DataFrame(x_test_norm, columns=x_test.columns, index=x_test.index)
    results = []

    
    for i in range(len(x_train_norm.columns) - 1):
        col1 = x_train_norm.columns[i]
      
        for j in range(i + 1, len(x_train_norm.columns)):
            col2 = x_train_norm.columns[j]
        

            print(f"\rRunning for {col1} & {col2}", end="           ", flush=False )    
            knn = KNeighborsClassifier(n_neighbors=k_neighbours)
            knn.fit(x_train_norm[[col1, col2]], y_train)
            y_pred = knn.predict(x_test_norm[[col1, col2]])
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Calculate accuracy
            accuracy = (cm[0][0] + cm[1][1]) / (sum(cm[0]) + sum(cm[1]))

            recall = recall_score(y_test, y_pred)
            
            # Calculate Kappa score
            kappa = cohen_kappa_score(y_test, y_pred)
                        
            #print(f" Accuracy: {100*(accuracy): .2f}%")
            #print(f"Cohen's Kappa: {kappa:.4f}")
            #print(cm)
            #print()

            results.append({
                "col1": col1,
                "col2": col2,
                "kappa": kappa,
                "recall": recall,
                "accuracy": accuracy
            })

    results_df = pd.DataFrame(results) 
    
    df2 = results_df.rename(columns={"col1": "temp", "col2": "col1"})
    df2 = df2.rename(columns={"temp": "col2"})
    df2 = df2[['col1', 'col2', 'kappa', 'recall', 'accuracy']]
    
    result_df = pd.concat([results_df, df2], axis=0)   
    return result_df

def test_weights(df, columns_to_weight, weight_values=[0, 0.5, 1], k_neighbors=5, rand_state=0):
    '''Evaluating the dataset for accuracy, kappa, and recall with a KNN ML model at different feature weights for selected columns.
    Takes a starting dataframe (heart_df), applies weights to specified columns, and evaluates model performance.
    Returns a dataframe of results.'''
    
    target = df['HeartDiseaseorAttack']
    features = df.drop('HeartDiseaseorAttack', axis=1)
    
    # Ensure columns_to_weight are in the features dataframe
    assert all(col in features.columns for col in columns_to_weight), \
        "All columns in 'columns_to_weight' must exist in the dataframe."

    # Split into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=rand_state)
    
    # Initialize results list
    results = []
    total_combinations = len(weight_values) ** len(columns_to_weight)  # Calculate the total number of weight combinations
    print(f"Total combinations: {total_combinations}")

    # Loop through all combinations of feature weights for specified columns using itertools.product
    for idx, weights in enumerate(itertools.product(weight_values, repeat=len(columns_to_weight))):
        print(f"Running combination {idx + 1}/{total_combinations} with weights: {weights}", end="            \r")
        
        # Create a copy of the original data for modification (no normalization applied)
        weighted_x_train = x_train.copy()
        weighted_x_test = x_test.copy()
        
        # Apply weights to the specified columns only
        for col, weight in zip(columns_to_weight, weights):
            weighted_x_train[col] *= weight
            weighted_x_test[col] *= weight
        
        # Train KNN classifier (with a fixed k, e.g., k=5)
        knn = KNeighborsClassifier(n_neighbors=k_neighbors)
        knn.fit(weighted_x_train, y_train)
        y_pred = knn.predict(weighted_x_test)
        
        # Compute performance metrics
        cm = confusion_matrix(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)      
        accuracy = (cm[0][0] + cm[1][1]) / (sum(cm[0]) + sum(cm[1]))
        
        # Store the result
        results.append({
            "Weights": weights,
            "Accuracy": accuracy,
            "Recall": recall,
            "Kappa": kappa
        })
    
    # Convert results to dataframe
    results_df = pd.DataFrame(results)

    # Sort by recall score in descending order
    results_df = results_df.sort_values(by="Recall", ascending=False)

    # Copy results to a new dataframe
    df_new = results_df.copy()

    # Dynamically create new columns for each column in columns_to_weight
    for i, col in enumerate(columns_to_weight):
        # Assign the corresponding weight from the 'Weights' tuple to each column
        df_new[col] = df_new['Weights'].apply(lambda x: x[i])
    
    # Drop the 'Weights' column as we now have separate columns for each feature
    df_new.drop(columns=['Weights'], inplace=True)

    return df_new