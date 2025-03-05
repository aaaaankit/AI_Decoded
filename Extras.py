## EDA
## get the columns of X_train
#columns = X_train.columns
#print(columns)
#
#"""
#Index(['sex', 'age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
#       'r_days_from_arrest', 'is_recid', 'race_Asian', 'race_Caucasian',
#       'race_Hispanic', 'race_Native American', 'race_Other',
#       'c_charge_degree_(CT)', 'c_charge_degree_(F1)', 'c_charge_degree_(F2)',
#       'c_charge_degree_(F3)', 'c_charge_degree_(F5)', 'c_charge_degree_(F6)',
#       'c_charge_degree_(F7)', 'c_charge_degree_(M1)', 'c_charge_degree_(M2)',
#       'c_charge_degree_(MO3)', 'c_charge_degree_(NI0)',
#       'c_charge_degree_(TCX)', 'c_charge_degree_(X)',
#       'c_charge_degree_Unknown', 'r_charge_degree_(F1)',
#       'r_charge_degree_(F2)', 'r_charge_degree_(F3)', 'r_charge_degree_(F5)',
#       'r_charge_degree_(F6)', 'r_charge_degree_(F7)', 'r_charge_degree_(M1)',
#       'r_charge_degree_(M2)', 'r_charge_degree_(MO3)',
#       'r_charge_degree_Unknown', 'vr_charge_degree_(F2)',
#       'vr_charge_degree_(F3)', 'vr_charge_degree_(F5)',
#       'vr_charge_degree_(F6)', 'vr_charge_degree_(F7)',
#       'vr_charge_degree_(M1)', 'vr_charge_degree_(M2)',
#       'vr_charge_degree_(MO3)', 'vr_charge_degree_Unknown'],
#      dtype='object')
#"""
#
#X_train_reduced_race = X_train[['race_Asian', 'race_Caucasian', 'race_Hispanic', 'race_Native American', 'race_Other']]
#
##X_train_reduced_c_charge_degree = X_train[['c_charge_degree_(CT)', 'c_charge_degree_(F1)', 'c_charge_degree_(F2)',
##       'c_charge_degree_(F3)', 'c_charge_degree_(F5)', 'c_charge_degree_(F6)',
##       'c_charge_degree_(F7)', 'c_charge_degree_(M1)', 'c_charge_degree_(M2)',
##       'c_charge_degree_(MO3)', 'c_charge_degree_(NI0)',
##       'c_charge_degree_(TCX)', 'c_charge_degree_(X)',
##       'c_charge_degree_Unknown']]
##
##X_train_reduced_r_charge_degree = X_train[['r_charge_degree_(F1)',
##       'r_charge_degree_(F2)', 'r_charge_degree_(F3)', 'r_charge_degree_(F5)',
##       'r_charge_degree_(F6)', 'r_charge_degree_(F7)', 'r_charge_degree_(M1)',
##       'r_charge_degree_(M2)', 'r_charge_degree_(MO3)',
##       'r_charge_degree_Unknown']]
##
##X_train_reduced_vr_charge_degree = X_train[['vr_charge_degree_(F2)',
##       'vr_charge_degree_(F3)', 'vr_charge_degree_(F5)',
##       'vr_charge_degree_(F6)', 'vr_charge_degree_(F7)',
##       'vr_charge_degree_(M1)', 'vr_charge_degree_(M2)',
##       'vr_charge_degree_(MO3)', 'vr_charge_degree_Unknown']]
#
#
## List of DataFrames and corresponding titles
#dataframes = [
#    (X_train, "Correlation Matrix X_train"),
#    (X_train_reduced_race, "Correlation Matrix X_train_reduced_race"),
#    #(X_train_reduced_c_charge_degree, "Correlation Matrix X_train_reduced_c_charge_degree"),
#    #(X_train_reduced_r_charge_degree, "Correlation Matrix X_train_reduced_r_charge_degree"),
#    #(X_train_reduced_vr_charge_degree, "Correlation Matrix X_train_reduced_vr_charge_degree")
#]
#
## Loop through each DataFrame, compute the correlation matrix, plot, and save the figure
#for df, title in dataframes:
#    corr_matrix = df.corr()
#
#    plt.figure(figsize=(12, 8))
#    if title != "Correlation Matrix X_train":
#        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
#    else:
#        sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5)
#    
#    plt.title(title)
#
#    # Save the figure as a PNG file
#    plt.savefig(f"{title}.png", bbox_inches='tight')
#
#    # Show the plot (optional)
#    plt.show()


#df = pd.read_csv('Dataset/cox-violent-parsed_filt_processed.csv')