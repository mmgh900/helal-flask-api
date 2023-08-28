import pickle
from http.client import HTTPException

import numpy as np
import pandas as pd
import pymssql
import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os


class DaysOpenModel:

    def __init__(self, conn, force_retrain= False):
        self.conn = conn
        directory = 'days-open-model'
        self.get_data()
        if os.path.exists(directory) and not force_retrain:
            self.model = tf.keras.models.load_model(filepath=directory)
            with open('days-open-params.pkl', 'rb') as dop:
                storage = pickle.load(dop)
            self.mean_values = storage['mean_values']
            self.scaler = storage['scaler']
            print("Days Open Model Loaded")
        else:
            self.train_model()

    def get_data(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        WITH cte AS (SELECT ca.Serial,
                        DATEDIFF(day, m.EngBDate, ca.EngZDate)                       AS AgeAtCalvingInDaysA,
                        DATEDIFF(month, m.EngBDate, ca.EngZDate)                     AS AgeAtCalvingInMonthsA,
                        DATEDIFF(day, m.EngBDate, cb.EngZDate)                       AS AgeAtCalvingInDays,
                        DATEDIFF(month, m.EngBDate, cb.EngZDate)                     AS AgeAtCalvingInMonths,

                        last_bc.Score                                                AS MostRecentBodyScore,
                        last_ms.Score                                                AS MostRecentMotionScore,
                        bc_avg.Score                                                 AS AverageBodyScore,
                        ms_avg.Score                                                 AS AverageMotionScore,
                        m.IntBlood                                                   AS InbreedingCoefficient,

                        DATEDIFF(month, tb.EngTDate, cb.EngZDate)                    AS PregnancyLengthMonths,
                        sm.AllMilk                                                   as Milk,
                        sm.MilkDays                                                  as MilkDays,
                        sm.AllOil                                                    as MilkFat,
                        sm.AllPro                                                    as MilkProtein,
                        CASE WHEN cb.Zsituation LIKE '%مرده%' THEN 1 ELSE 0 END AS StillBirth,
                        CASE WHEN cb.Zsituation LIKE '%سقط%' THEN 1 ELSE 0 END       AS Abortion,
                        DATEDIFF(day, ca.EngZDate, tb.EngTDate)                      AS DIMAtBreeding,
                        cb.MilkPeriod                                                AS Lactation,
                        CASE WHEN cb.BrithKind LIKE '%ت%' THEN 0 ELSE 1 END         AS MultiBirth,
                        CASE WHEN RTRIM(tb.TKind) = 'طبیعي' THEN 1 ELSE 0 END        AS NaturalBreeding,
                        DATEDIFF(day, ca.EngZDate, cb.EngZDate)                      AS LengthInDays,
                        DATEDIFF(day, ca.EngZDate, tb.EngTDate)                      AS DaysOpen,
                        t_count.Count                                                AS TimesBred,
                        CASE WHEN l.Serial IS NOT NULL THEN 1 ELSE 0 END             AS Lameness,
                        CASE WHEN mastites.Serial IS NOT NULL THEN 1 ELSE 0 END      AS Mastities,
                        CASE WHEN ketosis.Serial IS NOT NULL THEN 1 ELSE 0 END       AS Ketosis,
                        CASE WHEN retained.Serial IS NOT NULL THEN 1 ELSE 0 END      AS RetainedPlacentra,
                        CASE WHEN displaced.Serial IS NOT NULL THEN 1 ELSE 0 END     AS DisplacedAbomasam,
                        CASE
                            WHEN MONTH(cb.EngZDate) IN (1, 2, 12) OR (MONTH(cb.EngZDate) = 3 AND DAY(cb.EngZDate) <= 20) OR
                                 (MONTH(cb.EngZDate) = 12 AND DAY(cb.EngZDate) >= 21) THEN 1
                            ELSE 0 END                                               AS Winter,
                        CASE
                            WHEN MONTH(cb.EngZDate) IN (3, 4, 5) OR (MONTH(cb.EngZDate) = 6 AND DAY(cb.EngZDate) <= 20)
                                THEN 1
                            ELSE 0 END                                               AS Spring,
                        CASE
                            WHEN MONTH(cb.EngZDate) IN (6, 7, 8) OR (MONTH(cb.EngZDate) = 9 AND DAY(cb.EngZDate) <= 21)
                                THEN 1
                            ELSE 0 END                                               AS Summer,
                        CASE
                            WHEN MONTH(cb.EngZDate) IN (9, 10, 11) OR (MONTH(cb.EngZDate) = 12 AND DAY(cb.EngZDate) <= 20)
                                THEN 1
                            ELSE 0 END                                               AS Fall


                 FROM [ModiranFarmer].[dbo].[Zayesh] ca
                          LEFT JOIN [ModiranFarmer].[dbo].[StandardMilk] sm
                                    ON sm.Serial = ca.Serial AND sm.MilkPeriod = ca.MilkPeriod
                            JOIN [ModiranFarmer].[dbo].[Main] m
                                    ON ca.Serial = m.Serial
                          CROSS APPLY (SELECT TOP (1) *
                                       FROM [ModiranFarmer].[dbo].[Zayesh] c
                                       WHERE ca.Serial = c.Serial
                                         AND c.MilkPeriod = ca.MilkPeriod + 1
                                         AND MilkPeriodChange = 1
                                       ORDER BY c.ZDate ASC) AS cb
                          OUTER APPLY (SELECT count(*) Count
                                       FROM [ModiranFarmer].[dbo].[Talghih] t
                                       WHERE t.Serial = ca.Serial
                                         AND t.TDate > ca.ZDate
                                         AND t.TDate <= cb.PrInitDate) AS t_count


                          LEFT JOIN [ModiranFarmer].[dbo].[Talghih] tb
                                    ON tb.Serial = ca.Serial AND tb.TDate = cb.PrInitDate
                          OUTER APPLY (SELECT avg(bc.Score) AS Score
                                       FROM [ModiranFarmer].[dbo].[BodyCondition] bc
                                       WHERE bc.Serial = ca.Serial
                                         AND bc.SDate >= ca.ZDate
                                         AND bc.SDate <= cb.ZDate) AS bc_avg
                          OUTER APPLY (SELECT avg(ms.Score) AS Score
                                       FROM [ModiranFarmer].[dbo].[MotionScore] ms
                                       WHERE ms.Serial = ca.Serial
                                         AND ms.SDate >= ca.ZDate
                                         AND ms.SDate <= cb.ZDate) AS ms_avg
                          OUTER APPLY (SELECT TOP (1) *
                                       FROM [ModiranFarmer].[dbo].[BodyCondition] bc
                                       WHERE bc.Serial = cb.Serial
                                         AND bc.SDate >= ca.ZDate
                                         AND bc.SDate <= cb.ZDate) AS last_bc
                          OUTER APPLY (SELECT TOP (1) *
                                       FROM [ModiranFarmer].[dbo].[MotionScore] ms
                                       WHERE ms.Serial = cb.Serial
                                         AND ms.SDate >= ca.ZDate
                                         AND ms.SDate <= cb.ZDate) AS last_ms
                          OUTER APPLY (SELECT TOP (1) *
                                       FROM [ModiranFarmer].[dbo].[CaseHistory] ch
                                       WHERE ch.Serial = tb.Serial
                                         AND ch.CaseDate >= cb.ZDate
                                         AND ch.CaseDate >= ca.ZDate
                                         AND ch.Resone LIKE 'لنگش%') AS l
                          OUTER APPLY (SELECT TOP (1) *
                                       FROM [ModiranFarmer].[dbo].[CaseHistory] ch
                                       WHERE ch.Serial = tb.Serial
                                         AND ch.CaseDate >= cb.ZDate
                                         AND ch.CaseDate >= ca.ZDate
                                         AND ch.Resone LIKE '%ورم پستان%') AS mastites
                          OUTER APPLY (SELECT TOP (1) *
                                       FROM [ModiranFarmer].[dbo].[CaseHistory] ch
                                       WHERE ch.Serial = tb.Serial
                                         AND ch.CaseDate >= cb.ZDate
                                         AND ch.CaseDate >= ca.ZDate
                                         AND Resone LIKE 'كتوز%') AS ketosis
                          OUTER APPLY (SELECT TOP (1) *
                                       FROM [ModiranFarmer].[dbo].[CaseHistory] ch
                                       WHERE ch.Serial = tb.Serial
                                         AND ch.CaseDate >= cb.ZDate
                                         AND ch.CaseDate >= ca.ZDate
                                         AND (
                                           ch.Resone LIKE 'جفت ماند%' OR ch.WhatDocFind LIKE 'جفت ماند%'
                                           )) AS retained
                          OUTER APPLY (SELECT TOP (1) *
                                       FROM [ModiranFarmer].[dbo].[CaseHistory] ch
                                       WHERE ch.Serial = tb.Serial
                                         AND ch.CaseDate >= cb.ZDate
                                         AND ch.CaseDate >= ca.ZDate
                                         AND (
                                           RTRIM(ch.Resone) = 'جابجايي شيردان'
                                           )) AS displaced

                 WHERE cb.MilkPeriodChange = 1)
        SELECT *
        FROM cte
        WHERE AgeAtCalvingInMonths >= 20
        AND AgeAtCalvingInMonthsA >= 20
        AND DaysOpen > 70
        AND DaysOpen < 200
        AND PregnancyLengthMonths > 6
        AND PregnancyLengthMonths < 11
        ORDER BY StillBirth DESC

        """)
        rows = cursor.fetchall()
        columns = ['Serial', 'AgeAtCalvingInDaysA', 'AgeAtCalvingInMonthsA',
                   'AgeAtCalvingInDays', 'AgeAtCalvingInMonths', 'MostRecentBodyScore',
                   'MostRecentMotionScore', 'AverageBodyScore', 'AverageMotionScore',
                   'InbreedingCoefficient', 'PregnancyLengthMonths', 'Milk', 'MilkDays',
                   'MilkFat', 'MilkProtein', 'StillBirth', 'Abortion', 'DIMAtBreeding',
                   'Lactation', 'MultiBirth', 'NaturalBreeding', 'LengthInDays',
                   'DaysOpen', 'TimesBred', 'Lameness', 'Mastities', 'Ketosis',
                   'RetainedPlacentra', 'DisplacedAbomasam', 'Winter', 'Spring', 'Summer',
                   'Fall']
        self.data = pd.DataFrame(rows, columns=columns)

    def train_model(self):

        self.data['AverageMilk'] = self.data['Milk'] / self.data['MilkDays']
        mask = np.isnan(self.data["MilkFat"]) | np.isnan(self.data["MilkProtein"]) | np.isnan(self.data["Milk"])
        self.data["ECM"] = np.nan
        self.data.loc[~mask, "ECM"] = 0.327 * self.data["Milk"] + 12.95 * self.data["MilkFat"] + 7.65 * self.data[
            "MilkProtein"]
        self.data["FatProteinRatio"] = np.nan
        self.data.loc[~mask, "FatProteinRatio"] = self.data["MilkFat"] / self.data["MilkProtein"]
        # Detect outliers in numerical columns
        numeric_columns = ['AgeAtCalvingInDaysA', 'AgeAtCalvingInMonthsA', 'AgeAtCalvingInDays',
                           'AgeAtCalvingInMonths', 'MostRecentBodyScore', 'MostRecentMotionScore',
                           'AverageBodyScore', 'AverageMotionScore', 'InbreedingCoefficient',
                           'PregnancyLengthMonths', 'Milk',
                           'MilkDays', 'MilkFat', 'MilkProtein',
                           'DIMAtBreeding', 'Lactation',
                           'LengthInDays', 'DaysOpen', 'TimesBred', 'AverageMilk', 'ECM',
                           'FatProteinRatio']
        standard_data = self.data.copy()  # Create a copy of the original DataFrame
        # Handling outliers
        for column in numeric_columns:
            mean = standard_data[column].mean()
            std = standard_data[column].std()
            upper_bound = mean + 3 * std
            lower_bound = mean - 3 * std

            standard_data.loc[:, column] = standard_data.loc[:, column].clip(lower=lower_bound, upper=upper_bound)

        filled_data = standard_data.copy()  # Create a copy of the original DataFrame
        self.mean_values = {}  # Dictionary to store the mean values
        # fill NaN values with mean value of each column
        for col in numeric_columns:
            mean_val = filled_data[col].mean()
            self.mean_values[col] = mean_val  # Save the mean value for each column
            filled_data[col].fillna(mean_val, inplace=True)

        filled_data.drop(
            ['DIMAtBreeding', 'Abortion', 'AgeAtCalvingInDaysA', 'AgeAtCalvingInMonthsA', 'AgeAtCalvingInMonths',
             'MostRecentBodyScore', 'MostRecentMotionScore', 'PregnancyLengthMonths', 'Milk', 'MilkDays',
             'MilkProtein', 'MilkFat', 'LengthInDays'], axis=1, inplace=True)

        columns = filled_data.columns.to_list()
        columns_to_remove = ['Serial', 'Lactation']  # Columns to remove from the sequence
        feature_columns = [col for col in columns if col not in columns_to_remove]

        def generate_sequences(data, feature_cols, target_column):
            sequences = []
            targets = []

            for cow_serial in filled_data['Serial'].unique():
                cow_data = filled_data[filled_data['Serial'] == cow_serial].values

                # Order the lactation records for the cow based on the "Lactation" column
                cow_data = cow_data[np.argsort(cow_data[:, columns.index('Lactation')])]

                for i in range(1, len(cow_data)):
                    targets.append(cow_data[i, columns.index(target_column)])

                    # Get all previous lactation records for the cow
                    previous_records = cow_data[:i]

                    # Remove serial and lactation columns from the sequence
                    previous_records = np.delete(previous_records, [columns.index(col) for col in columns_to_remove],
                                                 axis=1)

                    sequences.append(previous_records)

            return sequences, targets

        sequences, targets = generate_sequences(filled_data, feature_columns, 'DaysOpen')

        # Padding sequences
        padded_sequences = pad_sequences(sequences, padding='post', dtype='float32')

        # Stacking padded sequences
        X = np.stack(padded_sequences)
        y = np.array(targets)


        # Fitting the normalizer to X_train
        self.scaler = StandardScaler()
        self.scaler.fit_transform(X.reshape(-1, X.shape[-1]))  # Reshape the data to 2D for fitting the scaler

        def train(model_type, X, y):
            model = Sequential()
            model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
            model.add(LSTM(32))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(1))
            model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

            # Define early stopping callback
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            train_seq_lengths = [len(seq) for seq in X]
            history = model.fit(X, y, epochs=20, batch_size=32, validation_data=(X, y),
                      sample_weight=np.array(train_seq_lengths), callbacks=[early_stopping], verbose=1)
            return model, history

        model, history = train('LSTM', X, y)
        tf.keras.saving.save_model(model, filepath='days-open-model')
        with open('days-open-params.pkl', 'wb') as dop:
            pickle.dump({"mean_values": self.mean_values, "scaler": self.scaler}, dop)
        self.model = model
        self.accuracy = history.history['mean_absolute_error'][-1]

    def data_to_sequence(self, data):
        data['AverageMilk'] = data['Milk'] / data['MilkDays']
        mask = np.isnan(data["MilkFat"]) | np.isnan(data["MilkProtein"]) | np.isnan(data["Milk"])
        data["ECM"] = np.nan
        data.loc[~mask, "ECM"] = 0.327 * data["Milk"] + 12.95 * data["MilkFat"] + 7.65 * data["MilkProtein"]
        data["FatProteinRatio"] = np.nan
        data.loc[~mask, "FatProteinRatio"] = data["MilkFat"] / data["MilkProtein"]
        numeric_columns = ['AgeAtCalvingInDaysA', 'AgeAtCalvingInMonthsA', 'AgeAtCalvingInDays',
                           'AgeAtCalvingInMonths', 'MostRecentBodyScore', 'MostRecentMotionScore',
                           'AverageBodyScore', 'AverageMotionScore', 'InbreedingCoefficient',
                           'PregnancyLengthMonths', 'Milk',
                           'MilkDays', 'MilkFat', 'MilkProtein',
                           'DIMAtBreeding', 'Lactation',
                           'LengthInDays', 'DaysOpen', 'TimesBred', 'AverageMilk', 'ECM',
                           'FatProteinRatio']

        # fill NaN values with mean value of each column
        for col in numeric_columns:
            data[col].fillna(self.mean_values[col], inplace=True)

        data.drop(
            ['DIMAtBreeding', 'Abortion', 'AgeAtCalvingInDaysA', 'AgeAtCalvingInMonthsA', 'AgeAtCalvingInMonths',
             'MostRecentBodyScore', 'MostRecentMotionScore', 'PregnancyLengthMonths', 'Milk', 'MilkDays', 'MilkProtein',
             'MilkFat', 'LengthInDays'], axis=1, inplace=True)

        columns = data.columns.to_list()
        columns_to_remove = ['Serial', 'Lactation']  # Columns to remove from the sequence
        sequences = []
        for cow_serial in data['Serial'].unique():
            cow_data = data[data['Serial'] == cow_serial].values
            # Order the lactation records for the cow based on the "Lactation" column
            cow_data = cow_data[np.argsort(cow_data[:, columns.index('Lactation')])]

            # Remove serial and lactation columns from the sequence
            cow_data = np.delete(cow_data, [columns.index(col) for col in columns_to_remove], axis=1)

            sequences.append(cow_data)  # Padding sequences

        padded_sequences = pad_sequences(sequences, padding='post', maxlen=7, dtype='float32')

        # Stacking padded sequences
        X = np.stack(padded_sequences)
        X = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        return X

    def predict(self, serial: str):
        data = self.data[self.data['Serial'] == serial]
        if data.shape[0] < 1:  raise HTTPException(status_code=404,
                                                   detail="شناسه نامعتبر است. این دام در دیتابیس وجود ندارد.")
        X = self.data_to_sequence(data)

        return int(np.round(self.model.predict(X)[0][0]))


if __name__ == '__main__':
    def create_connection():
        server = "192.168.32.2\\SQLEXPRESS"
        database = "ModiranFarmer"
        username = "mmgh900"
        password = "0936"
        conn = pymssql.connect(server, username, password, database)

        return conn


    model = DaysOpenModel(create_connection())
