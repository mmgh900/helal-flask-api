import pickle

import numpy as np
import pandas as pd
import pymssql
import tensorflow as tf
from fastapi import HTTPException
from keras import Sequential
from keras.layers import LSTM, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import jdatetime
import datetime


class InseminationResModel:

    def __init__(self, conn, force_retrain= False):
        self.conn = conn
        directory = 'InseminationResModel'

        if os.path.exists(directory) and not force_retrain:
            self.model = tf.keras.models.load_model(filepath=directory)
            with open('InseminationResModel-params.pkl', 'rb') as dop:
                storage = pickle.load(dop)
            self.mean_values = storage['mean_values']
            self.scaler = storage['scaler']
            print("InseminationResModel Loaded")
        else:
            self.data = self.get_data(True)
            self.train_model()

    def get_query(self, isTraining, serial="", date= None, naturalBreeding=False, milk_period=0):
        if isTraining:
            p = "CASE WHEN i.TDate = cc.PrInitDate THEN 1 ELSE 0 END      AS Pregnant,"
        else:
            p = ""
        if naturalBreeding:
            kind = 'طبیعي'
        else:
            kind = 'مصنوعي'

        if (isTraining):
            i = "[ModiranFarmer].[dbo].[Talghih] i"
        else:
            if date == None:
                persian_date = jdatetime.date.today().strftime('%Y/%m/%d')
                english_date = datetime.date.today().strftime('%Y-%m-%d %H:%M:%S.%f')
            else:
                persian_date = date
                try:
                    parsed_date = jdatetime.datetime.strptime(persian_date, '%Y/%m/%d')
                except:
                    raise HTTPException(status_code=400, detail="فرمت تاریخ ورودی اشتباه است. لطفا مشابه این نمونه وارد کنید: 1400/02/01")
                gregorian_date = parsed_date.togregorian()
                english_date = gregorian_date.strftime('%Y-%m-%d %H:%M:%S.%f')

            i = f"""
            (VALUES ('{serial}', '{persian_date}', '{english_date}', '{kind}', '{milk_period + 1}')) AS i (Serial, TDate, EngTDate, TKind, MilkPeriod)
            """

        query = f"""
        SELECT i.Serial,
       last_bc.Score                                            AS MostRecentBodyScore,
       last_ms.Score                                            AS MostRecentMotionScore,
       m.IntBlood                                               AS InbreedingCoefficient,
       DATEDIFF(day, m.EngBDate, cb.EngZDate)                   AS AgeAtCalvingInDays,
       sm.AllMilk                                               as Milk,
       sm.MilkDays                                              as MilkDays,
       sm.AllOil                                                as MilkFat,
       sm.AllPro                                                as MilkProtein,
       cb.MilkPeriod                                            AS Lactation,
       DATEDIFF(month, cb.EngZDate, i.EngTDate)                 AS StageOfLactation,
       DATEDIFF(day, ca.EngZDate, ib.EngTDate)                  AS PreviousDaysOpen,
       ptb.Count                                                AS PreviousTimesBred,
       ctb.Count                                                AS CurrentTimesBred,
       DATEDIFF(day, cb.EngZDate, i.EngTDate)                   AS DIMAtBreeding,
       CASE WHEN cb.Zsituation LIKE '%مرده%' THEN 1 ELSE 0 END  AS StillBirth,
       CASE WHEN cb.Zsituation LIKE '%سقط%' THEN 1 ELSE 0 END   AS Abortion,
       CASE WHEN cb.BrithKind LIKE '%ت%' THEN 0 ELSE 1 END      AS MultiBirth,
       CASE WHEN RTRIM(i.TKind) = 'طبیعي' THEN 1 ELSE 0 END     AS NaturalBreeding,
       {p}
              CASE WHEN l.Serial IS NOT NULL THEN 1 ELSE 0 END         AS Lameness,
       CASE WHEN mastites.Serial IS NOT NULL THEN 1 ELSE 0 END  AS Mastities,
       CASE WHEN ketosis.Serial IS NOT NULL THEN 1 ELSE 0 END   AS Ketosis,
       CASE WHEN retained.Serial IS NOT NULL THEN 1 ELSE 0 END  AS RetainedPlacentra,
       CASE WHEN displaced.Serial IS NOT NULL THEN 1 ELSE 0 END AS DisplacedAbomasa,
       CASE
           WHEN MONTH(i.EngTDate) IN (1, 2, 12) THEN 1
           WHEN MONTH(i.EngTDate) = 3 AND DAY(i.EngTDate) <= 20 THEN 1
           WHEN MONTH(i.EngTDate) = 12 AND DAY(i.EngTDate) >= 21 THEN 1
           ELSE 0
           END                                                  AS Winter,

       CASE
           WHEN MONTH(i.EngTDate) IN (4, 5) THEN 1
           WHEN MONTH(i.EngTDate) = 3 AND DAY(i.EngTDate) > 20 THEN 1
           WHEN MONTH(i.EngTDate) = 6 AND DAY(i.EngTDate) <= 20 THEN 1
           ELSE 0
           END                                                  AS Spring,

       CASE
           WHEN MONTH(i.EngTDate) IN (7, 8) THEN 1
           WHEN MONTH(i.EngTDate) = 6 AND DAY(i.EngTDate) > 20 THEN 1
           WHEN MONTH(i.EngTDate) = 9 AND DAY(i.EngTDate) <= 21 THEN 1
           ELSE 0
           END                                                  AS Summer,

       CASE
           WHEN MONTH(i.EngTDate) IN (10, 11) THEN 1
           WHEN MONTH(i.EngTDate) = 9 AND DAY(i.EngTDate) > 21 THEN 1
           WHEN MONTH(i.EngTDate) = 12 AND DAY(i.EngTDate) <= 20 THEN 1
           ELSE 0
           END                                                  AS Fall
        FROM {i}
                  JOIN [ModiranFarmer].[dbo].[Zayesh] cb ON i.Serial = cb.Serial AND i.MilkPeriod = cb.MilkPeriod + 1
        JOIN [ModiranFarmer].[dbo].[Main] m
                   ON cb.Serial = m.Serial
         LEFT JOIN [ModiranFarmer].[dbo].[StandardMilk] sm
                   ON sm.Serial = cb.Serial AND sm.MilkPeriod = cb.MilkPeriod

         OUTER APPLY (SELECT TOP (1) *
                      FROM [ModiranFarmer].[dbo].[Zayesh] c
                      WHERE c.Serial = i.Serial
                        AND c.ZDate < cb.PrInitDate
                      ORDER BY c.ZDate DESC) AS ca
         OUTER APPLY (SELECT TOP (1) *
                      FROM [ModiranFarmer].[dbo].[Zayesh] c
                      WHERE c.Serial = i.Serial
                        AND c.ZDate > cb.ZDate
                      ORDER BY c.ZDate ASC) AS cc
        OUTER APPLY (SELECT *
                      FROM [ModiranFarmer].[dbo].[Talghih] ib
                      WHERE ib.Serial = i.Serial
                        AND ib.TDate = cb.PrInitDate
                      ) AS ib
         OUTER APPLY (SELECT TOP (1) *
                      FROM [ModiranFarmer].[dbo].[BodyCondition] bc
                      WHERE bc.Serial = cb.Serial
                        AND bc.SDate >= cb.PrInitDate
                        AND bc.SDate <= i.TDate) AS last_bc
         OUTER APPLY (SELECT TOP (1) *
                      FROM [ModiranFarmer].[dbo].[MotionScore] ms
                      WHERE ms.Serial = cb.Serial
                        AND ms.SDate >= cb.PrInitDate
                        AND ms.SDate <= i.TDate) AS last_ms
         OUTER APPLY (SELECT COUNT(*) AS Count
                      FROM [ModiranFarmer].[dbo].[Talghih] pi
                      WHERE pi.Serial = i.Serial
                        AND pi.TDate > ca.ZDate
                        AND pi.TDate <= cb.PrInitDate) AS ptb
         OUTER APPLY (SELECT COUNT(*) AS Count
                      FROM [ModiranFarmer].[dbo].[Talghih] ci
                      WHERE ci.Serial = i.Serial
                        AND ci.TDate > cb.ZDate
                        AND ci.TDate <= i.TDate) AS ctb
         OUTER APPLY (SELECT TOP (1) *
                      FROM [ModiranFarmer].[dbo].[CaseHistory] ch
                      WHERE ch.Serial = i.Serial
                        AND ch.CaseDate >= cb.ZDate
                        AND ch.CaseDate >= i.TDate
                        AND RTRIM(ch.Resone) = 'لنگش') AS l
         OUTER APPLY (SELECT TOP (1) *
                      FROM [ModiranFarmer].[dbo].[CaseHistory] ch
                      WHERE ch.Serial = i.Serial
                        AND ch.CaseDate >= cb.ZDate
                        AND ch.CaseDate >= i.TDate
                        AND (
                                  RTRIM(ch.Resone) = 'اورام پستان'
                              OR RTRIM(ch.Resone) = 'ورم پستان قانقاريا'
                              OR RTRIM(ch.Resone) = 'ورم پستان فوق حاد'
                              OR RTRIM(ch.Resone) = 'ورم پستان حاد'
                              OR RTRIM(ch.Resone) = 'ورم پستان حاد'
                              OR RTRIM(ch.Resone) = 'ورم پستان گانگرنوز'
                              OR RTRIM(ch.Resone) = 'ورم پستان مزمن'
                              OR RTRIM(ch.Resone) = 'ورم پستان سپتيک'
                          )) AS mastites
         OUTER APPLY (SELECT TOP (1) *
                      FROM [ModiranFarmer].[dbo].[CaseHistory] ch
                      WHERE ch.Serial = i.Serial
                        AND ch.CaseDate >= cb.ZDate
                        AND ch.CaseDate >= i.TDate
                        AND (
                          RTRIM(ch.Resone) = 'كتوز'
                          )) AS ketosis
         OUTER APPLY (SELECT TOP (1) *
                      FROM [ModiranFarmer].[dbo].[CaseHistory] ch
                      WHERE ch.Serial = i.Serial
                        AND ch.CaseDate >= cb.ZDate
                        AND ch.CaseDate >= i.TDate
                        AND (
                          RTRIM(ch.Resone) = 'جفت ماندگي' OR RTRIM(WhatDocFind) = 'جفت مانده'
                          )) AS retained
         OUTER APPLY (SELECT TOP (1) *
                      FROM [ModiranFarmer].[dbo].[CaseHistory] ch
                      WHERE ch.Serial = i.Serial
                        AND ch.CaseDate >= cb.ZDate
                        AND ch.CaseDate >= i.TDate
                        AND (
                          RTRIM(ch.Resone) = 'جابجايي شيردان'
                          )) AS displaced

        WHERE i.TDate > cb.ZDate
        
        """
        return query

    def get_data(self, isTraining, serial="", date: str = None, naturalBreeding=False):
        max_milk_period = None
        if not isTraining:
            cursor = self.conn.cursor()
            cursor.execute(f"""
            SELECT MAX(c.MilkPeriod) 
            FROM [ModiranFarmer].[dbo].[Zayesh] c 
            WHERE c.Serial = '{serial}'
            """)
            rows = cursor.fetchall()
            max_milk_period = rows[0][0]
            if max_milk_period == None:
                raise HTTPException(status_code=400, detail="امکان پیش‌بینی برای تلیسه وجود ندارد.")
        cursor = self.conn.cursor()
        cursor.execute(self.get_query(isTraining, serial, date, naturalBreeding, max_milk_period))
        rows = cursor.fetchall()
        columns = [
            'Serial',
            'MostRecentBodyScore',
            'MostRecentMotionScore',
            'InbreedingCoefficient',
            'AgeAtCalvingInDays',
            'Milk',
            'MilkDays',
            'MilkFat',
            'MilkProtein',
            'Lactation',
            'StageOfLactation',
            'PreviousDaysOpen',
            'PreviousTimesBred',
            'CurrentTimesBred',
            'DIMAtBreeding',
            'StillBirth',
            'Abortion',
            'MultiBirth',
            'NaturalBreeding',
            'Pregnant',
            'Lameness',
            'Mastities',
            'Ketosis',
            'RetainedPlacentra',
            'DisplacedAbomasa',
            'Winter',
            'Spring',
            'Summer',
            'Fall'
        ]
        if not isTraining:
            columns.remove('Pregnant')
        return pd.DataFrame(rows, columns=columns)

    def train_model(self):
        data = self.data
        # Extracting Insementaion Season
        data['AverageMilk'] = data['Milk'] / data['MilkDays']
        # calculate ECM only for rows where all three columns are non-zero
        mask = (data['Milk'] != 0) & (data['MilkFat'] != 0) & (data['MilkProtein'] != 0)
        data.loc[mask, 'ECM'] = 0.327 * data.loc[mask, 'Milk'] + 12.95 * data.loc[mask, 'MilkFat'] + 7.65 * data.loc[
            mask, 'MilkProtein']
        data["FatProteinRatio"] = data["MilkFat"] / data["MilkProtein"]
        features = [
            'MostRecentBodyScore',
            'MostRecentMotionScore',
            'InbreedingCoefficient',
            'AgeAtCalvingInDays',
            'Milk',
            'MilkDays',
            'MilkFat',
            'MilkProtein',
            'Lactation',
            'StageOfLactation',
            'PreviousDaysOpen',
            'PreviousTimesBred',
            'CurrentTimesBred',
            'DIMAtBreeding',
            'StillBirth',
            'Abortion',
            'MultiBirth',
            'NaturalBreeding',
            'Pregnant',
            'Lameness',
            'Mastities',
            'Ketosis',
            'RetainedPlacentra',
            'DisplacedAbomasa',
            'Winter',
            'Spring',
            'Summer',
            'Fall',
            "ECM",
            "AverageMilk",
            "FatProteinRatio"

        ]

        numberical_features = [
            'MostRecentBodyScore',
            'MostRecentMotionScore',
            'InbreedingCoefficient',
            'AgeAtCalvingInDays',
            'Milk',
            'MilkDays',
            'MilkFat',
            'MilkProtein',
            'Lactation',
            'StageOfLactation',
            'PreviousDaysOpen',
            'PreviousTimesBred',
            'CurrentTimesBred',
            'DIMAtBreeding',
            "ECM",
            "AverageMilk",
            "FatProteinRatio"
        ]

        data = data[features]

        # Handling outliers
        for column in numberical_features:
            mean = data[column].mean()
            std = data[column].std()
            upper_bound = mean + 3 * std
            lower_bound = mean - 3 * std

            data.loc[:, column] = data.loc[:, column].clip(lower=lower_bound, upper=upper_bound)

        data['AgeAtCalvingInDays'] = data['AgeAtCalvingInDays'].where(
            data['AgeAtCalvingInDays'] >= 0, np.nan)

        # fill NaN values with mean value of each column

        self.mean_values = {}  # Dictionary to store the mean values
        # fill NaN values with mean value of each column
        for col in numberical_features:
            mean_val = data[col].mean()
            self.mean_values[col] = mean_val  # Save the mean value for each column
            data[col].fillna(mean_val, inplace=True)

        # Split the data into input features (X) and target variable (y)
        X = data.drop('Pregnant', axis=1)
        y = data['Pregnant']

        # Fitting the normalizer to X_train
        self.scaler = StandardScaler()
        self.scaler.fit_transform(X)  # Reshape the data to 2D for fitting the scaler

        # Define the model
        model = Sequential()

        model.add(Dense(256, input_dim=X.shape[1], activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        opt = Adam(learning_rate=0.001)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        # Define early stopping callback
        early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1)

        from sklearn.utils.class_weight import compute_class_weight

        # Get the class labels
        class_labels = np.unique(y)

        # Compute the class weights
        class_weights = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y)
        class_weight_dict = dict(zip(class_labels, class_weights))
        print(class_weight_dict)

        # Train the model with the class weights
        history = model.fit(X, y, batch_size=32, epochs=20, verbose=1,
                             callbacks=[early_stopping], class_weight=class_weight_dict)
        tf.keras.saving.save_model(model, filepath='InseminationResModel')
        with open('InseminationResModel-params.pkl', 'wb') as dop:
            pickle.dump({"mean_values": self.mean_values, "scaler": self.scaler}, dop)
        self.model = model
        self.accuracy = history.history['accuracy'][-1]

    def data_to_sequence(self, data):
        # Extracting Insementaion Season
        data['AverageMilk'] = data['Milk'] / data['MilkDays']
        # calculate ECM only for rows where all three columns are non-zero
        mask = (data['Milk'] != 0) & (data['MilkFat'] != 0) & (data['MilkProtein'] != 0)
        data.loc[mask, 'ECM'] = 0.327 * data.loc[mask, 'Milk'] + 12.95 * data.loc[mask, 'MilkFat'] + 7.65 * data.loc[
            mask, 'MilkProtein']
        data["FatProteinRatio"] = data["MilkFat"] / data["MilkProtein"]
        features = [
            'MostRecentBodyScore',
            'MostRecentMotionScore',
            'InbreedingCoefficient',
            'AgeAtCalvingInDays',
            'Milk',
            'MilkDays',
            'MilkFat',
            'MilkProtein',
            'Lactation',
            'StageOfLactation',
            'PreviousDaysOpen',
            'PreviousTimesBred',
            'CurrentTimesBred',
            'DIMAtBreeding',
            'StillBirth',
            'Abortion',
            'MultiBirth',
            'NaturalBreeding',
            'Lameness',
            'Mastities',
            'Ketosis',
            'RetainedPlacentra',
            'DisplacedAbomasa',
            'Winter',
            'Spring',
            'Summer',
            'Fall',
            "ECM",
            "AverageMilk",
            "FatProteinRatio"

        ]

        numberical_features = [
            'MostRecentBodyScore',
            'MostRecentMotionScore',
            'InbreedingCoefficient',
            'AgeAtCalvingInDays',
            'Milk',
            'MilkDays',
            'MilkFat',
            'MilkProtein',
            'Lactation',
            'StageOfLactation',
            'PreviousDaysOpen',
            'PreviousTimesBred',
            'CurrentTimesBred',
            'DIMAtBreeding',
            "ECM",
            "AverageMilk",
            "FatProteinRatio"
        ]

        data = data[features]

        for col in numberical_features:
            data[col].fillna(self.mean_values[col], inplace=True)

        self.scaler.transform(data)  # Reshape the data to 2D for fitting the scaler

        return data

    def predict(self, serial: str, date: str = None, naturalBreeding: bool = False):
        data = self.get_data(False, serial, date, naturalBreeding)
        if data.shape[0] < 1:  raise HTTPException(status_code=404,
                                                   detail="شناسه یا تاریخ نامعتبر است. این دام در دیتابیس وجود ندارد یا تاریخ لقاح قبل از آخرین لقاح این گاو است.")
        X = self.data_to_sequence(data)

        return self.model.predict(X)[0][0], X


if __name__ == '__main__':
    def create_connection():
        server = "192.168.32.2\\SQLEXPRESS"
        database = "ModiranFarmer"
        username = "mmgh900"
        password = "0936"
        conn = pymssql.connect(server, username, password, database)

        return conn


    model = InseminationResModel(create_connection())
    print(model.predict('3251370011400010'))

