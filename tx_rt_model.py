# Output the value of the Leap RT option/hedge product based upon specified parameters
# Will need to read data from Dremio instead of an Excel file
# Need to add code for Monthly constraint
import pandas as pd
import numpy as np
from datetime import datetime
from time import time
from scipy.stats import norm
from functools import reduce


# Used to "wrap" functions and calculate their time to execute
def performance(fn):  # used to measure time spent on a function
    def wrapper(*args, **kwargs):
        t1 = time()
        result = fn(*args, **kwargs)
        t2 = time()
        print(f'Function took {t2 - t1} seconds')
        return result

    return wrapper


# all the defined functions expect the load zones to be provided as *args
@performance
def add_ordc(*args):
    for arg in args:
        df_lmp[arg + '_Price'] = df_lmp[arg] + df_lmp['RTORPA'] + df_lmp['RTORDPA']


# Functions for ORDC, strikes, dispatches, events, durations, and value
@performance
def set_new_ordc(*args):
    df_lmp['new_ORDC_System Lamda'] = np.where(df_lmp['System Lamda'] > 5001, 5001, df_lmp['System Lamda'])
    df_lmp['new_ORDC_RTOFFPA'] = np.where((voll - df_lmp['new_ORDC_System Lamda']) > 0,
                                          0.5 * (voll - df_lmp['new_ORDC_System Lamda']), 0) *\
                                 np.where((df_lmp['RTOLCAP'] + np.where(df_lmp['RTOLCAP'] <= mcl, 0,
                                                                        df_lmp['RTOFFCAP'])) <= mcl, 1,
                                          (1 - norm.cdf((df_lmp['RTOLCAP'] + df_lmp['RTOFFCAP']) - mcl,
                                                        df_lmp['Average + (0.5 * Stand Deviation)'],
                                                        df_lmp['Standard Deviation'])))
    df_lmp['new_ORDC_RTORPA'] = df_lmp['new_ORDC_RTOFFPA'] + 0.5 *\
                                np.where((voll - df_lmp['new_ORDC_System Lamda']) > 0, voll -
                                         df_lmp['new_ORDC_System Lamda'], 0) *\
                                np.where(df_lmp['RTOLCAP'] <= mcl, 1,
                                         (1 - norm.cdf(df_lmp['RTOLCAP'] - mcl,
                                                       0.5 * df_lmp['Average + (0.5 * Stand Deviation)'],
                                                       0.707 * df_lmp['Standard Deviation'])))
    for arg in args:
        df_lmp['new_ORDC_' + arg] = df_lmp[arg] - (df_lmp['System Lamda'] - df_lmp['new_ORDC_System Lamda'])
    for arg in args:
        df_lmp['new_ORDC_' + arg + '_Price'] = df_lmp['new_ORDC_' + arg] + df_lmp['new_ORDC_RTORPA']


@performance
def calc_new_ordc_spp(*args):
    for arg in args:
        df_lmp['new_ORDC_' + arg + '_temp'] = df_lmp['new_ORDC_' + arg + '_Price'] * \
                                              (df_lmp['SCED total time'] / df_lmp['SPP total time'])
        df_lmp['new_ORDC_' + arg + '_SPP'] = df_lmp['new_ORDC_' + arg + '_temp'].groupby(
            [df_lmp['date'], df_lmp['hour'], df_lmp['interval']]).transform('sum')
        df_lmp.drop('new_ORDC_' + arg + '_temp', axis='columns', inplace=True)


@performance
def set_strike(*args):
    for arg in args:
        for strike in strike_range:
            df_lmp[strike] = np.where(df_lmp[arg + '_Price'] > df_lmp[strike + '_lmp'], 1, 0)
            df_lmp[arg + '_Strike_' + strike + '_SPP'] =\
                df_lmp[strike].groupby([df_lmp['date'], df_lmp['hour'], df_lmp['interval']]).cummax()
            df_lmp.drop(strike, axis='columns', inplace=True)
    for arg in args:
        for strike in strike_range:
            df_lmp[strike] = np.where(df_lmp['new_ORDC_' + arg + '_Price'] > df_lmp[strike + '_lmp'], 1, 0)
            df_lmp['new_ORDC_' + arg + '_Strike_' + strike + '_SPP'] =\
                df_lmp[strike].groupby([df_lmp['date'], df_lmp['hour'], df_lmp['interval']]).cummax()
            df_lmp.drop(strike, axis='columns', inplace=True)


@performance
def set_dispatch(*args):
    for arg in args:
        for strike in strike_range:
            df_lmp[arg + '_Dispatch_' + strike + '_SPP'] = np.where(
                (df_lmp[arg + '_Strike_' + strike + '_SPP'] == 1)
                & (df_lmp[arg + '_Strike_' + strike + '_SPP'].shift(1) == 0), 1, 0)
    for arg in args:
        for strike in strike_range:
            df_lmp['new_ORDC_' + arg + '_Dispatch_' + strike + '_SPP'] = np.where(
                (df_lmp['new_ORDC_' + arg + '_Strike_' + strike + '_SPP'] == 1)
                & (df_lmp['new_ORDC_' + arg + '_Strike_' + strike + '_SPP'].shift(1) == 0), 1, 0)


@performance
def set_events(*args):
    for arg in args:
        for strike in strike_range:
            df_lmp[strike] = np.where(
                (df_lmp[arg + '_Dispatch_' + strike + '_SPP'] == 1) |
                (df_lmp[arg + '_Strike_' + strike + '_SPP'] == 1) &
                (df_lmp['date'] != df_lmp['date'].shift(1)), 1, 0)
            df_lmp[strike + '_temp'] = df_lmp[strike].groupby(df_lmp['date']).cumsum()
            df_lmp[arg + '_Events_' + strike + '_SPP'] = np.where(df_lmp[strike + '_temp'] == 0, np.nan,
                                                                  df_lmp[strike + '_temp'])
            df_lmp.drop([strike, strike + '_temp'], axis='columns', inplace=True)
    for arg in args:
        for strike in strike_range:
            df_lmp[strike] = np.where(
                (df_lmp['new_ORDC_' + arg + '_Dispatch_' + strike + '_SPP'] == 1) |
                (df_lmp['new_ORDC_' + arg + '_Strike_' + strike + '_SPP'] == 1) &
                (df_lmp['date'] != df_lmp['date'].shift(1)), 1, 0)
            df_lmp[strike + '_temp'] = df_lmp[strike].groupby(df_lmp['date']).cumsum()
            df_lmp['new_ORDC_' + arg + '_Events_' + strike + '_SPP'] =\
                np.where(df_lmp[strike + '_temp'] == 0, np.nan, df_lmp[strike + '_temp'])
            df_lmp.drop([strike, strike + '_temp'], axis='columns', inplace=True)


@performance
def set_duration(*args):  # in minutes
    for arg in args:
        for strike in strike_range:
            df_lmp[arg + '_Duration_' + strike + '_SPP'] = (1/60) * np.where(
                df_lmp[arg + '_Strike_' + strike + '_SPP'] == 1,
                df_lmp['SCED total time'].groupby(
                    [df_lmp[arg + '_Events_' + strike + '_SPP'], df_lmp['date']]).cumsum(), np.nan)
    for arg in args:
        for strike in strike_range:
            df_lmp['new_ORDC_' + arg + '_Duration_' + strike + '_SPP'] = (1/60) * np.where(
                df_lmp['new_ORDC_' + arg + '_Strike_' + strike + '_SPP'] == 1,
                df_lmp['SCED total time'].groupby(
                    [df_lmp['new_ORDC_' + arg + '_Events_' + strike + '_SPP'], df_lmp['date']]).cumsum(), np.nan)


@performance
def calc_value(*args):
    for arg in args:
        for strike in strike_range:
            for col in df_strikes.columns:
                if int(col) in df_lmp['month'].values:
                    df_lmp[arg + '_Value_' + strike + '_SPP'] = np.where(
                        (df_lmp[arg + '_Strike_' + strike + '_SPP'] == 1)
                        & (df_lmp[arg + '_Strike_' + strike + '_SPP'].shift(1) == 0),
                        constraints['Initial Dispatch Capture Pct'] * (df_lmp[arg + '_SPP'] -
                                                                       df_strikes.loc[strike + '_spp', col]),
                        df_lmp[arg + '_Strike_' + strike + '_SPP'] *
                        (df_lmp[arg + '_SPP'] - df_strikes.loc[strike + '_spp', col])) * 0.25 * \
                                                             (df_lmp['SCED total time'] /
                                                              df_lmp['SPP total time'])
    for arg in args:
        for strike in strike_range:
            for col in df_strikes.columns:
                if int(col) in df_lmp['month'].values:
                    df_lmp['new_ORDC_' + arg + '_Value_' + strike + '_SPP'] = np.where(
                        (df_lmp['new_ORDC_' + arg + '_Strike_' + strike + '_SPP'] == 1)
                        & (df_lmp['new_ORDC_' + arg + '_Strike_' + strike + '_SPP'].shift(1) == 0),
                        constraints['Initial Dispatch Capture Pct'] * (df_lmp['new_ORDC_' + arg + '_SPP'] -
                                                                       df_strikes.loc[strike + '_spp', col]),
                        df_lmp['new_ORDC_' + arg + '_Strike_' + strike + '_SPP'] *
                        (df_lmp['new_ORDC_' + arg + '_SPP'] - df_strikes.loc[strike + '_spp', col])) * 0.25 *\
                                                                           (df_lmp['SCED total time'] /
                                                                            df_lmp['SPP total time'])


@performance
def constrained_dispatch(*args):
    for arg in args:
        for strike in strike_range:
            if constraints['Dispatch Outside Event Window']:
                df_lmp[arg + '_Dispatch_' + strike + '_SPP_constrained'] = \
                    np.where(df_lmp['Weekday Dispatch'] == 1,
                             np.where(((df_lmp['hour'] < df_lmp['Event Window Start Hour']) |
                                       (df_lmp['hour'] >= df_lmp['Event Window End Hour']))
                                      & (df_lmp['Event Duration Min (min)'] > 0),
                                      df_lmp[arg + '_Dispatch_' + strike + '_SPP'],
                                      np.where((df_lmp['hour'] >= df_lmp['Event Window Start Hour'])
                                               & (df_lmp['hour'] < df_lmp['Event Window End Hour']),
                                               np.where((df_lmp[arg + '_Strike_' + strike + '_SPP'] == 1)
                                                        & (df_lmp['hour'].shift(1) < df_lmp['Event Window Start Hour']),
                                                        1, df_lmp[arg + '_Dispatch_' + strike + '_SPP']), 0)), 0)
            else:
                df_lmp[arg + '_Dispatch_' + strike + '_SPP_constrained'] = np.where(
                    (df_lmp['hour'] < df_lmp['Event Window Start Hour'])
                    | (df_lmp['hour'] >= df_lmp['Event Window End Hour']), 0,
                    np.where((df_lmp['Weekday Dispatch'] == 1),
                             np.where((df_lmp[arg + '_Strike_' + strike + '_SPP'] == 1)
                                      & (df_lmp['hour'].shift(1) < df_lmp['Event Window Start Hour']), 1,
                                      df_lmp[arg + '_Dispatch_' + strike + '_SPP']), 0))
    for arg in args:
        for strike in strike_range:
            if constraints['Dispatch Outside Event Window']:
                if constraints['Dispatch Outside Event Window']:
                    df_lmp['new_ORDC_' + arg + '_Dispatch_' + strike + '_SPP_constrained'] = \
                        np.where(df_lmp['Weekday Dispatch'] == 1,
                                 np.where(((df_lmp['hour'] < df_lmp['Event Window Start Hour']) |
                                           (df_lmp['hour'] >= df_lmp['Event Window End Hour'])) &
                                          (df_lmp['Event Duration Min (min)'] > 0) & (df_lmp['Weekday Dispatch'] == 1),
                                          df_lmp['new_ORDC_' + arg + '_Dispatch_' + strike + '_SPP'],
                                          np.where((df_lmp['hour'] >= df_lmp['Event Window Start Hour']) &
                                                   (df_lmp['hour'] < df_lmp['Event Window End Hour']) &
                                                   (df_lmp['Weekday Dispatch'] == 1),
                                                   np.where((df_lmp['new_ORDC_' + arg + '_Strike_' + strike + '_SPP']
                                                             == 1) &
                                                            (df_lmp['hour'].shift(1) <
                                                             df_lmp['Event Window Start Hour']), 1,
                                                            df_lmp['new_ORDC_' + arg + '_Dispatch_' + strike + '_SPP']),
                                                   0)), 0)
            else:
                df_lmp['new_ORDC_' + arg + '_Dispatch_' + strike + '_SPP_constrained'] = np.where(
                    (df_lmp['hour'] < df_lmp['Event Window Start Hour'])
                    | (df_lmp['hour'] >= df_lmp['Event Window End Hour']), 0,
                    np.where((df_lmp['Weekday Dispatch'] == 1),
                             np.where((df_lmp['new_ORDC_' + arg + '_Strike_' + strike + '_SPP'] == 1)
                                      & (df_lmp['hour'].shift(1) < df_lmp['Event Window Start Hour']), 1,
                                      df_lmp['new_ORDC_' + arg + '_Dispatch_' + strike + '_SPP']), 0))


@performance
def constrained_events(*args):
    for arg in args:
        for strike in strike_range:
            if constraints['Dispatch Outside Event Window']:
                df_lmp[strike] = np.where(
                    (df_lmp[arg + '_Dispatch_' + strike + '_SPP_constrained'] == 1) |
                    (df_lmp[arg + '_Strike_' + strike + '_SPP'] == 1) &
                    (df_lmp['date'] != df_lmp['date'].shift(1)), 1, 0)
            else:
                df_lmp[strike] = np.where(
                    (df_lmp[arg + '_Dispatch_' + strike + '_SPP_constrained'] == 1) |
                    (df_lmp[arg + '_Strike_' + strike + '_SPP'] == 1) &
                    (df_lmp['date'] != df_lmp['date'].shift(1)) &
                    (df_lmp['hour'] >= df_lmp['Event Window Start Hour']) &
                    (df_lmp['hour'] < df_lmp['Event Window End Hour']), 1, 0)
            df_lmp[strike + '_temp'] = df_lmp[strike].groupby(df_lmp['date']).cumsum()
            df_lmp[arg + '_Events_' + strike + '_SPP_constrained'] = \
                np.where(df_lmp[strike + '_temp'] == 0, np.nan, df_lmp[strike + '_temp'])
            df_lmp[strike + '_max_dur'] = df_lmp[arg + '_Duration_' + strike + '_SPP'].groupby(
                [df_lmp['date'], df_lmp[arg + '_Events_' + strike + '_SPP']]).transform(max)
            df_lmp[strike + '_dispatch_over_min_dur'] = np.where((df_lmp[strike + '_max_dur'] >=
                                                                df_lmp['Event Duration Min (min)']) &
                                                                (df_lmp['hour'] >= df_lmp['Event Window Start Hour']) &
                                                                (df_lmp['hour'] < df_lmp['Event Window End Hour']) &
                                                                df_lmp[strike] == 1, 1, 0)
            df_lmp[strike + '_temp_Dly'] = df_lmp[strike + '_dispatch_over_min_dur'].groupby(df_lmp['date']).cumsum()
            df_lmp[arg + '_Events_Dly_' + strike + '_SPP_constrained'] = \
                np.where(df_lmp[strike + '_temp_Dly'] == 0, np.nan, df_lmp[strike + '_temp_Dly'])
            df_lmp[strike + '_temp_Wkly_dispatch'] = \
                np.where(df_lmp[arg + '_Events_Dly_' + strike + '_SPP_constrained'] <= df_lmp['Daily Events Max'],
                         df_lmp[strike + '_dispatch_over_min_dur'], 0)
            df_lmp[strike + '_temp_Wkly'] = \
                df_lmp[strike + '_temp_Wkly_dispatch'].groupby([df_lmp['week number'], df_lmp['year']]).cumsum()
            df_lmp[arg + '_Events_Wkly_' + strike + '_SPP_constrained'] = \
                np.where(df_lmp[strike + '_temp_Wkly'] == 0, np.nan, df_lmp[strike + '_temp_Wkly'])
            df_lmp.drop([strike, strike + '_temp', strike + '_temp_Dly', strike + '_temp_Wkly_dispatch',
                         strike + '_temp_Wkly', strike + '_max_dur', strike + '_dispatch_over_min_dur'],
                        axis='columns', inplace=True)
    for arg in args:
        for strike in strike_range:
            if constraints['Dispatch Outside Event Window']:
                df_lmp['new_ORDC_' + strike] = np.where(
                    (df_lmp['new_ORDC_' + arg + '_Dispatch_' + strike + '_SPP_constrained'] == 1) |
                    (df_lmp['new_ORDC_' + arg + '_Strike_' + strike + '_SPP'] == 1) &
                    (df_lmp['date'] != df_lmp['date'].shift(1)), 1, 0)
            else:
                df_lmp['new_ORDC_' + strike] = np.where(
                    (df_lmp['new_ORDC_' + arg + '_Dispatch_' + strike + '_SPP_constrained'] == 1) |
                    (df_lmp['new_ORDC_' + arg + '_Strike_' + strike + '_SPP'] == 1) &
                    (df_lmp['date'] != df_lmp['date'].shift(1)) &
                    (df_lmp['hour'] >= df_lmp['Event Window Start Hour']) &
                    (df_lmp['hour'] < df_lmp['Event Window End Hour']), 1, 0)
            df_lmp['new_ORDC_' + strike + '_temp'] = df_lmp['new_ORDC_' + strike].groupby(df_lmp['date']).cumsum()
            df_lmp['new_ORDC_' + arg + '_Events_' + strike + '_SPP_constrained'] = \
                np.where(df_lmp['new_ORDC_' + strike + '_temp'] == 0, np.nan, df_lmp['new_ORDC_' + strike + '_temp'])
            df_lmp['new_ORDC_' + strike + '_max_dur'] =\
                df_lmp['new_ORDC_' + arg + '_Duration_' + strike + '_SPP'].groupby(
                [df_lmp['date'], df_lmp['new_ORDC_' + arg + '_Events_' + strike + '_SPP']]).transform(max)
            df_lmp['new_ORDC_' + strike + '_dispatch_over_min_dur'] =\
                np.where((df_lmp['new_ORDC_' + strike + '_max_dur'] >= df_lmp['Event Duration Min (min)']) &
                         (df_lmp['hour'] >= df_lmp['Event Window Start Hour']) &
                         (df_lmp['hour'] < df_lmp['Event Window End Hour']) &
                         df_lmp['new_ORDC_' + strike] == 1, 1, 0)
            df_lmp['new_ORDC_' + strike + '_temp_Dly'] =\
                df_lmp['new_ORDC_' + strike + '_dispatch_over_min_dur'].groupby(df_lmp['date']).cumsum()
            df_lmp['new_ORDC_' + arg + '_Events_Dly_' + strike + '_SPP_constrained'] = \
                np.where(df_lmp['new_ORDC_' + strike + '_temp_Dly'] == 0, np.nan,
                         df_lmp['new_ORDC_' + strike + '_temp_Dly'])
            df_lmp['new_ORDC_' + strike + '_temp_Wkly_dispatch'] = \
                np.where(df_lmp['new_ORDC_' + arg + '_Events_Dly_' + strike + '_SPP_constrained'] <=
                         df_lmp['Daily Events Max'], df_lmp['new_ORDC_' + strike + '_dispatch_over_min_dur'], 0)
            df_lmp['new_ORDC_' + strike + '_temp_Wkly'] = \
                df_lmp['new_ORDC_' + strike + '_temp_Wkly_dispatch'].groupby([df_lmp['week number'],
                                                                              df_lmp['year']]).cumsum()
            df_lmp['new_ORDC_' + arg + '_Events_Wkly_' + strike + '_SPP_constrained'] = \
                np.where(df_lmp['new_ORDC_' + strike + '_temp_Wkly'] == 0, np.nan,
                         df_lmp['new_ORDC_' + strike + '_temp_Wkly'])
            df_lmp.drop(['new_ORDC_' + strike, 'new_ORDC_' + strike + '_temp', 'new_ORDC_' + strike + '_temp_Dly',
                         'new_ORDC_' + strike + '_temp_Wkly_dispatch', 'new_ORDC_' + strike + '_temp_Wkly',
                         'new_ORDC_' + strike + '_max_dur', 'new_ORDC_' + strike + '_dispatch_over_min_dur'],
                        axis='columns', inplace=True)


@performance
def constrained_duration(*args):  # in minutes
    for arg in args:
        for strike in strike_range:
            df_lmp[strike] = (1 / 60) * np.where(
                df_lmp[arg + '_Strike_' + strike + '_SPP'] == 1,
                df_lmp['SCED total time'].groupby(
                    [df_lmp[arg + '_Events_' + strike + '_SPP_constrained'],
                     df_lmp['date']]).cumsum(), np.nan)
            df_lmp[arg + '_Duration_' + strike + '_SPP_constrained'] = \
                np.where((df_lmp['hour'] >= df_lmp['Event Window Start Hour']) &
                         (df_lmp['hour'] < df_lmp['Event Window End Hour']),
                         np.where((df_lmp[arg + '_Events_Wkly_' + strike + '_SPP_constrained'] >
                                   df_lmp['Weekly Events Max']) & (df_lmp[strike] >=
                                                                   df_lmp['Event Duration Min (min)']), np.nan,
                                  np.where((df_lmp[arg + '_Events_Dly_' + strike + '_SPP_constrained'] >
                                            df_lmp['Daily Events Max']) &
                                           (df_lmp[strike] >= df_lmp['Event Duration Min (min)']), np.nan,
                                           np.where((df_lmp[strike] > df_lmp['Event Duration Max (min)']) |
                                                    (df_lmp['hour'] > df_lmp['Event Window End Hour']),
                                                    np.nan, df_lmp[strike]))),
                         np.where(constraints['Dispatch Outside Event Window'],
                                  np.where(df_lmp[arg + '_Duration_' + strike + '_SPP'] >=
                                           df_lmp['Event Duration Min (min)'], np.nan,
                                           df_lmp[arg + '_Duration_' + strike + '_SPP']), np.nan))
            df_lmp.drop([strike, arg + '_Events_Dly_' + strike + '_SPP_constrained',
                         arg + '_Events_Wkly_' + strike + '_SPP_constrained'], axis='columns',
                        inplace=True)
    for arg in args:
        for strike in strike_range:
            df_lmp[strike] = (1 / 60) * np.where(
                df_lmp['new_ORDC_' + arg + '_Strike_' + strike + '_SPP'] == 1,
                df_lmp['SCED total time'].groupby(
                    [df_lmp['new_ORDC_' + arg + '_Events_' + strike + '_SPP_constrained'],
                     df_lmp['date']]).cumsum(), np.nan)
            df_lmp['new_ORDC_' + arg + '_Duration_' + strike + '_SPP_constrained'] = \
                np.where((df_lmp['hour'] >= df_lmp['Event Window Start Hour']) &
                         (df_lmp['hour'] < df_lmp['Event Window End Hour']),
                         np.where((df_lmp['new_ORDC_' + arg + '_Events_Wkly_' + strike + '_SPP_constrained'] >
                                   df_lmp['Weekly Events Max']) & (df_lmp[strike] >=
                                                                   df_lmp['Event Duration Min (min)']), np.nan,
                                  np.where((df_lmp['new_ORDC_' + arg + '_Events_Dly_' + strike + '_SPP_constrained'] >
                                            df_lmp['Daily Events Max']) &
                                           (df_lmp[strike] >= df_lmp['Event Duration Min (min)']), np.nan,
                                           np.where((df_lmp[strike] > df_lmp['Event Duration Max (min)']) |
                                                    (df_lmp['hour'] > df_lmp['Event Window End Hour']),
                                                    np.nan, df_lmp[strike]))),
                         np.where(constraints['Dispatch Outside Event Window'],
                                  np.where(df_lmp['new_ORDC_' + arg + '_Duration_' + strike + '_SPP'] >=
                                           df_lmp['Event Duration Min (min)'], np.nan,
                                           df_lmp['new_ORDC_' + arg + '_Duration_' + strike + '_SPP']), np.nan))
            df_lmp.drop([strike, 'new_ORDC_' + arg + '_Events_Dly_' + strike + '_SPP_constrained',
                         'new_ORDC_' + arg + '_Events_Wkly_' + strike + '_SPP_constrained'], axis='columns',
                        inplace=True)


@performance
def constrained_value(*args):
    for arg in args:
        for strike in strike_range:
            for col in df_strikes.columns:
                if int(col) in df_lmp['month'].values:
                    df_lmp[arg + '_Value_' + strike + '_SPP_constrained'] =\
                        np.where((df_lmp[arg + '_Strike_' + strike + '_SPP'] == 1)
                                 & (df_lmp[arg + '_Strike_' + strike + '_SPP'].shift(1) == 0)
                                 & (~np.isnan(df_lmp[arg + '_Duration_' + strike + '_SPP_constrained'])),
                                 constraints['Initial Dispatch Capture Pct'] * (df_lmp[arg + '_SPP'] -
                                                                                df_strikes.loc[strike + '_spp', col]),
                                 np.where(~np.isnan(df_lmp[arg + '_Duration_' + strike + '_SPP_constrained']),
                                          df_lmp[arg + '_Strike_' + strike + '_SPP'] *
                                          (df_lmp[arg + '_SPP'] - df_strikes.loc[strike + '_spp', col]), 0))\
                        * 0.25 * (df_lmp['SCED total time'] / df_lmp['SPP total time'])
    for arg in args:
        for strike in strike_range:
            for col in df_strikes.columns:
                if int(col) in df_lmp['month'].values:
                    df_lmp['new_ORDC_' + arg + '_Value_' + strike + '_SPP_constrained'] =\
                        np.where((df_lmp['new_ORDC_' + arg + '_Strike_' + strike + '_SPP'] == 1)
                                 & (df_lmp['new_ORDC_' + arg + '_Strike_' + strike + '_SPP'].shift(1) == 0)
                                 & (~np.isnan(df_lmp['new_ORDC_' + arg + '_Duration_' + strike + '_SPP_constrained'])),
                                 constraints['Initial Dispatch Capture Pct'] * (df_lmp['new_ORDC_' + arg + '_SPP'] -
                                                                                df_strikes.loc[strike + '_spp', col]),
                                 np.where(~np.isnan(df_lmp['new_ORDC_' + arg + '_Duration_' + strike +
                                                           '_SPP_constrained']),
                                          df_lmp['new_ORDC_' + arg + '_Strike_' + strike + '_SPP'] *
                                          (df_lmp['new_ORDC_' + arg + '_SPP'] - df_strikes.loc[strike + '_spp', col]),
                                          0)) * 0.25 * (df_lmp['SCED total time'] / df_lmp['SPP total time'])


@performance
def value_pivot(*args):
    df_list = []
    df_new_ordc_list = []
    constrained_df_list = []
    constrained_df_new_ordc_list = []
    for arg in args:
        for strike in strike_range:
            temp_df = pd.pivot_table(df_lmp, values=arg + '_Value_' + strike + '_SPP',
                                     index='year', columns='month', aggfunc=np.sum)
            temp_df.insert(0, 'Strike', strike)
            temp_df.insert(1, 'Load Zone', arg)
            df_list.append(temp_df)
    pivot_df = pd.concat(df_list)
    for arg in args:
        for strike in strike_range:
            temp_df = pd.pivot_table(df_lmp, values='new_ORDC_' + arg + '_Value_' + strike + '_SPP',
                                     index='year', columns='month', aggfunc=np.sum)
            temp_df.insert(0, 'Strike', strike)
            temp_df.insert(1, 'Load Zone', arg)
            df_new_ordc_list.append(temp_df)
    new_ordc_pivot_df = pd.concat(df_new_ordc_list)
    for arg in args:
        for strike in strike_range:
            temp_df = pd.pivot_table(df_lmp, values=arg + '_Value_' + strike + '_SPP_constrained',
                                     index='year', columns='month', aggfunc=np.sum)
            temp_df.insert(0, 'Strike', strike)
            temp_df.insert(1, 'Load Zone', arg)
            constrained_df_list.append(temp_df)
    constrained_pivot_df = pd.concat(constrained_df_list)
    for arg in args:
        for strike in strike_range:
            temp_df = pd.pivot_table(df_lmp, values='new_ORDC_' + arg + '_Value_' + strike + '_SPP_constrained',
                                     index='year', columns='month', aggfunc=np.sum)
            temp_df.insert(0, 'Strike', strike)
            temp_df.insert(1, 'Load Zone', arg)
            constrained_df_new_ordc_list.append(temp_df)
    constrained_new_ordc_pivot_df = pd.concat(constrained_df_new_ordc_list)
    return pivot_df, new_ordc_pivot_df, constrained_pivot_df, constrained_new_ordc_pivot_df


@performance
def duration_pivot(*args):
    df_list_merged = []
    df_new_ordc_list_merged = []
    for arg in args:
        df_list = []
        for strike in strike_range:
            df_lmp[arg + '_Duration_' + strike + '_SPP'] = df_lmp[arg + '_Duration_' + strike + '_SPP'].round(1)
            index_list = ['year', 'month', 'day', arg + '_Events_low_SPP']
            temp_df = pd.pivot_table(df_lmp, values=arg + '_Duration_' + strike + '_SPP',
                                     index=index_list, aggfunc=np.max)
            temp_df.insert(0, 'Load Zone', arg)
            temp_df = temp_df.reset_index()
            temp_df = temp_df.rename({arg + '_Events_low_SPP': 'Event'}, axis=1)
            df_list.append(temp_df)
        for strike in strike_range:
            df_lmp[arg + '_Duration_' + strike + '_SPP_constrained'] = \
                df_lmp[arg + '_Duration_' + strike + '_SPP_constrained'].round(1)
            index_list = ['year', 'month', 'day', arg + '_Events_low_SPP']
            temp_df = pd.pivot_table(df_lmp, values=arg + '_Duration_' + strike + '_SPP_constrained',
                                     index=index_list, aggfunc=np.max)
            temp_df.insert(0, 'Load Zone', arg)
            temp_df = temp_df.reset_index()
            temp_df = temp_df.rename({arg + '_Events_low_SPP': 'Event'}, axis=1)
            df_list.append(temp_df)
        df = reduce(lambda x, y: pd.merge(x, y, how='left', on=['year', 'month', 'day', 'Event',
                                                                'Load Zone']), df_list)
        df_list_merged.append(df)
    pivot_df = pd.concat(df_list_merged)
    pivot_df = pivot_df.fillna('')
    pivot_df = pivot_df.reset_index()
    pivot_df = pivot_df.rename({'year': 'Year', 'month': 'Month', 'day': 'Day'}, axis=1)
    pivot_df = pivot_df.set_index('Year')
    pivot_df.drop('index', axis='columns', inplace=True)
    for arg in args:
        df_list = []
        for strike in strike_range:
            df_lmp['new_ORDC_' + arg + '_Duration_' + strike + '_SPP'] =\
                df_lmp['new_ORDC_' + arg + '_Duration_' + strike + '_SPP'].round(1)
            index_list = ['year', 'month', 'day', 'new_ORDC_' + arg + '_Events_low_SPP']
            temp_df = pd.pivot_table(df_lmp, values='new_ORDC_' + arg + '_Duration_' + strike + '_SPP',
                                     index=index_list, aggfunc=np.max)
            temp_df.insert(0, 'Load Zone', arg)
            temp_df = temp_df.reset_index()
            temp_df = temp_df.rename({'new_ORDC_' + arg + '_Events_low_SPP': 'Event'}, axis=1)
            df_list.append(temp_df)
        for strike in strike_range:
            df_lmp['new_ORDC_' + arg + '_Duration_' + strike + '_SPP_constrained'] =\
                df_lmp['new_ORDC_' + arg + '_Duration_' + strike + '_SPP_constrained'].round(1)
            index_list = ['year', 'month', 'day', 'new_ORDC_' + arg + '_Events_low_SPP_constrained']
            temp_df = pd.pivot_table(df_lmp, values='new_ORDC_' + arg + '_Duration_' + strike + '_SPP_constrained',
                                     index=index_list, aggfunc=np.max)
            temp_df.insert(0, 'Load Zone', arg)
            temp_df = temp_df.reset_index()
            temp_df = temp_df.rename({'new_ORDC_' + arg + '_Events_low_SPP_constrained': 'Event'}, axis=1)
            df_list.append(temp_df)
        df = reduce(lambda x, y: pd.merge(x, y, how='left', on=['year', 'month', 'day', 'Event',
                                                                'Load Zone']), df_list)
        df_new_ordc_list_merged.append(df)
    new_ordc_pivot_df = pd.concat(df_new_ordc_list_merged)
    new_ordc_pivot_df = new_ordc_pivot_df.fillna('')
    new_ordc_pivot_df = new_ordc_pivot_df.reset_index()
    new_ordc_pivot_df = new_ordc_pivot_df.rename({'year': 'Year', 'month': 'Month', 'day': 'Day'}, axis=1)
    new_ordc_pivot_df = new_ordc_pivot_df.set_index('Year')
    new_ordc_pivot_df.drop('index', axis='columns', inplace=True)
    return pivot_df, new_ordc_pivot_df


@performance
def strike_pivot(*args):
    df_list = []
    df_new_ordc_list = []
    for arg in args:
        for strike in strike_range:
            temp_df = pd.pivot_table(df_lmp, values=arg + '_Strike_' + strike + '_SPP',
                                     index=['year', 'hour'], columns='month', aggfunc=np.sum)
            temp_df.insert(0, 'Strike', strike)
            temp_df.insert(1, 'Load Zone', arg)
            df_list.append(temp_df)
    pivot_df = pd.concat(df_list)
    for arg in args:
        for strike in strike_range:
            temp_df = pd.pivot_table(df_lmp, values='new_ORDC_' + arg + '_Strike_' + strike + '_SPP',
                                     index=['year', 'hour'], columns='month', aggfunc=np.sum)
            temp_df.insert(0, 'Strike', strike)
            temp_df.insert(1, 'Load Zone', arg)
            df_new_ordc_list.append(temp_df)
    new_ordc_pivot_df = pd.concat(df_new_ordc_list)
    return pivot_df, new_ordc_pivot_df


print('Program started at ' + str(datetime.now()))
# set variables
folder_path = '/Users/rwilson/Desktop/Test/'
output_file = 'Hedge_Value_Chipotle_with_10min.xlsx'
excel_file = pd.ExcelWriter('/Users/rwilson/Desktop/' + output_file)
file_name = folder_path + 'Test.xlsx'
xls = pd.ExcelFile(file_name)
spp_columns = ['Delivery Date', 'Delivery Hour', 'Delivery Interval', 'Repeated Hour Flag', 'HB_NORTH',
               'LZ_NORTH', 'LZ_HOUSTON', 'LZ_SOUTH', 'LZ_WEST']
spp_types = ['HU', 'LZ', '']
lmp_columns = ['SCEDTimestamp', 'RepeatedHourFlag', 'HB_NORTH', 'LZ_NORTH', 'LZ_HOUSTON', 'LZ_SOUTH',
               'LZ_WEST', 'System Lamda', 'RTORPA', 'RTOFFPA', 'RTORDPA', 'RTOLCAP', 'RTOFFCAP']
final_lmp_columns_delete = ['HB_NORTH', 'LZ_HOUSTON', 'LZ_NORTH', 'LZ_SOUTH', 'LZ_WEST', 'System Lamda',
                            'RTORPA', 'RTOFFPA', 'RTORDPA', 'RTOLCAP', 'RTOFFCAP', 'low_spp', 'mid_spp',
                            'high_spp', 'low_lmp', 'mid_lmp', 'high_lmp', 'Season',
                            'Average + (0.5 * Stand Deviation)', 'Standard Deviation',
                            'new_ORDC_System Lamda', 'new_ORDC_LZ_HOUSTON', 'new_ORDC_LZ_NORTH',
                            'new_ORDC_LZ_SOUTH', 'new_ORDC_LZ_WEST', 'new_ORDC_RTOFFPA',
                            'new_ORDC_RTORPA', 'SCED total time', 'SPP total time', 'Daily Events Max',
                            'Weekly Events Max', 'Event Duration Min (min)', 'Event Duration Max (min)',
                            'Event Window Start Hour', 'Event Window End Hour', 'Monday', 'Tuesday',
                            'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
voll = 5000
mcl = 3000

# SPP and LMP strikes
strikes_spp = {'1': [100, 150, 200], '2': [100, 150, 200], '3': [55, 80, 110],
               '4': [55, 80, 110], '5': [55, 80, 110], '6': [100, 150, 200],
               '7': [100, 150, 200], '8': [100, 150, 200], '9': [100, 150, 200],
               '10': [55, 80, 110], '11': [55, 80, 110], '12': [100, 150, 200]}
strikes_lmp = {'1': [300, 450, 600], '2': [300, 450, 600], '3': [170, 250, 350],
               '4': [170, 250, 350], '5': [170, 250, 350], '6': [300, 450, 600],
               '7': [300, 450, 600], '8': [300, 450, 600], '9': [300, 450, 600],
               '10': [170, 250, 350], '11': [170, 250, 350], '12': [300, 450, 600]}
strike_range = ['low', 'mid', 'high']
df_strikes = pd.concat([pd.DataFrame(strikes_spp, index=['low_spp', 'mid_spp', 'high_spp']),
                        pd.DataFrame(strikes_lmp, index=['low_lmp', 'mid_lmp', 'high_lmp'])])

# constraints
constraints = {'Initial Dispatch Capture Pct': 0.2, 'Dispatch Outside Event Window': True}
event_window = {'1': [22, 5], '2': [22, 5], '3': [22, 5], '4': [22, 5], '5': [22, 5],
                '6': [22, 5], '7': [22, 5], '8': [22, 5], '9': [22, 5], '10': [22, 5],
                '11': [22, 5], '12': [22, 5]}
event_duration = {'1': [10, 100, 11, 360], '2': [10, 100, 11, 360], '3': [10, 100, 11, 360],
                  '4': [10, 100, 11, 360],  '5': [10, 100, 11, 360], '6': [10, 100, 11, 360],
                  '7': [10, 100, 11, 360], '8': [10, 100, 11, 360], '9': [10, 100, 11, 360],
                  '10': [10, 100, 11, 360], '11': [10, 100, 11, 360], '12': [10, 100, 11, 360]}
# event_day is binary by month with the array starting Monday and ending Sunday
event_day = {'1': [1, 1, 1, 1, 1, 1, 1], '2': [1, 1, 1, 1, 1, 1, 1], '3': [1, 1, 1, 1, 1, 1, 1],
             '4': [1, 1, 1, 1, 1, 1, 1], '5': [1, 1, 1, 1, 1, 1, 1], '6': [1, 1, 1, 1, 1, 1, 1],
             '7': [1, 1, 1, 1, 1, 1, 1], '8': [1, 1, 1, 1, 1, 1, 1], '9': [1, 1, 1, 1, 1, 1, 1],
             '10': [1, 1, 1, 1, 1, 1, 1], '11': [1, 1, 1, 1, 1, 1, 1], '12': [1, 1, 1, 1, 1, 1, 1]}
df_constraints = pd.concat([pd.DataFrame(event_duration, index=['Daily Events Max',
                                                                'Weekly Events Max',
                                                                'Event Duration Min (min)',
                                                                'Event Duration Max (min)']),
                            pd.DataFrame(event_window, index=['Event Window Start Hour',
                                                              'Event Window End Hour']),
                            pd.DataFrame(event_day, index=['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                                                           'Friday', 'Saturday', 'Sunday'])])
'''
min_duration_seconds = constraints['Initial Dispatch Capture Pct'] * 300
max_weekly_duration = 540
max_monthly_dispatch = 30
max_monthly_duration = 1200
delay_end_seconds = 600
recovery_time_minutes = 120
'''
# get data
print('Retrieving data at ' + str(datetime.now()))
df_lmp = pd.read_excel(xls, sheet_name='LMPs')
dict_spp = pd.read_excel(xls, sheet_name=['2018_SPP', '2019_SPP', '2020_SPP', '2021_SPP'])
df_spp = pd.concat(dict_spp)
df_lolp = pd.read_excel(xls, sheet_name='LOLP')

# format LOLP dataframe
print('Formatting dataframes at ' + str(datetime.now()))
df_lolp.drop(['Start Date', 'End Date', 'Start Hour', 'End Hour'], axis='columns', inplace=True)
df_lolp = df_lolp.drop_duplicates()
df_lolp = df_lolp.reset_index(drop=True)

# format SPP dataframe
df_spp = pd.pivot_table(df_spp, values='Settlement Point Price',
                        index=['Delivery Date', 'Delivery Hour', 'Delivery Interval', 'Repeated Hour Flag'],
                        columns=['Settlement Point Name', 'Settlement Point Type']).reset_index()
spp_columns_to_delete = [col for col in df_spp.columns if col[0] not in spp_columns or col[1]
                         not in spp_types]
df_spp.drop(spp_columns_to_delete, axis='columns', inplace=True)
df_spp.columns = df_spp.columns.droplevel(1)
df_spp['Delivery Hour'] = df_spp['Delivery Hour'].apply(lambda x: x - 1)
df_spp = df_spp.astype({'Delivery Hour': "int", 'Delivery Interval': "int"})
df_spp['Index'] = df_spp['Delivery Date'].map(str) + '-' + df_spp['Delivery Hour'].map(str) + '-' + \
                  df_spp['Delivery Interval'].map(str) + '-' + df_spp['Repeated Hour Flag'].map(str)
df_spp = df_spp.set_index('Index')
df_spp = df_spp.rename({'HB_NORTH': 'HB_NORTH_SPP', 'LZ_NORTH': 'LZ_NORTH_SPP',
                        'LZ_HOUSTON': 'LZ_HOUSTON_SPP', 'LZ_SOUTH': 'LZ_SOUTH_SPP',
                        'LZ_WEST': 'LZ_WEST_SPP'}, axis=1)

# format LMP dataframe
lmp_columns_to_delete = [col for col in df_lmp.columns if col not in lmp_columns]
df_lmp.drop(lmp_columns_to_delete, axis='columns', inplace=True)
df_lmp['SCEDTimestamp'] = pd.to_datetime(df_lmp['SCEDTimestamp'])
df_lmp['date'] = df_lmp['SCEDTimestamp'].dt.date
df_lmp['date'] = pd.to_datetime(df_lmp['date'])
df_lmp['date'] = df_lmp['date'].dt.strftime('%m/%d/%Y')
df_lmp['year'] = df_lmp['SCEDTimestamp'].dt.year
df_lmp['month'] = df_lmp['SCEDTimestamp'].dt.month
df_lmp['day'] = df_lmp['SCEDTimestamp'].dt.day
df_lmp['weekday'] = df_lmp['SCEDTimestamp'].dt.weekday
weekdays = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday',
            6: 'Sunday'}
df_lmp['weekday'] = df_lmp['weekday'].apply(lambda x: weekdays.get(x))
df_lmp['week number'] = df_lmp['SCEDTimestamp'].dt.isocalendar().week
df_lmp['hour'] = df_lmp['SCEDTimestamp'].dt.hour
df_lmp['minute'] = df_lmp['SCEDTimestamp'].dt.minute
df_lmp['second'] = df_lmp['SCEDTimestamp'].dt.second
df_lmp['interval'] = np.where(df_lmp['minute'] < 15, 1,
                              np.where((df_lmp['minute'] < 30) & (df_lmp['minute'] >= 15), 2,
                                       np.where((df_lmp['minute'] < 45) & (df_lmp['minute'] >= 30), 3, 4)))
df_lmp = df_lmp.astype({'interval': "int"})
df_lmp['SCED total time'] = np.where(pd.isna(df_lmp['SCEDTimestamp'].shift(-1) - df_lmp['SCEDTimestamp']),
                                     (60.0 - df_lmp['minute']) * 60,
                                     (df_lmp['SCEDTimestamp'].shift(-1) - df_lmp['SCEDTimestamp']).dt.total_seconds())
df_lmp['SPP total time'] = df_lmp['SCED total time'].groupby(
            [df_lmp['date'], df_lmp['hour'], df_lmp['interval']]).transform('sum')
df_lmp['Season'] = np.where((df_lmp['month'] >= 3) & (df_lmp['month'] <= 5), 'Spring',
                            np.where((df_lmp['month'] >= 6) & (df_lmp['month'] <= 8), 'Summer',
                                     np.where((df_lmp['month'] >= 9) & (df_lmp['month'] <= 11),
                                              'Fall', 'Winter')))
df_lmp['Index'] = df_lmp['date'].map(str) + '-' + df_lmp['hour'].map(str) + '-' + \
                  df_lmp['interval'].map(str) + '-' + df_lmp['RepeatedHourFlag'].map(str)

# Adding ORDC and SPPs
add_ordc('LZ_NORTH', 'LZ_HOUSTON', 'LZ_SOUTH', 'LZ_WEST')
df_lmp = pd.merge(df_lmp, df_spp, how='left', on='Index')
df_lmp.drop(['Index', 'Delivery Date', 'Delivery Hour', 'Delivery Interval', 'Repeated Hour Flag'],
            axis='columns', inplace=True)
df_lmp = pd.merge(df_lmp, df_lolp, how='left', on='Season')

# Adding Strikes and Constraints
df_strikes = df_strikes.transpose()
df_strikes.insert(0, 'month', df_strikes.index)
df_strikes['month'] = df_strikes['month'].astype(int)
df_lmp = pd.merge(df_lmp, df_strikes, how='left', on='month')
df_constraints = df_constraints.transpose()
df_constraints.insert(0, 'month', df_constraints.index)
df_constraints['month'] = df_constraints['month'].astype(int)
df_lmp = pd.merge(df_lmp, df_constraints, how='left', on='month')
df_lmp = df_lmp.set_index('SCEDTimestamp')
df_strikes.drop('month', axis='columns', inplace=True)
df_strikes = df_strikes.transpose()
df_constraints.drop('month', axis='columns', inplace=True)
df_constraints = df_constraints.transpose()
df_lmp['Weekday Dispatch'] = np.nan
for day in weekdays.values():
    df_lmp['Weekday Dispatch'] = np.where(df_lmp['weekday'] == day, df_lmp[day],
                                          df_lmp['Weekday Dispatch'])

# build data sheet
set_new_ordc('LZ_NORTH', 'LZ_HOUSTON', 'LZ_SOUTH', 'LZ_WEST')
calc_new_ordc_spp('LZ_NORTH', 'LZ_HOUSTON', 'LZ_SOUTH', 'LZ_WEST')
set_strike('LZ_NORTH', 'LZ_HOUSTON', 'LZ_SOUTH', 'LZ_WEST')
set_dispatch('LZ_NORTH', 'LZ_HOUSTON', 'LZ_SOUTH', 'LZ_WEST')
set_events('LZ_NORTH', 'LZ_HOUSTON', 'LZ_SOUTH', 'LZ_WEST')
set_duration('LZ_NORTH', 'LZ_HOUSTON', 'LZ_SOUTH', 'LZ_WEST')
calc_value('LZ_NORTH', 'LZ_HOUSTON', 'LZ_SOUTH', 'LZ_WEST')
constrained_dispatch('LZ_NORTH', 'LZ_HOUSTON', 'LZ_SOUTH', 'LZ_WEST')
constrained_events('LZ_NORTH', 'LZ_HOUSTON', 'LZ_SOUTH', 'LZ_WEST')
constrained_duration('LZ_NORTH', 'LZ_HOUSTON', 'LZ_SOUTH', 'LZ_WEST')
constrained_value('LZ_NORTH', 'LZ_HOUSTON', 'LZ_SOUTH', 'LZ_WEST')
df_value, new_ordc_df_value, constrained_df_value, constrained_new_ordc_df_value =\
    value_pivot('LZ_NORTH', 'LZ_HOUSTON', 'LZ_SOUTH', 'LZ_WEST')

# Format value pivots
df_value['Strike'] = pd.Categorical(df_value['Strike'], strike_range)
df_value = df_value.sort_values(['year', 'Strike'])
df_value = df_value.round(0)
new_ordc_df_value['Strike'] = pd.Categorical(new_ordc_df_value['Strike'], strike_range)
new_ordc_df_value = new_ordc_df_value.sort_values(['year', 'Strike'])
new_ordc_df_value = new_ordc_df_value.round(0)
constrained_df_value['Strike'] = pd.Categorical(constrained_df_value['Strike'], strike_range)
constrained_df_value = constrained_df_value.sort_values(['year', 'Strike'])
constrained_df_value = constrained_df_value.round(0)
constrained_new_ordc_df_value['Strike'] = pd.Categorical(constrained_new_ordc_df_value['Strike'], strike_range)
constrained_new_ordc_df_value = constrained_new_ordc_df_value.sort_values(['year', 'Strike'])
constrained_new_ordc_df_value = constrained_new_ordc_df_value.round(0)

df_duration, new_ordc_df_duration = duration_pivot('LZ_NORTH', 'LZ_HOUSTON', 'LZ_SOUTH', 'LZ_WEST')
df_lmp.drop(final_lmp_columns_delete, axis='columns', inplace=True)
df_lmp = df_lmp.set_index('date')

df_strike_pivot, new_ordc_df_strike_pivot = strike_pivot('LZ_NORTH')

# Build Excel file
print('Saving Strikes and Constraints to Excel file at ' + str(datetime.now()))
df_strikes.to_excel(excel_file, 'Strikes')  # second element is worksheet name
df_constraints.to_excel(excel_file, 'Constraints')
print('Saving Value to Excel file at ' + str(datetime.now()))
df_value.to_excel(excel_file, 'Value')
constrained_df_value.to_excel(excel_file, 'Value Constrained')
new_ordc_df_value.to_excel(excel_file, 'Value_New ORDC')
constrained_new_ordc_df_value.to_excel(excel_file, 'Value_ORDC Constrained')
print('Saving Duration to Excel file at ' + str(datetime.now()))
df_duration.to_excel(excel_file, 'Duration')
new_ordc_df_duration.to_excel(excel_file, 'Duration_New ORDC')
print('Saving Strikes to Excel file at ' + str(datetime.now()))
df_strike_pivot.to_excel(excel_file, 'Strike Pivot')
# new_ordc_df_strike_pivot.to_excel(excel_file, 'Strike Pivot_New ORDC')
# print('Saving Data to Excel file at ' + str(datetime.now()))
# df_lmp.to_excel(excel_file, 'Data')  # Only include if raw data is needed since
# the program slows down a lot -> over two hours if all load zones included
print('Saving Excel file at ' + str(datetime.now()))
excel_file.save()
print('Program completed at ' + str(datetime.now()))
