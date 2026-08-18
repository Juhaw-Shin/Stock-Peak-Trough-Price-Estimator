import numpy as np
import pandas as pd
import yfinance as yf
import scipy.signal as ss

Inflation_Rate = np.array(pd.read_excel("Data.xlsx", sheet_name="Inflation Rate", usecols="B"))
Interest_Rate = np.array(pd.read_excel("Data.xlsx", sheet_name="Interest Rate", usecols="B"))
Unemployment_Rate = np.array(pd.read_excel("Data.xlsx", sheet_name="Unemployment Rate", usecols="B"))
Credit_GDP = np.array(pd.read_excel("Data.xlsx", sheet_name="CreditGDP", usecols="B"))
Money_Supply = np.array(pd.read_excel("Data.xlsx", sheet_name="Money Supply", usecols="B"))
Real_GDP = np.array(pd.read_excel("Data.xlsx", sheet_name="Real GDP", usecols="B"))

def Arraying(Data):
    Data = Data.reshape((np.size(Data)))
    return Data


Inflation_Rate, Interest_Rate, Unemployment_Rate, Credit_GDP, Money_Supply, Real_GDP = [Arraying(k) for k in [Inflation_Rate, Interest_Rate, Unemployment_Rate, Credit_GDP, Money_Supply, Real_GDP]]

def Value_To_Rate(Data):
    Data_Pad = np.pad(Data[:-1], (1, 0), mode='constant', constant_values=0)
    Data = (Data-Data_Pad)/Data
    return np.delete(Data, 0)*100

Credit_GDP, Money_Supply, Real_GDP = [Value_To_Rate(k) for k in [Credit_GDP, Money_Supply, Real_GDP]]
Credit_GDP, Money_Supply, Real_GDP = [np.delete(k, np.size(k)-1) for k in [Credit_GDP, Money_Supply, Real_GDP]]

Credit_GDP = np.insert(Credit_GDP, 0, 0.455)
Money_Supply = np.insert(Money_Supply, 0, 0.09806)
Real_GDP = np.insert(Real_GDP, 0, 1.02021)


def Tripling(Data):
    Data = np.ones((np.size(Data),3))*Data[:,None]
    return Data.ravel()

Credit_GDP, Real_GDP = [Tripling(Data) for Data in [Credit_GDP, Real_GDP]]

Money_Supply = ss.savgol_filter(Money_Supply, window_length=15, polyorder=1)
Credit_GDP = ss.savgol_filter(Credit_GDP, window_length=15, polyorder=1)
Real_GDP = ss.savgol_filter(Real_GDP, window_length=15, polyorder=1)

Real_Rate = Inflation_Rate - Interest_Rate


df = yf.download("^VIX", start="2005-01-01", end="2025-12-31", interval="1d")
df = df.resample("ME").mean()
VIX = np.array((df["Close"]+df["Open"]+df["High"]+df["Low"])/4)
VIX = VIX.ravel()
VIX = ss.savgol_filter(VIX, window_length=15, polyorder=2)

np.savez("Macro_Coef_2.npz", Inflation_Rate=Inflation_Rate, Real_Rate=Real_Rate, Unemployment_Rate=Unemployment_Rate, Credit_GDP=Credit_GDP, Money_Supply=Money_Supply, Real_GDP=Real_GDP, VIX=VIX)
