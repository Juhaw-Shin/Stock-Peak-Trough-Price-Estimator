import Ai_Back
import numpy as np
import random


def build_index(files):
    index = []

    for file_id, f in enumerate(files):
        with np.load(f) as data:
            for key in data.files:
                index.append((file_id, f, key))

    return index


def shuffled_index(index):
    idx = index.copy()
    random.shuffle(idx)
    return idx


def load_sample(entry):
    _, file_path, key = entry

    with np.load(file_path) as data:
        return data[key]


def random_batch(index, batch_size=10):
    random.shuffle(index)

    batch = []

    for entry in index:
        x = load_sample(entry)
        batch.append(x)

        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


Ticker_List = ['MMM', 'ABT', 'ADBE', 'AES', 'AFL', 'A', 'APD', 'ALL', 'MO', 'AEE', 'AEP', 'AXP', 'AIG', 'AMGN', 'ADI', 'AON', 'APA', 'AAPL', 'AMAT', 'ADM', 'T', 'ADSK', 'ADP', 'AZO', 'AVY', 'BALL', 'BAC', 'BAX', 'BDX', 'BBY', 'BIIB', 'BNY', 'BA', 'BSX', 'BMY', 'BF-B', 'CPB', 'COF', 'CAH', 'CCL', 'CAT', 'COR', 'CNP', 'SCHW', 'CVX', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CLX', 'CMS', 'KO', 'CL', 'CMCSA', 'CAG', 'COP', 'ED', 'GLW', 'COST', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DE', 'DVN', 'DOV', 'DTE', 'DUK', 'ETN', 'EBAY', 'ECL', 'EIX', 'EA', 'ELV', 'EMR', 'ETR', 'EOG', 'EFX', 'EQR', 'EXC', 'XOM', 'FDX', 'FITB', 'FE', 'FISV', 'F', 'BEN', 'GE', 'GEN', 'GD', 'GIS', 'GPC', 'GILD', 'GL', 'GS', 'HAL', 'HIG', 'HAS', 'HSY', 'HD', 'HON', 'HPQ', 'HBAN', 'IBM', 'ITW', 'INTC', 'IFF', 'IP', 'INTU', 'JNJ', 'JPM', 'KEY', 'KMB', 'KLAC', 'KR', 'LH', 'LLY', 'LIN', 'LMT', 'L', 'LOW', 'MTB', 'MAR', 'MRSH', 'MAS', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'MET', 'MU', 'MSFT', 'TAP', 'MCO', 'MS', 'MSI', 'NTAP', 'NEM', 'NEE', 'NKE', 'NI', 'NSC', 'NTRS', 'NOC', 'NUE', 'NVDA', 'OXY', 'OMC', 'ORCL', 'PCAR', 'PSKY', 'PH', 'PAYX', 'PEP', 'PFE', 'PNW', 'PNC', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PHM', 'QCOM', 'DGX', 'RTX', 'RF', 'RVTY', 'ROK', 'SPGI', 'SLB', 'SHW', 'SPG', 'SNA', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'SYK', 'SYY', 'TPR', 'TGT', 'TXN', 'TXT', 'TMO', 'TJX', 'TRV', 'TFC', 'USB', 'UNP', 'UPS', 'UNH', 'VLO', 'VZ', 'VTRS', 'VMC', 'GWW', 'WMT', 'DIS', 'WM', 'WAT', 'WFC', 'WY', 'WMB', 'XEL', 'YUM', 'ZBH']


files = ["NPZ/Ticker/Month-Ext/"+t+".npz" for t in Ticker_List]

index = build_index(files)

for batch in random_batch(index, batch_size=10):
    print(batch)
