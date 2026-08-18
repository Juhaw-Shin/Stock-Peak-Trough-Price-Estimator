import os

def create_folders(base_path, folder_names):
    """
    Creates folders based on a list of names inside base_path.
    """

    os.makedirs(base_path, exist_ok=True)  # ensure base exists

    created = 0

    for name in folder_names:
        safe_name = str(name).strip()

        folder_path = os.path.join(base_path, safe_name)

        # THIS is the missing line
        os.makedirs(folder_path, exist_ok=True)

        created += 1

    print(f"Created {created} folders in {base_path}")


# Example usage:
names = ['MMM', 'ABT', 'ADBE', 'AES', 'AFL', 'A', 'APD', 'ALL', 'MO', 'AEE', 'AEP', 'AXP', 'AIG', 'AMGN', 'ADI', 'AON', 'APA', 'AAPL', 'AMAT', 'ADM', 'T', 'ADSK', 'ADP', 'AZO', 'AVY', 'BALL', 'BAC', 'BAX', 'BDX', 'BBY', 'BIIB', 'BNY', 'BA', 'BSX', 'BMY', 'BF-B', 'CPB', 'COF', 'CAH', 'CCL', 'CAT', 'COR', 'CNP', 'SCHW', 'CVX', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CLX', 'CMS', 'KO', 'CL', 'CMCSA', 'CAG', 'COP', 'ED', 'GLW', 'COST', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DE', 'DVN', 'DOV', 'DTE', 'DUK', 'ETN', 'EBAY', 'ECL', 'EIX', 'EA', 'ELV', 'EMR', 'ETR', 'EOG', 'EFX', 'EQR', 'EXC', 'XOM', 'FDX', 'FITB', 'FE', 'FISV', 'F', 'BEN', 'GE', 'GEN', 'GD', 'GIS', 'GPC', 'GILD', 'GL', 'GS', 'HAL', 'HIG', 'HAS', 'HSY', 'HD', 'HON', 'HPQ', 'HBAN', 'IBM', 'ITW', 'INTC', 'IFF', 'IP', 'INTU', 'JNJ', 'JPM', 'KEY', 'KMB', 'KLAC', 'KR', 'LH', 'LLY', 'LIN', 'LMT', 'L', 'LOW', 'MTB', 'MAR', 'MRSH', 'MAS', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'MET', 'MU', 'MSFT', 'TAP', 'MCO', 'MS', 'MSI', 'NTAP', 'NEM', 'NEE', 'NKE', 'NI', 'NSC', 'NTRS', 'NOC', 'NUE', 'NVDA', 'OXY', 'OMC', 'ORCL', 'PCAR', 'PSKY', 'PH', 'PAYX', 'PEP', 'PFE', 'PNW', 'PNC', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PHM', 'QCOM', 'DGX', 'RTX', 'RF', 'RVTY', 'ROK', 'SPGI', 'SLB', 'SHW', 'SPG', 'SNA', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'SYK', 'SYY', 'TPR', 'TGT', 'TXN', 'TXT', 'TMO', 'TJX', 'TRV', 'TFC', 'USB', 'UNP', 'UPS', 'UNH', 'VLO', 'VZ', 'VTRS', 'VMC', 'GWW', 'WMT', 'DIS', 'WM', 'WAT', 'WFC', 'WY', 'WMB', 'XEL', 'YUM', 'ZBH']

create_folders(r"C:\Users\shinj\Desktop\VS Project\NPZ\Ticker\Month-Ext", names)
