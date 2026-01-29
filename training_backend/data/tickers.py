"""
Massive list of tickers for large-scale data collection.
Includes S&P 500 components and top Cryptocurrencies.
"""

# Top ~500 US Stocks (S&P 500 approximation + Nasdaq 100)
US_STOCKS = [
    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "AVGO", "AMD",
    "CRM", "ORCL", "ADBE", "CSCO", "NFLX", "INTC", "QCOM", "TXN", "IBM", "AMAT",
    "NOW", "UBER", "ABNB", "PANW", "SNPS", "CDNS", "MU", "LRCX", "ADI", "KLAC",
    "CRWD", "ROP", "MRVL", "NXPI", "FTNT", "ADSK", "TEAM", "WDAY", "DDOG", "ZS",
    "PLTR", "MDB", "SQ", "NET", "TTD", "HUBS", "PATH", "U", "AI", "SMCI",
    
    # Financials
    "JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "SCHW", "AXP", "SPGI",
    "BLK", "C", "PGR", "CB", "MMC", "USB", "ICE", "CME", "COF", "TRV",
    "AIG", "PYPL", "DFS", "ALL", "HIG", "BK", "STT", "TROW", "AMP", "PFG",
    "KKR", "APO", "BX", "ARES", "WAL", "CMA", "ZION", "KEY", "FITB", "RF",
    
    # Healthcare
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "ISRG", "PFE",
    "AMGN", "VRTX", "REGN", "SYK", "BMY", "ELEV", "ZTS", "BSX", "GILD", "CI",
    "CVS", "BDX", "HCA", "MCK", "MDT", "EW", "HUM", "BIIB", "DXCM", "ILMN",
    "A", "BAX", "VEEV", "STE", "RMD", "WEST", "ALGN", "PODD", "GMED", "TECH",
    
    # Consumer Discretionary
    "HD", "COST", "MCD", "WMT", "DIS", "NKE", "SBUX", "LOW", "TJX", "BKNG",
    "MAR", "HLT", "CMG", "LULU", "TGT", "ROST", "ORLY", "O", "YUM", "FAST",
    "LEN", "DHI", "NVR", "PHM", "TOL", "HD", "LOW", "DRI", "DPZ", "WEN",
    
    # Consumer Staples
    "PG", "KO", "PEP", "PM", "MO", "CL", "EL", "K", "GIS", "HSY",
    "MNST", "STZ", "KDP", "ADM", "TSN", "CAG", "MKC", "CHD", "CLX", "KMB",
    
    # Industrials & Energy
    "XOM", "CVX", "GE", "CAT", "UNP", "HON", "UPS", "LMT", "RTX", "BA",
    "DE", "ADP", "COP", "SLB", "EOG", "PXD", "MPC", "PSX", "VLO", "OXY",
    "ETN", "ITW", "WM", "RSG", "GD", "NOC", "LHX", "EMR", "PH", "CMI",
    
    # Communication Services
    "T", "VZ", "TMUS", "CMCSA", "CHTR", "DIS", "WBD", "PARA", "OMC", "IPG",
    
    # Real Estate
    "PLD", "AMT", "CCI", "EQIX", "PSA", "DLR", "SPG", "O", "VICI", "WELL",
    
    # Utilities
    "NEE", "SO", "DUK", "SRE", "AEP", "D", "PEG", "EXC", "XEL", "ED",
    
    # ETFs (Broad & Sector)
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO", "BND", "AGG",
    "GLD", "SLV", "USO", "TLT", "SHY", "IEF", "LQD", "HYG", "JNK", "TIP",
    "XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE",
    "SMH", "XBI", "KRE", "KBE", "XRT", "XHB", "ITA", "JETS", "TAN", "ICLN"
]

# Top 50 Cryptocurrencies
CRYPTO_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    "TRXUSDT", "DOTUSDT", "MATICUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT", "UNIUSDT",
    "XLMUSDT", "OKBUSDT", "ETCUSDT", "FILUSDT", "HBARUSDT", "LDOUSDT", "APTUSDT", "NEARUSDT",
    "QNTUSDT", "VETUSDT", "ALGOUSDT", "AAVEUSDT", "GRTUSDT", "FTMUSDT", "SANDUSDT", "EOSUSDT",
    "MANAUSDT", "THETAUSDT", "XTZUSDT", "AXSUSDT", "CHZUSDT", "RUNEUSDT", "ZECUSDT", "MKRUSDT",
    "CRVUSDT", "SNXUSDT", "KAVAUSDT", "MINAUSDT", "DYDXUSDT", "FXSUSDT", "COMPUSDT", "GALAUSDT"
]
