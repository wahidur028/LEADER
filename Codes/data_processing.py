import pandas as pd
import pandas_ta as ta
import numpy as np

class BitcoinFeatureEngineer:
    def __init__(self, input_path, output_enhanced="ethereum_enhanced_dataset(Jan2015toMay2025).csv", output_cleaned="ethereum_price_data_with_added_features(Jan2015toMay2025).csv"):
        self.input_path = input_path
        self.output_enhanced = output_enhanced
        self.output_cleaned = output_cleaned
        self.df = None

    def load_and_prepare_data(self):
        self.df = pd.read_csv(self.input_path, encoding='utf-8-sig')
        self.df.columns = self.df.columns.str.strip()

        if 'Date' not in self.df.columns:
            raise KeyError("The column 'Date' was not found. Check column names.")

        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.set_index('Date', inplace=True)

        for col in ['Open', 'High', 'Low', 'Close']:
            self.df[col] = pd.to_numeric(self.df[col].astype(str).str.replace(',', ''), errors='coerce')

        def clean_volume(val):
            if isinstance(val, str):
                val = val.replace(',', '')
                if 'K' in val:
                    return float(val.replace('K', '')) * 1e3
                elif 'M' in val:
                    return float(val.replace('M', '')) * 1e6
            return pd.to_numeric(val, errors='coerce')

        self.df['Volume'] = self.df['Volume'].apply(clean_volume)
        self.df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

    def apply_technical_indicators(self):
        print("📊 Applying all major technical indicators...")

        # --- Trend Indicators ---
        self.df.ta.sma(append=True)
        self.df.ta.ema(append=True)
        self.df.ta.hma(append=True)
        self.df.ta.wma(append=True)
        self.df.ta.adx(append=True)
        self.df.ta.linreg(append=True)
        self.df.ta.midpoint(append=True)

        # --- Momentum Indicators ---
        self.df.ta.rsi(append=True)
        self.df.ta.macd(append=True)
        self.df.ta.roc(append=True)
        self.df.ta.mom(append=True)
        self.df.ta.stochrsi(append=True)
        self.df.ta.kdj(append=True)
        self.df.ta.ao(append=True)
        self.df.ta.apo(append=True)
        self.df.ta.ppo(append=True)
        self.df.ta.cmo(append=True)
        self.df.ta.tsi(append=True)

        # --- Volatility Indicators ---
        self.df.ta.atr(append=True)
        self.df.ta.natr(append=True)
        self.df.ta.bbands(append=True)
        self.df.ta.kc(append=True)
        self.df.ta.donchian(append=True)

        # --- Volume Indicators ---
        self.df.ta.obv(append=True)
        self.df.ta.mfi(append=True)
        self.df.ta.cmf(append=True)
        self.df.ta.vwap(append=True)
        self.df.ta.ad(append=True)
        self.df.ta.pvi(append=True)
        self.df.ta.nvi(append=True)
        self.df.ta.efi(append=True)

        # --- Custom Features ---
        self.df['log_return'] = np.log(self.df['Close'] / self.df['Close'].shift(1))

        self.df['fib_low'] = self.df['Low'].rolling(100).min()
        self.df['fib_high'] = self.df['High'].rolling(100).max()
        self.df['fib_0.382'] = self.df['fib_high'] - 0.382 * (self.df['fib_high'] - self.df['fib_low'])
        self.df['fib_0.5'] = self.df['fib_high'] - 0.5 * (self.df['fib_high'] - self.df['fib_low'])
        self.df['fib_0.618'] = self.df['fib_high'] - 0.618 * (self.df['fib_high'] - self.df['fib_low'])

        # --- Add Specific Missing Features ---
        self.df['WILLR_14'] = ta.willr(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], length=14)
        stoch = ta.stoch(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], k=14, d=3, smooth_k=3)
        self.df['STOCHk_14_3_3'] = stoch['STOCHk_14_3_3']
        self.df['STOCHd_14_3_3'] = stoch['STOCHd_14_3_3']

        bb_20 = ta.bbands(close=self.df['Close'], length=20, std=2)
        self.df['BBP_20_2.0'] = bb_20['BBP_20_2.0']
        self.df['BBL_20_2.0'] = bb_20['BBL_20_2.0']
        self.df['BBM_20_2.0'] = bb_20['BBM_20_2.0']
        self.df['BBU_20_2.0'] = bb_20['BBU_20_2.0']
        self.df['BBB_20_2.0'] = bb_20['BBB_20_2.0']

        # --- Add SMA_14 and EMA_14 ---
        self.df['SMA_14'] = ta.sma(self.df['Close'], length=14)
        self.df['EMA_14'] = ta.ema(self.df['Close'], length=14)

        self.df.to_csv(self.output_enhanced)
        print(f"✅ All indicators added and saved to '{self.output_enhanced}'")

    def clean_and_export_data(self):
        if self.df.index.name == 'Date':
            self.df.reset_index(inplace=True)

        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.sort_values('Date', inplace=True)
        self.df.set_index('Date', inplace=True)

        columns_to_drop = [col for col in ['SNo', 'Name', 'Symbol'] if col in self.df.columns]
        df_cleaned = self.df.drop(columns=columns_to_drop)
        df_cleaned.dropna(inplace=True)
        df_cleaned.to_csv(self.output_cleaned)
        print(f"✅ Cleaned dataset saved as '{self.output_cleaned}'")

    def run(self):
        self.load_and_prepare_data()
        self.apply_technical_indicators()
        self.clean_and_export_data()


if __name__ == "__main__":
    input_csv_path = "/home/infonet/wahid/projects/Fin/cryptotrade/data_processing/raw_data/ethereum_price_data(Jan-2015toMay2025).csv"
    engine = BitcoinFeatureEngineer(input_path=input_csv_path)
    engine.run()


