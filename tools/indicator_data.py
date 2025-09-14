from stockstats import wrap as stockstats_wrap
from langchain_core.tools import tool
from typing import Annotated
import yfinance as yf

@tool
def get_technical_indicators(

    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format - should be at least 3 months before end_date for meaningful technical analysis"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format - typically the analysis target date"],
) -> str:
    """Retrieve key technical indicators for a stock. Requires at least 90 days of historical data between start_date and end_date to calculate meaningful indicators like RSI, MACD, and moving averages."""
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if df.empty:
            return "No data to calculate indicators."
        stock_df = stockstats_wrap(df)
        indicators = stock_df[['macd', 'rsi_14', 'boll', 'boll_ub', 'boll_lb', 'close_50_sma', 'close_200_sma']]
        return indicators.tail().to_csv()
    except Exception as e:
        return f"Error calculating stockstats indicators: {e}"