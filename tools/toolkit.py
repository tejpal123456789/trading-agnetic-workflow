from .finance_data import get_yfinance_data
from .indicator_data import get_technical_indicators
from .finance_news import get_finnhub_news
from .social_media_sentiment import get_social_media_sentiment
from .fundamental_analysis import get_fundamental_analysis
from .macro_news import get_macroeconomic_news

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from dotenv import load_dotenv
load_dotenv()
# The Toolkit class aggregates all defined tools into a single, convenient object.
class Toolkit:
    def __init__(self, config):
        self.config = config
        self.get_yfinance_data = get_yfinance_data
        self.get_technical_indicators = get_technical_indicators
        self.get_finnhub_news = get_finnhub_news
        self.get_social_media_sentiment = get_social_media_sentiment
        self.get_fundamental_analysis = get_fundamental_analysis
        self.get_macroeconomic_news = get_macroeconomic_news

# Instantiate the Toolkit, making all tools available through this single object.
toolkit = Toolkit(config)
print(f"Toolkit class defined and instantiated with live data tools.")