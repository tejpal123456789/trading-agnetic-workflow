# 🤖 Multi-Agent Trading Analysis System

## 📊 Project Overview

This project is a sophisticated **Multi-Agent Trading Analysis System** built using **LangGraph** and **LangChain** that simulates a complete trading firm with specialized AI agents working collaboratively to analyze stocks, make investment decisions, and manage risk. The system uses advanced AI models to process real-time market data, news, social sentiment, and technical indicators to provide comprehensive trading insights.

### 🎯 **Why This Project Matters**

In modern financial markets, successful trading requires:
- **Multi-dimensional Analysis**: Technical, fundamental, sentiment, and news analysis
- **Real-time Decision Making**: Fast processing of market data and news
- **Risk Management**: Sophisticated risk assessment and portfolio management
- **Collaborative Intelligence**: Multiple specialized agents working together
- **Scalable Architecture**: Handle multiple stocks and market conditions simultaneously

This system addresses these challenges by creating a **collaborative AI trading firm** where specialized agents work together to make informed investment decisions.

---

## 🏗️ **System Architecture**

```
trading/
├── 📁 agents/                       # Specialized AI trading agents
│   ├── 📁 analyst_agent/           # Market analysis specialists
│   ├── 📁 bull_vs_bear/           # Investment debate agents
│   ├── 📁 portfolio_manager_agent/ # Portfolio management
│   ├── 📁 research_agent/         # Research and data gathering
│   ├── 📁 risk_agent/             # Risk assessment specialists
│   └── 📁 trader_agent/           # Execution and trading decisions
├── 📁 tools/                       # Data collection and analysis tools
│   ├── finance_data.py            # Financial data APIs
│   ├── indicator_data.py          # Technical indicators
│   ├── fundamental_analysis.py    # Fundamental analysis tools
│   ├── social_media_sentiment.py  # Social sentiment analysis
│   ├── macro_news.py              # Macroeconomic news
│   └── finance_news.py            # Financial news aggregation
├── 📁 memory/                      # Long-term memory and learning
├── 📁 data_cache/                  # Cached market data
├── 📁 reflection/                  # Agent reflection and learning
├── 📄 run.py                       # Main workflow orchestration
├── �� run1.py                      # Extended workflow with debates
├── 📄 run2.py                      # Advanced streaming workflow
├── 📄 stream.py                    # Real-time streaming analysis
├── 📄 config.py                    # System configuration
├── 📄 agent_state.py               # State management
└── 📄 llm.py                       # Language model configuration
```

---

## 🤖 **Agent Ecosystem**

### **1. Analyst Agents** 📈
**Purpose**: Specialized market analysis using different data sources

#### **Market Analyst**
- **Function**: Technical analysis and price action
- **Tools**: Yahoo Finance, technical indicators
- **Output**: Market trends, support/resistance levels, momentum analysis
- **Key Metrics**: RSI, MACD, Bollinger Bands, moving averages

#### **Social Sentiment Analyst**
- **Function**: Social media and sentiment analysis
- **Tools**: Tavily search, social media APIs
- **Output**: Public sentiment, social media buzz, influencer opinions
- **Key Metrics**: Sentiment scores, engagement rates, trending topics

#### **News Analyst**
- **Function**: News and event analysis
- **Tools**: Financial news APIs, macro news
- **Output**: News impact assessment, event correlation
- **Key Metrics**: News sentiment, event importance, market reaction

#### **Fundamental Analyst**
- **Function**: Company fundamentals and financial health
- **Tools**: Financial data APIs, earnings data
- **Output**: Financial ratios, growth metrics, valuation analysis
- **Key Metrics**: P/E ratio, revenue growth, debt levels, profitability

### **2. Bull vs Bear Debate Agents** 🐂🐻
**Purpose**: Simulate investment debate between optimistic and pessimistic viewpoints

#### **Bull Agent**
- **Role**: Optimistic investment advocate
- **Focus**: Growth potential, positive catalysts, upside scenarios
- **Arguments**: Market opportunities, company strengths, bullish indicators

#### **Bear Agent**
- **Role**: Pessimistic risk assessor
- **Focus**: Risks, challenges, downside scenarios
- **Arguments**: Market risks, company weaknesses, bearish indicators

#### **Investment Judge**
- **Role**: Neutral arbitrator
- **Function**: Weighs arguments and makes final investment decision
- **Output**: Balanced investment recommendation with reasoning

### **3. Risk Management Agents** ⚖️
**Purpose**: Comprehensive risk assessment and portfolio management

#### **Risky Agent**
- **Role**: Aggressive risk-taker
- **Strategy**: High-risk, high-reward investments
- **Focus**: Growth stocks, volatile markets, leverage opportunities

#### **Safe Agent**
- **Role**: Conservative risk manager
- **Strategy**: Low-risk, stable investments
- **Focus**: Blue-chip stocks, defensive sectors, capital preservation

#### **Neutral Agent**
- **Role**: Balanced risk assessor
- **Strategy**: Moderate risk, balanced approach
- **Focus**: Diversified portfolio, risk-adjusted returns

#### **Risk Judge**
- **Role**: Final risk decision maker
- **Function**: Synthesizes risk perspectives into actionable risk management plan

### **4. Portfolio Manager Agent** 💼
**Purpose**: Overall portfolio strategy and allocation

#### **Functions**:
- **Asset Allocation**: Determines optimal portfolio mix
- **Position Sizing**: Calculates appropriate position sizes
- **Risk Budgeting**: Allocates risk across different investments
- **Performance Monitoring**: Tracks portfolio performance
- **Rebalancing**: Adjusts portfolio based on market conditions

### **5. Trader Agent** 📊
**Purpose**: Execution and trading decisions

#### **Functions**:
- **Order Management**: Creates and manages trade orders
- **Execution Strategy**: Determines optimal execution methods
- **Market Timing**: Identifies best entry/exit points
- **Slippage Management**: Minimizes transaction costs
- **Position Management**: Monitors and adjusts positions

### **6. Research Agent** 🔍
**Purpose**: Data gathering and research coordination

#### **Functions**:
- **Data Collection**: Gathers data from multiple sources
- **Research Coordination**: Coordinates research across agents
- **Data Validation**: Ensures data quality and accuracy
- **Report Generation**: Creates comprehensive research reports

---

## 🛠️ **Tools and Data Sources**

### **Financial Data Tools**
```python
# Yahoo Finance integration
@tool
def get_yfinance_data(symbol: str, start_date: str, end_date: str) -> str:
    """Retrieve stock price data from Yahoo Finance"""
    ticker = yf.Ticker(symbol.upper())
    data = ticker.history(start=start_date, end=end_date)
    return data.to_csv()
```

### **Technical Indicators**
```python
# Technical analysis tools
@tool
def get_technical_indicators(symbol: str, start_date: str, end_date: str) -> str:
    """Calculate technical indicators using stockstats"""
    df = yf.download(symbol, start=start_date, end=end_date)
    stock_df = stockstats_wrap(df)
    indicators = stock_df[['macd', 'rsi_14', 'boll', 'boll_ub', 'boll_lb']]
    return indicators.tail().to_csv()
```

### **News and Sentiment Analysis**
```python
# Real-time news and sentiment
@tool
def get_social_media_sentiment(ticker: str, trade_date: str) -> str:
    """Analyze social media sentiment using Tavily search"""
    query = f"social media sentiment for {ticker} stock around {trade_date}"
    return tavily_tool.invoke({"query": query})
```

### **Fundamental Analysis**
```python
# Company fundamentals
@tool
def get_fundamental_analysis(ticker: str, trade_date: str) -> str:
    """Get fundamental analysis using web search"""
    query = f"fundamental analysis {ticker} stock financial metrics {trade_date}"
    return tavily_tool.invoke({"query": query})
```

---

## 🔄 **Workflow Orchestration**

### **Main Workflow (run.py)**
```python
class TradingAnalysisWorkflow:
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Core workflow nodes
        workflow.add_node("initialization", self.initialization_node)
        workflow.add_node("parallel_analysis", self.parallel_analysis_node)
        workflow.add_node("consolidation", self.consolidation_node)
        
        # Workflow edges
        workflow.add_edge(START, "initialization")
        workflow.add_edge("initialization", "parallel_analysis")
        workflow.add_edge("parallel_analysis", "consolidation")
        workflow.add_edge("consolidation", END)
        
        return workflow.compile()
```

### **Extended Workflow (run1.py)**
- **Parallel Analysis**: Multiple agents analyze simultaneously
- **Bull vs Bear Debate**: Investment perspective debate
- **Risk Assessment**: Comprehensive risk analysis
- **Portfolio Management**: Final investment decisions

### **Streaming Workflow (run2.py)**
- **Real-time Processing**: Continuous market data processing
- **Event-driven Updates**: Responds to market events
- **Dynamic Rebalancing**: Adjusts positions based on real-time data

---

## 📊 **State Management**

### **AgentState Structure**
```python
class AgentState(MessagesState):
    # Core analysis data
    company_of_interest: str          # Stock ticker being analyzed
    trade_date: str                   # Analysis date
    sender: str                       # Last agent to modify state
    
    # Analyst reports
    market_report: str                # Technical analysis report
    sentiment_report: str             # Sentiment analysis report
    news_report: str                  # News analysis report
    fundamentals_report: str          # Fundamental analysis report
    
    # Investment decisions
    investment_plan: str              # Research manager's plan
    trader_investment_plan: str       # Trader's actionable plan
    final_trade_decision: str         # Portfolio manager's final decision
    
    # Debate states
    investment_debate_state: InvestDebateState
    risk_debate_state: RiskDebateState
```

### **Debate States**
```python
class InvestDebateState(TypedDict):
    bull_history: str                 # Bull agent arguments
    bear_history: str                 # Bear agent arguments
    history: str                      # Full debate transcript
    current_response: str             # Latest argument
    judge_decision: str               # Final investment decision
    count: int                        # Debate round counter

class RiskDebateState(TypedDict):
    risky_history: str                # Aggressive risk arguments
    safe_history: str                 # Conservative risk arguments
    neutral_history: str              # Balanced risk arguments
    history: str                      # Full risk discussion
    latest_speaker: str               # Last agent to speak
    judge_decision: str               # Final risk decision
    count: int                        # Discussion round counter
```

---

## ⚙️ **Configuration System**

### **Core Configuration (config.py)**
```python
config = {
    # LLM Settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-4o",       # Complex reasoning
    "quick_think_llm": "gpt-4o-mini", # Fast processing
    
    # API Configuration
    "backend_url": "https://api.openai.com/v1",
    
    # Workflow Settings
    "max_debate_rounds": 2,           # Bull vs Bear rounds
    "max_risk_discuss_rounds": 1,     # Risk discussion rounds
    "max_recur_limit": 100,           # Safety limit for loops
    
    # Data Settings
    "online_tools": True,             # Use live APIs
    "data_cache_dir": "./data_cache", # Cache directory
    "results_dir": "./results"        # Results directory
}
```

---

## 🚀 **Usage Examples**

### **1. Basic Trading Analysis**
```python
from run import TradingAnalysisWorkflow
from agent_state import AgentState
from langchain_core.messages import HumanMessage

# Initialize workflow
workflow = TradingAnalysisWorkflow()

# Create initial state
initial_state = AgentState(
    messages=[HumanMessage(content="Analyze NVDA for trading on 2025-09-15")],
    company_of_interest="NVDA",
    trade_date="2025-09-15",
    # ... other required fields
)

# Run analysis
result = workflow.graph.invoke(initial_state)
print(result['final_trade_decision'])
```

### **2. Extended Analysis with Debates**
```python
from run1 import ExtendedTradingWorkflow

# Initialize extended workflow
workflow = ExtendedTradingWorkflow()

# Run comprehensive analysis
result = workflow.run_comprehensive_analysis("AAPL", "2025-09-15")

# Access debate results
bull_arguments = result['investment_debate_state']['bull_history']
bear_arguments = result['investment_debate_state']['bear_history']
final_decision = result['investment_debate_state']['judge_decision']
```

### **3. Real-time Streaming Analysis**
```python
from stream import StreamingTradingAnalysis

# Initialize streaming analysis
streamer = StreamingTradingAnalysis()

# Start real-time monitoring
streamer.start_monitoring("TSLA", update_interval=300)  # 5-minute updates

# Get latest analysis
latest_analysis = streamer.get_latest_analysis()
```

---

## 📈 **Performance Metrics**

### **Analysis Quality Metrics**
- **Data Coverage**: Percentage of available data sources utilized
- **Analysis Depth**: Number of indicators and metrics analyzed
- **Decision Confidence**: Confidence scores for investment decisions
- **Risk Assessment Accuracy**: Historical risk prediction accuracy

### **System Performance Metrics**
- **Processing Time**: Time to complete full analysis
- **Memory Usage**: System resource utilization
- **API Response Times**: Data source response times
- **Error Rates**: System reliability metrics

### **Trading Performance Metrics**
- **Decision Accuracy**: Historical decision success rate
- **Risk-Adjusted Returns**: Performance vs. risk taken
- **Drawdown Management**: Maximum portfolio drawdown
- **Sharpe Ratio**: Risk-adjusted performance measure

---

## 🔧 **Installation and Setup**

### **Prerequisites**
```bash
# Python 3.8+
python --version

# Required packages
pip install langchain langgraph openai
pip install yfinance pandas numpy
pip install tavily-python rich
pip install python-dotenv
```

### **Environment Setup**
```bash
# Clone repository
git clone <repository-url>
cd trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-api-key"
export TAVILY_API_KEY="your-tavily-key"
```

### **Configuration**
```python
# Update config.py with your settings
config = {
    "llm_provider": "openai",
    "deep_think_llm": "gpt-4o",
    "quick_think_llm": "gpt-4o-mini",
    "backend_url": "https://api.openai.com/v1",
    # ... other settings
}
```

---

## 🧪 **Testing and Validation**

### **Unit Tests**
```bash
# Run agent tests
python -m pytest tests/test_agents.py

# Run tool tests
python -m pytest tests/test_tools.py

# Run workflow tests
python -m pytest tests/test_workflows.py
```

### **Integration Tests**
```bash
# Test complete workflow
python -m pytest tests/test_integration.py

# Test with real data
python -m pytest tests/test_real_data.py
```

### **Performance Tests**
```bash
# Benchmark analysis speed
python scripts/benchmark.py

# Test with multiple stocks
python scripts/batch_analysis.py
```

---

## 📊 **Data Flow Architecture**

### **Input Data Sources**
1. **Market Data**: Yahoo Finance, Alpha Vantage
2. **News Data**: Financial news APIs, RSS feeds
3. **Social Data**: Twitter, Reddit, social media APIs
4. **Fundamental Data**: SEC filings, earnings reports
5. **Macro Data**: Economic indicators, central bank data

### **Processing Pipeline**
1. **Data Collection**: Agents gather data from multiple sources
2. **Data Validation**: Ensure data quality and consistency
3. **Analysis**: Specialized agents analyze different aspects
4. **Debate**: Bull vs Bear and Risk agents debate perspectives
5. **Decision**: Portfolio manager makes final investment decision
6. **Execution**: Trader agent creates actionable trading plan

### **Output Generation**
1. **Analysis Reports**: Detailed analysis from each agent
2. **Investment Recommendations**: Buy/sell/hold decisions
3. **Risk Assessment**: Risk levels and mitigation strategies
4. **Portfolio Allocation**: Optimal position sizing
5. **Trading Plan**: Specific entry/exit strategies

---

## 🔒 **Risk Management**

### **Built-in Risk Controls**
- **Position Limits**: Maximum position sizes per stock
- **Risk Budgets**: Total portfolio risk allocation
- **Stop Losses**: Automatic loss-cutting mechanisms
- **Diversification**: Portfolio diversification requirements
- **Liquidity Management**: Maintain sufficient cash reserves

### **Risk Monitoring**
- **Real-time Risk Metrics**: Continuous risk assessment
- **Stress Testing**: Scenario analysis and stress tests
- **VaR Calculation**: Value at Risk calculations
- **Correlation Analysis**: Asset correlation monitoring
- **Volatility Tracking**: Market volatility assessment

---

## 🚀 **Deployment Options**

### **Local Development**
```bash
# Run single analysis
python run.py

# Run extended analysis
python run1.py

# Run streaming analysis
python stream.py
```

### **Docker Deployment**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "run.py"]
```

### **Cloud Deployment**
```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-analysis
spec:
  replicas: 3
  selector:
    matchLabels:
      app: trading-analysis
  template:
    spec:
      containers:
      - name: trading-analysis
        image: trading-analysis:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai-key
```

---

## �� **API Documentation**

### **Core Workflow API**
```python
class TradingAnalysisWorkflow:
    def __init__(self):
        """Initialize the trading analysis workflow"""
        
    def run_analysis(self, symbol: str, date: str) -> dict:
        """Run complete trading analysis for a symbol"""
        
    def get_analysis_report(self, symbol: str) -> str:
        """Get formatted analysis report"""
        
    def get_investment_recommendation(self, symbol: str) -> dict:
        """Get investment recommendation with reasoning"""
```

### **Agent API**
```python
class MarketAnalyst:
    def analyze_technical(self, symbol: str) -> dict:
        """Perform technical analysis"""
        
    def analyze_sentiment(self, symbol: str) -> dict:
        """Analyze market sentiment"""
        
    def analyze_fundamentals(self, symbol: str) -> dict:
        """Analyze company fundamentals"""
```

---

## 🔮 **Future Enhancements**

### **Planned Features**
- [ ] **Multi-Asset Support**: Support for crypto, forex, commodities
- [ ] **Advanced ML Models**: Integration with custom ML models
- [ ] **Real-time Alerts**: Push notifications for trading opportunities
- [ ] **Backtesting Engine**: Historical performance testing
- [ ] **Paper Trading**: Simulated trading without real money
- [ ] **Portfolio Optimization**: Advanced portfolio optimization algorithms

### **Research Directions**
- [ ] **Reinforcement Learning**: RL-based trading strategies
- [ ] **Sentiment Analysis**: Advanced NLP for market sentiment
- [ ] **Alternative Data**: Satellite data, social media, news sentiment
- [ ] **High-Frequency Trading**: Microsecond-level trading decisions
- [ ] **Cross-Market Analysis**: Global market correlation analysis

---

## 🤝 **Contributing**

### **Development Guidelines**
1. **Code Style**: Follow PEP 8 guidelines
2. **Documentation**: Add docstrings to all functions
3. **Testing**: Write tests for new features
4. **Type Hints**: Use type hints for better code clarity
5. **Error Handling**: Implement proper error handling

### **Contribution Process**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🆘 **Support and Troubleshooting**

### **Common Issues**

1. **API Rate Limits**
   ```bash
   # Reduce API calls
   export ONLINE_TOOLS=false
   # Use cached data instead
   ```

2. **Memory Issues**
   ```bash
   # Reduce batch size
   export BATCH_SIZE=1
   # Use smaller models
   export QUICK_THINK_LLM=gpt-3.5-turbo
   ```

3. **Network Issues**
   ```bash
   # Check API connectivity
   python scripts/test_connectivity.py
   # Use offline mode
   export OFFLINE_MODE=true
   ```

### **Getting Help**
- Create an issue on GitHub
- Check the documentation
- Review the example scripts
- Contact the maintainers

---

## �� **Contact Information**

- **Project Maintainer**: [Tejpal Kumawat]
- **Email**: [tejpaliitb782@gmail.com]
- **GitHub**: [tejpal123456789]
- **LinkedIn**: [https://www.linkedin.com/in/tejpal-kumawat-722a061a9/]

---

## 🙏 **Acknowledgments**

- **LangChain Team**: For the excellent framework
- **LangGraph Team**: For the workflow orchestration
- **OpenAI**: For the powerful language models
- **Hugging Face**: For the transformers library
- **Financial Data Providers**: Yahoo Finance, Alpha Vantage, Tavily

---

*This project represents a cutting-edge approach to AI-powered trading analysis, combining multiple specialized agents to create a comprehensive trading intelligence system. The goal is to democratize sophisticated trading analysis and make it accessible to both individual traders and institutional investors.*
