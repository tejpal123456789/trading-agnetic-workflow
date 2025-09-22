#!/bin/bash

# Trading Project Structure Creation Script
# This script creates the complete directory structure for the trading project

echo "🚀 Creating Trading Project Structure..."

# Create main directories
mkdir -p agents/{analyst_agent,bull_vs_bear,portfolio_manager_agent,research_agent,risk_agent,trader_agent}
mkdir -p tools
mkdir -p memory
mkdir -p data_cache
mkdir -p reflection
mkdir -p results/{analysis,reports,logs}
mkdir -p tests/{unit,integration,performance}
mkdir -p scripts
mkdir -p docs
mkdir -p config
mkdir -p static/{images,diagrams}
mkdir -p notebooks
mkdir -p logs

# Create agent subdirectories with files
echo "📁 Creating agent directories..."

# Analyst Agent
mkdir -p agents/analyst_agent/{base,market,sentiment,news,fundamentals}
touch agents/analyst_agent/__init__.py
touch agents/analyst_agent/base.py
touch agents/analyst_agent/market.py
touch agents/analyst_agent/sentiment.py
touch agents/analyst_agent/news.py
touch agents/analyst_agent/fundamentals.py
touch agents/analyst_agent/analyst.py

# Bull vs Bear
mkdir -p agents/bull_vs_bear/{bull,bear,judge}
touch agents/bull_vs_bear/__init__.py
touch agents/bull_vs_bear/bull.py
touch agents/bull_vs_bear/bear.py
touch agents/bull_vs_bear/judge.py
touch agents/bull_vs_bear/base_bb.py

# Portfolio Manager
mkdir -p agents/portfolio_manager_agent/{allocation,monitoring,rebalancing}
touch agents/portfolio_manager_agent/__init__.py
touch agents/portfolio_manager_agent/portfolio_agent.py
touch agents/portfolio_manager_agent/allocation.py
touch agents/portfolio_manager_agent/monitoring.py
touch agents/portfolio_manager_agent/rebalancing.py

# Research Agent
mkdir -p agents/research_agent/{data_collection,validation,coordination}
touch agents/research_agent/__init__.py
touch agents/research_agent/research_agent.py
touch agents/research_agent/data_collection.py
touch agents/research_agent/validation.py
touch agents/research_agent/coordination.py

# Risk Agent
mkdir -p agents/risk_agent/{risky,safe,neutral,judge}
touch agents/risk_agent/__init__.py
touch agents/risk_agent/risky_agent.py
touch agents/risk_agent/safe_risk_agent.py
touch agents/risk_agent/neutral_risk_agent.py
touch agents/risk_agent/overall_risk.py

# Trader Agent
mkdir -p agents/trader_agent/{execution,orders,monitoring}
touch agents/trader_agent/__init__.py
touch agents/trader_agent/trader.py
touch agents/trader_agent/execution.py
touch agents/trader_agent/orders.py
touch agents/trader_agent/monitoring.py

# Tools directory
echo "🛠️ Creating tools directory..."
touch tools/__init__.py
touch tools/finance_data.py
touch tools/indicator_data.py
touch tools/fundamental_analysis.py
touch tools/social_media_sentiment.py
touch tools/macro_news.py
touch tools/finance_news.py
touch tools/toolkit.py

# Memory directory
echo "🧠 Creating memory directory..."
touch memory/__init__.py
touch memory/longterm_memory.py
touch memory/shortterm_memory.py
touch memory/experience_buffer.py

# Test directories
echo "🧪 Creating test directories..."
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/unit/test_agents.py
touch tests/unit/test_tools.py
touch tests/unit/test_workflows.py
touch tests/integration/__init__.py
touch tests/integration/test_integration.py
touch tests/integration/test_real_data.py
touch tests/performance/__init__.py
touch tests/performance/test_benchmark.py
touch tests/performance/test_load.py

# Scripts directory
echo "📜 Creating scripts directory..."
touch scripts/benchmark.py
touch scripts/batch_analysis.py
touch scripts/test_connectivity.py
touch scripts/setup_environment.py
touch scripts/deploy.py

# Documentation
echo "📚 Creating documentation..."
touch docs/API.md
touch docs/ARCHITECTURE.md
touch docs/DEPLOYMENT.md
touch docs/TROUBLESHOOTING.md
touch docs/CONTRIBUTING.md

# Configuration files
echo "⚙️ Creating configuration files..."
touch config/development.yaml
touch config/production.yaml
touch config/testing.yaml
touch config/agents.yaml
touch config/tools.yaml

# Static assets
echo "🖼️ Creating static assets..."
mkdir -p static/images/{logos,screenshots,diagrams}
mkdir -p static/diagrams/{workflows,architecture,data_flow}
touch static/images/README.md
touch static/diagrams/README.md

# Notebooks
echo "📓 Creating notebooks..."
touch notebooks/01_getting_started.ipynb
touch notebooks/02_agent_analysis.ipynb
touch notebooks/03_workflow_demo.ipynb
touch notebooks/04_performance_analysis.ipynb
touch notebooks/05_custom_agents.ipynb

# Main project files
echo "📄 Creating main project files..."
touch main.py
touch run.py
touch run1.py
touch run2.py
touch stream.py
touch demo.py
touch setup.py
touch requirements.txt
touch requirements-dev.txt
touch pyproject.toml
touch Dockerfile
touch docker-compose.yml
touch docker-compose.dev.yml
touch .env.example
touch .gitignore
touch Makefile

# Create __init__.py files for Python packages
echo "🐍 Creating Python package files..."
find . -type d -name "*" -exec touch {}/__init__.py \;

# Remove __init__.py from non-Python directories
rm -f __init__.py
rm -f static/__init__.py
rm -f static/images/__init__.py
rm -f static/diagrams/__init__.py
rm -f docs/__init__.py
rm -f config/__init__.py
rm -f scripts/__init__.py
rm -f logs/__init__.py
rm -f data_cache/__init__.py
rm -f reflection/__init__.py
rm -f results/__init__.py
rm -f results/analysis/__init__.py
rm -f results/reports/__init__.py
rm -f results/logs/__init__.py
rm -f notebooks/__init__.py

echo "✅ Project structure created successfully!"
echo ""
echo "📁 Directory structure:"
tree -I '__pycache__|*.pyc|.git' -a

echo ""
echo "🎯 Next steps:"
echo "1. Set up virtual environment: python -m venv venv"
echo "2. Activate environment: source venv/bin/activate"
echo "3. Install dependencies: pip install -r requirements.txt"
echo "4. Copy environment file: cp .env.example .env"
echo "5. Configure your API keys in .env"
echo "6. Run the project: python main.py"
