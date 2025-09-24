# Rural Credit Assessment Multi-Agent System

A comprehensive credit assessment system designed specifically for rural customers, built using Docker Compose for Agents architecture with Model Control Protocol (MCP).

## 🏗️ Architecture

This system follows the **Docker Compose for Agents** pattern with:

- **MCP Gateway**: Central tool management and coordination
- **Docker Model Runner**: Local model execution with OpenAI-compatible API
- **MCP Servers**: Containerized tools for specific assessment functions
- **LangGraph Agent**: Main orchestrator using workflow-based approach
- **Streamlit UI**: User-friendly interface for assessments

## 🔧 Components

### MCP Servers
- **Customer Assessment MCP**: Data collection and profiling
- **Risk Analyzer MCP**: AI-powered risk analysis  
- **Credit Scorer MCP**: Credit scoring and loan term calculation

### Agent System
- **LangGraph Orchestrator**: Workflow management
- **Multi-step Assessment**: Question → Profile → Risk → Score → Report

### Supporting Services
- **PostgreSQL**: Data persistence
- **Redis**: Caching and sessions
- **Streamlit UI**: Web interface
- **Nginx**: Load balancing (optional)

## 🚀 Quick Start

### Prerequisites
- Docker Desktop 4.40+ with Model Runner enabled
- Docker Compose 2.38.1+
- 8GB+ RAM recommended
- GPU support (optional, for local models)

### Installation

1. **Clone and setup:**
   ```bash
   git clone <repository>
   cd rural-credit-assessment
   cp mcp.env.example mcp.env
   # Edit mcp.env with your API keys if using remote models
   ```

2. **Start the system:**
   ```bash
   # For remote models (OpenAI/Anthropic)
   docker compose up --build

   # For local models with Docker Model Runner
   docker compose -f compose.yaml -f compose.dmr.yaml up --build
   ```

3. **Access the system:**
   - **Streamlit UI**: http://localhost:8501
   - **API Gateway**: http://localhost:8000
   - **MCP Gateway**: http://localhost:3001

### Using Docker Model Runner

To use local models:

1. **Enable Model Runner in Docker Desktop**
2. **Pull a model:**
   ```bash
   docker model run qwen2.5:7b
   ```
3. **Start with local model config:**
   ```bash
   docker compose -f compose.yaml -f compose.dmr.yaml up --build
   ```

## 📋 Usage

### Web Interface (Recommended)

1. Open http://localhost:8501
2. Navigate to "📋 New Assessment"
3. Fill customer information
4. Click "🚀 Start Assessment"
5. View results in "📊 Assessment Results"

### API Usage

**Start Assessment:**
```bash
curl -X POST http://localhost:8000/assess \
  -H "Content-Type: application/json" \
  -d '{
    "customer_data": {
      "name": "Test Customer",
      "age": 35,
      "location": "Test Village",
      "occupation": "farmer",
      "monthly_income": 15000
    },
    "assessment_type": "comprehensive"
  }'
```

**Check Status:**
```bash
curl http://localhost:8000/assess/{session_id}/status
```

## 🎯 Assessment Workflow

1. **Customer Data Collection**
   - Basic information validation
   - Context-aware question generation
   - Data quality assessment

2. **Profile Creation** 
   - Comprehensive customer profiling
   - Risk indicator identification
   - Financial literacy scoring

3. **Risk Analysis**
   - ML-powered risk assessment
   - Rural-specific risk factors
   - Scenario analysis capabilities

4. **Credit Scoring**
   - Rural-optimized scoring model
   - Grade assignment (Excellent → Very Poor)
   - Loan term recommendations

5. **Final Report**
   - Comprehensive assessment summary
   - Actionable recommendations
   - Loan offer document generation

## 🔧 Configuration

### Environment Variables

```bash
# API Keys (optional for local models)
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Database
DATABASE_URL=postgresql://credit_user:credit_pass@postgres:5432/credit_db

# MCP Configuration
MCP_GATEWAY_PORT=3001
```

### Model Selection

**Local Models (Docker Model Runner):**
- Qwen 2.5 (recommended)
- Llama 3.2
- Gemma 2

**Remote Models:**
- GPT-4, GPT-3.5 (OpenAI)
- Claude 3.5 Sonnet (Anthropic)

## 📊 Features

### Rural-Specific Assessment
- **Farmer-friendly criteria**: Seasonal income, crop types, land ownership
- **Collateral evaluation**: Agricultural assets, livestock, equipment
- **Financial inclusion**: Bank account, mobile phone, government ID checks
- **Flexible scoring**: Rural-optimized credit scoring models

### Multi-Agent Intelligence
- **Question Agent**: Dynamic, context-aware questioning
- **Risk Agent**: ML-powered risk analysis with explainability
- **Scoring Agent**: Comprehensive credit scoring with loan terms
- **Orchestrator**: Intelligent workflow management

### Production Ready
- **Containerized architecture**: Easy deployment and scaling
- **Health monitoring**: Built-in health checks and status monitoring
- **Data persistence**: PostgreSQL for reliable data storage
- **Security**: Isolated MCP servers, secure API endpoints

## 🐛 Troubleshooting

### Common Issues

**MCP Gateway not connecting:**
```bash
docker logs credit-mcp-gateway
# Check if all MCP servers are running
docker ps | grep mcp
```

**Model Runner issues:**
```bash
# Verify Docker Model Runner
docker model list
docker model run --help
```

**Database connection errors:**
```bash
# Check PostgreSQL logs
docker logs credit-postgres
# Verify database connectivity
docker exec -it credit-postgres psql -U credit_user -d credit_db
```

### System Status

Check system health at: http://localhost:8501 → "🔧 System Status"


