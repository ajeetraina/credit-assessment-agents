#!/bin/bash

# Rural Credit Assessment Multi-Agent System Setup Script
# Following Docker Compose for Agents Architecture with MCP

set -e

PROJECT_NAME="rural-credit-assessment"
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Setting up Rural Credit Assessment with MCP Architecture${NC}"
echo -e "${BLUE}📦 Project: ${PROJECT_NAME}${NC}"
echo ""

# Create main project directory
if [ -d "$PROJECT_NAME" ]; then
    echo -e "${YELLOW}⚠️  Directory $PROJECT_NAME already exists. Removing...${NC}"
    rm -rf "$PROJECT_NAME"
fi

mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

echo -e "${GREEN}✅ Created main project directory${NC}"

# Create directory structure following compose-for-agents pattern
echo -e "${BLUE}📁 Creating directory structure...${NC}"

directories=(
    "mcp-servers/customer-assessment"
    "mcp-servers/risk-analyzer" 
    "mcp-servers/credit-scorer"
    "mcp-servers/data-collector"
    "agents/orchestrator"
    "ui/streamlit"
    "data"
    "config"
    "scripts"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
done

echo -e "${GREEN}✅ Directory structure created${NC}"

# Create main Docker Compose file with proper MCP architecture
echo -e "${BLUE}🐳 Creating Docker Compose configuration...${NC}"

cat > compose.yaml << 'EOF'
# Rural Credit Assessment Multi-Agent System
# Following Docker Compose for Agents Architecture

services:
  # MCP Gateway - Central tool management
  mcp-gateway:
    image: docker.io/docker/mcp-gateway:latest
    container_name: credit-mcp-gateway
    ports:
      - "3001:3001"
    volumes:
      - ./mcp.env:/app/.env
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      - MCP_GATEWAY_PORT=3001
      - MCP_GATEWAY_LOG_LEVEL=info
    depends_on:
      - customer-assessment-mcp
      - risk-analyzer-mcp
      - credit-scorer-mcp
    networks:
      - agent-network
    restart: unless-stopped

  # Customer Assessment MCP Server
  customer-assessment-mcp:
    build: ./mcp-servers/customer-assessment
    container_name: credit-customer-mcp
    environment:
      - MCP_SERVER_NAME=customer-assessment
      - DATABASE_URL=postgresql://credit_user:credit_pass@postgres:5432/credit_db
    depends_on:
      - postgres
    networks:
      - agent-network
    restart: unless-stopped

  # Risk Analysis MCP Server
  risk-analyzer-mcp:
    build: ./mcp-servers/risk-analyzer
    container_name: credit-risk-mcp
    environment:
      - MCP_SERVER_NAME=risk-analyzer
      - MODEL_RUNNER_URL=http://host.docker.internal:11434
    networks:
      - agent-network
    restart: unless-stopped

  # Credit Scoring MCP Server
  credit-scorer-mcp:
    build: ./mcp-servers/credit-scorer
    container_name: credit-scorer-mcp
    environment:
      - MCP_SERVER_NAME=credit-scorer
      - DATABASE_URL=postgresql://credit_user:credit_pass@postgres:5432/credit_db
    depends_on:
      - postgres
    networks:
      - agent-network
    restart: unless-stopped

  # Main Agent using LangGraph
  credit-agent:
    build: ./agents/orchestrator
    container_name: credit-agent
    ports:
      - "8000:8000"
    environment:
      - MCP_GATEWAY_URL=http://mcp-gateway:3001
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - AGENT_MODE=production
    depends_on:
      - mcp-gateway
    volumes:
      - ./config:/app/config
    networks:
      - agent-network
    restart: unless-stopped

  # Streamlit UI
  streamlit-ui:
    build: ./ui/streamlit
    container_name: credit-ui
    ports:
      - "8501:8501"
    environment:
      - AGENT_API_URL=http://credit-agent:8000
    depends_on:
      - credit-agent
    networks:
      - agent-network
    restart: unless-stopped

  # Database
  postgres:
    image: postgres:15-alpine
    container_name: credit-postgres
    environment:
      - POSTGRES_DB=credit_db
      - POSTGRES_USER=credit_user
      - POSTGRES_PASSWORD=credit_pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - agent-network
    restart: unless-stopped

  # Redis for caching and sessions
  redis:
    image: redis:7-alpine
    container_name: credit-redis
    ports:
      - "6379:6379"
    networks:
      - agent-network
    restart: unless-stopped

networks:
  agent-network:
    driver: bridge

volumes:
  postgres_data:
EOF

# Create Docker Model Runner configuration
cat > compose.dmr.yaml << 'EOF'
# Docker Model Runner override for local models
services:
  # Override agent to use Docker Model Runner for local inference
  credit-agent:
    environment:
      - MCP_GATEWAY_URL=http://mcp-gateway:3001
      - USE_LOCAL_MODELS=true
      - MODEL_RUNNER_URL=http://host.docker.internal:11434
      - AGENT_MODE=local
EOF

# Create MCP environment file
cat > mcp.env.example << 'EOF'
# MCP Gateway Configuration
MCP_GATEWAY_PORT=3001

# API Keys (Optional - for remote model access)
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here

# Database Configuration
DATABASE_URL=postgresql://credit_user:credit_pass@postgres:5432/credit_db

# MCP Server Configuration
CUSTOMER_ASSESSMENT_MCP_URL=http://customer-assessment-mcp:8001
RISK_ANALYZER_MCP_URL=http://risk-analyzer-mcp:8002
CREDIT_SCORER_MCP_URL=http://credit-scorer-mcp:8003
EOF

# Create Customer Assessment MCP Server
echo -e "${BLUE}🔧 Creating Customer Assessment MCP Server...${NC}"

cat > mcp-servers/customer-assessment/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["python", "server.py"]
EOF

cat > mcp-servers/customer-assessment/requirements.txt << 'EOF'
mcp==1.0.0
fastapi==0.104.1
uvicorn==0.24.0
sqlalchemy==2.0.23
asyncpg==0.29.0
pydantic==2.5.0
python-multipart==0.0.6
httpx==0.25.2
EOF

cat > mcp-servers/customer-assessment/server.py << 'EOF'
#!/usr/bin/env python3
"""
Customer Assessment MCP Server
Handles customer data collection and basic profiling
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
from pydantic import BaseModel, Field
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
server = FastMCP("Customer Assessment")

class CustomerProfile(BaseModel):
    """Customer profile data structure"""
    customer_id: str
    name: str
    age: int
    location: str
    occupation: str
    income_source: str
    monthly_income: float
    family_size: int
    land_ownership: bool
    collateral_assets: List[str] = Field(default_factory=list)
    previous_loans: bool = False
    repayment_history: str = "none"
    education_level: str = "primary"
    financial_literacy_score: float = 0.0

class AssessmentRequest(BaseModel):
    """Assessment request structure"""
    customer_data: Dict[str, Any]
    assessment_type: str = "basic"

# Sample question sets for different customer types
QUESTION_SETS = {
    "farmer": [
        {
            "id": "crop_type",
            "question": "What types of crops do you cultivate?",
            "type": "multi_select",
            "options": ["rice", "wheat", "cotton", "sugarcane", "vegetables", "fruits"],
            "weight": 0.8
        },
        {
            "id": "land_size",
            "question": "How much land do you own/cultivate (in acres)?",
            "type": "number",
            "weight": 0.9
        },
        {
            "id": "irrigation",
            "question": "Do you have access to irrigation facilities?",
            "type": "boolean",
            "weight": 0.7
        },
        {
            "id": "seasonal_income",
            "question": "How many crop seasons do you have per year?",
            "type": "number",
            "weight": 0.6
        }
    ],
    "small_business": [
        {
            "id": "business_type",
            "question": "What type of business do you operate?",
            "type": "text",
            "weight": 0.8
        },
        {
            "id": "business_age",
            "question": "How long has your business been operating (in years)?",
            "type": "number",
            "weight": 0.7
        },
        {
            "id": "monthly_revenue",
            "question": "What is your average monthly business revenue?",
            "type": "number",
            "weight": 0.9
        },
        {
            "id": "employee_count",
            "question": "How many people do you employ?",
            "type": "number",
            "weight": 0.6
        }
    ],
    "daily_wage": [
        {
            "id": "work_type",
            "question": "What type of daily wage work do you do?",
            "type": "text",
            "weight": 0.8
        },
        {
            "id": "work_days_per_month",
            "question": "How many days per month do you typically get work?",
            "type": "number",
            "weight": 0.9
        },
        {
            "id": "daily_wage_amount",
            "question": "What is your average daily wage?",
            "type": "number",
            "weight": 0.8
        },
        {
            "id": "work_seasonality",
            "question": "Is your work seasonal or year-round?",
            "type": "select",
            "options": ["seasonal", "year_round", "irregular"],
            "weight": 0.7
        }
    ]
}

@server.tool()
async def generate_assessment_questions(
    customer_data: Dict[str, Any],
    assessment_type: str = "comprehensive"
) -> str:
    """
    Generate contextual assessment questions based on customer profile
    
    Args:
        customer_data: Basic customer information
        assessment_type: Type of assessment (basic, comprehensive, specific)
    
    Returns:
        JSON string containing relevant questions
    """
    try:
        occupation = customer_data.get("occupation", "").lower()
        
        # Determine customer category
        if any(word in occupation for word in ["farm", "agriculture", "crop", "livestock"]):
            question_set = QUESTION_SETS["farmer"]
            category = "farmer"
        elif any(word in occupation for word in ["business", "shop", "trade", "vendor"]):
            question_set = QUESTION_SETS["small_business"]
            category = "small_business"
        elif any(word in occupation for word in ["labor", "worker", "daily", "construction"]):
            question_set = QUESTION_SETS["daily_wage"]
            category = "daily_wage"
        else:
            # Default mixed question set
            question_set = QUESTION_SETS["farmer"][:2] + QUESTION_SETS["small_business"][:2]
            category = "mixed"
        
        # Add common questions for all categories
        common_questions = [
            {
                "id": "bank_account",
                "question": "Do you have a bank account?",
                "type": "boolean",
                "weight": 0.6
            },
            {
                "id": "mobile_phone",
                "question": "Do you have a mobile phone?",
                "type": "boolean",
                "weight": 0.5
            },
            {
                "id": "govt_id",
                "question": "Do you have government-issued ID (Aadhaar, PAN, etc.)?",
                "type": "boolean",
                "weight": 0.8
            }
        ]
        
        all_questions = question_set + common_questions
        
        result = {
            "customer_category": category,
            "total_questions": len(all_questions),
            "questions": all_questions,
            "estimated_time_minutes": len(all_questions) * 2,
            "assessment_metadata": {
                "generated_at": datetime.now().isoformat(),
                "assessment_type": assessment_type,
                "customer_id": customer_data.get("customer_id", "unknown")
            }
        }
        
        logger.info(f"Generated {len(all_questions)} questions for {category} customer")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        return json.dumps({"error": str(e)})

@server.tool()
async def create_customer_profile(
    customer_data: Dict[str, Any],
    responses: Dict[str, Any]
) -> str:
    """
    Create comprehensive customer profile from collected data and responses
    
    Args:
        customer_data: Basic customer information
        responses: Responses to assessment questions
    
    Returns:
        JSON string containing complete customer profile
    """
    try:
        # Calculate financial literacy score
        literacy_indicators = [
            responses.get("bank_account", False),
            responses.get("mobile_phone", False),
            responses.get("govt_id", False),
            customer_data.get("monthly_income", 0) > 0
        ]
        financial_literacy_score = sum(literacy_indicators) / len(literacy_indicators) * 100
        
        # Determine income stability
        occupation = customer_data.get("occupation", "").lower()
        if "farm" in occupation:
            income_stability = "seasonal"
        elif responses.get("work_seasonality") == "year_round":
            income_stability = "stable"
        elif responses.get("business_age", 0) > 2:
            income_stability = "moderate"
        else:
            income_stability = "irregular"
        
        # Create comprehensive profile
        profile = CustomerProfile(
            customer_id=customer_data.get("customer_id", f"CUST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            name=customer_data.get("name", ""),
            age=customer_data.get("age", 0),
            location=customer_data.get("location", ""),
            occupation=customer_data.get("occupation", ""),
            income_source=customer_data.get("income_source", occupation),
            monthly_income=customer_data.get("monthly_income", 0),
            family_size=customer_data.get("family_size", 1),
            land_ownership=responses.get("land_size", 0) > 0,
            collateral_assets=_extract_collateral_assets(customer_data, responses),
            previous_loans=customer_data.get("previous_loans", False),
            repayment_history=customer_data.get("repayment_history", "none"),
            education_level=customer_data.get("education_level", "primary"),
            financial_literacy_score=financial_literacy_score
        )
        
        # Add derived insights
        result = {
            "profile": profile.model_dump(),
            "insights": {
                "income_stability": income_stability,
                "financial_literacy_level": _categorize_literacy_score(financial_literacy_score),
                "risk_indicators": _identify_risk_indicators(profile, responses),
                "positive_factors": _identify_positive_factors(profile, responses),
                "recommended_loan_amount": _calculate_recommended_loan(profile, responses)
            },
            "next_steps": [
                "Proceed to risk analysis",
                "Verify provided information",
                "Check credit history if available"
            ]
        }
        
        logger.info(f"Created profile for customer {profile.customer_id}")
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Error creating profile: {str(e)}")
        return json.dumps({"error": str(e)})

def _extract_collateral_assets(customer_data: Dict, responses: Dict) -> List[str]:
    """Extract potential collateral assets from responses"""
    assets = []
    
    if responses.get("land_size", 0) > 0:
        assets.append(f"agricultural_land_{responses['land_size']}_acres")
    
    if responses.get("business_age", 0) > 1:
        assets.append("business_equipment")
    
    if customer_data.get("land_ownership", False):
        assets.append("residential_land")
    
    return assets

def _categorize_literacy_score(score: float) -> str:
    """Categorize financial literacy score"""
    if score >= 80:
        return "high"
    elif score >= 60:
        return "moderate"
    elif score >= 40:
        return "basic"
    else:
        return "low"

def _identify_risk_indicators(profile: CustomerProfile, responses: Dict) -> List[str]:
    """Identify potential risk factors"""
    risks = []
    
    if profile.monthly_income < 5000:
        risks.append("low_income")
    
    if profile.financial_literacy_score < 50:
        risks.append("low_financial_literacy")
    
    if not responses.get("bank_account"):
        risks.append("no_bank_account")
    
    if profile.age < 21 or profile.age > 65:
        risks.append("age_risk")
    
    if responses.get("work_seasonality") == "seasonal":
        risks.append("seasonal_income")
    
    return risks

def _identify_positive_factors(profile: CustomerProfile, responses: Dict) -> List[str]:
    """Identify positive factors"""
    positives = []
    
    if profile.land_ownership:
        positives.append("land_ownership")
    
    if responses.get("business_age", 0) > 2:
        positives.append("established_business")
    
    if responses.get("bank_account"):
        positives.append("banked_customer")
    
    if profile.financial_literacy_score > 70:
        positives.append("financially_literate")
    
    if responses.get("irrigation"):
        positives.append("irrigation_access")
    
    return positives

def _calculate_recommended_loan(profile: CustomerProfile, responses: Dict) -> Dict:
    """Calculate recommended loan amount and terms"""
    base_amount = profile.monthly_income * 6  # 6 months of income
    
    # Adjust based on factors
    multiplier = 1.0
    
    if profile.land_ownership:
        multiplier *= 1.5
    
    if responses.get("business_age", 0) > 2:
        multiplier *= 1.3
    
    if profile.financial_literacy_score > 70:
        multiplier *= 1.2
    
    if not responses.get("bank_account"):
        multiplier *= 0.7
    
    recommended_amount = min(base_amount * multiplier, 100000)  # Cap at 1 lakh
    
    return {
        "amount": int(recommended_amount),
        "tenure_months": 12 if recommended_amount < 25000 else 24,
        "interest_rate_range": "12-18%",
        "confidence_level": min(profile.financial_literacy_score, 95)
    }

@server.tool()
async def validate_customer_data(customer_data: Dict[str, Any]) -> str:
    """
    Validate customer data for completeness and consistency
    
    Args:
        customer_data: Customer data to validate
    
    Returns:
        JSON string containing validation results
    """
    try:
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "missing_fields": [],
            "data_quality_score": 0
        }
        
        # Required fields check
        required_fields = ["name", "age", "location", "occupation", "monthly_income"]
        for field in required_fields:
            if not customer_data.get(field):
                validation_results["missing_fields"].append(field)
                validation_results["is_valid"] = False
        
        # Data consistency checks
        age = customer_data.get("age", 0)
        if age < 18 or age > 80:
            validation_results["warnings"].append("Age outside typical range (18-80)")
        
        income = customer_data.get("monthly_income", 0)
        if income < 0:
            validation_results["errors"].append("Monthly income cannot be negative")
            validation_results["is_valid"] = False
        elif income < 3000:
            validation_results["warnings"].append("Very low income reported")
        
        # Calculate data quality score
        total_possible_fields = 10
        filled_fields = sum(1 for field in ["name", "age", "location", "occupation", 
                                          "monthly_income", "family_size", "education_level",
                                          "previous_loans", "land_ownership", "income_source"] 
                           if customer_data.get(field) is not None)
        
        validation_results["data_quality_score"] = (filled_fields / total_possible_fields) * 100
        
        logger.info(f"Validated customer data - Quality score: {validation_results['data_quality_score']:.1f}%")
        return json.dumps(validation_results, indent=2)
        
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    # Run the MCP server
    logger.info("Starting Customer Assessment MCP Server...")
    server.run(transport="stdio")
EOF

# Create Risk Analyzer MCP Server
echo -e "${BLUE}🔍 Creating Risk Analyzer MCP Server...${NC}"

cat > mcp-servers/risk-analyzer/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8002

CMD ["python", "server.py"]
EOF

cat > mcp-servers/risk-analyzer/requirements.txt << 'EOF'
mcp==1.0.0
fastapi==0.104.1
uvicorn==0.24.0
numpy==1.24.3
scikit-learn==1.3.2
pandas==2.1.4
pydantic==2.5.0
httpx==0.25.2
joblib==1.3.2
EOF

cat > mcp-servers/risk-analyzer/server.py << 'EOF'
#!/usr/bin/env python3
"""
Risk Analysis MCP Server
Performs credit risk assessment using ML models and rule-based analysis
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from datetime import datetime
import pickle
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
server = FastMCP("Risk Analyzer")

class RiskAssessmentRequest(BaseModel):
    """Risk assessment request structure"""
    customer_profile: Dict[str, Any]
    assessment_type: str = "comprehensive"
    use_ml_model: bool = True

class RiskScore(BaseModel):
    """Risk score structure"""
    overall_score: float
    category: str
    confidence: float
    factors: Dict[str, float]
    recommendations: List[str]

# Risk factors and their weights
RISK_FACTORS = {
    "income_stability": {
        "seasonal": 0.7,
        "irregular": 0.8,
        "moderate": 0.4,
        "stable": 0.1
    },
    "age_risk": {
        "under_21": 0.6,
        "21_35": 0.2,
        "36_50": 0.1,
        "51_65": 0.3,
        "over_65": 0.7
    },
    "financial_literacy": {
        "low": 0.6,
        "basic": 0.4,
        "moderate": 0.2,
        "high": 0.1
    },
    "collateral_availability": {
        "none": 0.8,
        "minimal": 0.5,
        "moderate": 0.3,
        "substantial": 0.1
    },
    "income_level": {
        "very_low": 0.9,  # < 5000
        "low": 0.6,       # 5000-10000
        "moderate": 0.3,  # 10000-25000
        "high": 0.1       # > 25000
    }
}

# Simple ML model simulation (in real implementation, load trained model)
class SimpleRiskModel:
    def __init__(self):
        self.feature_weights = {
            'monthly_income': -0.3,
            'age': -0.1,
            'financial_literacy_score': -0.2,
            'family_size': 0.1,
            'collateral_count': -0.4,
            'previous_loans': 0.2,
            'income_stability_score': -0.3
        }
    
    def predict_risk_score(self, features: Dict) -> float:
        """Predict risk score based on features"""
        score = 0.5  # Base risk score
        
        for feature, weight in self.feature_weights.items():
            value = features.get(feature, 0)
            score += value * weight
        
        # Normalize to 0-1 range
        return max(0, min(1, score))

# Initialize the simple model
risk_model = SimpleRiskModel()

@server.tool()
async def analyze_credit_risk(
    customer_profile: Dict[str, Any],
    assessment_type: str = "comprehensive",
    use_ml_model: bool = True
) -> str:
    """
    Perform comprehensive credit risk analysis
    
    Args:
        customer_profile: Complete customer profile data
        assessment_type: Type of analysis (basic, comprehensive, detailed)
        use_ml_model: Whether to use ML model prediction
    
    Returns:
        JSON string containing risk analysis results
    """
    try:
        logger.info(f"Starting risk analysis for customer {customer_profile.get('customer_id', 'unknown')}")
        
        # Extract profile data
        profile = customer_profile.get('profile', customer_profile)
        insights = customer_profile.get('insights', {})
        
        # Perform rule-based risk assessment
        risk_factors = _calculate_risk_factors(profile, insights)
        
        # Calculate overall risk score
        if use_ml_model:
            ml_features = _extract_ml_features(profile, insights)
            ml_risk_score = risk_model.predict_risk_score(ml_features)
            
            # Combine rule-based and ML scores
            rule_based_score = _calculate_rule_based_score(risk_factors)
            overall_score = (ml_risk_score * 0.6) + (rule_based_score * 0.4)
        else:
            overall_score = _calculate_rule_based_score(risk_factors)
        
        # Categorize risk
        risk_category = _categorize_risk(overall_score)
        
        # Generate recommendations
        recommendations = _generate_risk_recommendations(risk_factors, overall_score, profile)
        
        # Calculate confidence level
        confidence = _calculate_confidence(profile, insights, risk_factors)
        
        result = {
            "risk_assessment": {
                "overall_score": round(overall_score, 3),
                "category": risk_category,
                "confidence": round(confidence, 2),
                "assessment_date": datetime.now().isoformat()
            },
            "risk_factors": risk_factors,
            "detailed_analysis": {
                "primary_risks": _identify_primary_risks(risk_factors),
                "mitigating_factors": _identify_mitigating_factors(risk_factors, insights),
                "loan_suitability": _assess_loan_suitability(overall_score, risk_factors)
            },
            "recommendations": {
                "for_lender": recommendations["lender"],
                "for_customer": recommendations["customer"],
                "monitoring_required": recommendations["monitoring"]
            },
            "model_info": {
                "method": "hybrid" if use_ml_model else "rule_based",
                "version": "1.0",
                "last_updated": "2024-01-01"
            }
        }
        
        logger.info(f"Risk analysis completed - Score: {overall_score:.3f}, Category: {risk_category}")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error in risk analysis: {str(e)}")
        return json.dumps({"error": str(e)})

def _calculate_risk_factors(profile: Dict, insights: Dict) -> Dict[str, float]:
    """Calculate individual risk factor scores"""
    factors = {}
    
    # Income stability risk
    income_stability = insights.get("income_stability", "irregular")
    factors["income_stability"] = RISK_FACTORS["income_stability"].get(income_stability, 0.5)
    
    # Age-based risk
    age = profile.get("age", 35)
    if age < 21:
        age_category = "under_21"
    elif age <= 35:
        age_category = "21_35"
    elif age <= 50:
        age_category = "36_50"
    elif age <= 65:
        age_category = "51_65"
    else:
        age_category = "over_65"
    
    factors["age_risk"] = RISK_FACTORS["age_risk"][age_category]
    
    # Financial literacy risk
    literacy_level = insights.get("financial_literacy_level", "low")
    factors["financial_literacy"] = RISK_FACTORS["financial_literacy"].get(literacy_level, 0.5)
    
    # Collateral risk
    collateral_assets = profile.get("collateral_assets", [])
    if len(collateral_assets) == 0:
        collateral_category = "none"
    elif len(collateral_assets) <= 2:
        collateral_category = "minimal"
    elif len(collateral_assets) <= 4:
        collateral_category = "moderate"
    else:
        collateral_category = "substantial"
    
    factors["collateral_availability"] = RISK_FACTORS["collateral_availability"][collateral_category]
    
    # Income level risk
    monthly_income = profile.get("monthly_income", 0)
    if monthly_income < 5000:
        income_category = "very_low"
    elif monthly_income < 10000:
        income_category = "low"
    elif monthly_income < 25000:
        income_category = "moderate"
    else:
        income_category = "high"
    
    factors["income_level"] = RISK_FACTORS["income_level"][income_category]
    
    # Additional factors
    factors["repayment_history"] = 0.8 if profile.get("repayment_history") == "poor" else 0.2
    factors["location_risk"] = 0.3 if "rural" in profile.get("location", "").lower() else 0.2
    factors["occupation_risk"] = _calculate_occupation_risk(profile.get("occupation", ""))
    
    return factors

def _calculate_occupation_risk(occupation: str) -> float:
    """Calculate risk based on occupation"""
    occupation = occupation.lower()
    
    if any(word in occupation for word in ["government", "teacher", "clerk"]):
        return 0.1  # Very low risk
    elif any(word in occupation for word in ["business", "shop", "trade"]):
        return 0.3  # Moderate risk
    elif any(word in occupation for word in ["farm", "agriculture"]):
        return 0.4  # Seasonal risk
    elif any(word in occupation for word in ["labor", "daily", "construction"]):
        return 0.6  # High risk
    else:
        return 0.4  # Default moderate risk

def _extract_ml_features(profile: Dict, insights: Dict) -> Dict:
    """Extract features for ML model"""
    return {
        'monthly_income': profile.get('monthly_income', 0) / 10000,  # Normalize
        'age': profile.get('age', 35) / 100,
        'financial_literacy_score': profile.get('financial_literacy_score', 0) / 100,
        'family_size': min(profile.get('family_size', 1), 10) / 10,
        'collateral_count': len(profile.get('collateral_assets', [])) / 5,
        'previous_loans': 1 if profile.get('previous_loans') else 0,
        'income_stability_score': _get_stability_score(insights.get('income_stability', 'irregular'))
    }

def _get_stability_score(stability: str) -> float:
    """Convert stability category to numeric score"""
    return {
        'stable': 1.0,
        'moderate': 0.7,
        'seasonal': 0.4,
        'irregular': 0.1
    }.get(stability, 0.5)

def _calculate_rule_based_score(risk_factors: Dict[str, float]) -> float:
    """Calculate overall risk score from individual factors"""
    weights = {
        "income_level": 0.25,
        "income_stability": 0.20,
        "collateral_availability": 0.15,
        "financial_literacy": 0.15,
        "repayment_history": 0.10,
        "age_risk": 0.08,
        "occupation_risk": 0.07
    }
    
    weighted_score = sum(
        risk_factors.get(factor, 0.5) * weight 
        for factor, weight in weights.items()
    )
    
    return weighted_score

def _categorize_risk(score: float) -> str:
    """Categorize risk level based on score"""
    if score <= 0.3:
        return "low"
    elif score <= 0.5:
        return "moderate"
    elif score <= 0.7:
        return "high"
    else:
        return "very_high"

def _identify_primary_risks(risk_factors: Dict[str, float]) -> List[str]:
    """Identify the main risk factors"""
    sorted_risks = sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)
    return [factor for factor, score in sorted_risks[:3] if score > 0.4]

def _identify_mitigating_factors(risk_factors: Dict[str, float], insights: Dict) -> List[str]:
    """Identify factors that reduce risk"""
    mitigating = []
    
    if risk_factors.get("collateral_availability", 1) < 0.4:
        mitigating.append("strong_collateral")
    
    if risk_factors.get("financial_literacy", 1) < 0.3:
        mitigating.append("high_financial_literacy")
    
    if risk_factors.get("income_level", 1) < 0.4:
        mitigating.append("adequate_income")
    
    if "established_business" in insights.get("positive_factors", []):
        mitigating.append("business_track_record")
    
    return mitigating

def _assess_loan_suitability(overall_score: float, risk_factors: Dict) -> Dict:
    """Assess loan suitability based on risk analysis"""
    if overall_score <= 0.3:
        suitability = "highly_suitable"
        max_loan_ratio = 0.8
    elif overall_score <= 0.5:
        suitability = "suitable_with_conditions"
        max_loan_ratio = 0.6
    elif overall_score <= 0.7:
        suitability = "marginal"
        max_loan_ratio = 0.4
    else:
        suitability = "not_recommended"
        max_loan_ratio = 0.2
    
    return {
        "suitability": suitability,
        "max_loan_ratio": max_loan_ratio,
        "conditions": _generate_loan_conditions(overall_score, risk_factors)
    }

def _generate_loan_conditions(score: float, risk_factors: Dict) -> List[str]:
    """Generate specific conditions for loan approval"""
    conditions = []
    
    if risk_factors.get("collateral_availability", 0) > 0.5:
        conditions.append("Require additional collateral")
    
    if risk_factors.get("financial_literacy", 0) > 0.4:
        conditions.append("Mandatory financial literacy training")
    
    if risk_factors.get("income_stability", 0) > 0.5:
        conditions.append("Shorter repayment tenure")
    
    if score > 0.5:
        conditions.append("Regular monitoring required")
        conditions.append("Co-signer or guarantor required")
    
    return conditions

def _generate_risk_recommendations(risk_factors: Dict, overall_score: float, profile: Dict) -> Dict:
    """Generate recommendations for different stakeholders"""
    return {
        "lender": _generate_lender_recommendations(risk_factors, overall_score),
        "customer": _generate_customer_recommendations(risk_factors, profile),
        "monitoring": _generate_monitoring_recommendations(risk_factors, overall_score)
    }

def _generate_lender_recommendations(risk_factors: Dict, score: float) -> List[str]:
    """Generate recommendations for lender"""
    recommendations = []
    
    if score > 0.6:
        recommendations.append("Consider rejecting or require significant collateral")
    elif score > 0.4:
        recommendations.append("Approve with enhanced monitoring")
        recommendations.append("Reduce loan amount by 20-30%")
    
    if risk_factors.get("income_stability", 0) > 0.5:
        recommendations.append("Align repayment with income cycles")
    
    if risk_factors.get("financial_literacy", 0) > 0.4:
        recommendations.append("Provide financial education before disbursement")
    
    return recommendations

def _generate_customer_recommendations(risk_factors: Dict, profile: Dict) -> List[str]:
    """Generate recommendations for customer"""
    recommendations = []
    
    if risk_factors.get("financial_literacy", 0) > 0.4:
        recommendations.append("Improve financial literacy through training")
    
    if risk_factors.get("collateral_availability", 0) > 0.5:
        recommendations.append("Consider building collateral assets")
    
    if risk_factors.get("income_stability", 0) > 0.5:
        recommendations.append("Diversify income sources")
    
    if profile.get("monthly_income", 0) < 10000:
        recommendations.append("Focus on income generation activities")
    
    return recommendations

def _generate_monitoring_recommendations(risk_factors: Dict, score: float) -> List[str]:
    """Generate monitoring recommendations"""
    recommendations = []
    
    if score > 0.5:
        recommendations.append("Monthly check-ins required")
        recommendations.append("Income verification every quarter")
    
    if risk_factors.get("income_stability", 0) > 0.5:
        recommendations.append("Monitor seasonal income patterns")
    
    if risk_factors.get("occupation_risk", 0) > 0.5:
        recommendations.append("Track occupation-specific risk factors")
    
    return recommendations

def _calculate_confidence(profile: Dict, insights: Dict, risk_factors: Dict) -> float:
    """Calculate confidence level in the assessment"""
    confidence_factors = []
    
    # Data completeness
    required_fields = ['name', 'age', 'occupation', 'monthly_income', 'location']
    completeness = sum(1 for field in required_fields if profile.get(field)) / len(required_fields)
    confidence_factors.append(completeness * 30)
    
    # Data quality
    quality_score = insights.get('data_quality_score', 50) / 100
    confidence_factors.append(quality_score * 25)
    
    # Factor consistency
    consistency = 1.0 - (np.std(list(risk_factors.values())) / np.mean(list(risk_factors.values())))
    confidence_factors.append(max(0, consistency) * 25)
    
    # Model reliability
    model_confidence = 0.8  # Fixed for this simple model
    confidence_factors.append(model_confidence * 20)
    
    return sum(confidence_factors)

@server.tool()
async def compare_risk_scenarios(
    base_profile: Dict[str, Any],
    scenario_changes: List[Dict[str, Any]]
) -> str:
    """
    Compare risk scores under different scenarios
    
    Args:
        base_profile: Base customer profile
        scenario_changes: List of changes to apply for each scenario
    
    Returns:
        JSON string containing scenario comparison results
    """
    try:
        results = {
            "base_scenario": await analyze_credit_risk(base_profile),
            "alternative_scenarios": []
        }
        
        for i, changes in enumerate(scenario_changes):
            # Create modified profile
            modified_profile = base_profile.copy()
            if 'profile' in modified_profile:
                modified_profile['profile'].update(changes)
            else:
                modified_profile.update(changes)
            
            scenario_result = await analyze_credit_risk(modified_profile)
            
            results["alternative_scenarios"].append({
                "scenario_id": i + 1,
                "changes": changes,
                "result": json.loads(scenario_result)
            })
        
        # Add comparison summary
        base_score = json.loads(results["base_scenario"])["risk_assessment"]["overall_score"]
        scenario_scores = [
            json.loads(scenario["result"])["risk_assessment"]["overall_score"]
            for scenario in results["alternative_scenarios"]
        ]
        
        results["comparison_summary"] = {
            "base_score": base_score,
            "best_scenario_improvement": max(base_score - score for score in scenario_scores) if scenario_scores else 0,
            "worst_scenario_deterioration": max(score - base_score for score in scenario_scores) if scenario_scores else 0
        }
        
        return json.dumps(results, indent=2)
        
    except Exception as e:
        logger.error(f"Error in scenario comparison: {str(e)}")
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    logger.info("Starting Risk Analyzer MCP Server...")
    server.run(transport="stdio")
EOF

# Create Credit Scorer MCP Server
echo -e "${BLUE}🏦 Creating Credit Scorer MCP Server...${NC}"

cat > mcp-servers/credit-scorer/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8003

CMD ["python", "server.py"]
EOF

cat > mcp-servers/credit-scorer/requirements.txt << 'EOF'
mcp==1.0.0
fastapi==0.104.1
uvicorn==0.24.0
sqlalchemy==2.0.23
asyncpg==0.29.0
pydantic==2.5.0
httpx==0.25.2
numpy==1.24.3
EOF

cat > mcp-servers/credit-scorer/server.py << 'EOF'
#!/usr/bin/env python3
"""
Credit Scoring MCP Server
Generates final credit scores and loan recommendations
"""

import asyncio
import json
import logging
import math
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from datetime import datetime, timedelta
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
server = FastMCP("Credit Scorer")

class CreditScore(BaseModel):
    """Credit score structure"""
    score: int
    grade: str
    probability_default: float
    loan_amount: int
    interest_rate: float
    tenure_months: int
    monthly_emi: float
    confidence: float

class LoanTerms(BaseModel):
    """Loan terms structure"""
    principal: int
    interest_rate: float
    tenure_months: int
    monthly_emi: float
    total_amount: int
    processing_fee: float

# Credit scoring parameters
SCORE_RANGES = {
    "excellent": (750, 850),
    "good": (650, 749), 
    "fair": (550, 649),
    "poor": (450, 549),
    "very_poor": (300, 449)
}

BASE_INTEREST_RATES = {
    "excellent": 10.5,
    "good": 12.0,
    "fair": 14.5,
    "poor": 16.5,
    "very_poor": 18.5
}

@server.tool()
async def calculate_credit_score(
    customer_profile: Dict[str, Any],
    risk_analysis: Dict[str, Any],
    scoring_model: str = "rural_specific"
) -> str:
    """
    Calculate final credit score based on profile and risk analysis
    
    Args:
        customer_profile: Complete customer profile
        risk_analysis: Risk analysis results
        scoring_model: Scoring model to use (rural_specific, standard, conservative)
    
    Returns:
        JSON string containing credit score and details
    """
    try:
        logger.info(f"Calculating credit score using {scoring_model} model")
        
        # Extract data
        profile = customer_profile.get('profile', customer_profile)
        risk_assessment = risk_analysis.get('risk_assessment', {})
        risk_factors = risk_analysis.get('risk_factors', {})
        
        # Calculate base score
        base_score = _calculate_base_score(profile, scoring_model)
        
        # Apply risk adjustments
        risk_adjusted_score = _apply_risk_adjustments(base_score, risk_assessment, risk_factors)
        
        # Apply rural-specific adjustments
        final_score = _apply_rural_adjustments(risk_adjusted_score, profile, scoring_model)
        
        # Ensure score is within valid range
        final_score = max(300, min(850, int(final_score)))
        
        # Determine grade and default probability
        grade = _determine_grade(final_score)
        default_probability = _calculate_default_probability(final_score, risk_factors)
        
        # Calculate recommended loan terms
        loan_recommendation = _calculate_loan_terms(final_score, profile, risk_analysis)
        
        # Calculate confidence
        confidence = _calculate_scoring_confidence(profile, risk_analysis, final_score)
        
        result = {
            "credit_score": {
                "score": final_score,
                "grade": grade,
                "probability_default": round(default_probability, 4),
                "score_factors": _explain_score_factors(profile, risk_factors),
                "calculated_at": datetime.now().isoformat()
            },
            "loan_recommendation": loan_recommendation,
            "confidence_metrics": {
                "overall_confidence": round(confidence, 2),
                "data_quality": _assess_data_quality(profile),
                "model_reliability": 0.85  # Fixed for this implementation
            },
            "next_review_date": (datetime.now() + timedelta(days=180)).isoformat(),
            "model_info": {
                "model_type": scoring_model,
                "version": "1.0",
                "calibration_date": "2024-01-01"
            }
        }
        
        logger.info(f"Credit score calculated: {final_score} ({grade})")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error calculating credit score: {str(e)}")
        return json.dumps({"error": str(e)})

def _calculate_base_score(profile: Dict, model_type: str) -> float:
    """Calculate base credit score from profile data"""
    
    # Start with neutral score
    base = 550
    
    # Income factor (0-100 points)
    monthly_income = profile.get('monthly_income', 0)
    if monthly_income >= 25000:
        income_points = 100
    elif monthly_income >= 15000:
        income_points = 80
    elif monthly_income >= 10000:
        income_points = 60
    elif monthly_income >= 5000:
        income_points = 40
    else:
        income_points = 20
    
    # Age factor (0-50 points) - experience curve
    age = profile.get('age', 30)
    if 25 <= age <= 45:
        age_points = 50
    elif 21 <= age <= 55:
        age_points = 40
    elif 18 <= age <= 65:
        age_points = 30
    else:
        age_points = 10
    
    # Financial literacy (0-40 points)
    literacy_score = profile.get('financial_literacy_score', 0)
    literacy_points = min(40, literacy_score * 0.4)
    
    # Collateral factor (0-60 points)
    collateral_assets = profile.get('collateral_assets', [])
    collateral_points = min(60, len(collateral_assets) * 15)
    
    # Land ownership bonus for rural model
    if model_type == "rural_specific" and profile.get('land_ownership'):
        land_bonus = 30
    else:
        land_bonus = 0
    
    # Employment stability (0-30 points)
    occupation = profile.get('occupation', '').lower()
    if any(word in occupation for word in ['government', 'teacher', 'bank']):
        employment_points = 30
    elif any(word in occupation for word in ['business', 'shop']):
        employment_points = 20
    elif 'farm' in occupation:
        employment_points = 15 if model_type == "rural_specific" else 10
    else:
        employment_points = 10
    
    total_score = (base + income_points + age_points + literacy_points + 
                  collateral_points + land_bonus + employment_points)
    
    return total_score

def _apply_risk_adjustments(base_score: float, risk_assessment: Dict, risk_factors: Dict) -> float:
    """Apply risk-based adjustments to base score"""
    
    risk_score = risk_assessment.get('overall_score', 0.5)
    
    # Convert risk score (0-1) to point adjustment (-150 to +50)
    if risk_score <= 0.2:  # Very low risk
        risk_adjustment = 50
    elif risk_score <= 0.35:  # Low risk
        risk_adjustment = 25
    elif risk_score <= 0.5:  # Moderate risk
        risk_adjustment = 0
    elif risk_score <= 0.7:  # High risk
        risk_adjustment = -50
    else:  # Very high risk
        risk_adjustment = -100
    
    # Additional specific risk factor adjustments
    additional_adjustments = 0
    
    # Income stability adjustment
    if risk_factors.get('income_stability', 0) > 0.6:
        additional_adjustments -= 30
    elif risk_factors.get('income_stability', 0) < 0.3:
        additional_adjustments += 20
    
    # Repayment history adjustment
    if risk_factors.get('repayment_history', 0) > 0.7:
        additional_adjustments -= 40
    
    # Financial literacy adjustment
    if risk_factors.get('financial_literacy', 0) < 0.2:
        additional_adjustments += 30
    
    return base_score + risk_adjustment + additional_adjustments

def _apply_rural_adjustments(score: float, profile: Dict, model_type: str) -> float:
    """Apply rural-specific scoring adjustments"""
    
    if model_type != "rural_specific":
        return score
    
    adjustments = 0
    
    # Location-based adjustment
    location = profile.get('location', '').lower()
    if any(word in location for word in ['village', 'rural', 'district']):
        adjustments += 10  # Bonus for rural location in rural model
    
    # Occupation adjustments for rural context
    occupation = profile.get('occupation', '').lower()
    if 'farm' in occupation:
        # Farming gets bonus in rural model
        adjustments += 15
    elif any(word in occupation for word in ['livestock', 'dairy', 'poultry']):
        adjustments += 10
    
    # Family size consideration (larger families may have more income sources)
    family_size = profile.get('family_size', 1)
    if 3 <= family_size <= 6:  # Optimal family size for rural areas
        adjustments += 5
    elif family_size > 8:  # Too large may be burden
        adjustments -= 10
    
    # Education level in rural context
    education = profile.get('education_level', 'primary').lower()
    if education in ['graduate', 'postgraduate']:
        adjustments += 20  # Higher education is more valuable in rural areas
    elif education == 'secondary':
        adjustments += 10
    
    return score + adjustments

def _determine_grade(score: int) -> str:
    """Determine credit grade from score"""
    for grade, (min_score, max_score) in SCORE_RANGES.items():
        if min_score <= score <= max_score:
            return grade
    return "very_poor"

def _calculate_default_probability(score: int, risk_factors: Dict) -> float:
    """Calculate probability of default based on score and risk factors"""
    
    # Base probability from score (logistic function)
    base_prob = 1 / (1 + math.exp((score - 500) / 100))
    
    # Adjust for specific risk factors
    risk_multiplier = 1.0
    
    # High-impact risk factors
    if risk_factors.get('repayment_history', 0) > 0.7:
        risk_multiplier *= 1.5
    
    if risk_factors.get('income_level', 0) > 0.8:
        risk_multiplier *= 1.3
    
    if risk_factors.get('income_stability', 0) > 0.7:
        risk_multiplier *= 1.2
    
    # Cap the probability
    final_prob = min(0.95, base_prob * risk_multiplier)
    
    return final_prob

def _calculate_loan_terms(score: int, profile: Dict, risk_analysis: Dict) -> Dict:
    """Calculate recommended loan terms"""
    
    grade = _determine_grade(score)
    monthly_income = profile.get('monthly_income', 10000)
    
    # Base loan amount (multiple of monthly income)
    income_multiplier = {
        "excellent": 8,
        "good": 6,
        "fair": 4,
        "poor": 3,
        "very_poor": 2
    }.get(grade, 2)
    
    base_loan_amount = min(monthly_income * income_multiplier, 200000)  # Cap at 2 lakhs
    
    # Adjust based on collateral
    collateral_assets = profile.get('collateral_assets', [])
    if len(collateral_assets) >= 2:
        base_loan_amount = int(base_loan_amount * 1.2)
    elif len(collateral_assets) >= 1:
        base_loan_amount = int(base_loan_amount * 1.1)
    
    # Interest rate
    base_rate = BASE_INTEREST_RATES[grade]
    
    # Tenure based on loan amount and risk
    if base_loan_amount <= 25000:
        tenure = 12
    elif base_loan_amount <= 50000:
        tenure = 18
    elif base_loan_amount <= 100000:
        tenure = 24
    else:
        tenure = 36
    
    # Calculate EMI
    monthly_rate = base_rate / (12 * 100)
    emi = base_loan_amount * monthly_rate * ((1 + monthly_rate) ** tenure) / (((1 + monthly_rate) ** tenure) - 1)
    
    # Ensure EMI doesn't exceed 40% of income
    max_emi = monthly_income * 0.4
    if emi > max_emi:
        # Reduce loan amount to fit EMI constraint
        base_loan_amount = int(max_emi * (((1 + monthly_rate) ** tenure) - 1) / (monthly_rate * ((1 + monthly_rate) ** tenure)))
        emi = max_emi
    
    total_amount = int(emi * tenure)
    processing_fee = max(500, base_loan_amount * 0.01)  # 1% or min 500
    
    return {
        "recommended_amount": int(base_loan_amount),
        "interest_rate": base_rate,
        "tenure_months": tenure,
        "monthly_emi": int(emi),
        "total_repayment": total_amount,
        "processing_fee": int(processing_fee),
        "emi_to_income_ratio": round((emi / monthly_income) * 100, 1),
        "loan_conditions": _generate_loan_conditions(score, profile, risk_analysis)
    }

def _generate_loan_conditions(score: int, profile: Dict, risk_analysis: Dict) -> List[str]:
    """Generate specific loan conditions"""
    conditions = []
    
    grade = _determine_grade(score)
    risk_factors = risk_analysis.get('risk_factors', {})
    
    if grade in ['poor', 'very_poor']:
        conditions.append("Guarantor required")
        conditions.append("Monthly income verification")
    
    if grade == 'very_poor':
        conditions.append("Collateral mandatory")
        conditions.append("Financial literacy training required")
    
    if risk_factors.get('income_stability', 0) > 0.5:
        conditions.append("Seasonal repayment plan available")
    
    if risk_factors.get('financial_literacy', 0) > 0.4:
        conditions.append("Loan counseling mandatory")
    
    if profile.get('age', 35) > 60:
        conditions.append("Nominee declaration required")
    
    return conditions

def _explain_score_factors(profile: Dict, risk_factors: Dict) -> Dict:
    """Explain factors contributing to the credit score"""
    
    factors = {
        "positive_factors": [],
        "negative_factors": [],
        "neutral_factors": []
    }
    
    # Income analysis
    monthly_income = profile.get('monthly_income', 0)
    if monthly_income >= 15000:
        factors["positive_factors"].append(f"Good income level (₹{monthly_income:,})")
    elif monthly_income >= 8000:
        factors["neutral_factors"].append(f"Moderate income level (₹{monthly_income:,})")
    else:
        factors["negative_factors"].append(f"Low income level (₹{monthly_income:,})")
    
    # Age analysis
    age = profile.get('age', 30)
    if 25 <= age <= 50:
        factors["positive_factors"].append(f"Optimal age range ({age} years)")
    else:
        factors["neutral_factors"].append(f"Age factor ({age} years)")
    
    # Collateral analysis
    collateral_count = len(profile.get('collateral_assets', []))
    if collateral_count >= 2:
        factors["positive_factors"].append(f"Strong collateral ({collateral_count} assets)")
    elif collateral_count == 1:
        factors["neutral_factors"].append("Some collateral available")
    else:
        factors["negative_factors"].append("No collateral")
    
    # Financial literacy
    literacy_score = profile.get('financial_literacy_score', 0)
    if literacy_score >= 70:
        factors["positive_factors"].append(f"High financial literacy ({literacy_score:.0f}%)")
    elif literacy_score >= 40:
        factors["neutral_factors"].append(f"Moderate financial literacy ({literacy_score:.0f}%)")
    else:
        factors["negative_factors"].append(f"Low financial literacy ({literacy_score:.0f}%)")
    
    # Risk factors
    if risk_factors.get('income_stability', 0) < 0.3:
        factors["positive_factors"].append("Stable income source")
    elif risk_factors.get('income_stability', 0) > 0.6:
        factors["negative_factors"].append("Unstable income source")
    
    return factors

def _calculate_scoring_confidence(profile: Dict, risk_analysis: Dict, score: int) -> float:
    """Calculate confidence in the credit score"""
    
    confidence_components = []
    
    # Data completeness (0-30 points)
    required_fields = ['name', 'age', 'occupation', 'monthly_income', 'location']
    completeness = sum(1 for field in required_fields if profile.get(field)) / len(required_fields)
    confidence_components.append(completeness * 30)
    
    # Risk analysis confidence (0-25 points)
    risk_confidence = risk_analysis.get('risk_assessment', {}).get('confidence', 50) / 100
    confidence_components.append(risk_confidence * 25)
    
    # Score consistency (0-25 points)
    expected_score_range = _get_expected_score_range(profile)
    if expected_score_range[0] <= score <= expected_score_range[1]:
        consistency = 1.0
    else:
        deviation = min(abs(score - expected_score_range[0]), abs(score - expected_score_range[1]))
        consistency = max(0, 1 - (deviation / 100))
    
    confidence_components.append(consistency * 25)
    
    # Model calibration (0-20 points)
    confidence_components.append(0.85 * 20)  # Fixed model confidence
    
    return sum(confidence_components)

def _get_expected_score_range(profile: Dict) -> tuple:
    """Get expected score range based on profile"""
    
    monthly_income = profile.get('monthly_income', 0)
    
    if monthly_income >= 20000:
        return (650, 750)
    elif monthly_income >= 10000:
        return (550, 650)
    elif monthly_income >= 5000:
        return (450, 550)
    else:
        return (350, 450)

def _assess_data_quality(profile: Dict) -> float:
    """Assess overall data quality"""
    
    quality_score = 0
    total_checks = 0
    
    # Completeness check
    required_fields = ['name', 'age', 'occupation', 'monthly_income', 'location']
    completeness = sum(1 for field in required_fields if profile.get(field)) / len(required_fields)
    quality_score += completeness * 40
    
    # Consistency checks
    age = profile.get('age', 0)
    if 18 <= age <= 80:
        quality_score += 20
    
    income = profile.get('monthly_income', 0)
    if 0 < income <= 100000:  # Reasonable income range
        quality_score += 20
    
    # Additional data
    optional_fields = ['education_level', 'family_size', 'collateral_assets']
    optional_completeness = sum(1 for field in optional_fields if profile.get(field)) / len(optional_fields)
    quality_score += optional_completeness * 20
    
    return quality_score

@server.tool()
async def generate_loan_offer_document(
    credit_score_result: Dict[str, Any],
    customer_profile: Dict[str, Any]
) -> str:
    """
    Generate formal loan offer document
    
    Args:
        credit_score_result: Results from credit scoring
        customer_profile: Customer profile data
    
    Returns:
        JSON string containing loan offer document
    """
    try:
        profile = customer_profile.get('profile', customer_profile)
        credit_info = credit_score_result.get('credit_score', {})
        loan_info = credit_score_result.get('loan_recommendation', {})
        
        offer_document = {
            "offer_header": {
                "bank_name": "Rural Credit Bank",
                "offer_id": f"LOAN_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "offer_date": datetime.now().isoformat(),
                "valid_until": (datetime.now() + timedelta(days=30)).isoformat(),
                "customer_id": profile.get('customer_id', 'N/A')
            },
            "customer_details": {
                "name": profile.get('name', ''),
                "age": profile.get('age', ''),
                "location": profile.get('location', ''),
                "occupation": profile.get('occupation', ''),
                "monthly_income": profile.get('monthly_income', 0)
            },
            "credit_assessment": {
                "credit_score": credit_info.get('score', 0),
                "grade": credit_info.get('grade', ''),
                "assessment_date": credit_info.get('calculated_at', '')
            },
            "loan_offer": {
                "principal_amount": loan_info.get('recommended_amount', 0),
                "interest_rate": loan_info.get('interest_rate', 0),
                "tenure_months": loan_info.get('tenure_months', 0),
                "monthly_emi": loan_info.get('monthly_emi', 0),
                "processing_fee": loan_info.get('processing_fee', 0),
                "total_repayment": loan_info.get('total_repayment', 0)
            },
            "terms_and_conditions": loan_info.get('loan_conditions', []),
            "required_documents": [
                "Identity proof (Aadhaar Card)",
                "Address proof",
                "Income proof",
                "Bank statements (3 months)",
                "Passport size photographs"
            ],
            "next_steps": [
                "Visit nearest branch within 30 days",
                "Submit required documents",
                "Complete loan application form",
                "Await final approval"
            ],
            "contact_information": {
                "branch_address": "Rural Credit Bank, Main Branch",
                "phone": "+91-XXX-XXX-XXXX",
                "email": "loans@ruralcredit.com",
                "loan_officer": "Assigned upon application"
            }
        }
        
        logger.info(f"Generated loan offer document for {profile.get('name', 'N/A')}")
        return json.dumps(offer_document, indent=2)
        
    except Exception as e:
        logger.error(f"Error generating loan offer: {str(e)}")
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    logger.info("Starting Credit Scorer MCP Server...")
    server.run(transport="stdio")
EOF

# Create main orchestrator agent using LangGraph
echo -e "${BLUE}🤖 Creating Main Agent with LangGraph...${NC}"

cat > agents/orchestrator/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

cat > agents/orchestrator/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn==0.24.0
langgraph==0.2.3
langchain-mcp-adapters==0.1.0
langchain-core==0.3.0
langchain-openai==0.2.0
langchain-anthropic==0.2.0
pydantic==2.5.0
httpx==0.25.2
python-multipart==0.0.6
asyncio-redis==0.16.0
EOF

cat > agents/orchestrator/app.py << 'EOF'
#!/usr/bin/env python3
"""
Rural Credit Assessment Agent using LangGraph and MCP
Main orchestrator that coordinates the assessment workflow
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, TypedDict
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Rural Credit Assessment Agent",
    description="Multi-Agent Credit Assessment System using MCP",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global MCP client
mcp_client: Optional[MultiServerMCPClient] = None

class AssessmentRequest(BaseModel):
    customer_data: Dict[str, Any]
    assessment_type: str = "comprehensive"

class AssessmentResponse(BaseModel):
    session_id: str
    status: str
    message: str

class AssessmentState(TypedDict):
    """State for the assessment workflow"""
    customer_data: Dict[str, Any]
    questions: List[Dict[str, Any]]
    responses: Dict[str, Any]
    customer_profile: Dict[str, Any]
    risk_analysis: Dict[str, Any]
    credit_score: Dict[str, Any]
    final_report: Dict[str, Any]
    current_step: str
    messages: List[Any]
    session_id: str

async def initialize_mcp_client():
    """Initialize the MCP client with server connections"""
    global mcp_client
    
    try:
        # MCP server configuration
        server_config = {
            "customer-assessment": {
                "command": "docker",
                "args": [
                    "exec", "credit-customer-mcp",
                    "python", "/app/server.py"
                ],
                "transport": "stdio"
            },
            "risk-analyzer": {
                "command": "docker", 
                "args": [
                    "exec", "credit-risk-mcp",
                    "python", "/app/server.py"
                ],
                "transport": "stdio"
            },
            "credit-scorer": {
                "command": "docker",
                "args": [
                    "exec", "credit-scorer-mcp", 
                    "python", "/app/server.py"
                ],
                "transport": "stdio"
            }
        }
        
        mcp_client = MultiServerMCPClient(server_config)
        
        # Get available tools
        tools = await mcp_client.get_tools()
        logger.info(f"Initialized MCP client with {len(tools)} tools")
        
        return mcp_client
        
    except Exception as e:
        logger.error(f"Error initializing MCP client: {str(e)}")
        return None

async def get_mcp_tools():
    """Get MCP tools from the client"""
    global mcp_client
    if not mcp_client:
        mcp_client = await initialize_mcp_client()
    
    if mcp_client:
        return await mcp_client.get_tools()
    return []

# Agent workflow nodes
async def collect_customer_info(state: AssessmentState) -> AssessmentState:
    """Step 1: Collect and validate customer information"""
    logger.info(f"Starting customer info collection for session {state['session_id']}")
    
    try:
        tools = await get_mcp_tools()
        
        # Find the customer assessment tools
        generate_questions_tool = next(
            (tool for tool in tools if "generate_assessment_questions" in tool.name), 
            None
        )
        validate_data_tool = next(
            (tool for tool in tools if "validate_customer_data" in tool.name),
            None
        )
        
        if not generate_questions_tool or not validate_data_tool:
            raise Exception("Customer assessment tools not available")
        
        # Validate customer data first
        validation_result = await validate_data_tool.ainvoke({
            "customer_data": state["customer_data"]
        })
        
        validation_data = json.loads(validation_result)
        
        # Generate assessment questions
        questions_result = await generate_questions_tool.ainvoke({
            "customer_data": state["customer_data"],
            "assessment_type": "comprehensive"
        })
        
        questions_data = json.loads(questions_result)
        
        # Update state
        state["questions"] = questions_data.get("questions", [])
        state["current_step"] = "questions_generated"
        state["messages"].append(
            AIMessage(content=f"Generated {len(state['questions'])} assessment questions")
        )
        
        logger.info(f"Generated {len(state['questions'])} questions for customer assessment")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in collect_customer_info: {str(e)}")
        state["messages"].append(AIMessage(content=f"Error in customer info collection: {str(e)}"))
        return state

async def create_customer_profile(state: AssessmentState) -> AssessmentState:
    """Step 2: Create comprehensive customer profile"""
    logger.info(f"Creating customer profile for session {state['session_id']}")
    
    try:
        tools = await get_mcp_tools()
        
        create_profile_tool = next(
            (tool for tool in tools if "create_customer_profile" in tool.name),
            None
        )
        
        if not create_profile_tool:
            raise Exception("Create customer profile tool not available")
        
        # Use mock responses for demo (in real app, these would come from user input)
        mock_responses = _generate_mock_responses(state["customer_data"], state["questions"])
        
        # Create customer profile
        profile_result = await create_profile_tool.ainvoke({
            "customer_data": state["customer_data"],
            "responses": mock_responses
        })
        
        profile_data = json.loads(profile_result)
        
        # Update state
        state["customer_profile"] = profile_data
        state["responses"] = mock_responses
        state["current_step"] = "profile_created"
        state["messages"].append(
            AIMessage(content="Customer profile created successfully")
        )
        
        logger.info("Customer profile created successfully")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in create_customer_profile: {str(e)}")
        state["messages"].append(AIMessage(content=f"Error creating profile: {str(e)}"))
        return state

async def analyze_risk(state: AssessmentState) -> AssessmentState:
    """Step 3: Perform risk analysis"""
    logger.info(f"Analyzing risk for session {state['session_id']}")
    
    try:
        tools = await get_mcp_tools()
        
        analyze_risk_tool = next(
            (tool for tool in tools if "analyze_credit_risk" in tool.name),
            None
        )
        
        if not analyze_risk_tool:
            raise Exception("Risk analysis tool not available")
        
        # Perform risk analysis
        risk_result = await analyze_risk_tool.ainvoke({
            "customer_profile": state["customer_profile"],
            "assessment_type": "comprehensive",
            "use_ml_model": True
        })
        
        risk_data = json.loads(risk_result)
        
        # Update state
        state["risk_analysis"] = risk_data
        state["current_step"] = "risk_analyzed"
        
        risk_score = risk_data.get("risk_assessment", {}).get("overall_score", 0)
        risk_category = risk_data.get("risk_assessment", {}).get("category", "unknown")
        
        state["messages"].append(
            AIMessage(content=f"Risk analysis completed - Score: {risk_score:.3f}, Category: {risk_category}")
        )
        
        logger.info(f"Risk analysis completed - Score: {risk_score:.3f}")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in analyze_risk: {str(e)}")
        state["messages"].append(AIMessage(content=f"Error in risk analysis: {str(e)}"))
        return state

async def calculate_credit_score(state: AssessmentState) -> AssessmentState:
    """Step 4: Calculate final credit score and loan terms"""
    logger.info(f"Calculating credit score for session {state['session_id']}")
    
    try:
        tools = await get_mcp_tools()
        
        credit_score_tool = next(
            (tool for tool in tools if "calculate_credit_score" in tool.name),
            None
        )
        
        if not credit_score_tool:
            raise Exception("Credit scoring tool not available")
        
        # Calculate credit score
        score_result = await credit_score_tool.ainvoke({
            "customer_profile": state["customer_profile"],
            "risk_analysis": state["risk_analysis"],
            "scoring_model": "rural_specific"
        })
        
        score_data = json.loads(score_result)
        
        # Update state
        state["credit_score"] = score_data
        state["current_step"] = "score_calculated"
        
        credit_score = score_data.get("credit_score", {}).get("score", 0)
        grade = score_data.get("credit_score", {}).get("grade", "unknown")
        
        state["messages"].append(
            AIMessage(content=f"Credit score calculated - Score: {credit_score}, Grade: {grade}")
        )
        
        logger.info(f"Credit score calculated - Score: {credit_score}")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in calculate_credit_score: {str(e)}")
        state["messages"].append(AIMessage(content=f"Error calculating credit score: {str(e)}"))
        return state

async def generate_final_report(state: AssessmentState) -> AssessmentState:
    """Step 5: Generate final assessment report"""
    logger.info(f"Generating final report for session {state['session_id']}")
    
    try:
        # Compile final assessment report
        final_report = {
            "assessment_summary": {
                "session_id": state["session_id"],
                "customer_name": state["customer_data"].get("name", "N/A"),
                "assessment_date": datetime.now().isoformat(),
                "assessment_type": "comprehensive"
            },
            "customer_profile": state["customer_profile"],
            "risk_analysis": state["risk_analysis"],
            "credit_assessment": state["credit_score"],
            "workflow_status": "completed",
            "recommendations": _generate_final_recommendations(state),
            "next_steps": [
                "Review loan offer with customer",
                "Collect required documents", 
                "Schedule branch visit",
                "Process formal application"
            ]
        }
        
        # Update state
        state["final_report"] = final_report
        state["current_step"] = "completed"
        state["messages"].append(
            AIMessage(content="Final assessment report generated successfully")
        )
        
        logger.info(f"Assessment completed for session {state['session_id']}")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in generate_final_report: {str(e)}")
        state["messages"].append(AIMessage(content=f"Error generating final report: {str(e)}"))
        return state

def _generate_mock_responses(customer_data: Dict, questions: List[Dict]) -> Dict:
    """Generate mock responses for demo purposes"""
    responses = {}
    
    occupation = customer_data.get("occupation", "").lower()
    
    # Generate realistic responses based on occupation
    for question in questions:
        q_id = question.get("id")
        q_type = question.get("type", "text")
        
        if q_id == "crop_type" and "farm" in occupation:
            responses[q_id] = ["rice", "wheat"]
        elif q_id == "land_size":
            responses[q_id] = 5.0 if "farm" in occupation else 0.0
        elif q_id == "irrigation":
            responses[q_id] = True
        elif q_id == "business_age" and "business" in occupation:
            responses[q_id] = 3
        elif q_id == "monthly_revenue":
            responses[q_id] = customer_data.get("monthly_income", 10000) * 1.2
        elif q_id == "work_days_per_month":
            responses[q_id] = 22
        elif q_id == "daily_wage_amount":
            responses[q_id] = customer_data.get("monthly_income", 8000) / 22
        elif q_type == "boolean":
            responses[q_id] = True
        elif q_type == "number":
            responses[q_id] = 1
        else:
            responses[q_id] = "default_response"
    
    return responses

def _generate_final_recommendations(state: AssessmentState) -> List[str]:
    """Generate final recommendations based on assessment results"""
    recommendations = []
    
    # Get key metrics
    credit_score = state.get("credit_score", {}).get("credit_score", {}).get("score", 0)
    risk_category = state.get("risk_analysis", {}).get("risk_assessment", {}).get("category", "high")
    loan_amount = state.get("credit_score", {}).get("loan_recommendation", {}).get("recommended_amount", 0)
    
    if credit_score >= 650:
        recommendations.append("Approve loan with standard terms")
    elif credit_score >= 550:
        recommendations.append("Approve with enhanced monitoring")
    else:
        recommendations.append("Consider rejection or require guarantor")
    
    if risk_category == "high":
        recommendations.append("Implement monthly check-ins")
        recommendations.append("Consider collateral requirement")
    
    if loan_amount > 0:
        recommendations.append(f"Recommended loan amount: ₹{loan_amount:,}")
    
    return recommendations

# Create LangGraph workflow
def create_assessment_workflow():
    """Create the assessment workflow graph"""
    
    workflow = StateGraph(AssessmentState)
    
    # Add nodes
    workflow.add_node("collect_info", collect_customer_info)
    workflow.add_node("create_profile", create_customer_profile) 
    workflow.add_node("analyze_risk", analyze_risk)
    workflow.add_node("calculate_score", calculate_credit_score)
    workflow.add_node("generate_report", generate_final_report)
    
    # Add edges
    workflow.add_edge(START, "collect_info")
    workflow.add_edge("collect_info", "create_profile")
    workflow.add_edge("create_profile", "analyze_risk")
    workflow.add_edge("analyze_risk", "calculate_score")
    workflow.add_edge("calculate_score", "generate_report")
    workflow.add_edge("generate_report", END)
    
    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

# Global workflow app
workflow_app = None

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global workflow_app
    
    logger.info("Starting Rural Credit Assessment Agent...")
    
    # Initialize MCP client
    await initialize_mcp_client()
    
    # Create workflow
    workflow_app = create_assessment_workflow()
    
    logger.info("Agent initialized successfully")

@app.get("/")
async def root():
    return {
        "message": "Rural Credit Assessment Agent",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mcp_client_active": mcp_client is not None
    }

@app.post("/assess", response_model=AssessmentResponse)
async def start_assessment(request: AssessmentRequest):
    """Start a new credit assessment"""
    try:
        session_id = f"ASSESS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize state
        initial_state = AssessmentState(
            customer_data=request.customer_data,
            questions=[],
            responses={},
            customer_profile={},
            risk_analysis={},
            credit_score={},
            final_report={},
            current_step="initiated",
            messages=[HumanMessage(content="Starting credit assessment")],
            session_id=session_id
        )
        
        # Run workflow
        config = {"configurable": {"thread_id": session_id}}
        
        if workflow_app:
            # Run the assessment workflow
            final_state = await workflow_app.ainvoke(initial_state, config=config)
            
            logger.info(f"Assessment completed for session {session_id}")
            
            return AssessmentResponse(
                session_id=session_id,
                status="completed",
                message="Credit assessment completed successfully"
            )
        else:
            raise Exception("Workflow not initialized")
            
    except Exception as e:
        logger.error(f"Error in assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assess/{session_id}/status")
async def get_assessment_status(session_id: str):
    """Get assessment status and results"""
    try:
        if workflow_app:
            config = {"configurable": {"thread_id": session_id}}
            
            # Get state from checkpointer
            state = await workflow_app.aget_state(config)
            
            if state and state.values:
                return {
                    "session_id": session_id,
                    "current_step": state.values.get("current_step", "unknown"),
                    "status": "completed" if state.values.get("current_step") == "completed" else "in_progress",
                    "final_report": state.values.get("final_report"),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise HTTPException(status_code=404, detail="Session not found")
        else:
            raise Exception("Workflow not initialized")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools")
async def list_available_tools():
    """List available MCP tools"""
    try:
        tools = await get_mcp_tools()
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": getattr(tool, 'description', 'No description'),
                    "server": "MCP"
                }
                for tool in tools
            ],
            "total_count": len(tools)
        }
    except Exception as e:
        logger.error(f"Error listing tools: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# Create Streamlit UI
echo -e "${BLUE}🖥️  Creating Streamlit UI...${NC}"

cat > ui/streamlit/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF

cat > ui/streamlit/requirements.txt << 'EOF'
streamlit==1.28.1
requests==2.31.0
plotly==5.17.0
pandas==2.1.4
numpy==1.24.3
httpx==0.25.2
python-dateutil==2.8.2
EOF

cat > ui/streamlit/app.py << 'EOF'
#!/usr/bin/env python3
"""
Streamlit UI for Rural Credit Assessment System
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Rural Credit Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://credit-agent:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f1c0c7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🏦 Rural Credit Assessment System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/1f77b4/white?text=Rural+Bank", 
                caption="Rural Credit Bank")
        
        page = st.selectbox(
            "Navigate to:",
            ["🏠 Home", "📋 New Assessment", "📊 Assessment Results", "🔧 System Status"]
        )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📋 New Assessment":
        show_assessment_form()
    elif page == "📊 Assessment Results":
        show_results_page()
    elif page == "🔧 System Status":
        show_system_status()

def show_home_page():
    """Show home page with overview"""
    
    st.markdown("## Welcome to the Rural Credit Assessment System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📝 Total Assessments",
            value="1,234",
            delta="45 this month"
        )
    
    with col2:
        st.metric(
            label="✅ Approved Loans", 
            value="856",
            delta="32 this week"
        )
    
    with col3:
        st.metric(
            label="💰 Amount Disbursed",
            value="₹2.45 Cr",
            delta="₹15.6 L this month"
        )
    
    st.markdown("---")
    
    # System features
    st.markdown("### 🚀 System Features")
    
    feature_cols = st.columns(2)
    
    with feature_cols[0]:
        st.markdown("""
        **Multi-Agent Assessment:**
        - Customer profiling and data collection
        - AI-powered risk analysis
        - Dynamic credit scoring
        - Comprehensive reporting
        """)
    
    with feature_cols[1]:
        st.markdown("""
        **Rural-Specific Focus:**
        - Farmer-friendly assessment criteria
        - Seasonal income considerations
        - Collateral evaluation for rural assets
        - Local language support (planned)
        """)
    
    # Recent activity
    st.markdown("### 📈 Recent Activity")
    
    # Sample data for demonstration
    recent_data = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=30, freq='D'),
        'Applications': [15, 22, 18, 25, 30, 28, 35, 20, 25, 28, 32, 35, 40, 38, 42, 
                        45, 48, 52, 50, 55, 58, 62, 60, 65, 68, 70, 72, 75, 78, 80],
        'Approvals': [12, 18, 15, 20, 25, 23, 28, 16, 20, 23, 26, 28, 32, 30, 34,
                     36, 38, 42, 40, 44, 46, 50, 48, 52, 54, 56, 58, 60, 62, 64]
    })
    
    fig = px.line(recent_data, x='Date', y=['Applications', 'Approvals'], 
                  title="Daily Applications vs Approvals")
    st.plotly_chart(fig, use_container_width=True)

def show_assessment_form():
    """Show new assessment form"""
    
    st.markdown("## 📋 New Credit Assessment")
    
    with st.form("assessment_form"):
        st.markdown("### Customer Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name *", placeholder="Enter customer's full name")
            age = st.number_input("Age *", min_value=18, max_value=80, value=35)
            location = st.text_input("Location *", placeholder="Village, District, State")
            mobile = st.text_input("Mobile Number", placeholder="Enter 10-digit mobile number")
        
        with col2:
            occupation = st.selectbox(
                "Occupation *",
                ["Farmer", "Small Business Owner", "Daily Wage Worker", 
                 "Livestock Owner", "Shopkeeper", "Other"]
            )
            monthly_income = st.number_input(
                "Monthly Income (₹) *", 
                min_value=1000, 
                max_value=100000, 
                value=15000,
                step=1000
            )
            family_size = st.number_input("Family Size", min_value=1, max_value=15, value=4)
            education = st.selectbox(
                "Education Level",
                ["Primary", "Secondary", "Higher Secondary", "Graduate", "Post Graduate"]
            )
        
        st.markdown("### Additional Information")
        
        col3, col4 = st.columns(2)
        
        with col3:
            land_ownership = st.checkbox("Owns Land")
            bank_account = st.checkbox("Has Bank Account")
            previous_loans = st.checkbox("Had Previous Loans")
        
        with col4:
            govt_id = st.checkbox("Has Government ID (Aadhaar/PAN)")
            mobile_phone = st.checkbox("Owns Mobile Phone")
            
            if previous_loans:
                repayment_history = st.selectbox(
                    "Previous Repayment History",
                    ["Good", "Average", "Poor", "No History"]
                )
            else:
                repayment_history = "No History"
        
        # Assessment options
        st.markdown("### Assessment Options")
        assessment_type = st.selectbox(
            "Assessment Type",
            ["Comprehensive", "Quick Assessment", "Re-assessment"]
        )
        
        submit_button = st.form_submit_button("🚀 Start Assessment", use_container_width=True)
        
        if submit_button:
            # Validate required fields
            if not all([name, age, location, occupation, monthly_income]):
                st.error("Please fill all required fields marked with *")
                return
            
            # Prepare customer data
            customer_data = {
                "name": name,
                "age": age,
                "location": location,
                "occupation": occupation.lower(),
                "monthly_income": monthly_income,
                "family_size": family_size,
                "education_level": education.lower(),
                "land_ownership": land_ownership,
                "previous_loans": previous_loans,
                "repayment_history": repayment_history.lower() if previous_loans else "none",
                "mobile": mobile,
                "has_bank_account": bank_account,
                "has_govt_id": govt_id,
                "has_mobile_phone": mobile_phone,
                "customer_id": f"CUST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
            
            # Start assessment
            start_assessment(customer_data, assessment_type.lower())

def start_assessment(customer_data, assessment_type):
    """Start the credit assessment process"""
    
    try:
        # Show progress
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        with progress_placeholder.container():
            st.markdown("### 🔄 Assessment in Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # Call assessment API
        status_text.text("Connecting to assessment system...")
        progress_bar.progress(10)
        
        response = requests.post(
            f"{API_BASE_URL}/assess",
            json={
                "customer_data": customer_data,
                "assessment_type": assessment_type
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            session_id = result.get("session_id")
            
            # Poll for results
            status_text.text("Assessment started. Collecting customer information...")
            progress_bar.progress(25)
            time.sleep(2)
            
            status_text.text("Creating customer profile...")
            progress_bar.progress(50)
            time.sleep(2)
            
            status_text.text("Analyzing credit risk...")
            progress_bar.progress(75)
            time.sleep(2)
            
            status_text.text("Calculating credit score...")
            progress_bar.progress(90)
            time.sleep(2)
            
            # Get final results
            results_response = requests.get(f"{API_BASE_URL}/assess/{session_id}/status")
            
            if results_response.status_code == 200:
                results = results_response.json()
                progress_bar.progress(100)
                status_text.text("Assessment completed successfully!")
                
                # Store results in session state
                st.session_state.latest_assessment = results
                
                # Clear progress indicators
                time.sleep(1)
                progress_placeholder.empty()
                
                # Show results
                show_assessment_results(results)
                
            else:
                st.error("Failed to retrieve assessment results")
                
        else:
            st.error(f"Assessment failed: {response.text}")
            
    except requests.exceptions.Timeout:
        st.error("Assessment timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to assessment service. Please check system status.")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")

def show_assessment_results(results):
    """Display assessment results"""
    
    st.markdown("## 📊 Assessment Results")
    
    final_report = results.get("final_report", {})
    
    if not final_report:
        st.warning("Assessment results not available")
        return
    
    # Summary metrics
    st.markdown("### 🎯 Assessment Summary")
    
    credit_assessment = final_report.get("credit_assessment", {})
    credit_score_info = credit_assessment.get("credit_score", {})
    loan_rec = credit_assessment.get("loan_recommendation", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        credit_score = credit_score_info.get("score", 0)
        grade = credit_score_info.get("grade", "Unknown")
        st.metric(
            label="Credit Score",
            value=f"{credit_score}",
            delta=f"Grade: {grade.upper()}"
        )
    
    with col2:
        recommended_amount = loan_rec.get("recommended_amount", 0)
        st.metric(
            label="Loan Amount",
            value=f"₹{recommended_amount:,}"
        )
    
    with col3:
        interest_rate = loan_rec.get("interest_rate", 0)
        st.metric(
            label="Interest Rate",
            value=f"{interest_rate}%"
        )
    
    with col4:
        monthly_emi = loan_rec.get("monthly_emi", 0)
        st.metric(
            label="Monthly EMI",
            value=f"₹{monthly_emi:,}"
        )
    
    # Risk analysis
    st.markdown("### ⚠️ Risk Analysis")
    
    risk_analysis = final_report.get("risk_analysis", {})
    risk_assessment = risk_analysis.get("risk_assessment", {})
    
    col5, col6 = st.columns(2)
    
    with col5:
        risk_score = risk_assessment.get("overall_score", 0)
        risk_category = risk_assessment.get("category", "Unknown")
        
        # Risk gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 50], 'color': "yellow"},
                    {'range': [50, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col6:
        st.markdown(f"**Risk Category:** {risk_category.upper()}")
        st.markdown(f"**Confidence:** {risk_assessment.get('confidence', 0):.1f}%")
        
        # Primary risks
        detailed_analysis = risk_analysis.get("detailed_analysis", {})
        primary_risks = detailed_analysis.get("primary_risks", [])
        
        if primary_risks:
            st.markdown("**Primary Risk Factors:**")
            for risk in primary_risks:
                st.markdown(f"- {risk.replace('_', ' ').title()}")
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    recommendations = final_report.get("recommendations", [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
    
    # Loan terms details
    if loan_rec:
        st.markdown("### 📄 Loan Terms")
        
        col7, col8 = st.columns(2)
        
        with col7:
            st.markdown(f"**Principal Amount:** ₹{loan_rec.get('recommended_amount', 0):,}")
            st.markdown(f"**Interest Rate:** {loan_rec.get('interest_rate', 0)}% per annum")
            st.markdown(f"**Tenure:** {loan_rec.get('tenure_months', 0)} months")
            st.markdown(f"**Processing Fee:** ₹{loan_rec.get('processing_fee', 0):,}")
        
        with col8:
            st.markdown(f"**Monthly EMI:** ₹{loan_rec.get('monthly_emi', 0):,}")
            st.markdown(f"**Total Repayment:** ₹{loan_rec.get('total_repayment', 0):,}")
            st.markdown(f"**EMI-to-Income Ratio:** {loan_rec.get('emi_to_income_ratio', 0)}%")
        
        # Loan conditions
        conditions = loan_rec.get("loan_conditions", [])
        if conditions:
            st.markdown("**Loan Conditions:**")
            for condition in conditions:
                st.markdown(f"- {condition}")
    
    # Next steps
    next_steps = final_report.get("next_steps", [])
    if next_steps:
        st.markdown("### ⏭️ Next Steps")
        for i, step in enumerate(next_steps, 1):
            st.markdown(f"{i}. {step}")
    
    # Download report button
    if st.button("📥 Download Assessment Report", use_container_width=True):
        # Convert results to JSON for download
        report_json = json.dumps(final_report, indent=2)
        st.download_button(
            label="Download JSON Report",
            data=report_json,
            file_name=f"credit_assessment_{results.get('session_id', 'report')}.json",
            mime="application/json"
        )

def show_results_page():
    """Show assessment results page"""
    
    st.markdown("## 📊 Assessment Results")
    
    if 'latest_assessment' in st.session_state:
        show_assessment_results(st.session_state.latest_assessment)
    else:
        st.info("No recent assessment results available. Please complete an assessment first.")

def show_system_status():
    """Show system status and health"""
    
    st.markdown("## 🔧 System Status")
    
    try:
        # Check API health
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        
        if health_response.status_code == 200:
            health_data = health_response.json()
            
            st.markdown('<div class="success-box">✅ System is healthy and operational</div>', 
                       unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**System Information:**")
                st.markdown(f"- Status: {health_data.get('status', 'Unknown')}")
                st.markdown(f"- Timestamp: {health_data.get('timestamp', 'N/A')}")
                st.markdown(f"- MCP Client: {'Active' if health_data.get('mcp_client_active') else 'Inactive'}")
            
            with col2:
                # Get available tools
                try:
                    tools_response = requests.get(f"{API_BASE_URL}/tools", timeout=5)
                    if tools_response.status_code == 200:
                        tools_data = tools_response.json()
                        st.markdown("**Available Tools:**")
                        st.markdown(f"- Total Tools: {tools_data.get('total_count', 0)}")
                        
                        tools = tools_data.get('tools', [])
                        for tool in tools[:5]:  # Show first 5 tools
                            st.markdown(f"- {tool.get('name', 'Unknown')}")
                        
                        if len(tools) > 5:
                            st.markdown(f"- ... and {len(tools) - 5} more")
                            
                except:
                    st.markdown("**Tools:** Unable to fetch tool information")
        
        else:
            st.markdown('<div class="error-box">❌ System is not responding properly</div>', 
                       unsafe_allow_html=True)
            
    except requests.exceptions.ConnectionError:
        st.markdown('<div class="error-box">❌ Cannot connect to the assessment service</div>', 
                   unsafe_allow_html=True)
        st.markdown("**Troubleshooting:**")
        st.markdown("1. Check if the credit-agent container is running")
        st.markdown("2. Verify network connectivity")
        st.markdown("3. Check container logs for errors")
        
    except Exception as e:
        st.markdown(f'<div class="error-box">❌ System error: {str(e)}</div>', 
                   unsafe_allow_html=True)
    
    # System components status
    st.markdown("### 🔧 Component Status")
    
    components = [
        {"name": "API Gateway", "status": "Running", "port": "8000"},
        {"name": "MCP Gateway", "status": "Running", "port": "3001"},
        {"name": "Customer Assessment MCP", "status": "Running", "port": "8001"},
        {"name": "Risk Analyzer MCP", "status": "Running", "port": "8002"},
        {"name": "Credit Scorer MCP", "status": "Running", "port": "8003"},
        {"name": "PostgreSQL Database", "status": "Running", "port": "5432"},
        {"name": "Redis Cache", "status": "Running", "port": "6379"}
    ]
    
    for component in components:
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"**{component['name']}**")
        
        with col2:
            if component['status'] == 'Running':
                st.markdown("🟢 Running")
            else:
                st.markdown("🔴 Stopped")
        
        with col3:
            st.markdown(f"Port: {component['port']}")

if __name__ == "__main__":
    main()
EOF

# Create database initialization script
echo -e "${BLUE}🗄️  Creating database initialization script...${NC}"

cat > scripts/init.sql << 'EOF'
-- Rural Credit Assessment Database Schema

-- Customer profiles table
CREATE TABLE IF NOT EXISTS customer_profiles (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    age INTEGER,
    location VARCHAR(200),
    occupation VARCHAR(100),
    monthly_income DECIMAL(10, 2),
    family_size INTEGER,
    education_level VARCHAR(50),
    land_ownership BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Risk assessments table
CREATE TABLE IF NOT EXISTS risk_assessments (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customer_profiles(customer_id),
    session_id VARCHAR(100) NOT NULL,
    overall_score DECIMAL(5, 4),
    risk_category VARCHAR(20),
    confidence DECIMAL(5, 2),
    risk_factors JSONB,
    assessment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Credit scores table
CREATE TABLE IF NOT EXISTS credit_scores (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customer_profiles(customer_id),
    session_id VARCHAR(100) NOT NULL,
    credit_score INTEGER,
    grade VARCHAR(20),
    probability_default DECIMAL(6, 4),
    model_version VARCHAR(20),
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Loan applications table
CREATE TABLE IF NOT EXISTS loan_applications (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customer_profiles(customer_id),
    session_id VARCHAR(100) NOT NULL,
    application_id VARCHAR(50) UNIQUE,
    loan_amount DECIMAL(12, 2),
    interest_rate DECIMAL(5, 2),
    tenure_months INTEGER,
    monthly_emi DECIMAL(10, 2),
    status VARCHAR(20) DEFAULT 'pending',
    approved_amount DECIMAL(12, 2),
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Assessment sessions table
CREATE TABLE IF NOT EXISTS assessment_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) UNIQUE NOT NULL,
    customer_id VARCHAR(50),
    assessment_type VARCHAR(50),
    current_step VARCHAR(50),
    status VARCHAR(20) DEFAULT 'initiated',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    final_report JSONB
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_customer_profiles_customer_id ON customer_profiles(customer_id);
CREATE INDEX IF NOT EXISTS idx_risk_assessments_customer_id ON risk_assessments(customer_id);
CREATE INDEX IF NOT EXISTS idx_risk_assessments_session_id ON risk_assessments(session_id);
CREATE INDEX IF NOT EXISTS idx_credit_scores_customer_id ON credit_scores(customer_id);
CREATE INDEX IF NOT EXISTS idx_credit_scores_session_id ON credit_scores(session_id);
CREATE INDEX IF NOT EXISTS idx_loan_applications_customer_id ON loan_applications(customer_id);
CREATE INDEX IF NOT EXISTS idx_assessment_sessions_session_id ON assessment_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_assessment_sessions_customer_id ON assessment_sessions(customer_id);

-- Insert sample data for testing
INSERT INTO customer_profiles (customer_id, name, age, location, occupation, monthly_income, family_size, education_level, land_ownership) 
VALUES 
    ('CUST_001', 'Ramesh Kumar', 35, 'Khargone, Madhya Pradesh', 'farmer', 12000, 5, 'secondary', true),
    ('CUST_002', 'Sunita Devi', 28, 'Sikar, Rajasthan', 'small business', 8500, 3, 'primary', false),
    ('CUST_003', 'Mohan Singh', 42, 'Muzaffarpur, Bihar', 'dairy farmer', 15000, 6, 'secondary', true)
ON CONFLICT (customer_id) DO NOTHING;

-- Function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers to update timestamps
CREATE TRIGGER update_customer_profiles_updated_at BEFORE UPDATE ON customer_profiles FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_loan_applications_updated_at BEFORE UPDATE ON loan_applications FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
EOF

# Create README file
echo -e "${BLUE}📚 Creating README file...${NC}"

cat > README.md << 'EOF'
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

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE
