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
