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
