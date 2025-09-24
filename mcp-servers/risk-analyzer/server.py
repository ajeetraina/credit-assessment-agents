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
