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
