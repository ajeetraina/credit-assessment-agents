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
