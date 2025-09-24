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
