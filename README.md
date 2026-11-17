<div align="center">

# 💳 Credit Risk Prediction & Customer Segmentation System

### Production-Grade End-to-End Machine Learning Platform with FastAPI, PostgreSQL, Streamlit & Docker

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-316192?style=for-the-badge&logo=postgresql)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-ORM-D71F00?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

</p>

---

### 🚀 A Production-Style Machine Learning System for Credit Risk Assessment, Customer Segmentation, REST APIs, Interactive Analytics Dashboard, and Containerized Deployment.

Designed using modern Machine Learning and Software Engineering principles to demonstrate how an ML model moves from experimentation to a production-ready application.

</div>

---

# 📖 Overview

Financial institutions process thousands of loan applications every day. Evaluating each applicant manually is time-consuming, inconsistent, and prone to human error.

This project demonstrates how Machine Learning can automate the credit evaluation process by predicting the probability of customer default and classifying applicants into meaningful risk categories.

Unlike traditional notebook-based ML projects, this repository follows a **production-oriented architecture**, where a trained machine learning model is exposed through REST APIs, connected to a relational database, visualized through an interactive dashboard, and deployed using Docker.

The objective is not only to build an accurate prediction model but also to demonstrate how modern AI applications are engineered, deployed, monitored, and maintained in real-world environments.

---

# 🎯 Business Problem

Banks and financial organizations must answer a simple but critical question:

> **"Should we approve this customer's loan application?"**

Approving a high-risk customer may result in financial losses.

Rejecting a reliable customer may reduce business opportunities.

This system helps automate that decision-making process by predicting customer credit risk using historical financial attributes.

The predicted output can assist analysts, loan officers, and financial institutions in making faster, more consistent, and data-driven lending decisions.

---

# 🚀 Key Features

## 🤖 Machine Learning

- End-to-End ML Pipeline
- Data Validation
- Data Cleaning
- Feature Engineering
- Feature Scaling
- Model Training
- Model Evaluation
- Automatic Best Model Selection
- Credit Risk Prediction
- Customer Risk Segmentation

---

## ⚙ Backend

- FastAPI REST APIs
- SQLAlchemy ORM
- Pydantic Validation
- Dependency Injection
- Exception Handling
- Structured JSON Logging
- Request Tracking
- Health Monitoring

---

## 📊 Dashboard

- Executive Dashboard
- Customer Management
- Credit Risk Prediction
- Prediction History
- Interactive Analytics
- Feature Importance Visualization
- ROC Curve
- Confusion Matrix
- Precision-Recall Curve
- CSV Export

---

## 🗄 Database

- PostgreSQL
- Customer Records
- Prediction History
- Model Registry
- Model Metrics
- Relational Data Storage

---

## 🐳 DevOps

- Docker
- Docker Compose
- Environment Configuration
- GitHub Actions Ready
- Production Deployment Ready

---

# 🌟 Project Highlights

✔ Production-ready ML Architecture

✔ Modular Python Codebase

✔ FastAPI REST Backend

✔ PostgreSQL Integration

✔ Interactive Streamlit Dashboard

✔ Dockerized Deployment

✔ Automated Model Loading

✔ Clean Software Architecture

✔ RESTful API Design

✔ Industry Standard Project Structure

✔ Logging & Monitoring Support

✔ Testable and Scalable Design

---

# 💡 Why This Project?

Most machine learning repositories focus only on training a model inside a Jupyter Notebook.

This project goes several steps further by demonstrating the complete lifecycle of a machine learning application:

- Data Collection & Validation
- Data Cleaning
- Feature Engineering
- Model Training
- Model Evaluation
- Model Serialization
- Backend API Development
- Database Integration
- Interactive Dashboard
- Docker Containerization
- Production Deployment

The goal is to simulate how machine learning systems are designed and deployed in modern software companies rather than stopping after model training.

---

# 📌 Use Cases

This platform can serve as the foundation for:

- Banking Loan Approval Systems
- Financial Risk Assessment Platforms
- Customer Credit Analysis
- FinTech Applications
- Banking Analytics Dashboards
- Credit Scoring Systems
- Educational ML Deployment Projects
- End-to-End AI Portfolio Projects

---

# ⭐ Project Goals

- Build a complete production-ready ML application.
- Demonstrate software engineering best practices.
- Showcase deployment-ready machine learning architecture.
- Integrate ML with backend APIs and databases.
- Provide an interactive dashboard for business users.
- Create a scalable foundation for future AI-powered financial applications.

---

# 📚 Table of Contents

- Project Overview
- System Architecture
- Machine Learning Pipeline
- Backend Architecture
- Database Design
- Dashboard
- Project Structure
- Installation Guide
- Docker Deployment
- API Documentation
- Testing
- Security
- Performance
- Future Enhancements
- Contributing
- License

---

# 🏗️ System Architecture

The Credit Risk Prediction System follows a modular, production-oriented architecture where each component has a single responsibility.

Instead of combining machine learning, business logic, and data access into one application, the project separates them into independent layers that communicate through REST APIs. This architecture improves scalability, maintainability, and deployment flexibility.

---

# 📐 High-Level Architecture

```text
                         ┌──────────────────────────┐
                         │        End User          │
                         └────────────┬─────────────┘
                                      │
                                      │ HTTP Requests
                                      ▼
                    ┌──────────────────────────────────┐
                    │     Streamlit Dashboard (UI)      │
                    └────────────┬──────────────────────┘
                                 │
                                 │ REST API
                                 ▼
               ┌─────────────────────────────────────────┐
               │           FastAPI Backend               │
               │                                         │
               │  • Request Validation                   │
               │  • Business Logic                       │
               │  • Recommendation Engine                │
               │  • Prediction Service                   │
               └────────────┬────────────────────────────┘
                            │
              ┌─────────────┴──────────────┐
              │                            │
              ▼                            ▼
     ┌────────────────┐          ┌─────────────────────┐
     │ ML Model Bundle│          │ PostgreSQL Database │
     │ Random Forest  │          │ Customer Records    │
     │ Scikit-Learn   │          │ Prediction History  │
     └────────────────┘          │ Model Registry      │
                                 │ Model Metrics       │
                                 └─────────────────────┘
```

---

# ⚙️ Application Workflow

The prediction workflow follows these steps:

```text
Customer

↓

Streamlit Dashboard

↓

FastAPI Endpoint

↓

Input Validation

↓

Business Service Layer

↓

CreditRiskPredictor

↓

Random Forest Model

↓

Risk Probability

↓

Risk Segment

↓

Database Storage

↓

JSON Response

↓

Dashboard Visualization
```

---

# 🤖 Machine Learning Pipeline

The machine learning pipeline is completely separated from the backend.

The model is trained only once, serialized using Joblib, and later loaded by the API during application startup.

```text
Raw Dataset

↓

Data Validation

↓

Data Cleaning

↓

Exploratory Data Analysis

↓

Feature Engineering

↓

Train/Test Split

↓

SMOTE

↓

Model Training

↓

Model Evaluation

↓

Champion Model Selection

↓

Model Serialization

↓

saved_models/model_bundle.joblib

↓

Prediction API
```

---

# 🌐 Backend Architecture

The backend follows a layered architecture inspired by production software systems.

```text
HTTP Request

↓

API Router

↓

Service Layer

↓

Repository Layer

↓

SQLAlchemy ORM

↓

PostgreSQL
```

Each layer has a dedicated responsibility:

| Layer | Responsibility |
|--------|----------------|
| API Routers | Expose REST endpoints |
| Services | Business logic & prediction workflow |
| Repository | Database operations |
| Models | SQLAlchemy ORM entities |
| Schemas | Request & Response validation |
| Core | Configuration & Logging |
| Middleware | Request tracking & exception handling |

---

# 🗄️ Database Architecture

The backend stores prediction history and customer information inside PostgreSQL.

```text
Customers
      │
      │
      ├──────────────┐
      │              │
      ▼              ▼
Predictions     Prediction History

             │

             ▼

      Model Registry

             │

             ▼

      Model Metrics
```

---

# 🐳 Docker Architecture

Every major component runs inside an isolated container.

```text
                Docker Compose

       ┌─────────────┬─────────────┬─────────────┐
       │             │             │
       ▼             ▼             ▼
 PostgreSQL      FastAPI API    Streamlit UI

       │             │             │

       └─────────────┼─────────────┘

                     ▼

                User Browser
```

Containerized deployment ensures that the application behaves consistently across development, testing, and production environments.

---

# 📂 Project Structure

```text
credit-risk-prediction-system/

│
├── backend/
│   ├── api/
│   ├── core/
│   ├── database/
│   ├── middleware/
│   ├── schemas/
│   ├── services/
│   └── main.py
│
├── dashboard/
│   ├── views/
│   ├── charts.py
│   ├── forms.py
│   ├── api_client.py
│   └── app.py
│
├── ml/
│   ├── data/
│   ├── preprocessing/
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── pipeline.py
│
├── sql/
├── docs/
├── reports/
├── tests/
├── saved_models/
├── utils/
│
├── Dockerfile
├── Dockerfile.dashboard
├── docker-compose.yml
├── render.yaml
├── railway.toml
├── requirements.txt
└── README.md
```

---

# 🧩 Design Principles

The project follows modern software engineering practices to improve maintainability and scalability.

- Separation of Concerns (SoC)
- Layered Architecture
- Modular ML Pipeline
- Dependency Injection
- RESTful API Design
- Configuration through Environment Variables
- Containerized Deployment
- Structured Logging
- Type Hinting
- Unit & Integration Testing
- Reusable Components
- Production-Oriented Project Structure

---

# 🎯 Why This Architecture?

This architecture allows each component to evolve independently.

For example:

- The machine learning model can be replaced without changing the API.
- The dashboard can be redesigned without modifying backend services.
- PostgreSQL can be migrated without affecting the prediction logic.
- Additional models can be deployed using the existing service layer.
- Future authentication, monitoring, and cloud deployment can be integrated with minimal architectural changes.

This modular design closely resembles the architecture used in modern AI-powered production systems.
# ⚙️ System Architecture

The project follows a production-style layered architecture where Machine Learning, Backend APIs, Database, and Dashboard are completely separated.

```
                        +----------------------+
                        |  Streamlit Dashboard |
                        |  (Frontend UI)       |
                        +----------+-----------+
                                   |
                            REST API Requests
                                   |
                                   ▼
                     +-----------------------------+
                     |     FastAPI Backend         |
                     | Authentication (Future)     |
                     | Validation                  |
                     | Business Logic              |
                     +-------------+---------------+
                                   |
                +------------------+------------------+
                |                                     |
                ▼                                     ▼
     ML Prediction Engine                     PostgreSQL Database
  (Saved Random Forest Model)          Customers, Predictions, Metrics
                |
                ▼
     Credit Risk Prediction
      Customer Segmentation
```

---

# 📂 Project Structure

```
credit-risk-prediction-system/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── database/
│   │   ├── services/
│   │   ├── middleware/
│   │   ├── schemas/
│   │   ├── dependencies/
│   │   ├── core/
│   │   └── main.py
│   │
│   └── scripts/
│
├── dashboard/
│   ├── views/
│   ├── charts.py
│   ├── api_client.py
│   ├── forms.py
│   └── app.py
│
├── ml/
│   ├── data/
│   ├── preprocessing/
│   ├── artifacts.py
│   ├── pipeline.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── models.py
│   ├── segmentation.py
│   ├── imbalance.py
│   ├── splitting.py
│   ├── eda.py
│   ├── features.py
│   └── config.py
│
├── saved_models/
│
├── reports/
│
├── sql/
│
├── tests/
│
├── docs/
│
├── Dockerfile
├── Dockerfile.dashboard
├── docker-compose.yml
├── render.yaml
├── railway.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

# 🏗️ Backend Architecture

The backend follows a clean layered architecture.

```
API Layer
│
├── Request Validation
│
├── Dependency Injection
│
├── Service Layer
│
├── Database Layer
│
├── ML Prediction Layer
│
└── Response Serialization
```

Each layer has a single responsibility.

---

## API Layer

Responsible for

- HTTP Endpoints
- Request Validation
- Authentication (future)
- Error Handling
- Response Formatting

Technology

- FastAPI
- Pydantic

---

## Service Layer

Contains complete business logic.

Examples

- Customer creation
- Prediction generation
- Dashboard statistics
- Risk recommendation
- Analytics generation

---

## Database Layer

Uses SQLAlchemy ORM.

Responsibilities

- CRUD operations
- Database Sessions
- Transactions
- Model Mapping

Database Supported

- PostgreSQL
- SQLite (Development)

---

## ML Layer

Machine Learning module is completely independent from backend.

Responsibilities

- Load trained model
- Feature preprocessing
- Prediction
- Probability calculation
- Risk segmentation

No retraining happens during API requests.

Only inference is performed.

---

# 🗄️ Database Schema

Main Tables

### customers

Stores customer information.

Example fields

- Customer ID
- Name
- Age
- Income
- Credit Score
- Loan Amount
- Savings
- Expenses

---

### predictions

Stores every prediction.

Fields include

- Prediction ID
- Customer ID
- Default Probability
- Risk Segment
- Prediction Time

---

### prediction_history

Maintains historical predictions.

Useful for

- Tracking model performance
- Customer monitoring
- Dashboard analytics

---

### model_registry

Stores

- Active model
- Model version
- Training timestamp
- Performance metrics

---

### model_metrics

Stores evaluation metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

---

# 🔄 Prediction Workflow

```
Customer Data

      │

      ▼

Input Validation

      │

      ▼

Feature Engineering

      │

      ▼

Saved ML Model

      │

      ▼

Probability Prediction

      │

      ▼

Risk Classification

      │

      ▼

Store in PostgreSQL

      │

      ▼

Return JSON Response

      │

      ▼

Display on Dashboard
```

---

# 📈 Risk Segmentation

The predicted probability is converted into business-friendly risk categories.

| Probability | Risk |
|-------------|------|
| 0 – 30% | 🟢 Low Risk |
| 30 – 70% | 🟡 Medium Risk |
| 70 – 100% | 🔴 High Risk |

These thresholds are configurable through the project configuration.

---

# 🔐 Security Features

Current implementation includes

- Input validation
- Request logging
- JSON structured logs
- Environment variables
- Secret key configuration
- Docker isolation
- Health monitoring
- Error handling
- API documentation

Future improvements

- JWT Authentication
- OAuth2
- Rate Limiting
- HTTPS
- Role Based Access Control
- API Keys

# 🤖 Machine Learning Pipeline

The project implements an end-to-end Machine Learning pipeline for predicting customer credit risk using structured financial data.

The pipeline follows industry best practices to ensure reproducibility, scalability, and production readiness.

---

# 📊 Data Pipeline

```
Raw Dataset
      │
      ▼
Data Validation
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Train / Validation / Test Split
      │
      ▼
Class Balancing (SMOTE)
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Best Model Selection
      │
      ▼
Model Serialization
      │
      ▼
FastAPI Inference
```

---

# 🧹 Data Preprocessing

The preprocessing pipeline automatically handles common real-world data quality issues.

### Missing Value Handling

- Numerical Imputation
- Categorical Imputation
- Missing Value Reports

---

### Duplicate Removal

- Exact Duplicate Detection
- Duplicate Removal

---

### Invalid Data Handling

Examples

- Negative Income
- Impossible Age
- Invalid Credit Score
- Negative Savings
- Negative Loan Amount

Invalid values are converted to **NaN** and handled later by the preprocessing pipeline.

---

### Feature Scaling

Numerical features are normalized using Scikit-Learn preprocessing pipelines.

Examples

- Income
- Savings
- Expenses
- Loan Amount

---

### Categorical Encoding

Categorical columns are transformed into numerical representations before model training.

---

# ⚡ Feature Engineering

Several financial indicators are automatically generated.

Examples include

- Debt-to-Income Ratio
- Savings Ratio
- Expense Ratio
- Credit Utilization
- Payment Behaviour Indicators
- Financial Stability Score
- Income Categories
- Credit Categories
- Loan Burden Score

These engineered features improve prediction performance compared to raw attributes.

---

# 📈 Exploratory Data Analysis (EDA)

The project automatically generates analytical reports.

Generated Visualizations

- Class Distribution
- Missing Value Analysis
- Correlation Heatmap
- Feature Distribution
- Credit Score Distribution
- Income Distribution
- Loan Distribution
- Risk Segment Distribution
- Default Rate Analysis

Reports are stored inside

```
reports/
```

Generated summaries include

- Feature Statistics
- Correlation Tables
- Data Quality Reports
- Default Rate Reports

---

# 🔀 Dataset Splitting

The dataset is divided into three subsets.

| Dataset | Purpose |
|----------|---------|
| Training | Model Learning |
| Validation | Hyperparameter Tuning |
| Testing | Final Evaluation |

Stratified splitting is used to preserve class distribution across all datasets.

---

# ⚖️ Class Imbalance Handling

Real-world financial datasets are highly imbalanced.

This project uses

- SMOTE (Synthetic Minority Oversampling Technique)

Benefits

- Better Recall
- Improved Minority Detection
- Balanced Training Dataset

SMOTE is applied **only** to the training dataset to prevent data leakage.

---

# 🧠 Machine Learning Models

Multiple machine learning algorithms are trained and compared.

Implemented Models

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost *(Optional)*
- LightGBM *(Optional)*

Each model is evaluated using identical train-validation-test splits.

---

# 🎯 Hyperparameter Tuning

Randomized Search is used for parameter optimization.

Examples

- Number of Trees
- Maximum Depth
- Learning Rate
- Minimum Samples Split
- Feature Selection

The tuned model is selected automatically based on evaluation metrics.

---

# 📊 Model Evaluation

The project evaluates every model using multiple performance metrics.

Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

Additional evaluation includes

- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Classification Report

Evaluation plots are automatically generated and saved.

---

# 🏆 Champion Model Selection

Instead of selecting the highest accuracy model, the project compares multiple metrics.

Selection Criteria

- ROC-AUC
- Recall
- Precision
- F1 Score
- Stability

The best-performing model is automatically registered as the production model.

---

# 💾 Model Serialization

After training, the selected model is stored as a reusable artifact.

```
saved_models/
│
└── model_bundle.joblib
```

The bundle contains

- Trained Model
- Feature Pipeline
- Encoders
- Scalers
- Metadata
- Model Version

No retraining is required during deployment.

---

# 🚀 Model Inference

During prediction

```
Customer Input

      │

      ▼

Validation

      │

      ▼

Preprocessing Pipeline

      │

      ▼

Saved Model

      │

      ▼

Default Probability

      │

      ▼

Risk Segment

      │

      ▼

API Response
```

The backend only performs inference using the saved model, ensuring consistent predictions across environments.

---

# 📦 Model Output

Each prediction returns

- Default Probability
- Predicted Label
- Risk Segment
- Risk Level
- Prediction Timestamp
- Model Version

Example Response

```json
{
  "default_probability": 0.82,
  "predicted_label": 1,
  "risk_segment": "High",
  "risk_level": "High Risk"
}
```

---

# ✅ ML Pipeline Highlights

✔ End-to-End Automated Pipeline

✔ Data Validation

✔ Feature Engineering

✔ SMOTE Class Balancing

✔ Multiple Model Comparison

✔ Hyperparameter Optimization

✔ Automatic Champion Model Selection

✔ Production Model Serialization

✔ REST API Inference

✔ Docker Ready

✔ Deployment Ready

✔ Reproducible Training Pipeline

# 🚀 FastAPI Backend

The backend is built using **FastAPI**, providing high-performance REST APIs for customer management, credit risk prediction, analytics, and dashboard visualization.

The API acts as the central communication layer between the Machine Learning model, PostgreSQL database, and Streamlit dashboard.

---

# 🏛 Backend Architecture

```
                 Client / Dashboard

                        │

                        ▼

              FastAPI REST Endpoints

                        │

        ┌───────────────┼────────────────┐

        ▼               ▼                ▼

 Customer Service   Prediction      Analytics Service

        ▼               ▼                ▼

 SQLAlchemy ORM   ML Prediction     Aggregation Engine

        ▼

 PostgreSQL Database
```

---

# ⚡ API Features

✔ RESTful API

✔ Automatic Request Validation

✔ Response Serialization

✔ Dependency Injection

✔ Structured Logging

✔ Exception Handling

✔ OpenAPI Documentation

✔ Swagger UI

✔ ReDoc Documentation

✔ Health Monitoring

✔ Production Ready

---

# 📚 API Documentation

FastAPI automatically generates interactive API documentation.

| Documentation | URL |
|---------------|-----|
| Swagger UI | `http://localhost:8000/docs` |
| ReDoc | `http://localhost:8000/redoc` |

Developers can directly test every API endpoint from the browser.

---

# 📌 API Endpoints

## System

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Health Check |
| GET | `/model-info` | Current ML Model Information |
| GET | `/dashboard` | Dashboard Statistics |

---

## Customer Management

| Method | Endpoint |
|---------|----------|
| POST | `/customers` |
| GET | `/customers` |
| GET | `/customers/{id}` |
| PUT | `/customers/{id}` |
| DELETE | `/customers/{id}` |

Operations include

- Add Customer
- View Customer
- Update Customer
- Delete Customer
- Search Customer

---

## Prediction APIs

| Method | Endpoint |
|---------|----------|
| POST | `/predict` |
| POST | `/predict/customer/{id}` |

Returns

- Default Probability
- Risk Segment
- Risk Label
- Recommendation
- Prediction Timestamp

---

## Prediction History

| Method | Endpoint |
|---------|----------|
| GET | `/predictions` |
| GET | `/predictions/{id}` |

Supports filtering using

- Risk Segment
- Date Range
- Customer
- Probability

---

## Dashboard APIs

Dashboard statistics are served through REST APIs.

Examples

- Total Customers
- Average Credit Score
- Risk Distribution
- Default Rate
- Prediction Trend
- Recent Predictions

---

# 📤 Example Prediction Request

```json
{
  "age": 35,
  "income": 72000,
  "loan_amount": 25000,
  "credit_score": 690,
  "monthly_expense": 18000,
  "savings": 60000,
  "late_payments": 2
}
```

---

# 📥 Example Prediction Response

```json
{
  "customer_id": 101,
  "default_probability": 0.81,
  "predicted_label": 1,
  "risk_segment": "High",
  "risk_level": "High Risk",
  "recommendation": "Loan requires manual review."
}
```

---

# ⚠ Error Handling

The backend provides standardized error responses.

Example

```json
{
  "detail": "Customer not found."
}
```

Common HTTP Status Codes

| Code | Meaning |
|------|----------|
| 200 | Success |
| 201 | Resource Created |
| 400 | Invalid Request |
| 404 | Resource Not Found |
| 422 | Validation Error |
| 500 | Internal Server Error |

---

# 🛡 Input Validation

FastAPI validates every request before processing.

Validation includes

- Missing Fields
- Incorrect Data Types
- Invalid Numerical Values
- Required Parameters
- Schema Validation

This prevents invalid customer data from reaching the ML model.

---

# 📊 Streamlit Dashboard

A modern interactive dashboard has been developed using **Streamlit** for business users and analysts.

The dashboard communicates with the backend APIs instead of directly accessing the database or ML model.

---

# 🖥 Dashboard Pages

## 🏠 Dashboard

Displays overall business insights.

Features

- Total Customers
- Risk Distribution
- Average Credit Score
- Recent Predictions
- Prediction Trend
- KPI Cards

---

## 👥 Customer Management

Manage customer records.

Features

- Add Customer
- Update Customer
- Delete Customer
- Search Customer
- View Customer Profile

---

## 🤖 Prediction

Generate predictions for

- Existing Customers
- New Customers

Output includes

- Default Probability
- Risk Category
- Recommendation
- Prediction Summary

---

## 📈 Analytics

Visualizes important business insights.

Charts include

- Income Distribution
- Loan Distribution
- Credit Score Distribution
- Risk Segmentation
- Feature Importance
- ROC Curve
- Precision-Recall Curve
- Confusion Matrix

---

## 📜 Prediction History

Maintains prediction audit history.

Supports

- Search
- Filters
- Sorting
- CSV Export

---

# 📡 API Communication

```
Dashboard

     │

REST API

     │

FastAPI Backend

     │

Prediction Service

     │

ML Model

     │

Database

     │

JSON Response

     │

Dashboard Visualization
```

---

# 📈 Dashboard Features

✔ Interactive Charts

✔ Real-Time Predictions

✔ Customer Search

✔ KPI Cards

✔ Risk Visualization

✔ Analytics Dashboard

✔ Prediction History

✔ CSV Export

✔ Responsive Layout

✔ Production Ready UI

---

# 🔥 Backend Highlights

- High-performance FastAPI backend
- Fully documented REST APIs
- Automatic request validation
- Modular service architecture
- SQLAlchemy ORM integration
- PostgreSQL database support
- Streamlit frontend integration
- ML inference using saved model
- Interactive Swagger documentation
- Production-ready API design


# 🐳 Docker & Deployment

The entire application is fully containerized using **Docker** and **Docker Compose**, allowing developers to deploy the complete system with a single command.

The application consists of three independent services.

```
                 Docker Compose

        ┌──────────────────────────────┐

        │                              │

        ▼                              ▼

 FastAPI API Container         Streamlit Dashboard

        │                              │

        └──────────────┬───────────────┘
                       │

                       ▼

             PostgreSQL Database
```

---

# 📦 Docker Services

The project uses multiple Docker containers.

| Service | Description |
|----------|-------------|
| API | FastAPI Backend |
| Dashboard | Streamlit Frontend |
| Database | PostgreSQL |
| Seed (Optional) | Demo Data Generator |

Each service runs independently while communicating over Docker's internal network.

---

# 📁 Docker Files

```
Dockerfile
Dockerfile.dashboard
docker-compose.yml
.env.example
```

---

## Dockerfile

The backend Docker image includes

- Python Runtime
- Required Dependencies
- Machine Learning Model
- FastAPI Application
- SQLAlchemy
- Production Server

---

## Dashboard Dockerfile

Contains

- Streamlit
- Dashboard Components
- Charts
- API Client

---

## Docker Compose

Docker Compose automatically starts

- PostgreSQL
- FastAPI API
- Streamlit Dashboard

using a single command.

```bash
docker compose up --build
```

---

# 🌍 Environment Variables

Configuration is managed using environment variables.

Example

```env
ENVIRONMENT=development

API_HOST=0.0.0.0

API_PORT=8000

DATABASE_URL=postgresql://credit_risk:password@db:5432/credit_risk

SECRET_KEY=your-secret-key

MODEL_PATH=saved_models/model_bundle.joblib

API_URL=http://localhost:8000
```

Sensitive information is never hardcoded inside the application.

---

# 🚀 Running the Project

## Clone Repository

```bash
git clone https://github.com/yourusername/credit-risk-prediction-system.git

cd credit-risk-prediction-system
```

---

## Create Environment File

```bash
cp .env.example .env
```

Update

- Database Password
- Secret Key

---

## Start Complete Project

```bash
docker compose up --build
```

This automatically starts

- PostgreSQL
- FastAPI
- Streamlit

---

# 🌐 Application URLs

After successful deployment

| Service | URL |
|----------|-----|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| Health Check | http://localhost:8000/health |

---

# ☁️ Deployment Support

The project includes deployment configurations for multiple cloud platforms.

Supported Platforms

- Docker
- Docker Compose
- Render
- Railway

Deployment files

```
render.yaml

railway.toml

docker-compose.yml
```

---

# 🔄 CI/CD Pipeline

GitHub Actions is configured to automate project validation.

Pipeline includes

- Install Dependencies
- Code Formatting
- Linting
- Unit Testing
- Integration Testing
- Docker Image Build
- Build Verification

```
Git Push

    │

    ▼

GitHub Actions

    │

    ▼

Install Dependencies

    │

    ▼

Run Tests

    │

    ▼

Docker Build

    │

    ▼

Build Success
```

---

# 🧪 Testing

The project contains automated tests covering

- Machine Learning Pipeline
- API Endpoints
- Database Layer
- Dashboard Components

Run all tests

```bash
pytest
```

Run PostgreSQL integration tests

```bash
TEST_DATABASE_URL=postgresql://user:password@localhost/database pytest
```

---

# 📊 Logging

The backend uses structured JSON logging for production environments.

Features

- Request Logging
- Error Logging
- Response Time Monitoring
- Request ID Tracking
- Model Loading Logs
- Prediction Logs
- Database Logs

Example

```json
{
  "request_id": "abc123",
  "endpoint": "/predict",
  "status": 200,
  "duration_ms": 24
}
```

---

# 🔒 Security Features

Current implementation includes

- Environment Variables
- Secret Key Management
- Structured Logging
- Request Validation
- Docker Isolation
- SQLAlchemy ORM
- Input Validation
- Secure Configuration

Future Enhancements

- JWT Authentication
- OAuth2 Login
- HTTPS Support
- API Rate Limiting
- Role-Based Access Control
- API Keys
- Secrets Management

---

# 📈 Performance

The system is optimized for production workloads.

Highlights

- FastAPI asynchronous architecture
- SQLAlchemy ORM optimization
- Docker containerization
- Efficient ML inference
- Cached model loading
- Modular architecture
- Lightweight Streamlit frontend

---

# ✅ Deployment Checklist

Before deploying to production

- Configure `.env`
- Change `SECRET_KEY`
- Configure PostgreSQL credentials
- Verify API health endpoint
- Verify Docker containers
- Run automated tests
- Build Docker images
- Deploy using Render, Railway, or Docker

---

# 🎯 Production Highlights

✔ Dockerized Architecture

✔ Multi-Container Deployment

✔ PostgreSQL Integration

✔ Environment-Based Configuration

✔ GitHub Actions CI/CD

✔ Production Logging

✔ Health Monitoring

✔ Cloud Deployment Ready

✔ Scalable Architecture

✔ Enterprise Deployment Workflow

> **Note:** This project uses a reproducible synthetic financial dataset for educational and demonstration purposes. The architecture, deployment workflow, and software engineering practices closely resemble production-grade machine learning systems.

# ✨ Key Features

This project combines **Machine Learning**, **Backend Development**, **Database Engineering**, **Data Visualization**, and **DevOps** into a single production-style application.

---

## 🤖 Machine Learning

- End-to-End ML Pipeline
- Automated Data Validation
- Data Cleaning & Preprocessing
- Financial Feature Engineering
- Exploratory Data Analysis (EDA)
- Class Imbalance Handling (SMOTE)
- Multiple Model Training
- Hyperparameter Optimization
- Automatic Champion Model Selection
- Model Serialization
- Production Inference Pipeline

---

## 🚀 Backend Development

- FastAPI REST APIs
- Layered Architecture
- Dependency Injection
- SQLAlchemy ORM
- Pydantic Validation
- Exception Handling
- OpenAPI Documentation
- Health Monitoring
- Modular Service Design

---

## 🗄 Database

- PostgreSQL Integration
- SQLAlchemy ORM
- Customer Management
- Prediction Storage
- Prediction History
- Model Registry
- Model Metrics
- Automatic Table Creation

---

## 📊 Dashboard

Interactive Streamlit dashboard including

- KPI Cards
- Customer Management
- Credit Risk Prediction
- Risk Distribution
- Analytics Dashboard
- Feature Importance
- ROC Curve
- Confusion Matrix
- Prediction History
- CSV Export

---

## 🐳 DevOps

- Docker
- Docker Compose
- Multi-Container Architecture
- Environment Variables
- Health Checks
- GitHub Actions CI
- Render Deployment
- Railway Deployment

---

## 🔐 Security

- Environment Configuration
- Secret Key Management
- Input Validation
- Structured Logging
- Request Tracking
- Secure Database Connections

---

# 🛠 Technology Stack

## Programming Languages

- Python 3.12
- SQL

---

## Machine Learning

- Scikit-Learn
- Pandas
- NumPy
- Joblib
- Imbalanced-Learn (SMOTE)

---

## Backend

- FastAPI
- Uvicorn
- Pydantic
- SQLAlchemy

---

## Database

- PostgreSQL
- SQLite

---

## Frontend

- Streamlit

---

## Visualization

- Matplotlib
- Plotly

---

## DevOps

- Docker
- Docker Compose
- GitHub Actions

---

## Deployment

- Render
- Railway

---

## Version Control

- Git
- GitHub

---

# 📊 Project Statistics

| Category | Details |
|----------|----------|
| Architecture | Production Style |
| Machine Learning Models | Multiple |
| REST APIs | FastAPI |
| Database | PostgreSQL |
| Dashboard | Streamlit |
| Docker Support | Yes |
| CI/CD | GitHub Actions |
| Documentation | Comprehensive |
| Testing | Unit + Integration |
| Deployment Ready | Yes |

---

# 📸 Application Preview

## Dashboard

```
+------------------------------------------------------------+
|                Credit Risk Dashboard                       |
+------------------------------------------------------------+

Customers           Predictions        High Risk

     1,240               1,018              132

--------------------------------------------------------------

Risk Distribution

██████████

Analytics

██████████████████

Recent Predictions

██████████████
```

---

## Prediction Screen

```
Customer Information

Income

Savings

Loan Amount

Credit Score

Expenses

Late Payments

[ Predict Risk ]
```

---

## Prediction Result

```
Customer Risk Report

Probability of Default

82%

Risk Segment

🔴 HIGH RISK

Recommendation

Manual Review Recommended
```

---

# 📈 Learning Outcomes

This project demonstrates practical knowledge of

### Machine Learning

- Data Cleaning
- Feature Engineering
- Model Training
- Model Evaluation
- Model Deployment

---

### Backend Engineering

- REST API Development
- Dependency Injection
- ORM Design
- Layered Architecture
- Validation

---

### Database Engineering

- Schema Design
- CRUD Operations
- SQL Optimization
- ORM Relationships

---

### Data Science

- EDA
- Visualization
- Feature Selection
- Model Comparison
- Risk Analysis

---

### DevOps

- Docker
- Containerization
- CI/CD
- Environment Configuration
- Cloud Deployment

---

# 🎯 Resume Highlights

This project demonstrates experience with

- Production ML Pipeline
- Credit Risk Prediction
- Customer Segmentation
- FastAPI Backend
- PostgreSQL Database
- SQLAlchemy ORM
- Streamlit Dashboard
- Docker & Docker Compose
- REST API Development
- Model Deployment
- Data Visualization
- GitHub Actions CI/CD

---

# 🌟 Why This Project Stands Out

Unlike a traditional Machine Learning notebook, this project represents a complete production-ready application.

It demonstrates skills across multiple domains including

- Machine Learning Engineering
- Backend Development
- Database Design
- API Development
- Dashboard Development
- Docker Containerization
- Cloud Deployment
- DevOps Practices

The trained model is packaged and served through REST APIs, integrated with a PostgreSQL database, visualized using Streamlit, and deployed using Docker, closely resembling real-world enterprise applications.

---

# ⭐ If You Like This Project

If you found this repository useful,

⭐ Star the repository

🍴 Fork the repository

🐛 Report issues

💡 Suggest improvements

🤝 Contribute to the project

Every contribution is appreciated!

# 🛣️ Project Roadmap

The project is actively evolving. Planned enhancements include:

## Version 2.0

- JWT Authentication
- Role-Based Access Control (RBAC)
- User Login & Registration
- API Key Authentication
- Password Hashing

---

## Version 2.1

- Explainable AI (SHAP)
- Feature Importance per Prediction
- Risk Explanation Dashboard
- Individual Prediction Reports

---

## Version 2.2

- Batch Prediction API
- CSV Upload Support
- Bulk Customer Processing
- Excel Report Export

---

## Version 2.3

- Model Monitoring
- Data Drift Detection
- Performance Monitoring
- Automated Retraining Pipeline

---

## Version 3.0

- Microservices Architecture
- Kubernetes Deployment
- Redis Caching
- RabbitMQ Background Jobs
- Grafana Monitoring
- Prometheus Metrics

---

# 📊 Project Highlights

✅ End-to-End Machine Learning Pipeline

✅ Financial Data Processing

✅ Credit Risk Prediction

✅ Customer Risk Segmentation

✅ FastAPI Backend

✅ PostgreSQL Database

✅ SQLAlchemy ORM

✅ Streamlit Dashboard

✅ RESTful APIs

✅ Docker & Docker Compose

✅ GitHub Actions CI/CD

✅ Production-Ready Project Structure

---

# 📚 Documentation

Project documentation includes:

- System Architecture
- API Documentation
- Database Design
- Machine Learning Pipeline
- Deployment Guide
- Docker Setup
- Configuration Guide
- Testing Guide

Refer to the `docs/` directory for detailed documentation.

---

# 🤝 Contributing

Contributions are always welcome.

If you would like to improve this project:

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Push your branch.
5. Open a Pull Request.

Please ensure that:

- Code follows PEP 8 guidelines.
- New features include appropriate tests.
- Documentation is updated when necessary.

---

# 🧪 Running Tests

Run all tests:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov
```

---

# 📝 License

This project is licensed under the **MIT License**.

You are free to:

- Use
- Modify
- Distribute
- Fork

while preserving the original license.

---

# 👨‍💻 Author

**Karan Sharma**

B.Tech Computer Science Engineering (Artificial Intelligence & Machine Learning)

Passionate about

- Machine Learning
- Artificial Intelligence
- Backend Development
- Data Science
- MLOps
- Production AI Systems

---

# 📫 Connect With Me

- GitHub: https://github.com/karan-sharma-aiml
- LinkedIn: *(Add your LinkedIn profile URL here)*
- Email: *(Add your professional email here)*

---

# 🙏 Acknowledgements

Special thanks to the open-source community and the creators of:

- Python
- FastAPI
- Streamlit
- Scikit-Learn
- SQLAlchemy
- PostgreSQL
- Docker
- GitHub Actions
- Pandas
- NumPy
- Matplotlib
- Plotly

Their tools and libraries made this project possible.

---

# ⭐ Support

If you found this project useful,

⭐ Star this repository

🍴 Fork the project

🐞 Report bugs

💡 Suggest new features

🤝 Contribute to the project

Your support helps improve the project and encourages future development.

---

# 📌 Final Note

This project was developed as a **production-style Credit Risk Prediction & Customer Segmentation Platform**, integrating Machine Learning, Backend Engineering, Database Design, Data Visualization, and DevOps into a single application.

It demonstrates the complete lifecycle of deploying a Machine Learning model—from data preprocessing and model training to REST API development, database integration, interactive dashboards, Docker containerization, and cloud deployment.

The primary objective is to showcase practical software engineering and machine learning skills through a scalable, modular, and deployment-ready architecture.

---

<div align="center">

### ⭐ If you like this project, consider giving it a Star!

**Built with ❤️ using Python, FastAPI, Scikit-Learn, PostgreSQL, Streamlit, Docker, and GitHub Actions**

</div>
