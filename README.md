# 🤖 AI Data Analyst

An intelligent, agentic data analysis platform that allows users to upload datasets and interact with them using natural language.

The system automatically understands the uploaded data, performs data cleaning and preprocessing, generates statistical analysis and insights, creates meaningful visualizations, writes and executes Python analysis code, trains machine learning models for prediction tasks, and answers user questions about the dataset through a RAG-powered conversational interface.

The platform is built using **LangChain**, **LangGraph**, **LangSmith**, and **Hugging Face** models, with **FastAPI** as the backend and **HTML/CSS/JavaScript** as the frontend.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Key Features](#-key-features)
- [System Architecture](#️-system-architecture)
- [High-Level Architecture](#-high-level-architecture)
- [Agent Architecture](#-agent-architecture)
- [LangGraph Workflow](#-langgraph-workflow)
- [RAG Architecture](#-rag-architecture)
- [RAG Knowledge Base](#-rag-knowledge-base)
- [Data Processing Pipeline](#-data-processing-pipeline)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Frontend Architecture](#-frontend-architecture)
- [Web Application Pages](#-web-application-pages)
- [Dashboard Layout](#️-dashboard-layout)
- [Visualization Workspace](#-visualization-workspace)
- [Database Architecture](#️-database-architecture)
- [Prerequisites & User Configuration](#-prerequisites--user-configuration-requirements)
- [Database Tables](#database-tables)
- [CRUD Operations](#-crud-operations)
- [Recommended Project Structure](#-recommended-project-structure)
- [Technology Stack](#️-technology-stack)
- [AI Models](#-hugging-face-integration)
- [LangChain Usage](#-langchain-usage)
- [LangGraph Usage](#-langgraph-usage)
- [LangSmith Usage](#-langsmith-usage)
- [Code Generation and Execution](#-code-generation-and-execution)
- [Security Architecture](#-security-architecture)
- [API Architecture](#-api-architecture)
- [Example User Workflows](#-example-user-workflows)
- [Session-Based Architecture](#-session-based-architecture)
- [Intelligent Decision Making](#-intelligent-decision-making)
- [Analysis Report](#-analysis-report)
- [Recommendation Engine](#-recommendation-engine)
- [Dashboard Components](#-dashboard-components)
- [Development Phases](#️-development-phases)
- [Future Enhancements](#-future-enhancements)
- [Final Agent Concept](#-final-agent-concept)
- [Final Architecture](#-final-architecture)
- [Final Project Goal](#-final-project-goal)
- [Recommended Implementation Order](#-recommended-implementation-order)
- [Project Vision](#-project-vision)

---

## 🚀 Project Overview

**AI Data Analyst** is an agentic AI-powered data analysis platform where a user can upload a CSV or TXT dataset and interact with the dataset using natural language.

Instead of manually performing every step of data analysis, the user can simply ask:

- "Clean my dataset."
- "Analyze this dataset."
- "Show me meaningful visualizations."
- "Create a correlation heatmap."
- "What are the important insights?"
- "Find outliers in the dataset."
- "What columns have missing values?"
- "Which features are most important?"
- "Train a model to predict sales."
- "Which machine learning algorithm performed best?"
- "Why is the accuracy low?"
- "Recommend what I should do with this data."

The AI Data Analyst understands the request and determines which operations need to be performed.

The system can:

- Understand the uploaded dataset
- Profile the data
- Detect data quality issues
- Clean the data
- Preprocess the data
- Perform exploratory data analysis
- Generate Python analysis code
- Execute analysis code in a controlled environment
- Generate visualizations
- Explain observations
- Answer questions about the dataset
- Train machine learning models
- Evaluate model performance
- Recommend appropriate analytical approaches
- Store analytical history in the database
- Store useful information in a RAG knowledge base
- Use previous analytical context to provide better answers

---

## 🎯 Problem Statement

Traditional data analysis requires users to have knowledge of:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Statistics
- Machine Learning
- Data preprocessing
- SQL
- Data visualization

This creates a barrier for non-technical users.

The goal of this project is to build an AI Data Analyst that converts natural-language instructions into complete data analysis workflows.

**Example:**

> **User:** Clean this dataset.
>
> **AI:** The system analyzes the dataset and determines:
> - Missing values detected
> - Duplicate rows detected
> - Outliers detected
> - Incorrect data types detected
> - Constant columns detected
> - High-cardinality columns detected
>
> It then creates an appropriate cleaning pipeline.

---

## 🎯 Project Objectives

The main objectives are:

1. Build an intelligent AI-powered data analyst
2. Support CSV and TXT datasets
3. Automatically understand dataset structure
4. Automatically identify data quality problems
5. Perform intelligent preprocessing
6. Generate Python analysis code
7. Execute generated code
8. Generate meaningful visualizations
9. Provide natural-language observations
10. Allow users to ask questions about their datasets
11. Support machine learning model training
12. Compare different ML algorithms
13. Provide model metrics
14. Maintain conversational context
15. Build a persistent knowledge base using RAG
16. Store users, sessions, datasets, conversations, and analysis history
17. Use LangGraph for agentic workflow orchestration
18. Use LangChain for LLM, tools, and RAG integration
19. Use LangSmith for tracing, debugging, and evaluation

---

## ⭐ Key Features

### 1. Dataset Upload

**Supported formats:**
- CSV
- TXT

**Future:**
- Excel
- JSON
- Parquet
- SQL databases

After upload, the system automatically performs initial dataset profiling.

### 2. Automatic Dataset Understanding

The system determines:

- Number of rows
- Number of columns
- Column names
- Data types
- Missing values
- Duplicate records
- Unique values
- Cardinality
- Numerical columns
- Categorical columns
- Date/time columns
- Text columns
- Statistical summaries
- Potential target variables
- Potential identifier columns

**Example:**

```
Dataset: sales.csv

Rows: 50,000
Columns: 14

Numerical columns: 8
Categorical columns: 4
Date columns: 2

Missing values: 3.2%
Duplicate rows: 184

Potential target: Revenue
```

### 3. Intelligent Data Cleaning

When the user says "Clean the data," the AI does not blindly apply the same operations to every dataset. Instead, the system first profiles the dataset.

**Possible operations:**

**Missing Values**
- Mean imputation
- Median imputation
- Mode imputation
- Forward fill
- Backward fill
- Dropping rows
- Dropping columns

> The decision depends on the characteristics of the column.

**Duplicate Detection** — detects:
- Exact duplicate rows
- Duplicate IDs
- Potential duplicate records

**Outlier Detection** — possible techniques:
- IQR
- Z-score
- Isolation Forest

> The AI determines which approach is appropriate based on the dataset.

**Data Type Correction** — examples:
- `"100"` → `100`
- `"2026-01-01"` → `datetime`
- `"₹50,000"` → numeric
- `"25%"` → `0.25`

**Invalid Values** — detects values such as:
- `Age = -10`
- `Salary = -5000`
- `Percentage = 150`

### 4. Data Preprocessing

The preprocessing pipeline can include:

```
Missing Value Handling
        ↓
Duplicate Removal
        ↓
Outlier Treatment
        ↓
Data Type Conversion
        ↓
Categorical Encoding
        ↓
Feature Scaling
        ↓
Feature Selection
        ↓
Train/Test Split
```

**Techniques may include:**

*Encoding*
- One-Hot Encoding
- Ordinal Encoding
- Label Encoding

*Scaling*
- StandardScaler
- MinMaxScaler
- RobustScaler

*Feature Selection*
- Correlation
- Mutual Information
- Feature Importance
- Recursive Feature Elimination

### 5. Exploratory Data Analysis

The AI performs:

**Univariate Analysis**
- Distribution
- Mean
- Median
- Mode
- Variance
- Standard deviation
- Skewness
- Kurtosis

**Bivariate Analysis**
- Correlation
- Categorical vs numerical relationships
- Numerical vs numerical relationships

**Multivariate Analysis**
- Correlation matrix
- Feature relationships
- Cluster patterns

### 6. Automatic Visualization

When the user asks to "Visualize this dataset," the AI identifies meaningful visualizations based on the dataset.

**Possible charts:**
- Bar Chart
- Line Chart
- Histogram
- Box Plot
- Scatter Plot
- Pie Chart
- Heatmap
- Count Plot
- Area Chart
- Violin Plot
- Pair Plot

The system should avoid generating meaningless charts. For example:

| Data Pattern | Chart Type |
|---|---|
| Time + Sales | Line Chart |
| Category + Revenue | Bar Chart |
| Numerical distribution | Histogram |
| Numerical relationships | Scatter Plot |
| Multiple numerical variables | Correlation Heatmap |

### 7. Specific Visualization Requests

The user can request things like:

- "Create a bar chart of sales by region."
- "Create a scatter plot between price and quantity."
- "Show correlation heatmap."

The system generates the required Python code, executes it, and displays the chart.

### 8. Automatic Observations

Every visualization should be accompanied by AI-generated observations.

**Example:**

> The West region generates the highest revenue, accounting for approximately 34% of total sales.
>
> Revenue increased significantly during Q4.
>
> The relationship between advertising spend and sales appears moderately positive.

### 9. Natural Language Data Chatbot

The right-side chatbot allows users to ask questions about their dataset, such as:

- "What is the average salary?"
- "Which region has the highest sales?"
- "Are there any outliers?"
- "Which columns contain missing values?"
- "What are the most important insights?"
- "Why did sales decrease?"
- "What visualization should I use?"

The chatbot uses **Dataset Context** + **Analysis Results** + **Conversation History** + **Knowledge Base** + **RAG** to generate responses.

### 10. Machine Learning

The user can ask to "Train a model to predict sales." The AI determines whether the task is **Regression** or **Classification**.

**Regression Algorithms**
- Linear Regression
- Ridge
- Lasso
- Decision Tree
- Random Forest
- Gradient Boosting
- Random Forest Regressor

**Classification Algorithms**
- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- SVM
- Gradient Boosting

### 11. Automatic Model Selection

The AI can create a model training pipeline:

```
Dataset
   ↓
Target Detection
   ↓
Problem Type Detection
   ↓
Feature Selection
   ↓
Preprocessing
   ↓
Train/Test Split
   ↓
Multiple Algorithms
   ↓
Training
   ↓
Evaluation
   ↓
Model Comparison
   ↓
Best Model
```

### 12. Model Metrics

**Classification**
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

**Regression**
- MAE
- MSE
- RMSE
- R²
- MAPE

**Example:**

```
Best Model: Random Forest

Task: Regression

R² Score: 0.91
RMSE: 12.43
MAE: 8.72
```

---

## 🏗️ System Architecture

```
                    ┌─────────────────────┐
                    │     Web Browser     │
                    │  HTML/CSS/JS UI     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │    API Gateway      │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
                 ▼             ▼             ▼
          Authentication    Dataset      Chat API
                 │           Service         │
                 │             │             │
                 │             ▼             ▼
                 │       Data Profiler   LangGraph
                 │             │             │
                 │             ▼             │
                 │      Data Processing      │
                 │             │             │
                 │             ▼             │
                 │          Pandas           │
                 │             │             │
                 │             ▼             │
                 │       ML / Analytics      │
                 │                           │
                 │                           ▼
                 │                    LangChain Tools
                 │                           │
                 │             ┌─────────────┼─────────────┐
                 │             │             │             │
                 │             ▼             ▼             ▼
                 │           Python        RAG          LLM
                 │           Tool       Retriever     HuggingFace
                 │             │             │             │
                 │             └─────────────┼─────────────┘
                 │                           │
                 ▼                           ▼
          ┌─────────────────────────────────────────┐
          │              PostgreSQL                 │
          │ Users / Sessions / Files / Chats /      │
          │ Analysis / Models / Results             │
          └─────────────────────────────────────────┘
```

---

## 🧠 High-Level Architecture

The system contains six major layers.

```
┌───────────────────────────────────────────┐
│              PRESENTATION                 │
│           HTML + CSS + JavaScript         │
└─────────────────────┬─────────────────────┘
                       │
┌──────────────────────▼─────────────────────┐
│               API LAYER                    │
│                 FastAPI                    │
└──────────────────────┬─────────────────────┘
                       │
┌──────────────────────▼─────────────────────┐
│              AGENT LAYER                    │
│               LangGraph                     │
└──────────────────────┬─────────────────────┘
                       │
┌──────────────────────▼─────────────────────┐
│             AI / TOOL LAYER                 │
│ LangChain + HuggingFace + Python Tools      │
└──────────────────────┬─────────────────────┘
                       │
┌──────────────────────▼─────────────────────┐
│             KNOWLEDGE LAYER                 │
│          RAG + Vector Database              │
└──────────────────────┬─────────────────────┘
                       │
┌──────────────────────▼─────────────────────┐
│              DATA LAYER                     │
│ PostgreSQL + File Storage + Cache           │
└──────────────────────────────────────────────┘
```

---

## 🤖 Agent Architecture

The core intelligence is implemented using LangGraph. Instead of one giant LLM prompt, the system uses specialized nodes.

```
                    User Query
                         │
                         ▼
                 ┌───────────────┐
                 │ Query Analyzer │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ Intent Router  │
                 └───────┬───────┘
                         │
       ┌─────────────────┼──────────────────┐
       │                 │                  │
       ▼                 ▼                  ▼
   Cleaning          Analysis          Visualization
       │                 │                  │
       └─────────────────┼──────────────────┘
                         │
                         ▼
                  ┌─────────────┐
                  │ ML Training │
                  └──────┬──────┘
                         │
                         ▼
                  ┌─────────────┐
                  │ RAG Search  │
                  └──────┬──────┘
                         │
                         ▼
                  ┌─────────────┐
                  │   Response  │
                  └─────────────┘
```

---

## 🔄 LangGraph Workflow

A possible LangGraph state:

```python
AgentState = {
    "user_id": ...,
    "session_id": ...,
    "file_id": ...,
    "dataset": ...,
    "user_query": ...,
    "intent": ...,
    "analysis_plan": ...,
    "generated_code": ...,
    "execution_result": ...,
    "visualization": ...,
    "observations": ...,
    "ml_result": ...,
    "retrieved_context": ...,
    "final_response": ...,
}
```

### LangGraph Nodes

**1. Dataset Loader**
Loads the user's uploaded dataset.
- CSV → Pandas DataFrame
- TXT → Parsed DataFrame/Text

**2. Dataset Profiler**
Creates a dataset profile: shape, columns, data types, missing values, duplicates, statistics, unique values, outliers.

**3. Query Analyzer**
Determines what the user wants.

| User Input | Resulting Intent |
|---|---|
| "Clean my data" | `CLEANING` |
| "Create a correlation heatmap" | `VISUALIZATION` |
| "Predict customer churn" | `MACHINE_LEARNING` |

**4. Planner Node**
Creates an analysis plan. Example for "Clean dataset":
1. Check missing values
2. Check duplicates
3. Detect outliers
4. Check invalid values
5. Correct data types
6. Apply appropriate transformations
7. Generate cleaning report

**5. Data Cleaning Node**
Executes cleaning operations.

**6. Analysis Node**
Performs statistical analysis and EDA.

**7. Visualization Node**
Generates Python visualization code, e.g.:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(...)
plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.show()
```

**8. Python Execution Node**
Executes generated code in a controlled environment. Responsible for:
- Data analysis
- Visualization
- Feature engineering
- Model training
- Metric calculation

> The Python execution environment must be sandboxed to prevent unsafe operations.

**9. ML Node**
Responsible for:
- Target detection
- Problem classification
- Feature preprocessing
- Model training
- Model evaluation
- Model comparison

**10. RAG Node**
Retrieves relevant information from:
- Dataset profile
- Previous analysis
- Previous user questions
- Generated observations
- Cleaning decisions
- Visualization explanations
- ML results
- User session history

**11. Response Node**
Combines all results and generates the final natural-language answer.

---

## 🧠 RAG Architecture

```
User Query
     │
     ▼
Embedding Model
     │
     ▼
Vector Search
     │
     ▼
Relevant Knowledge
     │
     ├── Dataset Information
     ├── Previous Analysis
     ├── User Questions
     ├── Observations
     ├── ML Results
     └── Visualization Results
     │
     ▼
Context + User Query
     │
     ▼
HuggingFace LLM
     │
     ▼
Final Answer
```

---

## 📚 RAG Knowledge Base

The knowledge base can store:

**Dataset Knowledge**
- Dataset description
- Column descriptions
- Data types
- Statistics
- Missing value information

**Analysis Knowledge**
- Analysis performed
- Insights
- Observations
- Relationships
- Correlations

**Visualization Knowledge**
- Chart type
- Chart purpose
- Chart observations
- Generated code

**Machine Learning Knowledge**
- Target variable
- Features
- Problem type
- Algorithms
- Metrics
- Best model
- Results

**Conversation Knowledge**
- User question
- AI answer
- Context
- Actions performed

This allows the chatbot to answer follow-up questions.

**Example:**

> **User:** Which model performed best?
>
> **AI:** Random Forest performed best with an R² score of 0.91.
>
> **User:** Why did it perform better?
>
> *The system retrieves the previous ML analysis from the knowledge base.*

---

## 🔬 Data Processing Pipeline

```
                 Upload Dataset
                       │
                       ▼
                File Validation
                       │
                       ▼
                 Dataset Loading
                       │
                       ▼
                Dataset Profiling
                       │
                       ▼
              Data Quality Report
                       │
                       ▼
             Intelligent Cleaning
                       │
                       ▼
               Preprocessing
                       │
                       ▼
              Prepared Dataset
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
            EDA                 ML
             │                   │
             ▼                   ▼
      Visualizations        Model Training
             │                   │
             └─────────┬─────────┘
                       ▼
                    Insights
                       │
                       ▼
                  RAG Storage
```

---

## 📊 Machine Learning Pipeline

```
User Request
     │
     ▼
Identify Target
     │
     ▼
Determine Task
     │
     ├───────────────┐
     ▼               ▼
Classification    Regression
     │               │
     ▼               ▼
Preprocessing    Preprocessing
     │               │
     ▼               ▼
Train Models     Train Models
     │               │
     └───────┬───────┘
             ▼
       Model Evaluation
             │
             ▼
       Model Comparison
             │
             ▼
        Best Model
             │
             ▼
       Metrics + Report
```

---

## 🎨 Frontend Architecture

The frontend uses HTML, CSS, and JavaScript, following a dashboard-style layout.

---

## 🌐 Web Application Pages

**1. Login Page**
- Logo
- Email
- Password
- Login button
- Register link
- Forgot password

**2. Register Page**
- Full Name
- Username
- Email
- Password
- Confirm Password

**3. Home Dashboard**
- Header
- Sidebar
- Main Workspace
- Chat Sidebar
- Footer

**4. Sidebar** — Navigation:
- Dashboard
- New Analysis
- Datasets
- Analysis History
- Visualizations
- ML Models
- Chat History
- Knowledge Base
- About
- Settings
- Logout

**5. Header**
- AI Data Analyst Logo
- Current Dataset
- Search
- Notifications
- User Profile

**6. Main Workspace** — displays:
- Dataset Summary
- Data Preview
- Data Quality
- Statistics
- Visualizations
- Analysis Results
- ML Results
- Generated Code
- Insights

**7. Right Chat Sidebar** — contains the AI chatbot.

```
┌─────────────────────────────┐
│ 🤖 AI Data Analyst          │
├─────────────────────────────┤
│                             │
│ User: Clean my data         │
│                             │
│ AI: I found 3 issues...     │
│                             │
│ User: Show me a chart       │
│                             │
│ AI: I created a bar chart   │
│                             │
├─────────────────────────────┤
│ Ask about your data...  ➤   │
└─────────────────────────────┘
```

---

## 🖥️ Dashboard Layout

Recommended layout:

```
┌──────────────────────────────────────────────────────────────┐
│                         HEADER                                │
├──────────────┬───────────────────────────────┬────────────────┤
│              │                               │                │
│   SIDEBAR    │       MAIN WORKSPACE          │   AI CHAT      │
│              │                               │                │
│ Dashboard    │ Dataset Overview              │                │
│ Datasets     │                               │ User           │
│ Analysis     │ Data Preview                  │                │
│ Charts       │                               │ AI             │
│ ML Models    │ Visualizations                │                │
│ History      │                               │ User           │
│ Knowledge    │ Insights                      │                │
│ Settings     │                               │ AI             │
│              │ ML Results                    │                │
│              │                               │                │
├──────────────┴───────────────────────────────┴────────────────┤
│                         FOOTER                                 │
└──────────────────────────────────────────────────────────────┘
```

---

## 📈 Visualization Workspace

Charts should be displayed in cards.

```
┌───────────────────────────┐
│ Sales by Region           │
│                           │
│       📊 CHART            │
│                           │
│ Observation:              │
│ West region has the       │
│ highest sales.            │
│                           │
│ [View Code] [Download]    │
└───────────────────────────┘
```

---

## 🗄️ Database Architecture

**Core DBMS:** PostgreSQL (Primary Relational Database)

All application state — including users, active sessions, file metadata, dataset profiles, conversation history, analysis records, visualization metadata, ML trained model records, and knowledge base references — is stored persistently in PostgreSQL using SQLAlchemy ORM.

### 🔑 Prerequisites & User Configuration Requirements

To run this platform:

1. **Hugging Face User Access Token (`HUGGINGFACEHUB_API_TOKEN`)**
   Required for API inference calls (LLM & Embeddings). Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
   Set in `backend/.env`:
   ```
   HUGGINGFACEHUB_API_TOKEN=hf_xxxx...
   ```

2. **PostgreSQL DBMS Instance (`DATABASE_URL`)**
   Active PostgreSQL instance running locally or hosted. Create a database (e.g., `analytica_db`).
   Set connection string in `backend/.env`:
   ```
   DATABASE_URL=postgresql://<user>:<password>@localhost:5432/analytica_db
   ```

3. **Python 3.11+ Environment** (Installed)

4. **LangSmith API Key** (Optional, for LLM tracing/evaluations)

### Database Tables

**`users`**
| Column |
|---|
| id |
| username |
| email |
| password_hash |
| full_name |
| created_at |
| updated_at |

**`sessions`**
| Column |
|---|
| id |
| user_id |
| name |
| created_at |
| updated_at |

**`files`**
| Column |
|---|
| id |
| user_id |
| session_id |
| filename |
| file_type |
| file_path |
| file_size |
| row_count |
| column_count |
| uploaded_at |

**`dataset_profiles`**
| Column |
|---|
| id |
| file_id |
| columns |
| data_types |
| missing_values |
| duplicates |
| statistics |
| profile_json |
| created_at |

**`conversations`**
| Column |
|---|
| id |
| user_id |
| session_id |
| title |
| created_at |
| updated_at |

**`messages`**
| Column |
|---|
| id |
| conversation_id |
| role |
| content |
| created_at |

**`analyses`**
| Column |
|---|
| id |
| session_id |
| file_id |
| analysis_type |
| request |
| code |
| result |
| observations |
| created_at |

**`visualizations`**
| Column |
|---|
| id |
| analysis_id |
| chart_type |
| title |
| code |
| image_path |
| observations |
| created_at |

**`ml_models`**
| Column |
|---|
| id |
| session_id |
| file_id |
| task_type |
| target_column |
| algorithm |
| parameters |
| metrics |
| model_path |
| created_at |

**`knowledge`**
| Column |
|---|
| id |
| user_id |
| session_id |
| source_type |
| content |
| embedding_id |
| metadata |
| created_at |

### 🔄 CRUD Operations

The backend should support CRUD operations for: Users, Sessions, Files, Conversations, Messages, Analyses, Visualizations, ML Models, and Knowledge.

**Example:**

```
POST   /users
GET    /users/{id}
PUT    /users/{id}
DELETE /users/{id}
```

Similar CRUD APIs should exist for datasets, sessions, and analysis records.

---

## 📁 Recommended Project Structure

```
ai-data-analyst/
│
├── backend/
│   │
│   ├── app/
│   │   ├── main.py
│   │   │
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── security.py
│   │   │   └── logging.py
│   │   │
│   │   ├── api/
│   │   │   ├── auth.py
│   │   │   ├── users.py
│   │   │   ├── datasets.py
│   │   │   ├── analysis.py
│   │   │   ├── visualization.py
│   │   │   ├── ml.py
│   │   │   ├── chat.py
│   │   │   └── sessions.py
│   │   │
│   │   ├── models/
│   │   │   ├── user.py
│   │   │   ├── session.py
│   │   │   ├── file.py
│   │   │   ├── message.py
│   │   │   ├── analysis.py
│   │   │   ├── visualization.py
│   │   │   └── ml_model.py
│   │   │
│   │   ├── schemas/
│   │   │   ├── user.py
│   │   │   ├── dataset.py
│   │   │   ├── chat.py
│   │   │   └── analysis.py
│   │   │
│   │   ├── services/
│   │   │   ├── dataset_service.py
│   │   │   ├── preprocessing_service.py
│   │   │   ├── analysis_service.py
│   │   │   ├── visualization_service.py
│   │   │   └── ml_service.py
│   │   │
│   │   ├── agents/
│   │   │   ├── state.py
│   │   │   ├── graph.py
│   │   │   ├── router.py
│   │   │   ├── planner.py
│   │   │   ├── analyst.py
│   │   │   ├── cleaner.py
│   │   │   ├── visualizer.py
│   │   │   ├── ml_agent.py
│   │   │   └── responder.py
│   │   │
│   │   ├── tools/
│   │   │   ├── pandas_tool.py
│   │   │   ├── python_tool.py
│   │   │   ├── visualization_tool.py
│   │   │   ├── statistics_tool.py
│   │   │   └── ml_tool.py
│   │   │
│   │   ├── rag/
│   │   │   ├── embeddings.py
│   │   │   ├── vectorstore.py
│   │   │   ├── retriever.py
│   │   │   └── ingestion.py
│   │   │
│   │   └── db/
│   │       ├── database.py
│   │       └── crud.py
│   │
│   ├── uploads/
│   ├── generated/
│   ├── requirements.txt
│   └── .env
│
├── frontend/
│   │
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── about.html
│   │
│   ├── css/
│   │   ├── style.css
│   │   ├── dashboard.css
│   │   └── chat.css
│   │
│   ├── js/
│   │   ├── auth.js
│   │   ├── dashboard.js
│   │   ├── dataset.js
│   │   ├── chat.js
│   │   ├── visualization.js
│   │   └── api.js
│   │
│   └── assets/
│
├── tests/
│
├── README.md
└── .gitignore
```

---

## 🛠️ Technology Stack

**Frontend**
- HTML5
- CSS3
- JavaScript
- Chart.js / Plotly

**Backend**
- Python
- FastAPI
- Uvicorn
- Pydantic
- SQLAlchemy

**Data Analysis**
- Pandas
- NumPy
- SciPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly

**AI / LLM**
- LangChain
- LangGraph
- LangSmith
- Hugging Face

### Embeddings

Possible models:
- `BAAI/bge-small-en-v1.5`
- `BAAI/bge-base-en-v1.5`
- sentence-transformers models

> The embedding model should be selected based on local hardware and dataset size.

### LLM

Hugging Face models can be used for:
- Query understanding
- Planning
- Code generation
- Insight generation
- RAG responses

The model should ideally be an instruction-following/code-capable model. Examples include Qwen, Llama, Mistral, or DeepSeek. The exact model can be changed through configuration.

---

## 🔗 LangChain Usage

LangChain will be responsible for:

- LLM integration
- Prompt templates
- Tools
- Retrievers
- Embeddings
- Vector stores
- Document processing
- Structured output
- Agent utilities

**Example conceptual flow:**

```
User Query
    ↓
LangChain Prompt
    ↓
HuggingFace LLM
    ↓
Structured Intent
    ↓
LangGraph
```

---

## 🔀 LangGraph Usage

LangGraph is the core orchestration framework. It manages:

- State
- Nodes
- Edges
- Conditional routing
- Agent loops
- Tool execution
- Error handling
- Human approval

**Example:**

```
START
  ↓
Profile Dataset
  ↓
Analyze Query
  ↓
Route Intent
  │
  ├── CLEAN
  ├── ANALYZE
  ├── VISUALIZE
  ├── ML
  └── CHAT
  ↓
Execute Tools
  ↓
Generate Results
  ↓
Store Results
  ↓
RAG Update
  ↓
Generate Response
  ↓
END
```

---

## 🔎 LangSmith Usage

LangSmith is used for:

- Tracing
- Debugging
- Monitoring
- Prompt evaluation
- Agent evaluation
- Latency monitoring
- Token usage
- Error tracking
- Workflow visualization

**Example:**

```
User Query
    ↓
LangGraph
    ↓
Query Analyzer
    ↓
Planner
    ↓
Python Tool
    ↓
Visualization
    ↓
Response
```

Each step can be traced through LangSmith.

---

## 🧠 Hugging Face Integration

The AI layer should use Hugging Face models.

**Possible architecture:**

```
FastAPI
   │
   ▼
LangChain
   │
   ▼
HuggingFace Endpoint / Local Model
   │
   ▼
LLM
```

**Embeddings:**

```
Dataset Knowledge
       ↓
HuggingFace Embedding Model
       ↓
Vector Database
```

---

## 🧪 Code Generation and Execution

A major feature of the system is automatic Python code generation.

**Example request:** "Show the distribution of customer ages."

The LLM generates:

```python
import matplotlib.pyplot as plt

plt.hist(df["age"])
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()
```

The execution engine runs the code and returns: **Chart** + **Statistics** + **Execution result** + **Observation**.

### Important Security Requirement

Generated code must not be executed directly with unrestricted system access.

The Python execution environment should eventually use:
- Sandbox
- Container
- Restricted subprocess
- Resource limits
- Timeouts
- Allowed libraries
- Filesystem restrictions

> This becomes especially important if the application is deployed publicly.

---

## 🔐 Security Architecture

Authentication should use:
- JWT
- Password hashing
- Access tokens
- Refresh tokens

Users should only be able to access their own:
- Datasets
- Sessions
- Chats
- Analysis
- Models
- Knowledge

---

## 📡 API Architecture

```
/api/v1/auth
/api/v1/users
/api/v1/sessions
/api/v1/files
/api/v1/datasets
/api/v1/analysis
/api/v1/visualizations
/api/v1/ml
/api/v1/chat
/api/v1/knowledge
```

### Example APIs

**Authentication**
```
POST /api/v1/auth/register
POST /api/v1/auth/login
POST /api/v1/auth/refresh
```

**Dataset**
```
POST   /api/v1/files/upload
GET    /api/v1/files
GET    /api/v1/files/{id}
DELETE /api/v1/files/{id}
```

**Analysis**
```
POST   /api/v1/analysis
GET    /api/v1/analysis/{id}
DELETE /api/v1/analysis/{id}
```

**Chat**
```
POST   /api/v1/chat
GET    /api/v1/chat/history
DELETE /api/v1/chat/{id}
```

**Visualization**
```
POST /api/v1/visualizations
GET  /api/v1/visualizations
GET  /api/v1/visualizations/{id}
```

**Machine Learning**
```
POST   /api/v1/ml/train
GET    /api/v1/ml/models
GET    /api/v1/ml/models/{id}
DELETE /api/v1/ml/models/{id}
```

---

## 🧩 Example User Workflows

### Workflow 1 — Upload Dataset

```
User
 ↓
Upload CSV
 ↓
FastAPI
 ↓
Validate File
 ↓
Store File
 ↓
Load Pandas DataFrame
 ↓
Profile Dataset
 ↓
Generate Dataset Report
 ↓
Store Metadata
 ↓
Display Dashboard
```

### Workflow 2 — Clean Dataset

**User:** "Clean my dataset."

```
Query Analyzer
      ↓
Cleaning Intent
      ↓
Dataset Profiler
      ↓
Cleaning Planner
      ↓
Missing Values
      ↓
Duplicates
      ↓
Outliers
      ↓
Invalid Values
      ↓
Data Types
      ↓
Clean Dataset
      ↓
Cleaning Report
      ↓
Store Result
      ↓
RAG Knowledge Base
```

### Workflow 3 — Visualization

**User:** "Show me meaningful visualizations."

```
Dataset Profile
      ↓
Visualization Planner
      ↓
Select Chart Types
      ↓
Generate Python Code
      ↓
Execute Code
      ↓
Generate Charts
      ↓
Generate Observations
      ↓
Store Results
      ↓
Display Dashboard
```

### Workflow 4 — Specific Chart

**User:** "Create a scatter plot between price and sales."

```
Query
 ↓
Intent Detection
 ↓
Column Validation
 ↓
Chart Planning
 ↓
Python Code
 ↓
Execution
 ↓
Chart
 ↓
Observation
```

### Workflow 5 — Machine Learning

**User:** "Train a model to predict house prices."

```
Identify Target
       ↓
Detect Regression
       ↓
Feature Analysis
       ↓
Preprocessing
       ↓
Train Multiple Models
       ↓
Evaluate
       ↓
Compare
       ↓
Select Best Model
       ↓
Generate Report
       ↓
Store Model
       ↓
Display Metrics
```

### Workflow 6 — RAG Chat

**User:** "Why did you remove these rows?"

```
User Query
    ↓
Embedding
    ↓
Vector Search
    ↓
Retrieve Cleaning History
    ↓
Retrieve Dataset Profile
    ↓
LLM
    ↓
Explanation
```

---

## 💾 Session-Based Architecture

Every analysis should belong to a session.

**Example:**

```
User
 │
 ├── Session 1: Sales Analysis
 │      ├── sales.csv
 │      ├── Cleaning
 │      ├── EDA
 │      ├── Charts
 │      ├── ML Model
 │      └── Chat
 │
 ├── Session 2: Customer Analysis
 │      ├── customers.csv
 │      ├── Cleaning
 │      ├── EDA
 │      └── Chat
 │
 └── Session 3: Marketing Analysis
        └── marketing.csv
```

This allows users to return to previous analyses.

---

## 🧠 Intelligent Decision Making

One of the most important goals of the project is: **the AI should make analytical decisions based on the dataset rather than blindly applying predefined operations.**

**Example — missing values:**

```
IF missing values exist
        ↓
Analyze column type
        ↓
Analyze missing percentage
        ↓
Analyze distribution
        ↓
Select suitable strategy
```

**Example — outliers:**

```
IF outliers exist
        ↓
Determine whether outliers are legitimate
        ↓
Check distribution
        ↓
Select treatment
```

**Example — visualization:**

```
IF user asks for visualization
        ↓
Understand dataset
        ↓
Understand requested objective
        ↓
Select meaningful chart
```

---

## 📋 Analysis Report

Every major analysis should generate a structured report:

```
Dataset Overview
----------------
Rows:
Columns:
Numerical Features:
Categorical Features:

Data Quality
------------
Missing Values:
Duplicates:
Outliers:

Preprocessing
-------------
Operations Applied:

EDA
---
Important Statistics:

Visualizations
--------------
Charts Generated:

Insights
--------
1.
2.
3.

Recommendations
---------------
1.
2.
3.
```

---

## 💡 Recommendation Engine

The AI should provide recommendations such as:

> Your dataset contains significant missing values in three columns.
>
> **Recommendation:** Investigate why these values are missing before performing model training.

> The target variable is highly imbalanced.
>
> **Recommendation:** Consider class weighting, resampling, or appropriate imbalance-aware metrics.

> Two features have very high correlation.
>
> **Recommendation:** Consider removing one of them to reduce multicollinearity.

---

## 📊 Dashboard Components

The dashboard should contain cards such as:

```
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Rows        │ │ Columns     │ │ Missing     │
│ 50,000      │ │ 14          │ │ 3.2%        │
└─────────────┘ └─────────────┘ └─────────────┘

┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Duplicates  │ │ Numeric     │ │ Categorical │
│ 184         │ │ 8           │ │ 4           │
└─────────────┘ └─────────────┘ └─────────────┘
```

Followed by:
- Data Preview
- Data Quality
- Charts
- Insights
- ML Results
- Generated Code

---

## 🛠️ Development Phases

The project should be implemented incrementally.

### Phase 1 — Project Setup
**Goal:** Create the basic project.
- Create Git repository
- Create backend
- Create frontend
- Setup Python environment
- Setup FastAPI
- Setup database
- Setup environment variables
- Configure CORS
- Create initial README

### Phase 2 — Authentication
**Goal:** Implement user management.
- User model
- Registration
- Login
- Password hashing
- JWT authentication
- Protected APIs
- Logout
- User profile

### Phase 3 — Dataset Upload
**Goal:** Allow users to upload datasets.
- CSV upload
- TXT upload
- File validation
- File storage
- Dataset loading
- Dataset preview
- File database records

### Phase 4 — Dataset Profiling
**Goal:** Automatically understand uploaded datasets.
- Shape detection
- Data types
- Missing values
- Duplicate detection
- Statistical summary
- Unique values
- Cardinality
- Outlier detection
- Dataset profile generation

### Phase 5 — Data Cleaning Engine
**Goal:** Build intelligent preprocessing.
- Missing-value handling
- Duplicate removal
- Outlier detection
- Invalid-value detection
- Data type correction
- Cleaning report
- Save cleaned dataset

### Phase 6 — LangChain Integration
**Goal:** Connect Hugging Face models.
- Configure Hugging Face LLM
- Configure embedding model
- Create prompts
- Create structured outputs
- Create LangChain tools
- Create dataset analysis chain

### Phase 7 — LangGraph Agent
**Goal:** Create the central AI workflow.
- Define AgentState
- Dataset profiler node
- Query analyzer
- Intent router
- Planner
- Cleaning node
- Analysis node
- Visualization node
- ML node
- Response node
- Error handling
- Conditional routing

### Phase 8 — Python Analysis Engine
**Goal:** Allow AI to generate and execute analysis code.
- Pandas tool
- Statistics tool
- Visualization tool
- Python execution tool
- Code validation
- Execution timeout
- Error recovery
- Result extraction

### Phase 9 — Visualization Engine
**Goal:** Automatically generate charts.
- Automatic chart selection
- Bar chart
- Line chart
- Histogram
- Scatter plot
- Box plot
- Heatmap
- Pie chart
- Custom visualization requests
- Chart observations
- Save charts

### Phase 10 — Machine Learning Engine
**Goal:** Enable AI-driven model training.
- Target detection
- Classification detection
- Regression detection
- Feature preprocessing
- Model training
- Model comparison
- Metrics
- Best model selection
- Model persistence
- ML report

### Phase 11 — RAG System
**Goal:** Build persistent analytical memory.
- Document creation
- Chunking
- Embeddings
- Vector database
- Retriever
- Metadata filtering
- Dataset-specific retrieval
- Session-specific retrieval
- Conversation retrieval
- Analysis-result retrieval

### Phase 12 — Chatbot
**Goal:** Create the conversational data analyst.
- Chat UI
- Chat API
- Conversation storage
- Context retrieval
- RAG integration
- LangGraph integration
- Streaming responses
- Follow-up questions
- Session memory

### Phase 13 — Frontend Dashboard
**Goal:** Build the complete user interface.
- Login
- Register
- Dashboard
- Sidebar
- Header
- Footer
- Dataset upload
- Dataset preview
- Charts
- Analysis results
- ML results
- Chat sidebar
- History
- About page
- Responsive design

### Phase 14 — Database CRUD
**Goal:** Persist everything.
- User CRUD
- Session CRUD
- File CRUD
- Conversation CRUD
- Message CRUD
- Analysis CRUD
- Visualization CRUD
- ML model CRUD
- Knowledge CRUD

### Phase 15 — LangSmith
**Goal:** Monitor and evaluate the AI system.
- LangSmith project
- Tracing
- Agent traces
- Tool traces
- Error monitoring
- Prompt evaluation
- Response evaluation
- Agent performance evaluation

### Phase 16 — Testing

Testing should cover:
- Unit Tests
- Integration Tests
- API Tests
- Database Tests
- Agent Tests
- RAG Tests
- ML Tests
- Frontend Tests
- Security Tests

**Example tests:**
- Upload valid CSV → success
- Upload invalid file → rejected
- Clean dataset → expected transformations
- Generate visualization → chart created
- Ask dataset question → correct answer
- Train model → metrics generated
- Unauthorized user → access denied

### Phase 17 — Deployment

Possible deployment architecture:

```
                 Internet
                    │
                    ▼
             Nginx / Reverse Proxy
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
       Frontend             FastAPI
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
                PostgreSQL   Vector DB    Worker
```

**Potential deployment technologies:**
- Docker
- Docker Compose
- Nginx
- PostgreSQL
- Redis
- Cloud VM

---

## 🔮 Future Enhancements

Future versions can support:

- Excel
- JSON
- Parquet
- SQL databases
- Multiple datasets
- Data merging
- Data joining
- Time-series forecasting
- Deep learning
- Automated feature engineering
- AutoML
- Natural language SQL
- Report generation
- PDF export
- PowerPoint export
- Excel report generation
- Data drift detection
- Model monitoring
- Scheduled analysis
- Multi-agent architecture
- Voice-based data analysis

---

## 🧠 Final Agent Concept

The final AI Data Analyst should behave like a real data analyst.

**Example:**

> **User:** Analyze my sales data.
>
> **AI:** I analyzed the dataset.
>
> **Dataset:** 50,000 rows × 14 columns
>
> **Data Quality:**
> - 3.2% missing values
> - 184 duplicate rows
> - 3 columns contain potential outliers
>
> **Important Findings:**
> - West region has the highest revenue.
> - Sales increased significantly during Q4.
> - Advertising spend has a positive relationship with sales.
>
> **Recommended Visualizations:**
> - Sales by region
> - Monthly sales trend
> - Advertising vs sales
> - Correlation heatmap

Then the user can continue the conversation:

> **User:** Clean the dataset.
> *(The system performs cleaning.)*
>
> **User:** Show me visualizations.
> *(The system generates charts.)*
>
> **User:** Train a model to predict sales.
> *(The system trains models.)*
>
> **User:** Why did you choose Random Forest?
> *(The RAG system retrieves the previous model-training result and explains the decision.)*

---

## 🏆 Final Architecture

```
                         USER
                          │
                          ▼
                ┌──────────────────┐
                │   Web Interface  │
                │    HTML/CSS/JS   │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │      FastAPI     │
                │    REST APIs     │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │    LangGraph     │
                │ Agent Orchestrator│
                └────────┬─────────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
              ▼          ▼          ▼
          ANALYSIS    CLEANING     ML
              │          │          │
              └──────────┼──────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  LangChain Tools │
                │ Pandas / Python  │
                │ Visualization    │
                │ Statistics / ML  │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ HuggingFace LLM  │
                │ + Embeddings     │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │       RAG        │
                │ Knowledge Base   │
                └────────┬─────────┘
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       PostgreSQL              Vector Database
             │                       │
             └───────────┬───────────┘
                         ▼
                  FINAL RESPONSE
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          Insights     Charts      ML Results
```

---

## 🎯 Final Project Goal

The final system should provide an experience similar to having a virtual data analyst.

Instead of requiring the user to know Python, Pandas, statistics, and machine learning, the user communicates with the system using natural language.

The AI should:

```
Understand
    ↓
Plan
    ↓
Analyze
    ↓
Execute
    ↓
Visualize
    ↓
Explain
    ↓
Remember
    ↓
Recommend
```

The combination of **FastAPI** + **LangChain** + **LangGraph** + **LangSmith** + **Hugging Face** + **RAG** + **Pandas** + **Scikit-learn** + **PostgreSQL** + **HTML/CSS/JavaScript** creates a complete agentic AI Data Analytics platform capable of performing end-to-end data analysis through natural-language interaction.

---

## 📌 Recommended Implementation Order

Do not start by building the entire system simultaneously. Build it in this order:

1. FastAPI + Database
2. Authentication
3. File Upload
4. Pandas Dataset Profiling
5. Data Cleaning Engine
6. EDA Engine
7. Visualization Engine
8. ML Engine
9. HuggingFace LLM
10. LangChain Tools
11. LangGraph Agent
12. RAG
13. Chatbot
14. Frontend Dashboard
15. LangSmith
16. Testing
17. Docker + Deployment

This order allows each component to be tested independently before introducing the next layer.

---

## 🚀 Project Vision

AI Data Analyst is not simply a chatbot that talks about data. It is an agentic data-analysis system that can:

**Understand → Decide → Execute → Analyze → Visualize → Train → Explain → Remember → Recommend**

using LangChain, LangGraph, RAG, and Hugging Face models.

The ultimate goal is to make complex data analysis accessible through a simple conversational interface while maintaining reproducible analysis, persistent history, explainable decisions, and measurable machine-learning results.
