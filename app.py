from flask import Flask, request, jsonify
import shutil
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import ast
import urllib
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import requests

# Load environment variables
load_dotenv()

# Configure GenAI Key
api_key = os.getenv("GENAI_API_KEY")
genai.configure(api_key=api_key)

# Temporary CSV file path
TEMP_CSV = "temp_data.csv"

# Initialize Flask app
app = Flask(__name__)
CORS(app) 

# # Function to generate Python code using LLM
# def generate_code(prompt):
#     try:
#         model = genai.GenerativeModel('gemini-1.5-pro')
#         response = model.generate_content(prompt)
#         response_text = response.text.strip()
#         response_text = response_text.strip("").strip("```python").strip()
#         print(response_text)
#         return response_text
#     except Exception as e:
#         return jsonify({"error": f"Error generating code: {e}"}), 500

# Function to generate Python code using Ollama
def generate_code(prompt):
    try:
        # Ollama API endpoint (running locally)
        url = "http://localhost:11434/api/generate"
        
        # Payload for the API call
        payload = {
            # "model": "mistral",  # or "codellama"
            "model": "codellama",  
            "prompt": prompt,
            "stream": False  # Set to False to get the full response at once
        }
        
        # Make the API call
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Parse the response
        response_data = response.json()
        generated_code = response_data.get("response", "").strip()
        
        # Clean up the generated code
        generated_code = generated_code.strip("```python").strip("```").strip()
        # generated_code = generated_code.replace("```python", "").replace("```", "").strip()
        
        print("Generated Code:", generated_code)
        return generated_code
    except Exception as e:
        print(f"Error generating code: {e}")
        return None

def create_dynamic_prompt(df, dq_checks_input):
    sample_data = df.head(5).to_dict(orient='records')  # Take the first 5 rows as a sample
    sample_data_str = "\n".join([str(row) for row in sample_data])
    
    prompt = f"""
    We have a DataFrame with the following columns: {', '.join(df.columns)}. The following data quality checks need to be applied to the entire DataFrame:

    {dq_checks_input}

    Here is a sample of the data:

    {sample_data_str}

    Please generate Python code to apply these DQ checks to the entire DataFrame df. The code should:
    - Load the df from a csv file named {TEMP_CSV}.
    - Process all rows of the DataFrame df, not just a sample.
    - Ensure that the operations apply to the entire DataFrame without limitations.
    - Avoid any hardcoded limits or specific conditions that restrict the operation to a subset of the data.
    - The resultant df is to be overwritten to a csv file named {TEMP_CSV}.
    - If there is last update like column then update the values to current datetime.
    - Only generate python do not generate extra text,information or comments.
    """
    return prompt

def validate_code(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        return False

def debug_code_execution(df, code):
    local_context = {'df': df.copy(), 'pd': pd, 'np': np}
    try:
        exec(code, {}, local_context)
    except Exception as e:
        return None

    return local_context.get('df', df)

# Save DataFrame to temporary CSV
def save_to_temp_csv(df):
    df.to_csv(TEMP_CSV, index=False)
    return True

# Load DataFrame from temporary CSV, or initialize it if CSV doesn't exist
def load_or_initialize_temp_csv(df):
    if os.path.exists(TEMP_CSV):
        return pd.read_csv(TEMP_CSV)
    else:
        save_to_temp_csv(df)  # Create the temp CSV with the original data
        return df

# Function to execute DQ checks with retries and improved debugging
def execute_dq_checks_with_retry(df, dq_checks_input, max_attempt=11):
    attempt = 0
    while attempt < max_attempt:
        try:
            prompt = create_dynamic_prompt(df, dq_checks_input)
            dq_check_code = generate_code(prompt)
            if not dq_check_code:
                return None, None

            if not validate_code(dq_check_code):
                return None, None

            df_checked = debug_code_execution(df, dq_check_code)

            if df_checked is not None:
                save_to_temp_csv(df_checked)  # Save changes to temporary CSV
                return df_checked, dq_check_code
            else:
                return None, None
        except Exception as e:
            attempt += 1
    return None, None

def connect_to_sqlserver(db_name):
    try:
        server_name = r"CSGVTVLT199\SQLEXPRESS"
        connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server_name};DATABASE={db_name};Trusted_Connection=yes;"
        connection_uri = f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(connection_string)}"
        
        engine = create_engine(connection_uri)
        return engine
    except SQLAlchemyError as e:
        return None

# Function to get list of databases
def get_databases():
    try:
        admin_engine = connect_to_sqlserver("master")  # Connect to the master database to list all databases
        with admin_engine.connect() as connection:
            result = connection.execute(text("SELECT name FROM sys.databases WHERE database_id > 4;"))  # Exclude system databases
            databases = [row[0] for row in result]
        return databases
    except Exception as e:
        return []

# Function to fetch list of schemas
def get_schemas(engine):
    try:
        query = "SELECT schema_name FROM information_schema.schemata;"
        with engine.connect() as connection:
            result = connection.execute(text(query))
            schemas = [row[0] for row in result]
        return schemas
    except Exception as e:
        return []

# Function to fetch list of tables in a specific schema
def get_tables(engine, schema):
    try:
        query = """
        SELECT table_name 
        FROM information_schema.tables
        WHERE table_schema = :schema
          AND table_type = 'BASE TABLE';
        """
        with engine.connect() as connection:
            result = connection.execute(text(query), {'schema': schema})
            tables = [row[0] for row in result]
        return tables
    except Exception as e:
        return []

# Function to load data from a specific table in PostgreSQL
def load_data_from_query(engine, query):
    try:
        with engine.connect() as connection:
            result = connection.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        return df
    except Exception as e:
        return None

# Function to save DataFrame to microsoft SQL Server
def save_df_to_db(engine, df, table_name):
    try:
        df.to_sql(table_name, engine, index=False, if_exists='replace')
        return True
    except Exception as e:
        return False

# API Endpoints

@app.route('/databases', methods=['GET'])
def list_databases():
    databases = get_databases()
    return jsonify(databases)

@app.route('/schemas', methods=['GET'])
def list_schemas():
    db_name = request.args.get('db_name')
    if not db_name:
        return jsonify({"error": "Database name is required"}), 400
    engine = connect_to_sqlserver(db_name)
    if not engine:
        return jsonify({"error": "Unable to connect to the database"}), 500
    schemas = get_schemas(engine)
    return jsonify(schemas)

@app.route('/tables', methods=['GET'])
def list_tables():
    db_name = request.args.get('db_name')
    schema = request.args.get('schema')
    if not db_name or not schema:
        return jsonify({"error": "Database name and schema are required"}), 400
    engine = connect_to_sqlserver(db_name)
    if not engine:
        return jsonify({"error": "Unable to connect to the database"}), 500
    tables = get_tables(engine, schema)
    return jsonify(tables)

@app.route('/data', methods=['GET'])
def get_table_data():
    db_name = request.args.get('db_name')
    schema = request.args.get('schema')
    table = request.args.get('table')

    if not db_name or not schema or not table:
        return jsonify({"error": "Database name, schema, and table are required"}), 400

    engine = connect_to_sqlserver(db_name)
    if not engine:
        return jsonify({"error": "Unable to connect to the database"}), 500

    query = f"SELECT * FROM {schema}.{table}"
    df = load_data_from_query(engine, query)

    if df is None:
        return jsonify({"error": "Failed to load data from the table"}), 500

    # Save data to temp CSV
    load_or_initialize_temp_csv(df)

    return jsonify(df.to_dict(orient='records'))

@app.route('/dq_checks', methods=['POST'])
def apply_dq_checks():
    try:
        data = request.json
        db_name = data.get('db_name')
        schema = data.get('schema')
        table = data.get('table')
        dq_checks_input = data.get('dq_checks_input')

        # Validate input
        if not db_name or not schema or not table or not dq_checks_input:
            return jsonify({"error": "Database name, schema, table, and DQ checks input are required"}), 400

        # Connect to the database
        engine = connect_to_sqlserver(db_name)
        if not engine:
            return jsonify({"error": "Unable to connect to the database"}), 500

        # Fetch data from the table
        query = f"SELECT * FROM {schema}.{table}"
        df = load_data_from_query(engine, query)
        if df is None:
            return jsonify({"error": "Failed to load data from the table"}), 500

        # Apply DQ checks
        df_checked, dq_check_code = execute_dq_checks_with_retry(df, dq_checks_input)
        if df_checked is None:
            return jsonify({"error": "Failed to apply data quality checks"}), 500

        # Return the result
        return jsonify({
            "data": df_checked.to_dict(orient='records'),
            "code": dq_check_code
        })

    except Exception as e:
        # Log the error
        print(f"Error in /dq_checks: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/save', methods=['POST'])
def save_data():
    data = request.json
    db_name = data.get('db_name')
    schema = data.get('schema')
    table = data.get('table')
    new_table_name = data.get('new_table_name')
    if not db_name or not schema or not table:
        return jsonify({"error": "Database name, schema, and table are required"}), 400
    engine = connect_to_sqlserver(db_name)
    if not engine:
        return jsonify({"error": "Unable to connect to the database"}), 500
    df_final = pd.read_csv(TEMP_CSV)
    if new_table_name:
        table_name = new_table_name
    else:
        table_name = table
    success = save_df_to_db(engine, df_final, table_name)
    if not success:
        return jsonify({"error": "Failed to save data to the database"}), 500
    if os.path.exists(TEMP_CSV):
        os.remove(TEMP_CSV)
    return jsonify({"message": "Data saved successfully"})

if __name__ == '__main__':
    app.run(debug=True)