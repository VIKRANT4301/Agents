from typing import Dict, Any, List, Optional
import re
import json
import datetime
from sqlalchemy.orm import Session
from sqlalchemy.sql import text
from app.utils.logger import logger
from app.llm.client import llm_client
from app.utils.config import settings

SCHEMA_PROMPT = """
You are a SQL generator for a manufacturing assistant system. Given a natural language question, your job is to output a single valid SQL query matching the question.

Here is the database schema:

1. Table: users
   Columns:
     - id: INTEGER (primary key)
     - username: VARCHAR(100) (unique)
     - role: VARCHAR(50) ('OPERATOR', 'ENGINEER', 'PLANT_MANAGER')
     - plant_id: VARCHAR(50)
     - is_active: BOOLEAN

2. Table: lines
   Columns:
     - id: VARCHAR(50) (primary key)
     - name: VARCHAR(100)
     - status: VARCHAR(50) ('ACTIVE', 'MAINTENANCE', 'CHANGEOVER')

3. Table: machines
   Columns:
     - id: VARCHAR(50) (primary key)
     - line_id: VARCHAR(50) (foreign key to lines.id)
     - name: VARCHAR(100)
     - type: VARCHAR(50)
     - manufacturer: VARCHAR(100)
     - model_number: VARCHAR(100)
     - last_maintenance_date: DATETIME

4. Table: recipes
   Columns:
     - id: INTEGER (primary key)
     - sku: VARCHAR(100) (unique)
     - description: TEXT
     - target_yield_rate: VARCHAR(50)
     - scrap_tolerance_pct: FLOAT

5. Table: schedules
   Columns:
     - id: INTEGER (primary key)
     - line_id: VARCHAR(50) (foreign key to lines.id)
     - source_sku: VARCHAR(100)
     - target_sku: VARCHAR(100)
     - batch_size: INTEGER
     - scheduled_date: DATETIME
     - status: VARCHAR(50) ('SCHEDULED', 'PENDING_APPROVAL', 'IN_PROGRESS', 'COMPLETED')

6. Table: changeover_logs
   Columns:
     - id: INTEGER (primary key)
     - line_id: VARCHAR(50) (foreign key to lines.id)
     - source_sku: VARCHAR(100)
     - target_sku: VARCHAR(100)
     - operator_id: INTEGER (foreign key to users.id)
     - actual_duration_minutes: INTEGER
     - setup_errors_count: INTEGER
     - waste_produced_kg: FLOAT
     - energy_consumed_kwh: FLOAT
     - started_at: DATETIME
     - completed_at: DATETIME
     - status: VARCHAR(50) ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')

7. Table: defect_history
   Columns:
     - id: INTEGER (primary key)
     - changeover_log_id: INTEGER (foreign key to changeover_logs.id)
     - defect_type: VARCHAR(100)
     - rejections_count: INTEGER
     - severity: VARCHAR(20) ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
     - notes: TEXT
     - scrap_cost_usd: FLOAT
     - downtime_minutes: INTEGER
     - reported_at: DATETIME

8. Table: operator_skills
   Columns:
     - id: INTEGER (primary key)
     - operator_id: INTEGER (foreign key to users.id)
     - machine_id: VARCHAR(50) (foreign key to machines.id)
     - skill_level: INTEGER (1 = Basic, 2 = Competent, 3 = Expert)
     - years_experience: FLOAT
     - certification_status: VARCHAR(50)

9. Table: approvals
   Columns:
     - id: INTEGER (primary key)
     - resource_type: VARCHAR(50) ('RECIPE_CHANGE', 'SEQUENCE_OVERRIDE')
     - resource_id: INTEGER
     - engineer_id: INTEGER (foreign key to users.id)
     - status: VARCHAR(50) ('DRAFT', 'UNDER_REVIEW', 'APPROVED', 'REJECTED')
     - signature_hash: VARCHAR(256)
     - created_at: DATETIME
     - updated_at: DATETIME

10. Table: audit_logs
    Columns:
      - id: INTEGER (primary key)
      - timestamp: DATETIME
      - action_type: VARCHAR(100)
      - details: TEXT
      - user_id: INTEGER (foreign key to users.id)
      - tamper_hash: VARCHAR(256)

Instructions:
- Write a valid, standard SQL query.
- Use explicit column names when possible.
- If joining tables, use correct join conditions (e.g. `changeover_logs JOIN users ON changeover_logs.operator_id = users.id`).
- Output ONLY the clean SQL query. Do not explain anything. Do not wrap in markdown quotes if possible, or if you must wrap, use ```sql ... ``` format.
- Output ONLY a SELECT statement. Any query executing WRITE operations (INSERT, UPDATE, DELETE, ALTER, etc.) is strictly prohibited.
"""

class NL2SQLAgent:
    """
    Agent responsible for translating Natural Language questions into SQL queries,
    running them safely against the SQLite/PostgreSQL database, and returning the dataset.
    """

    def generate_and_execute(self, query: str, db: Session) -> Dict[str, Any]:
        logger.info(f"NL2SQL Agent executing query: '{query}'")
        
        # Step 1: Generate SQL statement
        sql_query = ""
        provider = settings.LLM_PROVIDER.lower()
        
        if provider == "mock":
            sql_query = self._generate_mock_sql(query)
        else:
            prompt = f"Question: {query}\nSQL Query:"
            try:
                raw_response = llm_client.call_primary_llm(prompt, SCHEMA_PROMPT)
                sql_query = self._clean_sql_response(raw_response)
            except Exception as e:
                logger.error(f"Failed to generate SQL via primary LLM: {e}")
                # Fallback to mock generation if LLM fails
                sql_query = self._generate_mock_sql(query)

        logger.info(f"Generated SQL: {sql_query}")

        # Step 2: Validate SQL safety (Read-Only validation)
        if not self._is_query_safe(sql_query):
            err_msg = "Security check failed: Only read-only SELECT statements are allowed."
            logger.warning(f"NL2SQL safety block on query: '{sql_query}'")
            return {
                "sql": sql_query,
                "columns": [],
                "rows": [],
                "error": err_msg
            }

        # Step 3: Execute query against database
        try:
            result = db.execute(text(sql_query))
            
            # For some queries, result.keys() might not be populated or the query returns no rows
            columns = list(result.keys()) if result.returns_rows else []
            rows = []
            
            if result.returns_rows:
                for row in result.all():
                    row_dict = {}
                    for col in columns:
                        val = getattr(row, col)
                        # Handle datetime serialization
                        if isinstance(val, (datetime.datetime, datetime.date)):
                            row_dict[col] = val.isoformat()
                        else:
                            row_dict[col] = val
                    rows.append(row_dict)
                    
            return {
                "sql": sql_query,
                "columns": columns,
                "rows": rows,
                "error": None
            }
        except Exception as e:
            logger.error(f"Failed to execute SQL query '{sql_query}': {e}")
            return {
                "sql": sql_query,
                "columns": [],
                "rows": [],
                "error": f"Database Error: {str(e)}"
            }

    def _generate_mock_sql(self, query: str) -> str:
        """
        Hardcoded mapper for testing and mock mode.
        """
        q = query.lower()
        if "user" in q or "operator" in q or "people" in q:
            return "SELECT username, role, is_active FROM users;"
        elif "average" in q or "avg" in q or "duration" in q or "how long" in q:
            return "SELECT AVG(actual_duration_minutes) as average_duration FROM changeover_logs WHERE status = 'COMPLETED';"
        elif "defect" in q or "reject" in q or "rejections" in q or "scrap" in q or "waste" in q:
            return "SELECT defect_type, SUM(rejections_count) as total_rejections FROM defect_history GROUP BY defect_type;"
        elif "machine" in q or "maintenance" in q:
            return "SELECT id, name, type, manufacturer, last_maintenance_date FROM machines ORDER BY last_maintenance_date ASC;"
        elif "schedule" in q or "planned" in q or "job" in q:
            return "SELECT line_id, source_sku, target_sku, batch_size, status FROM schedules;"
        elif "log" in q or "history" in q or "record" in q:
            return "SELECT cl.id, cl.line_id, cl.source_sku, cl.target_sku, u.username, cl.actual_duration_minutes, cl.status FROM changeover_logs cl JOIN users u ON cl.operator_id = u.id ORDER BY cl.started_at DESC LIMIT 10;"
        else:
            return "SELECT * FROM changeover_logs WHERE line_id = 'Line-3' ORDER BY started_at DESC LIMIT 5;"

    def _clean_sql_response(self, raw: str) -> str:
        clean = raw.strip()
        # Remove markdown codeblock tags if returned by LLM
        if clean.startswith("```sql"):
            clean = clean[6:]
        elif clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        return clean.strip()

    def _is_query_safe(self, sql: str) -> bool:
        sql_upper = sql.upper().strip()
        if not sql_upper.startswith("SELECT"):
            return False
            
        mutating_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE", "TRUNCATE", "GRANT", "REVOKE"]
        for keyword in mutating_keywords:
            pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(pattern, sql_upper):
                return False
        return True

nl2sql_agent = NL2SQLAgent()
