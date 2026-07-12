from typing import Dict, Any, List, Optional
from app.utils.logger import logger
from app.llm.client import llm_client
from app.database.postgres.vector_client import vector_db
import json

class SequenceToolPlannerAgent:
    """
    Sequence & Tool Planner Agent (Consolidates Planning, Tools, and Duration).
    1. Retrieves line changeover SOPs from the vector store.
    2. Builds the step-by-step changeover checklist matching target SKU machine requirements.
    3. Suggests physical tools/materials required.
    4. Predicts changeover duration adjusted for operator capability.
    """
    
    def generate_plan(self, line_id: str, source_sku: str, target_sku: str, operator_skill_level: int,
                      telemetry: Optional[Any] = None) -> Dict[str, Any]:
        """
        Executes sequence generation by gathering RAG documents and querying the LLM.
        """
        logger.info(f"Planner Agent starting sequence generation for {line_id} from {source_sku} to {target_sku}")
        
        # Step 1: Retrieve manuals / SOPs via RAG
        rag_query = f"SOP setup manuals for {target_sku} on {line_id} changeover"
        rag_contexts = vector_db.search(rag_query, top_k=2)
        
        sop_text = "\n".join([f"- [{doc['category']}] {doc['title']}: {doc['text']}" for doc in rag_contexts])
        
        # Step 2: Formulate prompt for ollama model
        system_prompt = (
            "You are the Sequence & Tool Planner Agent for a manufacturing changeover assistant. "
            "Your task is to generate a step-by-step setup checklist, identify tools/materials, "
            "and calculate expected duration. You must output ONLY a valid JSON object matching the requested schema."
        )
        
        prompt = f"""
        Generate a detailed changeover sequence plan.
        
        CONTEXT AND CONSTRAINTS:
        - Line ID: {line_id}
        - Current SKU (Source): {source_sku}
        - Target SKU (Destination): {target_sku}
        - Operator Skill Rating (1=Basic, 2=Competent, 3=Expert): {operator_skill_level}
        
        SOP REFERENCE MATERIAL (RAG):
        {sop_text}
        
        DURATION CALCULATION RULES:
        - Base duration: 30 minutes.
        - If Operator Skill is 1 (Basic): Add 15 minutes.
        - If Operator Skill is 3 (Expert): Deduct 5 minutes.
        
        OUTPUT FORMAT (JSON):
        Provide a JSON object with:
        1. "explanation": A text summary of the changeover layout.
        2. "duration_minutes": Predicted duration (integer).
        3. "tools": A list of physical tool names.
        4. "steps": An array of steps, where each step has:
           - "index" (int)
           - "step_id" (str)
           - "machine_id" (str)
           - "instruction" (str)
           - "target_value" (str)
           - "required_tool" (str)
           - "safety_warning" (str)
        
        Begin JSON response:
        """
        
        # Step 3: Run LLM call
        raw_response = llm_client.call_primary_llm(prompt, system_prompt)
        
        if telemetry:
            telemetry.add_mock_tokens_for_prompt(prompt, raw_response)
        
        # Step 4: Parse response and return structured dict
        try:
            # Strip markdown code blocks if the model returned them
            clean_str = raw_response.strip()
            if clean_str.startswith("```json"):
                clean_str = clean_str[7:]
            if clean_str.endswith("```"):
                clean_str = clean_str[:-3]
            clean_str = clean_str.strip()
            
            plan_data = json.loads(clean_str)
            logger.info("Planner Agent plan successfully generated and parsed.")
            return plan_data
        except Exception as e:
            logger.error(f"Failed to parse LLM planner JSON output: {e}. Raw: {raw_response[:200]}")
            
            # Sturdy fallback mapping in case of JSON parse failure
            fallback_duration = 45 if operator_skill_level == 1 else (25 if operator_skill_level == 3 else 30)
            return {
                "explanation": f"Standard changeover sequence for {line_id}. Duration calculated with fallback rules.",
                "duration_minutes": fallback_duration,
                "tools": ["LOTO Lockout Kit", "Hex Wrench Set", "Adjustable Spanner", "Purging Compound HDPE-P"],
                "steps": [
                    {
                        "index": 1,
                        "step_id": "LOTO-01",
                        "machine_id": "L3-EXTRUDER",
                        "instruction": "Initiate LOTO procedure. Isolate power breaker L3-E1.",
                        "target_value": "LOCKED",
                        "required_tool": "LOTO Lock Kit #4",
                        "safety_warning": "DANGER: High electrical voltage."
                    },
                    {
                        "index": 2,
                        "step_id": "PURGE-02",
                        "machine_id": "L3-EXTRUDER",
                        "instruction": "Purge extruder barrel with purging compound.",
                        "target_value": "5kg purged",
                        "required_tool": "Purging Compound HDPE-P",
                        "safety_warning": "Extremely hot melt."
                    }
                ]
            }

planner_agent = SequenceToolPlannerAgent()
