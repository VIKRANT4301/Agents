from typing import Dict, Any, List, Optional
from app.utils.logger import logger
from app.llm.client import llm_client
from app.database.postgres.vector_client import vector_db
from app.models.relational import Recipe
import json

class RecipeSafetyAdvisorAgent:
    """
    Recipe & Safety Advisor Agent (Consolidates Recipes, Safety, and Guidance).
    1. Looks up recipe parameter configurations.
    2. Performs mathematical parameter delta comparisons (current vs. target).
    3. Retrieves Lockout/Tagout (LOTO) safety protocols from the vector database.
    4. Customizes guidance language based on operator skill level (Basic vs. Expert).
    """

    def compile_advice(self, line_id: str, source_sku: str, target_sku: str, 
                       source_params: Dict[str, Any], target_params: Dict[str, Any], 
                       operator_username: str, operator_skill_level: int,
                       telemetry: Optional[Any] = None) -> Dict[str, Any]:
        """
        Compares parameter values, queries LOTO safety context, and adjusts guidance tone.
        """
        logger.info(f"Advisor Agent compiling recipe differences and safety context for {operator_username}")
        
        # Step 1: Compute raw parameter deltas using de-duplicated Recipe helper
        deltas = Recipe.compare_parameters(source_params, target_params)
            
        # Step 2: Retrieve safety instructions via RAG (Vector Search)
        safety_query = f"Safety Lockout Tagout LOTO PPE instructions for {line_id} extruder and packers"
        try:
            safety_docs = vector_db.search(safety_query, top_k=2)
            safety_text = "\n".join([doc["text"] for doc in safety_docs])
        except Exception as e:
            logger.error(f"Failed to query vector database for safety LOTO docs: {e}")
            safety_text = ""
        
        # Step 3: Run LLM to summarize safety guidelines and customize guidance tone
        system_prompt = (
            "You are the Recipe & Safety Advisor Agent. Your task is to analyze LOTO (Lockout/Tagout) requirements "
            "and required PPE (Personal Protective Equipment) based on database context, and adapt instructions "
            "based on the operator's skill level. You must output ONLY a valid JSON object matching the requested schema."
        )
        
        prompt = f"""
### System Role
You are the Recipe & Safety Advisor Agent. Your job is to extract, summarize, and customize LOTO safety steps, required PPE, and safety warnings for the changeover process.

### Operator Information
- **Name**: {operator_username}
- **Skill Rating**: {operator_skill_level} (1 = Basic/New, 2 = Competent, 3 = Expert)

### LOTO Reference Context (RAG)
{safety_text if safety_text.strip() else "No specific LOTO reference context found."}

### Instruction Adaptation Guidelines
Tailor the tone and level of detail based on the Operator Skill Rating:
- **Skill Level 1 (Basic)**: Provide highly detailed, step-by-step guidance. Use clear, simple, and encouraging terms. Highlight warning notices prominently.
- **Skill Level 2 (Competent)**: Provide standard operational guidance with moderate detail. Balance technical terminology with clear, actionable steps.
- **Skill Level 3 (Expert)**: Provide high-level, compact, procedural commands. Skip basic explanations and focus on critical technical parameter values and advanced checkpoints.

### Output Requirements
Output MUST be a single, valid JSON object. Do not include any conversational preamble, explanation, or markdown wrappers outside the JSON block.

Expected JSON schema:
{{
  "loto_steps": [
    "Step 1...",
    "Step 2..."
  ],
  "required_ppe": [
    "PPE Item 1...",
    "PPE Item 2..."
  ],
  "general_warnings": [
    "Warning 1...",
    "Warning 2..."
  ],
  "guidance_tone": "Explanation of how the guidance tone was adapted for the operator's skill level."
}}

Begin JSON response:
"""
        
        raw_response = llm_client.call_primary_llm(prompt, system_prompt)
        
        if telemetry:
            telemetry.add_mock_tokens_for_prompt(prompt, raw_response)
        
        # Robust parsing of JSON from LLM response
        safety_data = self._extract_and_parse_json(raw_response)
            
        return {
            "parameter_deltas": deltas,
            "safety": safety_data
        }

    def _extract_and_parse_json(self, raw_response: str) -> Dict[str, Any]:
        """
        Robustly extracts and parses JSON content from LLM response.
        Falls back to standard template on failure.
        """
        clean_str = raw_response.strip()
        
        # Try direct parse
        try:
            return json.loads(clean_str)
        except json.JSONDecodeError:
            pass
            
        # Strip markdown block wrappers if present
        if clean_str.startswith("```"):
            lines = clean_str.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            clean_str = "\n".join(lines).strip()
            
            try:
                return json.loads(clean_str)
            except json.JSONDecodeError:
                pass

        # Locate outermost brackets if JSON is wrapped in extra text
        start_idx = clean_str.find("{")
        end_idx = clean_str.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_candidate = clean_str[start_idx:end_idx + 1]
            try:
                return json.loads(json_candidate)
            except json.JSONDecodeError as e:
                logger.warning(f"Extracted candidate JSON block but failed to parse: {e}")

        # Final Fallback block if parsing fails
        logger.error(f"Failed to parse LLM advisor safety JSON. Using fallback. Raw response: {raw_response[:200]}")
        return {
            "loto_steps": [
                "Locate circuit breaker L3-E1 on Wall Box B-04. Turn off.",
                "Attach safety padlock and tag #24 stating 'DO NOT OPERATE'."
            ],
            "required_ppe": ["Safety Glasses", "High-Temperature Gloves", "Steel-Toed Boots"],
            "general_warnings": ["Check hot barrel surfaces before disassembly to avoid burns."],
            "guidance_tone": "Standard safety instructions provided (default fallback)."
        }

advisor_agent = RecipeSafetyAdvisorAgent()
