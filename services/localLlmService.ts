import type { SystemInventory, GeneratedWorkflowResponse, ComfyUIWorkflow, ComfyUIApiWorkflow, ValidationResponse, DebugResponse, WorkflowFormat } from '../types';
import { 
    SYSTEM_INSTRUCTION_TEMPLATE, 
    SYSTEM_INSTRUCTION_VALIDATOR, 
    SYSTEM_INSTRUCTION_API_VALIDATOR,
    SYSTEM_INSTRUCTION_DEBUGGER, 
    SYSTEM_INSTRUCTION_API_DEBUGGER,
    GRAPH_FORMAT_INSTRUCTION, 
    API_FORMAT_INSTRUCTION 
} from './prompts';

/**
 * Robust JSON Extractor (The "Surgical Parser")
 * Identifies the actual workflow JSON within a mix of reasoning text and markdown.
 */
function extractJson(text: string): string {
  // Strategy 1: Explicit Marker
  const marker = "###JSON_START###";
  if (text.includes(marker)) {
    const parts = text.split(marker);
    return parts[1].trim(); 
  }

  // Strategy 2: ComfyUI JSON Start Pattern Matching
  // Searching for typical keys to avoid false positives from thinking-text braces.
  const patterns = ['{"workflow"', '{"nodes"', '{"requirements"', '{"last_node_id"'];
  
  let bestStartIndex = -1;
  for (const pattern of patterns) {
    const idx = text.indexOf(pattern);
    if (idx > -1) {
      if (bestStartIndex === -1 || idx < bestStartIndex) {
        bestStartIndex = idx;
      }
    }
  }

  if (bestStartIndex > -1) {
    // Balanced brace counting to find the actual end of the primary object
    let openBraces = 0;
    let endIndex = -1;
    let insideString = false;
    let escape = false;

    for (let i = bestStartIndex; i < text.length; i++) {
      const char = text[i];
      if (escape) { escape = false; continue; }
      if (char === '\\') { escape = true; continue; }
      if (char === '"') { insideString = !insideString; continue; }

      if (!insideString) {
        if (char === '{') openBraces++;
        if (char === '}') {
          openBraces--;
          if (openBraces === 0) {
            endIndex = i;
            break;
          }
        }
      }
    }
    
    if (endIndex > -1) {
      return text.substring(bestStartIndex, endIndex + 1);
    }
  }

  // Strategy 3: Fallback (Legacy)
  const firstBrace = text.indexOf('{');
  const lastBrace = text.lastIndexOf('}');
  if (firstBrace > -1 && lastBrace > firstBrace) {
      return text.substring(firstBrace, lastBrace + 1);
  }

  throw new Error("No valid JSON object found in model response.");
}

/**
 * STREAMING GENERATION - SMART NDJSON HANDLER
 * Processes real-time tokens (thoughts) and status messages from the Python backend.
 */
export const generateWorkflowLocal = async (
    description: string,
    localLlmApiUrl: string, 
    localLlmModel: string,
    inventory: SystemInventory | null,
    imageName: string | undefined,
    ragApiUrl: string, 
    format: WorkflowFormat = 'graph',
    systemInstructionTemplate: string = SYSTEM_INSTRUCTION_TEMPLATE,
    onThoughtsUpdate: (thoughtChunk: string) => void,
    onStatusUpdate?: (status: string) => void // Optional status callback
): Promise<GeneratedWorkflowResponse> => {
    
    console.log("🚀 Starting Smart NDJSON Stream Request...");
    
    // --- 1. PROMPT PREPARATION ---
    let ragContextBlock = '';
    if (ragApiUrl) {
        try {
            const ragContext = await queryRag(description, ragApiUrl, localLlmModel);
            if (ragContext && ragContext.trim()) {
                ragContextBlock = `\n**RAG-CONTEXT:**\n${ragContext.trim()}\n`;
            }
        } catch (e) { console.warn("RAG retrieval failed", e); }
    }

    const formatInstruction = format === 'api' ? API_FORMAT_INSTRUCTION : GRAPH_FORMAT_INSTRUCTION;
    const finalSystemInstruction = systemInstructionTemplate
        .replace('{{RAG_CONTEXT_PLACEHOLDER}}', ragContextBlock)
        .replace('{{IMAGE_CONTEXT_PLACEHOLDER}}', imageName ? `\nUser Image: ${imageName}\n` : '')
        .replace('{{SYSTEM_INVENTORY_PLACEHOLDER}}', inventory ? `\nInventory: ${JSON.stringify(inventory)}\n` : '')
        .replace('{{FORMAT_INSTRUCTION_PLACEHOLDER}}', formatInstruction);

    // --- 2. STREAM REQUEST ---
    const API_URL = 'http://192.168.1.73:8000/v1/generate_workflow_stream';

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: description,
                model: localLlmModel,
                system_prompt: finalSystemInstruction,
                ollama_url: localLlmApiUrl 
            })
        });

        if (!response.body) throw new Error("No ReadableStream received from server.");

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        let fullText = '';
        let buffer = ''; 

        // --- PHASE 1: STREAM PROCESSING (NDJSON) ---
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split(/\r?\n/);
            buffer = lines.pop() || ''; // Keep incomplete line in buffer

            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const msg = JSON.parse(line);
                    if (msg.type === 'status') {
                        if (onStatusUpdate) onStatusUpdate(msg.data);
                        console.log("Backend Status:", msg.data);
                    } else if (msg.type === 'token') {
                        fullText += msg.data;
                        onThoughtsUpdate(fullText); // Stream combined thoughts to UI
                    }
                } catch (e) {
                    // If parsing fails, it might be raw text or partial stream; try appending
                    fullText += line;
                    onThoughtsUpdate(fullText);
                }
            }
        }

        console.log("🏁 Stream finished. Extracting JSON payload...");

        // --- PHASE 2: EXTRACTION & REPAIR ---
        let thoughts = fullText;
        let jsonString = "";

        try {
            jsonString = extractJson(fullText);
            
            // Isolate thoughts (everything before the JSON starts)
            const jsonStartIdx = fullText.indexOf(jsonString);
            if (jsonStartIdx > -1) {
                thoughts = fullText.substring(0, jsonStartIdx).trim();
                thoughts = thoughts.replace("###JSON_START###", "").trim();
            }
        } catch (e) {
            console.error("Extraction error:", e);
            throw new Error("AI responded but no valid ComfyUI JSON could be extracted.");
        }

        // Clean markdown remnants
        jsonString = jsonString.replace(/```json/g, '').replace(/```/g, '').trim();

        // Final Parsing
        let parsedData: any;
        try {
            parsedData = JSON.parse(jsonString);
        } catch (e: any) {
            console.error("❌ JSON Syntax Error in string:", jsonString);
            throw new Error(`Syntax Error in generated JSON: ${e.message}`);
        }

        // Auto-Fix structure if wrapper is missing
        if (!parsedData.workflow && (parsedData.nodes || parsedData.links)) {
            console.warn("⚠️ Repairing missing wrapper structure...");
            parsedData = {
                workflow: parsedData,
                requirements: { models: [], custom_nodes: [] }
            };
        }

        if (parsedData.workflow && !parsedData.requirements) {
            parsedData.requirements = { models: [], custom_nodes: [] };
        }

        if (!parsedData.workflow) {
             throw new Error("Workflow structure invalid (No 'workflow' key found).");
        }

        return {
            thoughts: thoughts || "No reasoning logged.",
            workflow: parsedData.workflow,
            requirements: parsedData.requirements
        };

    } catch (error: any) {
        console.error("🔥 Fatal Error in Local Generation:", error);
        throw error;
    }
};

/**
 * Standard Chat Implementation for non-streaming tasks (Validation/Debug)
 */
async function callLocalLlmChat(apiUrl: string, model: string, messages: Array<{role: string, content: string}>): Promise<string> {
    const endpoint = new URL('/v1/chat/completions', apiUrl).toString();
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: model,
                messages: messages,
                temperature: 0.2,
                stream: false
            }),
        });

        if (!response.ok) {
             const errorText = await response.text();
             throw new Error(`Local LLM Error (${response.status}): ${errorText}`);
        }

        const data = await response.json();
        return data.choices[0].message.content;
    } catch (error) {
        throw error;
    }
}

export const validateAndCorrectWorkflowLocal = async (
    workflow: ComfyUIWorkflow | ComfyUIApiWorkflow, 
    localLlmApiUrl: string, 
    localLlmModel: string,
    ragApiUrl?: string
): Promise<ValidationResponse> => {
    const isGraph = typeof workflow === 'object' && 'nodes' in workflow;
    const basePrompt = isGraph ? SYSTEM_INSTRUCTION_VALIDATOR : SYSTEM_INSTRUCTION_API_VALIDATOR;
    try {
        const content = await callLocalLlmChat(localLlmApiUrl, localLlmModel, [
            { role: "system", content: basePrompt },
            { role: "user", content: `Validate:\n\n${JSON.stringify(workflow)}` }
        ]);
        const { json } = { json: JSON.parse(extractJson(content)) }; // Quick hack using extractJson
        return json;
    } catch (e: any) { throw new Error(`Validation failed: ${e.message}`); }
};

export const debugAndCorrectWorkflowLocal = async (
    workflow: ComfyUIWorkflow | ComfyUIApiWorkflow, 
    errorMessage: string,
    localLlmApiUrl: string, 
    localLlmModel: string,
    ragApiUrl?: string
): Promise<DebugResponse> => {
    const isGraph = typeof workflow === 'object' && 'nodes' in workflow;
    const basePrompt = isGraph ? SYSTEM_INSTRUCTION_DEBUGGER : SYSTEM_INSTRUCTION_API_DEBUGGER;
    try {
        const content = await callLocalLlmChat(localLlmApiUrl, localLlmModel, [
            { role: "system", content: basePrompt },
            { role: "user", content: JSON.stringify({ workflow, errorMessage }) }
        ]);
        const { json } = { json: JSON.parse(extractJson(content)) };
        return json;
    } catch (e: any) { throw new Error(`Debug failed: ${e.message}`); }
};

export const uploadRagDocument = async (file: File, apiUrl: string) => { return { message: "ok" }; };
export const queryRag = async (prompt: string, apiUrl: string, model?: string) => { return ""; };
export const learnWorkflow = async (type: any, prompt: string, workflow: any, apiUrl: string) => { return { message: "ok" }; };
export const startFineTuning = async (data: string, apiUrl: string) => { return { job_id: "1" }; };
export const getServerInventory = async (apiUrl: string) => { return {}; };
export const testLocalLlmConnection = async (url: string) => { return { success: true, message: "ok" }; };
export const testRagConnection = async (url: string) => { return { success: true, message: "ok" }; };
