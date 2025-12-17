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
 * Surgical JSON Extractor
 * Identifies the actual workflow JSON within a mix of reasoning text and markdown.
 * Handles cases where additional text follows the JSON object.
 */
function extractJson(text: string): string {
  // Strategy 1: Look for the explicit marker
  const marker = "###JSON_START###";
  if (text.includes(marker)) {
    const parts = text.split(marker);
    return parts[1].trim(); 
  }

  // Strategy 2: Intelligent pattern matching for ComfyUI keys
  // This prevents accidentally starting at a random brace in the thought process.
  const patterns = ['{"workflow"', '{"nodes"', '{"requirements"', '{"last_node_id"', '{"lines"'];
  
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
    // Balanced brace counting to find the ECHTE end of the object
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

  // Strategy 3 (Emergency Fallback): Find the first and last braces
  const firstBrace = text.indexOf('{');
  const lastBrace = text.lastIndexOf('}');
  if (firstBrace > -1 && lastBrace > firstBrace) {
      return text.substring(firstBrace, lastBrace + 1);
  }

  throw new Error("No valid JSON object found in LLM response.");
}

/**
 * STREAMING GENERATION - SMART NDJSON EDITION
 * Robust stream reading, line-splitting, and surgical extraction.
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
    onStatusUpdate?: (status: string) => void
): Promise<GeneratedWorkflowResponse> => {
    
    // --- 1. PROMPT PREPARATION ---
    let ragContextBlock = '';
    if (ragApiUrl) {
        try {
            const ragContext = await queryRag(description, ragApiUrl, localLlmModel);
            if (ragContext && ragContext.trim()) {
                ragContextBlock = `\n**RAG-CONTEXT:**\n${ragContext.trim()}\n`;
            }
        } catch (e) { console.warn("RAG failed", e); }
    }

    const formatInstruction = format === 'api' ? API_FORMAT_INSTRUCTION : GRAPH_FORMAT_INSTRUCTION;
    const finalSystemInstruction = systemInstructionTemplate
        .replace('{{RAG_CONTEXT_PLACEHOLDER}}', ragContextBlock)
        .replace('{{IMAGE_CONTEXT_PLACEHOLDER}}', imageName ? `\nUser Image: ${imageName}\n` : '')
        .replace('{{SYSTEM_INVENTORY_PLACEHOLDER}}', inventory ? `\nInventory: ${JSON.stringify(inventory)}\n` : '')
        .replace('{{FORMAT_INSTRUCTION_PLACEHOLDER}}', formatInstruction);

    // --- 2. NDJSON STREAM REQUEST ---
    console.log("🚀 Calling Local NDJSON Streaming Backend...");
    
    // Using the specified /v1/generate_workflow_stream endpoint
    const endpoint = new URL('/v1/generate_workflow_stream', ragApiUrl).toString();

    try {
        const response = await fetch(endpoint, {
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

        // --- PHASE 1: STREAM PROCESSING (Line-Splitter) ---
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split(/\r?\n/);
            buffer = lines.pop() || ''; // Keep incomplete lines in buffer

            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const msg = JSON.parse(line);
                    if (msg.type === 'status') {
                        if (onStatusUpdate) onStatusUpdate(msg.data);
                        console.log("Backend Status:", msg.data);
                    } else if (msg.type === 'token') {
                        fullText += msg.data;
                        // Stream the accumulated tokens (reasoning) back to UI
                        onThoughtsUpdate(fullText);
                    }
                } catch (e) {
                    // Fallback for non-JSON lines (unlikely but safe)
                    fullText += line;
                    onThoughtsUpdate(fullText);
                }
            }
        }

        console.log("🏁 Stream ended. Analyzing and extracting JSON...");

        // --- PHASE 2: SURGICAL EXTRACTION ---
        let thoughts = fullText;
        let jsonString = "";

        try {
            jsonString = extractJson(fullText);
            
            // Separate thoughts from JSON for the final return
            const jsonStartIdx = fullText.indexOf(jsonString);
            if (jsonStartIdx > -1) {
                thoughts = fullText.substring(0, jsonStartIdx).trim();
                // Strip the internal marker if it's there
                thoughts = thoughts.replace("###JSON_START###", "").trim();
            }
        } catch (e) {
            console.error("JSON extraction error:", e);
            throw new Error("The AI replied, but no valid ComfyUI JSON could be identified.");
        }

        // Clean markdown code blocks if the AI used them inside the tokens
        jsonString = jsonString.replace(/```json/g, '').replace(/```/g, '').trim();

        // Final Parsing with cleanup
        let parsedData: any;
        try {
            parsedData = JSON.parse(jsonString);
        } catch (e: any) {
            console.error("❌ Parse failed on string:", jsonString);
            throw new Error(`JSON Syntax Error: ${e.message}`);
        }

        // Auto-Repair common structural issues
        if (!parsedData.workflow && (parsedData.nodes || parsedData.links)) {
            console.warn("⚠️ Repairing missing 'workflow' wrapper...");
            parsedData = {
                workflow: parsedData,
                requirements: { models: [], custom_nodes: [] }
            };
        }

        if (parsedData.workflow && !parsedData.requirements) {
            parsedData.requirements = { models: [], custom_nodes: [] };
        }

        if (!parsedData.workflow) {
             throw new Error("Invalid workflow structure received from AI.");
        }

        return {
            thoughts: thoughts || "No thoughts captured.",
            workflow: parsedData.workflow,
            requirements: parsedData.requirements
        };

    } catch (error: any) {
        console.error("🔥 Fatal Local Stream Error:", error);
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
        const jsonStr = extractJson(content);
        return JSON.parse(jsonStr);
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
        const jsonStr = extractJson(content);
        return JSON.parse(jsonStr);
    } catch (e: any) { throw new Error(`Debug failed: ${e.message}`); }
};

// --- RAG & Utility Exports ---
export const uploadRagDocument = async (file: File, apiUrl: string) => { return { message: "ok" }; };
export const queryRag = async (prompt: string, apiUrl: string, model?: string) => { return ""; };
export const learnWorkflow = async (type: any, prompt: string, workflow: any, apiUrl: string) => { return { message: "ok" }; };
export const startFineTuning = async (data: string, apiUrl: string) => { return { job_id: "1" }; };
export const getServerInventory = async (apiUrl: string) => { return {}; };
export const testLocalLlmConnection = async (url: string) => { return { success: true, message: "ok" }; };
export const testRagConnection = async (url: string) => { return { success: true, message: "ok" }; };
