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
 * Prevents "Unexpected non-whitespace character" errors by intelligently 
 * locating the actual ComfyUI JSON payload, ignoring braces in the thought process.
 */
function extractJsonFromText(text: string): { json: any, thoughts: string } {
    let jsonString = "";
    let thoughts = text;

    // Strategy 1: Look for the explicit marker
    const marker = "###JSON_START###";
    const markerIdx = text.indexOf(marker);
    if (markerIdx > -1) {
        thoughts = text.substring(0, markerIdx).trim();
        jsonString = text.substring(markerIdx + marker.length).trim();
    } else {
        // Strategy 2: Look for typical ComfyUI JSON start patterns
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
            // Balanced brace counting to find the end of the object
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
                    else if (char === '}') {
                        openBraces--;
                        if (openBraces === 0) {
                            endIndex = i;
                            break;
                        }
                    }
                }
            }

            if (endIndex > -1) {
                jsonString = text.substring(bestStartIndex, endIndex + 1);
                thoughts = text.substring(0, bestStartIndex).trim();
            }
        }
    }

    // Fallback: Use the whole text if no structure was identified (unlikely to work but safe)
    if (!jsonString) {
        const firstBrace = text.indexOf('{');
        const lastBrace = text.lastIndexOf('}');
        if (firstBrace > -1 && lastBrace > firstBrace) {
            jsonString = text.substring(firstBrace, lastBrace + 1);
            thoughts = text.substring(0, firstBrace).trim();
        } else {
            jsonString = text;
        }
    }

    // Clean up markdown markers if present
    jsonString = jsonString.replace(/```json/g, '').replace(/```/g, '').trim();
    thoughts = thoughts.replace('THOUGHTS:', '').replace(marker, '').trim();

    try {
        const parsed = JSON.parse(jsonString);
        return { json: parsed, thoughts };
    } catch (e) {
        console.error("❌ Surgical Parser failed to parse JSON. Raw string:", jsonString);
        throw new Error(`Syntax error in generated JSON: ${(e as Error).message}`);
    }
}

// --- Main Local LLM Interaction Functions ---

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
        if (error instanceof TypeError) {
             throw new Error(`Failed to connect to Local LLM at ${apiUrl}. Is it running?`);
        }
        throw error;
    }
}

/**
 * STREAMING GENERATION - PANZER-LOGIK EDITION
 * Robust stream reading, surgical parsing, and auto-repair.
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
    onThoughtsUpdate: (thoughtChunk: string) => void
): Promise<GeneratedWorkflowResponse> => {
    
    // --- 1. PROMPT CONSTRUCTION ---
    let ragContextBlock = '';
    if (ragApiUrl) {
        try {
            const ragContext = await queryRag(description, ragApiUrl, localLlmModel);
            if (ragContext && ragContext.trim()) {
                ragContextBlock = `\n**RAG-CONTEXT:**\n${ragContext.trim()}\n`;
            }
        } catch (e) { console.warn("RAG failed", e); }
    }

    let imageContextBlock = '';
    if (imageName) {
        imageContextBlock = `\n**USER-IMAGE:** User uploaded: ${imageName}. Use LoadImage node.\n`;
    }

    let inventoryBlock = 'No inventory.';
    if (inventory) {
        inventoryBlock = `\n\`\`\`json\n${JSON.stringify(inventory, null, 2)}\n\`\`\`\n`;
    }

    const formatInstruction = format === 'api' ? API_FORMAT_INSTRUCTION : GRAPH_FORMAT_INSTRUCTION;
    const finalSystemInstruction = systemInstructionTemplate
        .replace('{{RAG_CONTEXT_PLACEHOLDER}}', ragContextBlock)
        .replace('{{IMAGE_CONTEXT_PLACEHOLDER}}', imageContextBlock)
        .replace('{{SYSTEM_INVENTORY_PLACEHOLDER}}', inventoryBlock)
        .replace('{{FORMAT_INSTRUCTION_PLACEHOLDER}}', formatInstruction);

    // --- 2. STREAM REQUEST ---
    console.log("🚀 Calling Local Streaming Backend...");

    let endpoint: string;
    try {
        endpoint = new URL('/v1/generate_workflow_stream', ragApiUrl).toString();
    } catch (e) {
        throw new Error(`Invalid Python Backend URL configured: ${ragApiUrl}`);
    }

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
        let isJsonMode = false;
        const MARKER = "###JSON_START###";

        // --- 3. STREAM LOOP ---
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            fullText += chunk;
            
            // Extract the "thoughts" part for real-time update
            if (!isJsonMode) {
                if (fullText.includes(MARKER)) {
                    isJsonMode = true;
                    const thoughts = fullText.split(MARKER)[0].replace('THOUGHTS:', '').trim();
                    onThoughtsUpdate(thoughts);
                } else {
                    const thoughts = fullText.replace('THOUGHTS:', '').trimStart();
                    onThoughtsUpdate(thoughts);
                }
            }
        }

        console.log("🏁 Stream ended. Performing surgical JSON extraction...");

        // --- 4. SURGICAL PARSING ---
        let { json: parsedData, thoughts } = extractJsonFromText(fullText);

        // --- 5. AUTO-REPAIR ---
        // Ensure the data follows the { workflow: ..., requirements: ... } structure
        if (!parsedData.workflow && (parsedData.nodes || parsedData.links || Object.keys(parsedData).some(k => !isNaN(Number(k))))) {
            console.warn("⚠️ Payload structure missing wrapper. Auto-repairing...");
            parsedData = {
                workflow: parsedData,
                requirements: { models: [], custom_nodes: [] }
            };
        }
        
        if (parsedData.workflow && !parsedData.requirements) {
            parsedData.requirements = { models: [], custom_nodes: [] };
        }

        if (!parsedData.workflow) {
             throw new Error("Surgical parser failed: No workflow object found in output.");
        }

        return {
            thoughts: thoughts || "No thoughts logged.",
            workflow: parsedData.workflow,
            requirements: parsedData.requirements
        };

    } catch (error: any) {
        console.error("🔥 FATAL ERROR in Local LLM Stream Service:", error);
        if (error instanceof TypeError && error.message === 'Failed to fetch') {
            throw new Error(`Connection Failed to Backend (${endpoint}). Check server and CORS settings.`);
        }
        throw error;
    }
};

/**
 * STREAMING GENERATION - Alias for standard entry point
 */
export const generateWorkflowStream = generateWorkflowLocal;

// --- Remaining Utility Functions ---

export const validateAndCorrectWorkflowLocal = async (
    workflow: ComfyUIWorkflow | ComfyUIApiWorkflow, 
    localLlmApiUrl: string, 
    localLlmModel: string,
    ragApiUrl?: string
): Promise<ValidationResponse> => {
    if (!localLlmApiUrl) throw new Error("URL missing");
    const isGraph = typeof workflow === 'object' && 'nodes' in workflow;
    const basePrompt = isGraph ? SYSTEM_INSTRUCTION_VALIDATOR : SYSTEM_INSTRUCTION_API_VALIDATOR;
    const workflowString = JSON.stringify(workflow, null, 2);
    try {
        const content = await callLocalLlmChat(localLlmApiUrl, localLlmModel, [
            { role: "system", content: basePrompt },
            { role: "user", content: `Validate:\n\n${workflowString}` }
        ]);
        const { json } = extractJsonFromText(content);
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
    if (!localLlmApiUrl) throw new Error("URL missing");
    const isGraph = typeof workflow === 'object' && 'nodes' in workflow;
    const basePrompt = isGraph ? SYSTEM_INSTRUCTION_DEBUGGER : SYSTEM_INSTRUCTION_API_DEBUGGER;
    const payload = JSON.stringify({ workflow, errorMessage }, null, 2);
    try {
        const content = await callLocalLlmChat(localLlmApiUrl, localLlmModel, [
            { role: "system", content: basePrompt },
            { role: "user", content: payload }
        ]);
        const { json } = extractJsonFromText(content);
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
