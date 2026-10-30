import { ExecuteNode } from '@gravitydock/types/src/node';
import type { InputDefinition, OutputDefinition, PropertyDefinition, NodeCategory } from '@gravitydock/types/src/property';
import type { GravityDockAPI } from '@gravitydock/types';

declare const gravityDock: GravityDockAPI;

export class GenerateEmbeddingNode extends ExecuteNode {
    readonly typeId = 'llm.embedding';
    readonly displayName = 'Generate Embedding';
    readonly category: NodeCategory = { name: 'AI', accent: 'cyan-500' };
    readonly visible = true;

    readonly properties: PropertyDefinition[] = [
        { name: 'provider', label: 'Provider', type: 'select', options: [{label: 'Ollama', value: 'Ollama'}, {label: 'Gemini', value: 'Gemini'}], defaultValue: '' },
        { name: 'model', label: 'Model', type: 'string', defaultValue: '' }
    ];

    readonly inputs: InputDefinition[] = [
        { name: 'Text', acceptsType: 'core:string' }
    ];

    readonly outputs: OutputDefinition[] = [
        { name: 'Success', outputType: 'core:boolean' },
        { name: 'Embedding', outputType: 'core:array' }
    ];

    private async getProvider(properties: Record<string, any>): Promise<string> {
        const defaultProvider = await gravityDock.settings.get('defaultProvider') || 'Ollama';
        return properties.provider || defaultProvider;
    }

    private async getModel(provider: string, properties: Record<string, any>): Promise<string> {
        if (properties.model) return properties.model;
        if (provider === 'Ollama') {
            return await gravityDock.settings.get('defaultOllamaModel') || 'llama3';
        } else {
            return await gravityDock.settings.get('defaultGeminiModel') || 'gemini-1.5-flash';
        }
    }

    async evaluate(inputs: Record<string, any>, properties: Record<string, any>): Promise<Record<string, any>> {
        const provider = await this.getProvider(properties);
        const model = await this.getModel(provider, properties);
        const text = inputs.Text || '';
        
        try {
            if (provider === 'Ollama') {
                const baseUrl = await gravityDock.settings.get('ollamaUrl') || 'http://localhost:11434';
                const response = await gravityDock.net.fetch(`${baseUrl}/api/embeddings`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: model,
                        prompt: text
                    })
                });
                
                const data = JSON.parse(response.text);
                if (data.error) throw new Error(data.error);
                
                return { Success: true, Embedding: data.embedding, Event: true };
                
            } else if (provider === 'Gemini') {
                const apiKey = await gravityDock.settings.get('geminiKey');
                if (!apiKey) throw new Error("Gemini API Key is missing in package settings.");
                
                const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:embedContent?key=${apiKey}`;
                
                const response = await gravityDock.net.fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: `models/${model}`,
                        content: { parts: [{ text }] }
                    })
                });
                
                const data = JSON.parse(response.text);
                if (data.error) throw new Error(data.error.message || "Unknown Gemini error");
                
                const embedding = data.embedding?.values || [];
                return { Success: true, Embedding: embedding, Event: true };
            }
        } catch (e: any) {
            return { Success: false, Embedding: null, Event: true };
        }
        
        return { Success: false, Embedding: null, Event: true };
    }
}
