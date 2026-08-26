import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import axios from 'axios';

export function activate(context: vscode.ExtensionContext) {
    console.log('Ollama Assistant is now active!');

    // Register the main command to open assistant
    const startCommand = vscode.commands.registerCommand('ollama-assistant.start', () => {
        OllamaAssistantPanel.createOrShow(context.extensionUri);
    });

    // Register context menu command
    const analyzeCommand = vscode.commands.registerCommand('ollama-assistant.analyzeFile', async () => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            const document = editor.document;
            const selection = editor.selection;
            const text = selection.isEmpty ? document.getText() : document.getText(selection);
            
            // Open assistant with selected text
            OllamaAssistantPanel.createOrShow(context.extensionUri);
            // Send the selected text to the panel
            vscode.commands.executeCommand('ollama-assistant.start');
        }
    });

    context.subscriptions.push(startCommand, analyzeCommand);
}

class OllamaAssistantPanel {
    public static currentPanel: OllamaAssistantPanel | undefined;
    private readonly _panel: vscode.WebviewPanel;
    private _disposables: vscode.Disposable[] = [];

    private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri) {
        this._panel = panel;
        
        // Set the webview's initial html content
        this._panel.webview.html = this._getHtmlForWebview(panel.webview, extensionUri);
        
        // Handle messages from the webview
        this._panel.webview.onDidReceiveMessage(
            async message => {
                switch (message.command) {
                    case 'sendMessage':
                        await this.handleChatMessage(message.text);
                        break;
                    case 'readFile':
                        await this.readFile(message.filePath);
                        break;
                    case 'editFile':
                        await this.editFile(message.filePath, message.content);
                        break;
                    case 'getWorkspaceFiles':
                        await this.getWorkspaceFiles();
                        break;
                    case 'createFile':
                        await this.createFile(message.filePath, message.content);
                        break;
                    case 'deleteFile':
                        await this.deleteFile(message.filePath);
                        break;
                    case 'searchFiles':
                        await this.searchFiles(message.pattern);
                        break;
                }
            },
            null,
            this._disposables
        );

        // Clean up when panel is closed
        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);
    }

    public static createOrShow(extensionUri: vscode.Uri) {
        const column = vscode.ViewColumn.Beside;

        // If we already have a panel, show it
        if (OllamaAssistantPanel.currentPanel) {
            OllamaAssistantPanel.currentPanel._panel.reveal(column);
            return;
        }

        // Otherwise, create a new panel
        const panel = vscode.window.createWebviewPanel(
            'ollamaAssistant',
            'Ollama Workspace Assistant',
            column,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [extensionUri]
            }
        );

        OllamaAssistantPanel.currentPanel = new OllamaAssistantPanel(panel, extensionUri);
    }

    private async handleChatMessage(message: string) {
        try {
            const config = vscode.workspace.getConfiguration('ollamaAssistant');
            const model = config.get('model', 'codellama');
            const ollamaUrl = config.get('url', 'http://localhost:11434');

            // Get workspace context
            const workspaceContext = await this.getWorkspaceContext();

            const prompt = `You are an AI assistant that can read and edit files in VS Code. 
Current workspace context:
${workspaceContext}

User request: ${message}

To perform file operations, use these exact formats:
- To read a file: [READ_FILE:path/to/file]
- To edit a file: [EDIT_FILE:path/to/file] then on next line [CONTENT:new content]
- To create a file: [CREATE_FILE:path/to/file] then on next line [CONTENT:file content]
- To delete a file: [DELETE_FILE:path/to/file]
- To search: [SEARCH:pattern]

Respond to the user's request naturally. If you need to perform file operations, include the appropriate commands in your response.`;

            // Call Ollama
            const response = await axios.post(`${ollamaUrl}/api/generate`, {
                model: model,
                prompt: prompt,
                stream: false
            });

            const aiResponse = response.data.response;
            
            // Parse and execute file commands
            await this.executeFileCommands(aiResponse);
            
            // Send response back to webview
            this._panel.webview.postMessage({
                command: 'response',
                text: aiResponse
            });

        } catch (error: any) {
            vscode.window.showErrorMessage(`Ollama error: ${error.message}`);
            this._panel.webview.postMessage({
                command: 'response',
                text: `Error: ${error.message}. Make sure Ollama is running locally.`
            });
        }
    }

    private async getWorkspaceContext(): Promise<string> {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (!workspaceFolders) {
            return 'No workspace open. Please open a folder to use file operations.';
        }

        let context = `Workspace: ${workspaceFolders[0].uri.fsPath}\n\n`;
        
        // Get list of files
        const files = await vscode.workspace.findFiles('**/*', '**/node_modules/**', 20);
        context += `Files in workspace (${files.length} total, showing first 20):\n`;
        files.forEach(file => {
            context += `- ${file.fsPath.replace(workspaceFolders[0].uri.fsPath, '')}\n`;
        });

        return context;
    }

    private async getWorkspaceFiles() {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (!workspaceFolders) {
            this._panel.webview.postMessage({
                command: 'filesList',
                files: []
            });
            return;
        }

        const files = await vscode.workspace.findFiles('**/*', '**/node_modules/**', 100);
        const fileList = files.map(f => f.fsPath);
        
        this._panel.webview.postMessage({
            command: 'filesList',
            files: fileList
        });
    }

    private async readFile(filePath: string) {
        try {
            const document = await vscode.workspace.openTextDocument(filePath);
            const content = document.getText();
            
            this._panel.webview.postMessage({
                command: 'fileContent',
                path: filePath,
                content: content
            });
        } catch (error: any) {
            vscode.window.showErrorMessage(`Error reading file: ${error.message}`);
        }
    }

    private async editFile(filePath: string, content: string) {
        try {
            const document = await vscode.workspace.openTextDocument(filePath);
            const edit = new vscode.WorkspaceEdit();
            
            const fullRange = new vscode.Range(
                document.positionAt(0),
                document.positionAt(document.getText().length)
            );
            
            edit.replace(document.uri, fullRange, content);
            await vscode.workspace.applyEdit(edit);
            await document.save();
            
            vscode.window.showInformationMessage(`File updated: ${path.basename(filePath)}`);
        } catch (error: any) {
            vscode.window.showErrorMessage(`Error editing file: ${error.message}`);
        }
    }

    private async createFile(filePath: string, content: string) {
        try {
            const workspaceFolders = vscode.workspace.workspaceFolders;
            if (!workspaceFolders) {
                throw new Error('No workspace open');
            }

            const fullPath = path.isAbsolute(filePath) ? filePath : path.join(workspaceFolders[0].uri.fsPath, filePath);
            
            // Ensure directory exists
            const dir = path.dirname(fullPath);
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
            }
            
            // Write file
            fs.writeFileSync(fullPath, content);
            
            // Open in editor
            const document = await vscode.workspace.openTextDocument(fullPath);
            await vscode.window.showTextDocument(document);
            
            vscode.window.showInformationMessage(`File created: ${path.basename(fullPath)}`);
        } catch (error: any) {
            vscode.window.showErrorMessage(`Error creating file: ${error.message}`);
        }
    }

    private async deleteFile(filePath: string) {
        try {
            const uri = vscode.Uri.file(filePath);
            await vscode.workspace.fs.delete(uri);
            vscode.window.showInformationMessage(`File deleted: ${path.basename(filePath)}`);
        } catch (error: any) {
            vscode.window.showErrorMessage(`Error deleting file: ${error.message}`);
        }
    }

    private async searchFiles(pattern: string) {
        try {
            const files = await vscode.workspace.findFiles(pattern, '**/node_modules/**');
            const fileList = files.map(f => f.fsPath);
            
            this._panel.webview.postMessage({
                command: 'searchResults',
                pattern: pattern,
                files: fileList
            });
        } catch (error: any) {
            vscode.window.showErrorMessage(`Error searching files: ${error.message}`);
        }
    }

    private async executeFileCommands(response: string) {
        // Parse READ_FILE commands
        const readFileRegex = /\[READ_FILE:(.*?)\]/g;
        let match;
        while ((match = readFileRegex.exec(response)) !== null) {
            await this.readFile(match[1].trim());
        }

        // Parse EDIT_FILE commands
        const editFileRegex = /\[EDIT_FILE:(.*?)\]\s*\[CONTENT:(.*?)\]/gs;
        while ((match = editFileRegex.exec(response)) !== null) {
            await this.editFile(match[1].trim(), match[2].trim());
        }

        // Parse CREATE_FILE commands
        const createFileRegex = /\[CREATE_FILE:(.*?)\]\s*\[CONTENT:(.*?)\]/gs;
        while ((match = createFileRegex.exec(response)) !== null) {
            await this.createFile(match[1].trim(), match[2].trim());
        }

        // Parse DELETE_FILE commands
        const deleteFileRegex = /\[DELETE_FILE:(.*?)\]/g;
        while ((match = deleteFileRegex.exec(response)) !== null) {
            await this.deleteFile(match[1].trim());
        }

        // Parse SEARCH commands
        const searchRegex = /\[SEARCH:(.*?)\]/g;
        while ((match = searchRegex.exec(response)) !== null) {
            await this.searchFiles(match[1].trim());
        }
    }

    private _getHtmlForWebview(webview: vscode.Webview, extensionUri: vscode.Uri) {
        return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {
                    padding: 0;
                    margin: 0;
                    font-family: var(--vscode-font-family);
                    color: var(--vscode-editor-foreground);
                    background-color: var(--vscode-editor-background);
                    height: 100vh;
                    display: flex;
                    flex-direction: column;
                }
                
                .toolbar {
                    padding: 10px;
                    background-color: var(--vscode-editor-inactiveSelectionBackground);
                    border-bottom: 1px solid var(--vscode-panel-border);
                }
                
                button {
                    background-color: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                    border: none;
                    padding: 6px 12px;
                    margin-right: 8px;
                    cursor: pointer;
                    border-radius: 2px;
                    font-size: 12px;
                }
                
                button:hover {
                    background-color: var(--vscode-button-hoverBackground);
                }
                
                .chat-container {
                    flex: 1;
                    overflow-y: auto;
                    padding: 10px;
                    display: flex;
                    flex-direction: column;
                }
                
                .message {
                    margin: 8px 0;
                    padding: 10px;
                    border-radius: 5px;
                    max-width: 80%;
                    word-wrap: break-word;
                }
                
                .user {
                    align-self: flex-end;
                    background-color: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                }
                
                .assistant {
                    align-self: flex-start;
                    background-color: var(--vscode-editor-inactiveSelectionBackground);
                }
                
                .file-list {
                    max-height: 200px;
                    overflow-y: auto;
                    border: 1px solid var(--vscode-panel-border);
                    margin: 10px 0;
                    padding: 5px;
                }
                
                .file-item {
                    padding: 4px 8px;
                    cursor: pointer;
                    border-radius: 2px;
                }
                
                .file-item:hover {
                    background-color: var(--vscode-list-hoverBackground);
                }
                
                .input-area {
                    padding: 10px;
                    background-color: var(--vscode-editor-inactiveSelectionBackground);
                    border-top: 1px solid var(--vscode-panel-border);
                    display: flex;
                }
                
                textarea {
                    flex: 1;
                    background-color: var(--vscode-input-background);
                    color: var(--vscode-input-foreground);
                    border: 1px solid var(--vscode-input-border);
                    padding: 8px;
                    border-radius: 2px;
                    font-family: var(--vscode-font-family);
                    resize: none;
                    min-height: 60px;
                }
                
                textarea:focus {
                    outline: 1px solid var(--vscode-focusBorder);
                }
                
                .status {
                    padding: 4px 10px;
                    font-size: 11px;
                    color: var(--vscode-descriptionForeground);
                    background-color: var(--vscode-editor-background);
                }
                
                .model-select {
                    background-color: var(--vscode-dropdown-background);
                    color: var(--vscode-dropdown-foreground);
                    border: 1px solid var(--vscode-dropdown-border);
                    padding: 4px;
                    margin-left: 10px;
                }
            </style>
        </head>
        <body>
            <div class="toolbar">
                <button onclick="getWorkspaceFiles()">📁 Refresh Files</button>
                <button onclick="clearChat()">🗑️ Clear Chat</button>
                <button onclick="openFile()">📄 Open File</button>
                <select id="modelSelect" class="model-select">
                    <option value="codellama">CodeLlama</option>
                    <option value="llama2">Llama 2</option>
                    <option value="mistral">Mistral</option>
                    <option value="phi">Phi</option>
                </select>
            </div>
            
            <div class="status" id="status">Ready</div>
            
            <div class="file-list" id="fileList" style="display: none;"></div>
            
            <div class="chat-container" id="chat"></div>
            
            <div class="input-area">
                <textarea id="messageInput" placeholder="Ask me to help with your code... (Press Ctrl+Enter to send)"></textarea>
                <button onclick="sendMessage()" style="margin-left: 8px;">Send</button>
            </div>

            <script>
                const vscode = acquireVsCodeApi();
                let fileListVisible = false;
                
                function sendMessage() {
                    const input = document.getElementById('messageInput');
                    const message = input.value.trim();
                    const model = document.getElementById('modelSelect').value;
                    
                    if (message) {
                        addMessage('user', message);
                        vscode.postMessage({ command: 'sendMessage', text: message, model: model });
                        input.value = '';
                        updateStatus('Thinking...');
                    }
                }
                
                function addMessage(sender, text) {
                    const chat = document.getElementById('chat');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = 'message ' + sender;
                    
                    // Format the text (simple markdown-like formatting)
                    let formattedText = text
                        .replace(/\\n/g, '<br>')
                        .replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>')
                        .replace(/\`(.*?)\`/g, '<code>$1</code>');
                    
                    messageDiv.innerHTML = '<strong>' + (sender === 'user' ? 'You' : 'Assistant') + ':</strong><br>' + formattedText;
                    chat.appendChild(messageDiv);
                    chat.scrollTop = chat.scrollHeight;
                }
                
                function getWorkspaceFiles() {
                    vscode.postMessage({ command: 'getWorkspaceFiles' });
                    updateStatus('Loading files...');
                }
                
                function clearChat() {
                    document.getElementById('chat').innerHTML = '';
                    updateStatus('Chat cleared');
                }
                
                function openFile() {
                    const fileInput = document.createElement('input');
                    fileInput.type = 'file';
                    fileInput.webkitdirectory = true;
                    fileInput.onchange = () => {
                        if (fileInput.files.length > 0) {
                            addMessage('system', 'Selected ' + fileInput.files.length + ' files');
                        }
                    };
                    fileInput.click();
                }
                
                function readFile(filePath) {
                    vscode.postMessage({ command: 'readFile', filePath: filePath });
                    updateStatus('Reading file...');
                }
                
                function updateStatus(text) {
                    document.getElementById('status').textContent = 'Status: ' + text;
                }
                
                window.addEventListener('message', event => {
                    const message = event.data;
                    
                    switch (message.command) {
                        case 'response':
                            addMessage('assistant', message.text);
                            updateStatus('Ready');
                            break;
                            
                        case 'filesList':
                            const fileList = document.getElementById('fileList');
                            fileList.style.display = 'block';
                            fileList.innerHTML = '<strong>Workspace Files:</strong>';
                            
                            if (message.files.length === 0) {
                                fileList.innerHTML += '<div>No files found</div>';
                            } else {
                                message.files.forEach(file => {
                                    const div = document.createElement('div');
                                    div.className = 'file-item';
                                    div.textContent = file.split(/[\\\\/]/).pop();
                                    div.title = file;
                                    div.onclick = () => readFile(file);
                                    fileList.appendChild(div);
                                });
                            }
                            updateStatus('Files loaded: ' + message.files.length);
                            break;
                            
                        case 'fileContent':
                            addMessage('assistant', '📄 **' + message.path + '**\\n\\n```\\n' + message.content + '\\n```');
                            updateStatus('File loaded');
                            break;
                            
                        case 'searchResults':
                            addMessage('assistant', '🔍 Found ' + message.files.length + ' files matching "' + message.pattern + '":\\n' + 
                                message.files.map(f => '- ' + f).join('\\n'));
                            break;
                    }
                });
                
                document.getElementById('messageInput').addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && e.ctrlKey) {
                        sendMessage();
                    }
                });
                
                // Check if Ollama is running
                updateStatus('Checking Ollama connection...');
                setTimeout(() => {
                    updateStatus('Ready (make sure Ollama is running)');
                }, 2000);
            </script>
        </body>
        </html>
        `;
    }

    public dispose() {
        OllamaAssistantPanel.currentPanel = undefined;
        this._panel.dispose();
        while (this._disposables.length) {
            const x = this._disposables.pop();
            if (x) {
                x.dispose();
            }
        }
    }
}

export function deactivate() {}