/**
 * Angular Root Component for RAG X-Ray UI (With Workspace Switcher & History)
 * ============================================================================
 * Manages the interactive dashboard:
 * - Workspace Switcher: Switch between different workspaces at any time (all saved locally).
 * - Document uploads and live status polling per workspace.
 * - Asking questions with top-k context chunk retrieval.
 * - Local Query History: Browse and reload past questions and their X-Ray journeys.
 * - Inspecting step-by-step X-Ray transformations (Data Journey).
 * - Rendering ChatGPT-style formatted Markdown with citation badges.
 */
import { CommonModule } from '@angular/common';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { environment } from '../environments/environment';

// -------------------------------------------------------------
// Type Definitions
// -------------------------------------------------------------

/** Represents a single transformation step (e.g. Clean text, Embeddings) */
interface Stage {
  name: string;
  summary?: string;
  before: unknown;
  operation: string;
  after: unknown;
  next: string;
}

/** Represents an uploaded document and its metadata */
interface Doc {
  id: string;
  filename: string;
  status: string; // 'queued' | 'processing' | 'ready' | 'failed'
  stage_count?: number;
  chunk_count?: number;
  error?: string | null;
  stages?: Stage[];
  chunks?: any[];
}

/** Represents a single retrieved chunk with its cosine similarity score */
interface Rank {
  id: string;
  filename: string;
  page: number;
  text: string;
  score: number;
  vector_preview: number[];
  vector_norm: number;
}

/** Represents a past query record in the local history */
interface HistoryItem {
  id: string;
  question: string;
  answer: string;
  answer_mode: string;
  sources: Rank[];
  ranked: Rank[];
  steps: Stage[];
  top_k: number;
  total_candidates: number;
  model: string;
  provider?: string;
  timestamp: string;
  thinking_time?: string;
}

/** Represents an existing local workspace */
interface Workspace {
  id: string;
  name: string;
  doc_count: number;
  history_count: number;
  is_latest?: boolean;
  time_ago?: string;
  last_active?: string;
  updated_at?: number;
}

/** Response received from the /api/ask endpoint */
interface Answer {
  id?: string;
  question?: string;
  answer: string;
  answer_mode: string;
  sources: Rank[];
  ranked: Rank[];
  steps: Stage[];
  top_k: number;
  total_candidates: number;
  model: string;
  provider?: string;
  timestamp?: string;
  thinking_time?: string;
}

/**
 * Resolves the backend API URL dynamically across environments:
 * 1. Runtime window environment override: (window as any).__ENV__?.API_URL
 * 2. LocalStorage override for testing without rebuild: localStorage.getItem('RAG_API_URL')
 * 3. Angular environment configuration (environment.apiUrl)
 * 4. Auto-detect: if on localhost/127.0.0.1 dev server, use http://localhost:8000/api
 * 5. Production fallback: '/api' (relative for Vercel rewrites or reverse proxies)
 */
export function getApiUrl(): string {
  if (typeof window !== 'undefined' && (window as any).__ENV__?.API_URL) {
    return (window as any).__ENV__.API_URL.replace(/\/$/, '');
  }
  const stored = typeof localStorage !== 'undefined' ? localStorage.getItem('RAG_API_URL') : null;
  if (stored) {
    return stored.replace(/\/$/, '');
  }
  if (environment.apiUrl) {
    return environment.apiUrl.replace(/\/$/, '');
  }
  if (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')) {
    return 'http://localhost:8000/api';
  }
  return '/api';
}

const API_URL = getApiUrl();

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent implements OnInit, OnDestroy {
  // Local browser session ID (ensures documents & questions are private to this workspace)
  sessionId: string = this.initSessionId();

  // Workspaces list and dropdown toggle
  workspaces: Workspace[] = [];
  showWorkspaceDropdown = false;

  // Document list and selections
  docs: Doc[] = [];
  selectedIds = new Set<string>();
  currentDoc: Doc | null = null;

  // Question & Answer state (ChatGPT style follow-up flow)
  question = '';
  activeQuestion = '';
  answer: Answer | null = null;
  topK = 3;
  selectedSource: Rank | null = null;

  // ChatGPT-style Live Thinking Journey Pipeline
  thinkingStep = 0;
  thinkingSeconds = 0;
  private thinkingInterval?: any;
  showThoughtProcess = false;

  // Mobile UX: toggle between sidebar (library/history) and main inspector
  mobileTab: 'sidebar' | 'main' = 'main';
  uploadingSample = false;
  uploading = false;

  // Multi-Provider LLM Configuration (Ollama, Groq, OpenAI, Gemini)
  selectedProvider: 'ollama' | 'groq' | 'openai' | 'gemini' = 'groq';
  customModel = 'llama-3.3-70b-versatile';
  apiKey = '';
  showModelSettings = false;
  showApiKey = false;

  // Independent API key storage per provider so user never loses their keys when toggling
  providerKeys: { [key: string]: string } = {
    ollama: '',
    groq: '',
    openai: '',
    gemini: ''
  };

  // Independent model presets per provider
  providerModels: { [key: string]: string } = {
    ollama: 'qwen2.5-coder:7b',
    groq: 'llama-3.3-70b-versatile',
    openai: 'gpt-4o-mini',
    gemini: 'gemini-1.5-flash'
  };

  testTesting = false;
  testResult: { success: boolean; message: string } | null = null;

  // Local query history list
  historyList: HistoryItem[] = [];

  // X-Ray Journey state (document upload steps vs question answer steps)
  journey: 'document' | 'question' | null = null;
  stepIndex = 0;
  visualPulse = false;

  // UI state toggles
  busy = false;
  error = '';
  showMore = false;
  showWhy = false;
  showExperiment = false;

  // Polling interval timer
  private pollTimer?: ReturnType<typeof setInterval>;

  constructor(
    private http: HttpClient,
    private sanitizer: DomSanitizer
  ) {}

  // -----------------------------------------------------------
  // Session & Workspace Management
  // -----------------------------------------------------------

  /**
   * Initializes or restores the browser's local private session ID.
   * Stored in localStorage so refreshing the page preserves the local session.
   */
  private initSessionId(): string {
    let sid = localStorage.getItem('rag_xray_session_id');
    if (!sid) {
      sid = 'default';
      localStorage.setItem('rag_xray_session_id', sid);
    }
    return sid;
  }

  /** HTTP options including the session header for private workspace isolation */
  private get httpOptions() {
    return {
      headers: new HttpHeaders({
        'X-Session-ID': this.sessionId
      })
    };
  }

  /** Name of the currently active workspace */
  get currentWorkspaceName(): string {
    const found = this.workspaces.find(w => w.id === this.sessionId);
    if (found) {
      return this.cleanWorkspaceName(found.name);
    }
    if (this.docs.length > 0) {
      return this.docs[0].filename;
    }
    return 'New Workspace';
  }

  /** Whether the currently selected workspace is the latest active workspace */
  get isCurrentWorkspaceLatest(): boolean {
    const found = this.workspaces.find(w => w.id === this.sessionId);
    return !!found?.is_latest;
  }

  /** Helper to strip repetitive 'Workspace: ' prefix from display titles */
  cleanWorkspaceName(name: string): string {
    if (!name) return 'Workspace';
    return name.replace(/^Workspace:\s*/i, '');
  }

  /** Loads all available local workspaces that contain documents */
  loadWorkspaces(): void {
    this.http.get<Workspace[]>(`${API_URL}/workspaces`).subscribe({
      next: (list) => {
        this.workspaces = list || [];
      },
      error: () => {}
    });
  }

  /** Switches back to an existing workspace */
  switchWorkspace(ws: Workspace): void {
    this.sessionId = ws.id;
    localStorage.setItem('rag_xray_session_id', ws.id);
    this.showWorkspaceDropdown = false;
    this.reset();
    this.refresh();
    this.loadHistory();
    this.loadWorkspaces();
  }

  /** Starts a fresh workspace (only gets saved into list when a document is uploaded) */
  createNewWorkspace(): void {
    const newSid = 'sess_' + Math.random().toString(36).substring(2, 9) + '_' + Date.now().toString(36);
    this.sessionId = newSid;
    localStorage.setItem('rag_xray_session_id', newSid);
    this.showWorkspaceDropdown = false;
    this.reset();
    this.refresh();
    this.loadHistory();
    this.loadWorkspaces();
  }

  /** Deletes an existing workspace and cleans up its data */
  deleteWorkspace(ws: Workspace, event: Event): void {
    event.stopPropagation();
    if (!confirm(`Delete workspace "${ws.name}" and all its documents?`)) {
      return;
    }

    this.http.delete(`${API_URL}/workspaces/${ws.id}`).subscribe({
      next: () => {
        if (this.sessionId === ws.id) {
          const remaining = this.workspaces.filter(w => w.id !== ws.id);
          if (remaining.length > 0) {
            this.switchWorkspace(remaining[0]);
          } else {
            this.createNewWorkspace();
          }
        } else {
          this.loadWorkspaces();
        }
      },
      error: (err) => {
        this.error = this.extractErrorMessage(err);
      }
    });
  }

  // -----------------------------------------------------------
  // Lifecycle Hooks
  // -----------------------------------------------------------

  ngOnInit(): void {
    this.loadModelSettings();
    this.loadWorkspaces();
    this.refresh();
    this.loadHistory();

    // Poll the backend every 1.1s to update document processing status
    this.pollTimer = setInterval(() => this.refresh(), 1100);
  }

  // -----------------------------------------------------------
  // Multi-Provider LLM Configuration
  // -----------------------------------------------------------

  loadModelSettings(): void {
    try {
      const saved = localStorage.getItem('rag_model_settings');
      if (saved) {
        const parsed = JSON.parse(saved);
        this.selectedProvider = parsed.provider || 'groq';
        if (parsed.providerKeys) {
          this.providerKeys = { ...this.providerKeys, ...parsed.providerKeys };
        } else if (parsed.apiKey && parsed.provider) {
          this.providerKeys[parsed.provider] = parsed.apiKey;
        }
        if (parsed.providerModels) {
          this.providerModels = { ...this.providerModels, ...parsed.providerModels };
        }
        this.customModel = this.providerModels[this.selectedProvider] || parsed.model || this.getDefaultModel(this.selectedProvider);
        this.apiKey = this.providerKeys[this.selectedProvider] || '';
      } else {
        this.selectedProvider = 'groq';
        this.customModel = 'llama-3.3-70b-versatile';
        this.apiKey = '';
      }
    } catch (e) {}
  }

  saveModelSettingsToStorage(): void {
    this.providerKeys[this.selectedProvider] = this.apiKey;
    this.providerModels[this.selectedProvider] = this.customModel;
    const data = {
      provider: this.selectedProvider,
      model: this.customModel,
      apiKey: this.apiKey,
      providerKeys: this.providerKeys,
      providerModels: this.providerModels
    };
    localStorage.setItem('rag_model_settings', JSON.stringify(data));
  }

  saveModelSettings(): void {
    this.saveModelSettingsToStorage();
    this.showModelSettings = false;
    this.testResult = null;
  }

  selectProvider(p: 'ollama' | 'groq' | 'openai' | 'gemini'): void {
    // Preserve current provider key and model
    if (this.selectedProvider) {
      this.providerKeys[this.selectedProvider] = this.apiKey;
      this.providerModels[this.selectedProvider] = this.customModel;
    }
    this.selectedProvider = p;
    this.customModel = this.providerModels[p] || this.getDefaultModel(p);
    this.apiKey = this.providerKeys[p] || '';
    this.testResult = null;
    this.saveModelSettingsToStorage();

    // If switching to cloud provider without a key set, open settings dialog automatically
    if (p !== 'ollama' && (!this.apiKey || !this.apiKey.trim())) {
      this.showModelSettings = true;
    }
  }

  getDefaultModel(p: string): string {
    switch (p) {
      case 'groq': return 'llama-3.3-70b-versatile';
      case 'gemini': return 'gemini-1.5-flash';
      case 'openai': return 'gpt-4o-mini';
      default: return 'qwen2.5-coder:7b';
    }
  }

  hasApiKeyForProvider(p: string): boolean {
    if (p === 'ollama') return true;
    const key = (this.selectedProvider === p ? this.apiKey : this.providerKeys[p]) || '';
    return key.trim().length > 5;
  }

  getApiKeyStatusText(p: string): string {
    if (p === 'ollama') return 'Local';
    return this.hasApiKeyForProvider(p) ? 'Ready ✓' : 'Add Key ⚙';
  }

  getProviderName(p: string): string {
    switch (p) {
      case 'groq': return 'Groq Cloud';
      case 'openai': return 'OpenAI';
      case 'gemini': return 'Google Gemini';
      default: return 'Ollama (Local)';
    }
  }

  getProviderIcon(p: string): string {
    switch (p) {
      case 'groq': return '⚡';
      case 'openai': return '🤖';
      case 'gemini': return '✨';
      default: return '🖥️';
    }
  }

  testConnection(): void {
    this.testTesting = true;
    this.testResult = null;
    this.providerKeys[this.selectedProvider] = this.apiKey;
    const payload = {
      provider: this.selectedProvider,
      model: this.customModel,
      api_key: this.apiKey
    };
    this.http.post<any>(`${API_URL}/models/test`, payload).subscribe({
      next: (res) => {
        this.testTesting = false;
        if (res.status === 'ok') {
          this.testResult = {
            success: true,
            message: res.message || `Connected to ${res.model} successfully!`
          };
        } else {
          this.testResult = {
            success: false,
            message: res.message || 'Connection failed.'
          };
        }
      },
      error: (err) => {
        this.testTesting = false;
        this.testResult = {
          success: false,
          message: this.extractErrorMessage(err)
        };
      }
    });
  }

  ngOnDestroy(): void {
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
    }
    this.stopThinkingTimer();
  }

  // -----------------------------------------------------------
  // Computed Properties (Getters)
  // -----------------------------------------------------------

  /** Documents that have finished indexing and are ready for search */
  get ready(): Doc[] {
    return this.docs.filter(d => d.status === 'ready');
  }

  /** The stages to display in the current X-Ray journey */
  get stages(): Stage[] {
    if (this.journey === 'document') {
      return this.currentDoc?.stages || [];
    }
    if (this.journey === 'question') {
      return this.answer?.steps || [];
    }
    return [];
  }

  /** Currently selected step in the X-Ray journey */
  get currentStep(): Stage | null {
    return this.stages[this.stepIndex] || null;
  }

  /** Documents checked/selected by the user for querying */
  get chosenDocs(): Doc[] {
    return this.docs.filter(d => this.selectedIds.has(d.id) && d.status === 'ready');
  }

  /** The top-K ranked chunks used as context */
  get activeRanked(): Rank[] {
    return this.answer?.ranked.slice(0, this.topK) || [];
  }

  /** Total stages completed for the current document */
  get totalStages(): number {
    return this.currentDoc?.stages?.length || this.currentDoc?.stage_count || 0;
  }

  /** Percentage progress of document processing */
  get stageProgress(): number {
    if (this.currentDoc?.status === 'ready') {
      return 100;
    }
    return Math.min(95, Math.round((this.totalStages / 6) * 100));
  }

  // -----------------------------------------------------------
  // Document Operations
  // -----------------------------------------------------------

  /** Fetches the list of documents belonging to this local session */
  refresh(): void {
    this.http.get<Doc[]>(`${API_URL}/documents`, this.httpOptions).subscribe({
      next: (updatedDocs) => {
        // Auto-select newly added ready documents
        for (const doc of updatedDocs) {
          const isNewlyReady = doc.status === 'ready' &&
            !this.selectedIds.has(doc.id) &&
            !this.docs.some(old => old.id === doc.id && old.status === 'ready');

          if (isNewlyReady) {
            this.selectedIds.add(doc.id);
          }
        }

        this.docs = updatedDocs;

        // Auto-select first document if none is selected yet
        if (!this.currentDoc && updatedDocs.length > 0) {
          const defaultDoc = updatedDocs.find(d => d.status === 'ready') || updatedDocs[0];
          this.loadDoc(defaultDoc.id, false);
        } else if (this.currentDoc) {
          // Refresh currently viewed document if its stage count or status changed
          const matchingDoc = updatedDocs.find(d => d.id === this.currentDoc?.id);
          const needsReload = matchingDoc && (
            matchingDoc.stage_count !== this.totalStages ||
            matchingDoc.status !== this.currentDoc.status
          );
          if (needsReload) {
            this.loadDoc(matchingDoc.id, false);
          }
        }
      },
      error: () => {}
    });
  }

  /** Triggers the native OS file picker dynamically via a trusted user click gesture */
  triggerFilePicker(event?: Event): void {
    if (event) {
      event.stopPropagation();
    }
    const input = document.createElement('input');
    input.type = 'file';
    input.multiple = true;
    input.accept = '.pdf,.txt,.md';
    input.style.position = 'fixed';
    input.style.left = '-9999px';
    input.style.top = '-9999px';
    document.body.appendChild(input);

    input.onchange = (e: Event) => {
      const target = e.target as HTMLInputElement;
      if (target && target.files && target.files.length > 0) {
        this.uploadFileArray(Array.from(target.files));
      }
      if (document.body.contains(input)) {
        document.body.removeChild(input);
      }
    };

    // Clean up if cancelled or closed
    window.addEventListener('focus', () => {
      setTimeout(() => {
        if (document.body.contains(input) && (!input.files || input.files.length === 0)) {
          document.body.removeChild(input);
        }
      }, 500);
    }, { once: true });

    input.click();
  }

  /** Handles native file input change event */
  onFileInputChange(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input && input.files && input.files.length > 0) {
      const files = Array.from(input.files);
      input.value = ''; // Reset so the same file can be selected again
      this.uploadFileArray(files);
    }
  }

  /** Uploads an array of File objects */
  uploadFileArray(files: File[]): void {
    if (!files || files.length === 0) return;
    this.error = '';
    this.uploading = true;

    let pending = files.length;
    for (const file of files) {
      const formData = new FormData();
      formData.append('file', file);

      this.http.post<{ document_id: string }>(`${API_URL}/documents`, formData, this.httpOptions).subscribe({
        next: (result) => {
          pending--;
          if (pending <= 0) {
            this.uploading = false;
          }
          this.selectedIds.add(result.document_id);
          this.loadDoc(result.document_id, true);
          this.refresh();
          this.loadWorkspaces();
          this.mobileTab = 'main';
        },
        error: (err) => {
          pending--;
          if (pending <= 0) {
            this.uploading = false;
          }
          this.error = this.extractErrorMessage(err);
        }
      });
    }
  }

  /** Uploads one or more selected files into the local session (for drag & drop) */
  uploadFiles(fileList: FileList | null): void {
    if (!fileList || fileList.length === 0) {
      return;
    }
    this.uploadFileArray(Array.from(fileList));
  }

  /** Handles drag and drop file uploads */
  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.uploadFiles(event.dataTransfer?.files || null);
  }

  /** Loads full document metadata and stage details */
  loadDoc(id: string, openJourney = true): void {
    this.http.get<Doc>(`${API_URL}/documents/${id}`, this.httpOptions).subscribe({
      next: (docData) => {
        this.currentDoc = docData;
        if (openJourney) {
          this.journey = 'document';
          this.stepIndex = 0;
          this.showMore = false;
        }
      },
      error: (err) => {
        this.error = this.extractErrorMessage(err);
      }
    });
  }

  /** Toggles document selection for querying */
  toggleDoc(id: string, isChecked: boolean): void {
    if (isChecked) {
      this.selectedIds.add(id);
    } else {
      this.selectedIds.delete(id);
    }
    this.selectedIds = new Set(this.selectedIds);
  }

  /** Deletes a document and its ChromaDB chunk vectors */
  removeDoc(doc: Doc): void {
    const confirmed = confirm(`Delete ${doc.filename} and its indexed chunks?`);
    if (!confirmed) {
      return;
    }

    this.http.delete(`${API_URL}/documents/${doc.id}`, this.httpOptions).subscribe({
      next: () => {
        this.selectedIds.delete(doc.id);
        if (this.currentDoc?.id === doc.id) {
          this.currentDoc = null;
          this.journey = null;
        }
        this.refresh();
        this.loadWorkspaces();
      },
      error: (err) => {
        this.error = this.extractErrorMessage(err);
      }
    });
  }

  // -----------------------------------------------------------
  // Question & Query History
  // -----------------------------------------------------------

  /** Loads past queries and answers for this local session */
  loadHistory(): void {
    this.http.get<HistoryItem[]>(`${API_URL}/history`, this.httpOptions).subscribe({
      next: (list) => {
        this.historyList = list || [];
      },
      error: () => {}
    });
  }

  /**
   * Sets the suggested prompt in the question text box without auto-submitting.
   * Gives user full control to edit or review before clicking Ask.
   */
  setPrompt(promptText: string): void {
    this.question = promptText;
  }

  /** Backwards compatibility alias for setPrompt */
  askSuggestion(promptText: string): void {
    this.setPrompt(promptText);
  }

  /** Starts real-time simulation of live RAG pipeline steps during AI thinking */
  startThinkingTimer(): void {
    this.stopThinkingTimer();
    this.thinkingSeconds = 0;
    this.thinkingStep = 1;
    this.thinkingInterval = setInterval(() => {
      this.thinkingSeconds = Math.round((this.thinkingSeconds + 0.1) * 10) / 10;
      if (this.thinkingSeconds >= 0.4 && this.thinkingStep < 2) {
        this.thinkingStep = 2; // Query embedding
      }
      if (this.thinkingSeconds >= 1.0 && this.thinkingStep < 3) {
        this.thinkingStep = 3; // Compare vectors (ChromaDB cosine similarity)
      }
      if (this.thinkingSeconds >= 1.6 && this.thinkingStep < 4) {
        this.thinkingStep = 4; // Rank & choose top-K
      }
      if (this.thinkingSeconds >= 2.2 && this.thinkingStep < 5) {
        this.thinkingStep = 5; // LLM synthesis (Qwen 2.5)
      }
    }, 100);
  }

  stopThinkingTimer(): void {
    if (this.thinkingInterval) {
      clearInterval(this.thinkingInterval);
      this.thinkingInterval = undefined;
    }
  }

  /** Handles Enter key in chat input (Enter to send, Shift+Enter for newline) */
  onEnterKey(event: Event): void {
    const keyboardEvent = event as KeyboardEvent;
    if (!keyboardEvent.shiftKey) {
      keyboardEvent.preventDefault();
      this.ask();
    }
  }

  /** Sends the question and selected document IDs to /api/ask */
  ask(overrideQuestion?: string): void {
    const q = (overrideQuestion || this.question || this.activeQuestion || this.answer?.question || '').trim();
    if (!q || this.chosenDocs.length === 0 || this.busy) {
      return;
    }

    // If cloud provider is selected but API key is missing, prompt user
    if (this.selectedProvider !== 'ollama' && (!this.apiKey || !this.apiKey.trim())) {
      this.showModelSettings = true;
      this.error = `Please enter your ${this.getProviderName(this.selectedProvider)} API key in Settings (or switch to Ollama for offline use).`;
      return;
    }

    this.activeQuestion = q;
    this.question = ''; // Clears input box for subsequent follow-up, ChatGPT style!
    this.error = '';
    this.busy = true;
    this.answer = null;
    this.journey = null;
    this.showThoughtProcess = false;

    this.startThinkingTimer();

    const payload = {
      question: q,
      document_ids: this.chosenDocs.map(d => d.id),
      top_k: this.topK,
      provider: this.selectedProvider,
      model: this.customModel,
      api_key: this.apiKey
    };

    this.http.post<Answer>(`${API_URL}/ask`, payload, this.httpOptions).subscribe({
      next: (answerResponse) => {
        this.stopThinkingTimer();
        this.thinkingStep = 5;
        answerResponse.steps = this.normalizeSteps(
          answerResponse.steps,
          answerResponse.question,
          answerResponse.model,
          answerResponse.provider || this.selectedProvider
        );
        this.answer = answerResponse;
        this.answer.thinking_time = this.thinkingSeconds > 0 ? this.thinkingSeconds.toFixed(1) : '1.4';
        this.selectedSource = answerResponse.sources[0] || null;
        this.busy = false;
        // Refresh local history list and workspaces count
        this.loadHistory();
        this.loadWorkspaces();
      },
      error: (err) => {
        this.stopThinkingTimer();
        this.error = this.extractErrorMessage(err);
        this.busy = false;
        // Restore question text if request failed so user doesn't lose their input
        this.question = q;
      }
    });
  }

  /**
   * Re-runs the RAG pipeline directly from the Top-K experiment slider
   * Keeps the journey drawer open at Step 4 and updates results live
   */
  reRunExperiment(): void {
    const q = (this.activeQuestion || this.answer?.question || this.question || '').trim();
    if (!q || this.chosenDocs.length === 0 || this.busy) {
      return;
    }

    if (this.selectedProvider !== 'ollama' && (!this.apiKey || !this.apiKey.trim())) {
      this.showModelSettings = true;
      this.error = `Please enter your ${this.getProviderName(this.selectedProvider)} API key in Settings (or switch to Ollama for offline use).`;
      return;
    }

    this.busy = true;
    this.activeQuestion = q;
    this.error = '';
    this.startThinkingTimer();

    const payload = {
      question: q,
      document_ids: this.chosenDocs.map(d => d.id),
      top_k: this.topK,
      provider: this.selectedProvider,
      model: this.customModel,
      api_key: this.apiKey
    };

    this.http.post<Answer>(`${API_URL}/ask`, payload, this.httpOptions).subscribe({
      next: (answerResponse) => {
        this.stopThinkingTimer();
        this.thinkingStep = 5;
        answerResponse.steps = this.normalizeSteps(
          answerResponse.steps,
          answerResponse.question,
          answerResponse.model,
          answerResponse.provider || this.selectedProvider
        );
        this.answer = answerResponse;
        this.answer.thinking_time = this.thinkingSeconds > 0 ? this.thinkingSeconds.toFixed(1) : '1.4';
        this.selectedSource = answerResponse.sources[0] || null;
        this.busy = false;

        // Keep journey open at Step 4 (Rank & choose) and refresh state
        this.journey = 'question';
        this.chooseStep(3);
        this.showExperiment = true;
        this.scrollToExperiment();

        this.loadHistory();
        this.loadWorkspaces();
      },
      error: (err) => {
        this.stopThinkingTimer();
        this.error = this.extractErrorMessage(err);
        this.busy = false;
      }
    });
  }

  /**
   * Cleans legacy static pseudocode from earlier versions so the user
   * always sees meaningful, dynamic runtime descriptions for every step.
   */
  normalizeSteps(steps: Stage[], question?: string, model?: string, provider?: string): Stage[] {
    if (!steps || !steps.length) return [];
    const qText = question || this.activeQuestion || '';
    const wordCount = qText ? qText.split(/\s+/).filter(Boolean).length : 0;
    const m = model || this.customModel || 'LLM';
    const p = provider || this.selectedProvider || 'AI';
    const pName = this.getProviderName(p);

    return steps.map((st) => {
      if (st.name === 'Your question' && (!st.summary || st.operation === 'Capture user question string')) {
        return {
          ...st,
          summary: `Query: "${qText}" (${wordCount} words)`,
          operation: `Captured and tokenized query string: "${qText}"`
        };
      }
      if (st.name === 'Query embedding' && (!st.summary || st.operation?.includes('model.encode'))) {
        return {
          ...st,
          summary: `Dense vector created · 384 dimensions`,
          operation: `SentenceTransformer(all-MiniLM-L6-v2).encode(["${qText}"], normalize_embeddings=True)`
        };
      }
      if (st.name === 'Compare vectors' && (!st.summary || st.operation?.includes('candidate') || st.operation?.includes('cosine_similarity('))) {
        return {
          ...st,
          summary: `Calculated cosine similarity dot products across indexed document chunks`,
          operation: `Calculated cosine similarity dot products between query vector and document chunk vectors in ChromaDB`
        };
      }
      if (st.name === 'Rank & choose' && (!st.summary || st.operation?.includes('candidate') || st.operation?.includes('sorted(candidates'))) {
        return {
          ...st,
          summary: `Ranked document chunks descending and selected Top matching chunks`,
          operation: `Sorted document chunks by similarity descending and retained top chunks for context`
        };
      }
      if (st.name === 'Generate answer' && (!st.summary || st.operation?.includes('Generate grounded answer with Ollama'))) {
        return {
          ...st,
          summary: `Grounded answer synthesized using ${m} (${pName})`,
          operation: `Grounded LLM synthesis using model "${m}" via ${pName}`
        };
      }
      return st;
    });
  }

  /** Selects a past query from history and restores its answer and X-Ray stages */
  selectHistoryItem(item: HistoryItem): void {
    this.activeQuestion = item.question;
    this.question = '';
    const cloned = { ...item };
    if (cloned.steps) {
      cloned.steps = this.normalizeSteps(cloned.steps, cloned.question, cloned.model, cloned.provider);
    }
    this.answer = cloned;
    this.selectedSource = cloned.sources[0] || null;
    this.topK = cloned.top_k || 3;
    this.journey = null;
    this.showThoughtProcess = false;
  }

  // -----------------------------------------------------------
  // 1-Click Sample Document Loader
  // -----------------------------------------------------------

  /** Loads the built-in sample Employee Handbook with 1 click */
  loadSampleDoc(): void {
    if (this.uploadingSample) return;
    this.uploadingSample = true;
    this.error = '';

    this.http.post<{ document_id: string; filename: string; status: string }>(
      `${API_URL}/documents/sample`,
      {},
      this.httpOptions
    ).subscribe({
      next: (res) => {
        this.uploadingSample = false;
        this.selectedIds.add(res.document_id);
        this.loadDoc(res.document_id, true);
        this.refresh();
        this.loadWorkspaces();
        this.mobileTab = 'main';
      },
      error: (err) => {
        this.uploadingSample = false;
        this.error = this.extractErrorMessage(err);
      }
    });
  }

  /** Clears local question history */
  clearHistory(): void {
    if (!confirm('Clear your local query history?')) {
      return;
    }
    this.http.delete(`${API_URL}/history`, this.httpOptions).subscribe({
      next: () => {
        this.historyList = [];
        this.loadWorkspaces();
      },
      error: (err) => {
        this.error = this.extractErrorMessage(err);
      }
    });
  }

  // -----------------------------------------------------------
  // X-Ray Journey Navigation
  // -----------------------------------------------------------

  /** Opens or closes the X-Ray data journey panel */
  openJourney(which: 'document' | 'question'): void {
    if (which === 'document' && !this.currentDoc) {
      const readyDoc = this.docs.find(d => d.status === 'ready') || this.docs[0];
      if (readyDoc) {
        this.loadDoc(readyDoc.id);
        return;
      }
    }

    this.journey = this.journey === which ? null : which;
    this.stepIndex = 0;
    this.showMore = false;
    this.showWhy = false;
  }

  /** Switches to a specific step in the journey */
  chooseStep(stepNum: number): void {
    this.stepIndex = stepNum;
    this.showMore = false;
    this.showWhy = false;
    this.visualPulse = true;
    setTimeout(() => {
      this.visualPulse = false;
    }, 400);
  }

  /** Steps forward or backward in the journey */
  step(delta: number): void {
    const total = this.stages.length;
    if (total > 0) {
      this.chooseStep((this.stepIndex + delta + total) % total);
    }
  }

  /** Jump directly to Step 4 (Rank & choose) to experiment with Top-K slider */
  compareDoc(): void {
    if (this.answer) {
      this.journey = 'question';
      this.chooseStep(3);
      this.showExperiment = true;
      this.mobileTab = 'main'; // Ensure mobile drawer view is visible
      this.scrollToExperiment();
    }
  }

  /** Smoothly scrolls the journey drawer content directly down to the Top-K Experiment section */
  scrollToExperiment(): void {
    const tryScroll = () => {
      const el = document.getElementById('topk-experiment');
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        el.classList.add('highlight-pulse');
        setTimeout(() => el.classList.remove('highlight-pulse'), 1500);
        return true;
      }
      return false;
    };

    // First attempt shortly after Angular DOM change detection
    setTimeout(() => {
      if (!tryScroll()) {
        // Fallback retry if drawer animation/DOM takes an extra frame
        setTimeout(tryScroll, 120);
      }
    }, 80);
  }

  selectSource(source: Rank): void {
    this.selectedSource = source;
  }

  reset(): void {
    this.answer = null;
    this.journey = null;
    this.currentDoc = null;
    this.question = '';
    this.activeQuestion = '';
    this.error = '';
    this.busy = false;
    this.stopThinkingTimer();
  }

  // -----------------------------------------------------------
  // Interactive JSON Tree & Copy Helpers
  // -----------------------------------------------------------
  collapsedPaths = new Set<string>();
  copiedBlock: string | null = null;
  jsonViewModes: Record<string, 'tree' | 'raw'> = {
    before: 'tree',
    after: 'tree'
  };

  toggleViewMode(blockId: string): void {
    const current = this.jsonViewModes[blockId] || 'tree';
    this.jsonViewModes[blockId] = current === 'raw' ? 'tree' : 'raw';
  }

  toggleCollapse(path: string): void {
    if (this.collapsedPaths.has(path)) {
      this.collapsedPaths.delete(path);
    } else {
      this.collapsedPaths.add(path);
    }
  }

  isCollapsed(path: string): boolean {
    return this.collapsedPaths.has(path);
  }

  collapseAll(blockId: string, val: unknown): void {
    const data = this.clarifyJsonKeys(val);
    this.collectAllPaths(data, blockId, this.collapsedPaths);
  }

  expandAll(blockId: string): void {
    for (const p of Array.from(this.collapsedPaths)) {
      if (p.startsWith(blockId)) {
        this.collapsedPaths.delete(p);
      }
    }
  }

  private collectAllPaths(val: unknown, currentPath: string, set: Set<string>): void {
    if (this.isObject(val)) {
      set.add(currentPath);
      for (const k of this.getKeys(val)) {
        this.collectAllPaths((val as Record<string, unknown>)[k], `${currentPath}.${k}`, set);
      }
    } else if (this.isArray(val)) {
      set.add(currentPath);
      (val as unknown[]).forEach((item, i) => {
        this.collectAllPaths(item, `${currentPath}[${i}]`, set);
      });
    }
  }

  copyJson(value: unknown, blockId: string): void {
    const data = this.clarifyJsonKeys(value);
    const text = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
    if (navigator?.clipboard?.writeText) {
      navigator.clipboard.writeText(text).then(() => {
        this.copiedBlock = blockId;
        setTimeout(() => {
          if (this.copiedBlock === blockId) {
            this.copiedBlock = null;
          }
        }, 2000);
      }).catch(() => {
        this.fallbackCopy(text, blockId);
      });
    } else {
      this.fallbackCopy(text, blockId);
    }
  }

  private fallbackCopy(text: string, blockId: string): void {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand('copy');
      this.copiedBlock = blockId;
      setTimeout(() => {
        if (this.copiedBlock === blockId) {
          this.copiedBlock = null;
        }
      }, 2000);
    } catch {}
    document.body.removeChild(ta);
  }

  isObject(val: unknown): boolean {
    return typeof val === 'object' && val !== null && !Array.isArray(val);
  }

  isArray(val: unknown): boolean {
    return Array.isArray(val);
  }

  isString(val: unknown): boolean {
    return typeof val === 'string';
  }

  isNumber(val: unknown): boolean {
    return typeof val === 'number';
  }

  isBoolean(val: unknown): boolean {
    return typeof val === 'boolean';
  }

  isNull(val: unknown): boolean {
    return val === null;
  }

  isShortPrimitiveArray(val: unknown): boolean {
    if (!Array.isArray(val)) return false;
    return val.length <= 4 && val.every(item => typeof item === 'number' || typeof item === 'string' || typeof item === 'boolean');
  }

  getKeys(obj: unknown): string[] {
    if (!this.isObject(obj)) return [];
    return Object.keys(obj as Record<string, unknown>);
  }

  // -----------------------------------------------------------
  // Markdown & Text Rendering
  // -----------------------------------------------------------

  /** Maps older or ambiguous JSON keys to clear, beginner-friendly names */
  clarifyJsonKeys(obj: unknown): unknown {
    if (obj === null || obj === undefined) return obj;
    if (Array.isArray(obj)) {
      return obj.map(item => this.clarifyJsonKeys(item));
    }
    if (typeof obj === 'object') {
      const keyMap: Record<string, string> = {
        candidate_pool_size: 'total_document_chunks_in_scope',
        candidate_count: 'total_document_chunks_evaluated',
        stored_chunks: 'stored_document_chunks_to_compare',
        total_scored: 'total_document_chunks_scored',
        highest_similarity: 'top_similarity_score',
        lowest_similarity: 'lowest_similarity_score',
        selected_docs_count: 'selected_documents_count',
        target_top_k: 'top_k_chunks_to_retrieve',
        top_k: 'top_k_chunks_to_retrieve',
        query_text: 'user_question',
        bytes: 'file_size_bytes',
        filename: 'uploaded_filename',
        count: 'total_document_chunks_created',
        preview: 'document_chunks_sample',
        chunks: 'document_chunks_to_embed',
        first_chunk: 'first_document_chunk_sample',
        shape: 'vector_matrix_dimensions',
        first_vectors: 'document_chunk_vectors_sample',
        first_vector_norm: 'first_chunk_vector_length_norm',
        vector_count: 'document_chunk_vectors_to_store',
        stored_count: 'stored_document_chunks_in_db',
        first_ids: 'stored_chunk_ids_sample',
        context_count: 'grounded_context_chunks_count',
        citations_count: 'cited_document_chunks_count',
        answer_words: 'generated_answer_words',
        excluded_count: 'excluded_lower_score_chunks_count',
        excluded_preview: 'excluded_chunks_sample',
        scores: 'top_matching_document_chunks',
        selected: 'selected_top_document_chunks'
      };

      const clarified: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
        const newKey = keyMap[k] || k;
        clarified[newKey] = this.clarifyJsonKeys(v);
      }
      return clarified;
    }
    return obj;
  }

  /** Formats a JSON value or string for preview */
  preview(value: unknown): string {
    const normalized = this.clarifyJsonKeys(value);
    const raw = typeof normalized === 'string' ? normalized : JSON.stringify(normalized, null, 2);
    if (raw === undefined) {
      return '—';
    }
    if (!this.showMore && raw.length > 700) {
      return raw.slice(0, 700) + '\n…';
    }
    return raw;
  }

  hasMore(value: unknown): boolean {
    const normalized = this.clarifyJsonKeys(value);
    const raw = typeof normalized === 'string' ? normalized : JSON.stringify(normalized, null, 2) || '';
    return raw.length > 700;
  }

  scoreWidth(score: number): string {
    const percentage = Math.max(0, Math.min(100, score * 100));
    return `${percentage}%`;
  }

  MathMin(a: number, b: number): number {
    return Math.min(a, b);
  }

  /** Formats an ISO date string to readable time */
  formatTime(isoString?: string): string {
    if (!isoString) return '';
    try {
      const d = new Date(isoString);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch {
      return '';
    }
  }

  /**
   * Converts markdown syntax into clean HTML:
   * - Headings (#, ##, ###)
   * - Bold (**text**) and Italics (*text*)
   * - Bullet lists (- item) and Numbered lists (1. item)
   * - Code blocks (```code```) and Inline code (`code`)
   * - Citation badges ([1], [2])
   */
  renderMarkdown(content: string): SafeHtml {
    if (!content) {
      return '';
    }

    // 1. Escape HTML special characters to prevent XSS
    let html = content
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // 2. Format code blocks and inline code
    html = html.replace(/```([\s\S]*?)```/g, '<pre class="chat-code"><code>$1</code></pre>');
    html = html.replace(/`([^`]+)`/g, '<code class="chat-inline-code">$1</code>');

    // 3. Format headings
    html = html.replace(/^### (.*$)/gim, '<h4 class="chat-h4">$1</h4>');
    html = html.replace(/^## (.*$)/gim, '<h3 class="chat-h3">$1</h3>');
    html = html.replace(/^# (.*$)/gim, '<h2 class="chat-h2">$1</h2>');

    // 4. Format bold and italics
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // 5. Format citations [1], [2] as styled badges
    html = html.replace(/\[(\d+)\]/g, '<span class="cite-pill">[$1]</span>');

    // 6. Format lists and paragraphs
    const lines = html.split('\n');
    let inList = false;
    let listType = '';
    const output: string[] = [];

    for (const line of lines) {
      const trimmed = line.trim();
      const numberedMatch = trimmed.match(/^(\d+)\.\s+(.*)$/);
      const bulletMatch = trimmed.match(/^[-*•]\s+(.*)$/);

      if (numberedMatch) {
        if (!inList || listType !== 'ol') {
          if (inList) output.push(`</${listType}>`);
          output.push('<ol class="chat-list ol-list">');
          inList = true;
          listType = 'ol';
        }
        output.push(`<li>${numberedMatch[2]}</li>`);
      } else if (bulletMatch) {
        if (!inList || listType !== 'ul') {
          if (inList) output.push(`</${listType}>`);
          output.push('<ul class="chat-list ul-list">');
          inList = true;
          listType = 'ul';
        }
        output.push(`<li>${bulletMatch[1]}</li>`);
      } else {
        if (inList) {
          output.push(`</${listType}>`);
          inList = false;
          listType = '';
        }

        const isHeadingOrCode = trimmed.startsWith('<h2') ||
          trimmed.startsWith('<h3') ||
          trimmed.startsWith('<h4') ||
          trimmed.startsWith('<pre');

        if (isHeadingOrCode) {
          output.push(trimmed);
        } else if (trimmed) {
          output.push(`<p class="chat-p">${trimmed}</p>`);
        }
      }
    }

    if (inList) {
      output.push(`</${listType}>`);
    }

    return this.sanitizer.bypassSecurityTrustHtml(output.join(''));
  }

  loadSample(): void {
    this.http.get('/assets/employee_handbook.txt', { responseType: 'text' }).subscribe({
      next: (textData) => {
        const file = new File([textData], 'employee_handbook.txt', { type: 'text/plain' });
        this.uploadFiles(new DataTransferFiles([file]).files);
      },
      error: () => {
        this.error = 'Use the included sample_docs/employee_handbook.txt with Choose files.';
      }
    });
  }

  private extractErrorMessage(err: any): string {
    return err?.error?.detail || err?.message || 'Something went wrong';
  }
}

/** Helper adapter for mock FileList in the sample load button */
class DataTransferFiles {
  readonly files: FileList;
  constructor(files: File[]) {
    const dt = new DataTransfer();
    files.forEach(f => dt.items.add(f));
    this.files = dt.files;
  }
}
