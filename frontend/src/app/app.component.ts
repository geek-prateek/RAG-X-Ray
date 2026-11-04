/**
 * Angular Root Component for RAG X-Ray UI
 * =======================================
 * Manages the interactive dashboard:
 * - Document uploads and live status polling.
 * - Asking questions with top-k context chunk retrieval.
 * - Inspecting step-by-step X-Ray transformations (Data Journey).
 * - Rendering ChatGPT-style formatted Markdown with citation badges.
 */
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';

// -------------------------------------------------------------
// Type Definitions
// -------------------------------------------------------------

/** Represents a single transformation step (e.g. Clean text, Embeddings) */
interface Stage {
  name: string;
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

/** Response received from the /api/ask endpoint */
interface Answer {
  answer: string;
  answer_mode: string;
  sources: Rank[];
  ranked: Rank[];
  steps: Stage[];
  top_k: number;
  total_candidates: number;
  model: string;
}

// Backend API Base URL
const API_URL = 'http://localhost:8000/api';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent implements OnInit, OnDestroy {
  // Document list and selections
  docs: Doc[] = [];
  selectedIds = new Set<string>();
  currentDoc: Doc | null = null;

  // Question & Answer state
  question = 'How do employees request annual leave?';
  answer: Answer | null = null;
  topK = 3;
  selectedSource: Rank | null = null;

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
  // Lifecycle Hooks
  // -----------------------------------------------------------

  ngOnInit(): void {
    // Initial fetch of document list
    this.refresh();

    // Poll the backend every 1.1s to update document processing status
    this.pollTimer = setInterval(() => this.refresh(), 1100);
  }

  ngOnDestroy(): void {
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
    }
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

  /** Fetches the latest list of documents from the backend */
  refresh(): void {
    this.http.get<Doc[]>(`${API_URL}/documents`).subscribe({
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

        // Refresh currently viewed document if its stage count or status changed
        if (this.currentDoc) {
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

  /** Uploads one or more selected files to the backend */
  uploadFiles(fileList: FileList | null): void {
    if (!fileList || fileList.length === 0) {
      return;
    }
    this.error = '';

    for (const file of Array.from(fileList)) {
      const formData = new FormData();
      formData.append('file', file);

      this.http.post<{ document_id: string }>(`${API_URL}/documents`, formData).subscribe({
        next: (result) => {
          this.selectedIds.add(result.document_id);
          this.loadDoc(result.document_id, true);
          this.refresh();
        },
        error: (err) => {
          this.error = this.extractErrorMessage(err);
        }
      });
    }
  }

  /** Handles drag and drop file uploads */
  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.uploadFiles(event.dataTransfer?.files || null);
  }

  /** Loads full document metadata and stage details */
  loadDoc(id: string, openJourney = true): void {
    this.http.get<Doc>(`${API_URL}/documents/${id}`).subscribe({
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

  /** Toggles document selection for questioning */
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

    this.http.delete(`${API_URL}/documents/${doc.id}`).subscribe({
      next: () => {
        this.selectedIds.delete(doc.id);
        if (this.currentDoc?.id === doc.id) {
          this.currentDoc = null;
          this.journey = null;
        }
        this.refresh();
      },
      error: (err) => {
        this.error = this.extractErrorMessage(err);
      }
    });
  }

  // -----------------------------------------------------------
  // Question & RAG Query
  // -----------------------------------------------------------

  /** Sends the question and selected document IDs to /api/ask */
  ask(): void {
    const trimmedQuestion = this.question.trim();
    if (!trimmedQuestion || this.chosenDocs.length === 0 || this.busy) {
      return;
    }

    this.error = '';
    this.busy = true;
    this.answer = null;
    this.journey = null;

    const payload = {
      question: trimmedQuestion,
      document_ids: this.chosenDocs.map(d => d.id),
      top_k: this.topK
    };

    this.http.post<Answer>(`${API_URL}/ask`, payload).subscribe({
      next: (answerResponse) => {
        this.answer = answerResponse;
        this.selectedSource = answerResponse.sources[0] || null;
        this.busy = false;
      },
      error: (err) => {
        this.error = this.extractErrorMessage(err);
        this.busy = false;
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
    }
  }

  selectSource(source: Rank): void {
    this.selectedSource = source;
  }

  reset(): void {
    this.answer = null;
    this.journey = null;
    this.currentDoc = null;
    this.error = '';
  }

  // -----------------------------------------------------------
  // Markdown & Text Rendering
  // -----------------------------------------------------------

  /** Formats a JSON value or string for preview */
  preview(value: unknown): string {
    const raw = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
    if (raw === undefined) {
      return '—';
    }
    if (!this.showMore && raw.length > 700) {
      return raw.slice(0, 700) + '\n…';
    }
    return raw;
  }

  hasMore(value: unknown): boolean {
    const raw = typeof value === 'string' ? value : JSON.stringify(value, null, 2) || '';
    return raw.length > 700;
  }

  scoreWidth(score: number): string {
    const percentage = Math.max(0, Math.min(100, score * 100));
    return `${percentage}%`;
  }

  MathMin(a: number, b: number): number {
    return Math.min(a, b);
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
