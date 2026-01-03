function triageApp() {
    return {
        // --- STATE ---
        step: 'upload', // 'upload' | 'review'
        loading: false,
        processed: false,
        filename: null,
        file: null,
        errorMessage: null,
        
        // Configuration
        textColumn: 'feedback',
        riskMode: 'balanced',
        
        // Data from Backend
        stats: { auto: 0, human: 0, auto_coverage: 0 },
        needsHuman: [],
        currentIndex: 0,
        
        // Human Decisions
        labels: { intent: '', issue: '', safe: false },
        pendingLabels: [],
        flashSuccess: false,

        // --- COMPUTED ---
        get currentTicket() {
            return this.needsHuman[this.currentIndex] || null;
        },

        // --- ACTIONS ---
        handleFile(e) {
            this.file = e.target.files[0];
            if(this.file) {
                this.filename = this.file.name;
                this.errorMessage = null;
            }
        },

        async processData() {
            if(!this.file) return;
            this.loading = true;
            this.errorMessage = null;

            const fd = new FormData();
            fd.append('file', this.file);
            fd.append('text_column', this.textColumn);
            
            try {
                // 1. Upload
                const uploadRes = await fetch('/upload_csv', { method: 'POST', body: fd });
                const uploadData = await uploadRes.json();
                if (uploadData.error) throw new Error(uploadData.error);
                
                // 2. Process
                const res = await fetch('/process', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ risk_tolerance: this.riskMode })
                });
                
                const data = await res.json();
                if (data.error) throw new Error(data.error);

                this.stats = data.stats;
                this.needsHuman = data.needs_human || [];
                this.currentIndex = 0;
                this.processed = true;
                
                if(this.needsHuman.length > 0) {
                    this.step = 'review';
                    this.syncForm();
                } else {
                    alert("All tickets automated! No human review needed.");
                }

            } catch (err) {
                console.error(err);
                this.errorMessage = err.message;
            } finally {
                this.loading = false;
            }
        },

        applyModelSuggestion() {
            if(!this.currentTicket) return;
            this.labels.intent = this.currentTicket.intent;
            this.labels.issue = this.currentTicket.issue;
        },

        syncForm() {
            if(!this.currentTicket) return;
            
            const existingLabel = this.pendingLabels.find(l => l.ticket_id === this.currentTicket.id);
            if (existingLabel) {
                this.labels.intent = existingLabel.intent;
                this.labels.issue = existingLabel.issue;
                this.labels.safe = existingLabel.safe_for_automation;
            } else {
                this.labels.intent = ''; 
                this.labels.issue = '';
                this.labels.safe = false;
            }
        },

        nextTicket() {
            if(this.currentIndex < this.needsHuman.length - 1) {
                this.currentIndex++;
                this.syncForm();
            }
        },

        prevTicket() {
            if(this.currentIndex > 0) {
                this.currentIndex--;
                this.syncForm();
            }
        },

        saveAndNext() {
            if(!this.currentTicket) return;
            
            if (!this.labels.intent || !this.labels.issue) {
                alert("Please select both an Intent and an Issue.");
                return;
            }

            const decision = {
                ticket_id: this.currentTicket.id,
                intent: this.labels.intent,
                issue: this.labels.issue,
                safe_for_automation: this.labels.safe
            };
            
            const existingIdx = this.pendingLabels.findIndex(l => l.ticket_id === decision.ticket_id);
            if (existingIdx > -1) {
                this.pendingLabels[existingIdx] = decision;
            } else {
                this.pendingLabels.push(decision);
            }
            
            this.flashSuccess = true; 
            setTimeout(() => {
                this.flashSuccess = false;
                if(this.currentIndex < this.needsHuman.length - 1) {
                    this.nextTicket();
                } else {
                    alert("Review complete! Export your labels now.");
                }
            }, 400); 
        },

        async exportLabels() {
            if(this.pendingLabels.length === 0) return;
            
            try {
                const res = await fetch('/label_bulk', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ human_labels: this.pendingLabels })
                });
                const data = await res.json();
                alert(`Successfully submitted ${data.count} labels.`);
                this.pendingLabels = [];
            } catch (e) {
                alert("Failed to export labels.");
            }
        },

        download(kind) {
            window.location.href = `/download/${kind}`;
        },
        
        reset() {
            // FIXED: Fully reset state AND file input value
            this.step = 'upload';
            this.processed = false;
            this.file = null;
            this.filename = null;
            this.errorMessage = null;
            this.pendingLabels = [];
            this.stats = { auto: 0, human: 0, auto_coverage: 0 };
            
            // Critical fix for allowing re-upload of same file
            const input = document.getElementById('fileInput');
            if(input) input.value = ''; 
        }
    }
}