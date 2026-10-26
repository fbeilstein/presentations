/**
 * viewer_download.js
 * In-browser client-side bundle repacking and download manager using JSZip.
 * Supports:
 *   1. Molecule mode: Selective orbital filtering (Frontier, σ, π, anti-bonds, ESP).
 *   2. Reaction mode: Choice between "Reaction Only" vs "Reaction + Molecules".
 * Activated when ?download is present in the URL query parameters.
 */

export class ViewerDownloadModal {
  constructor(app) {
    this.app = app;
    this.elBtn = document.getElementById('btn-header-download');
    this.elModal = document.getElementById('download-modal');
    this.elTitle = document.getElementById('download-modal-title');
    this.elBody = document.getElementById('download-modal-body');
    this.elClose = document.getElementById('btn-close-download');
    this.elCancel = document.getElementById('btn-cancel-download');
    this.elConfirm = document.getElementById('btn-confirm-download');
    this.init();
  }

  init() {
    const params = new URLSearchParams(window.location.search);
    const hasDownloadKey = params.has('download');

    if (this.elBtn) {
      this.elBtn.style.display = hasDownloadKey ? 'inline-flex' : 'none';
      this.elBtn.addEventListener('click', () => this.show());
    }

    this.elClose?.addEventListener('click', () => this.close());
    this.elCancel?.addEventListener('click', () => this.close());
    this.elConfirm?.addEventListener('click', () => this.handleConfirm());

    // Close on clicking backdrop outside dialog
    this.elModal?.addEventListener('click', (e) => {
      if (e.target === this.elModal) this.close();
    });
  }

  show() {
    const bundle = this.app.bundle;
    if (!bundle) return;

    if (bundle.isSingleMolecule) {
      this.renderMoleculeDialog(bundle);
    } else {
      this.renderReactionDialog(bundle);
    }

    if (this.elModal) {
      this.elModal.style.display = 'flex';
    }
  }

  close() {
    if (this.elModal) {
      this.elModal.style.display = 'none';
    }
  }

  renderMoleculeDialog(bundle) {
    if (this.elTitle) {
      const name = bundle.manifest?.title || bundle.manifest?.molKey || 'Molecule';
      this.elTitle.textContent = `📦 Download Custom Bundle: ${name}`;
    }
    if (!this.elBody) return;

    let html = `
      <div style="color:var(--text-secondary); margin-bottom: 6px;">
        Select the orbitals to include in the exported archive:
      </div>
      <div style="display:flex; justify-content:space-between; align-items:center; padding: 4px 6px; background: rgba(255,255,255,0.03); border-radius: 4px; margin-bottom: 8px;">
        <label style="font-weight:600; cursor:pointer; display:flex; align-items:center; gap:6px;">
          <input type="checkbox" id="dl-select-all" checked style="accent-color:var(--accent-color);">
          <span>Select All / Deselect All</span>
        </label>
        <span id="dl-selected-count" style="color:var(--text-secondary); font-size:11px;"></span>
      </div>
      <div class="download-checklist" id="dl-orbital-checklist">
    `;

    // 1. ESP Surface if present
    if (bundle.hasEsp) {
      html += `
        <label class="download-check-item">
          <input type="checkbox" class="dl-orb-chk" value="esp" checked>
          <span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#06b6d4;"></span>
          <span style="font-weight:600;">⚡ ESP Surface</span>
        </label>
      `;
    }

    // 2. Orbital categories and items
    const categories = bundle.categories || [];
    categories.forEach(cat => {
      html += `<div style="font-weight:700; color:#cbd5e1; margin-top:8px; margin-bottom:2px; padding-left:4px; font-size:11px; border-bottom:1px solid rgba(255,255,255,0.1);">${cat.name}</div>`;
      (cat.items || []).forEach(item => {
        const k = item.key || item.file || item.label;
        const color = item.color || '#38bdf8';
        const label = item.label || item.key;
        const ev = item.energy_ev !== undefined ? ` (${item.energy_ev.toFixed(2)} eV)` : '';
        html += `
          <label class="download-check-item">
            <input type="checkbox" class="dl-orb-chk" value="${k}" checked>
            <span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${color};"></span>
            <span>${label}${ev}</span>
          </label>
        `;
      });
    });

    html += `</div>`;
    this.elBody.innerHTML = html;

    // Bind Select All toggle
    const selectAll = this.elBody.querySelector('#dl-select-all');
    const chks = this.elBody.querySelectorAll('.dl-orb-chk');
    const countEl = this.elBody.querySelector('#dl-selected-count');

    const updateCount = () => {
      const total = chks.length;
      const checked = Array.from(chks).filter(c => c.checked).length;
      if (countEl) countEl.textContent = `${checked} / ${total} selected`;
      if (selectAll) selectAll.checked = checked === total;
    };

    selectAll?.addEventListener('change', (e) => {
      chks.forEach(c => c.checked = e.target.checked);
      updateCount();
    });

    chks.forEach(c => c.addEventListener('change', updateCount));
    updateCount();
  }

  renderReactionDialog(bundle) {
    if (this.elTitle) {
      const title = bundle.manifest?.title || 'Reaction Pathway';
      this.elTitle.textContent = `📦 Download Bundle: ${title}`;
    }
    if (!this.elBody) return;

    this.elBody.innerHTML = `
      <div style="color:var(--text-secondary); margin-bottom: 8px;">
        Choose which contents to include in the downloaded archive:
      </div>
      <div style="display:flex; flex-direction:column; gap:10px;">
        <label class="download-option-card">
          <input type="radio" name="dl_rxn_scope" value="reaction_only" checked>
          <div>
            <div class="download-option-title">Reaction Pathway Only</div>
            <div class="download-option-desc">
              Includes reaction coordinates, trajectory frames, PES energy profile, mechanism SVG, and orbital tracks. Excludes embedded single molecule sub-bundles for a compact file size.
            </div>
          </div>
        </label>
        <label class="download-option-card">
          <input type="radio" name="dl_rxn_scope" value="reaction_and_molecules">
          <div>
            <div class="download-option-title">Reaction + Molecule Bundles</div>
            <div class="download-option-desc">
              Includes full reaction pathway plus all individual 3D molecule archives for reactants, products, and intermediates.
            </div>
          </div>
        </label>
      </div>
    `;
  }

  async handleConfirm() {
    const bundle = this.app.bundle;
    if (!bundle || !bundle.rawZip) {
      alert('Raw archive data unavailable for repacking.');
      return;
    }

    if (this.elConfirm) {
      this.elConfirm.disabled = true;
      this.elConfirm.textContent = '⏳ Repacking...';
    }

    try {
      if (bundle.isSingleMolecule) {
        await this.repackMolecule(bundle);
      } else {
        await this.repackReaction(bundle);
      }
      this.close();
    } catch (err) {
      console.error('Repacking failed:', err);
      alert(`Error repacking bundle: ${err.message}`);
    } finally {
      if (this.elConfirm) {
        this.elConfirm.disabled = false;
        this.elConfirm.textContent = '📦 Download';
      }
    }
  }

  async repackMolecule(bundle) {
    const rawZip = bundle.rawZip;
    const newZip = new window.JSZip();

    // Determine kept orbital keys
    const chks = this.elBody.querySelectorAll('.dl-orb-chk:checked');
    const keptKeys = new Set(Array.from(chks).map(c => c.value));
    const keepEsp = keptKeys.has('esp');

    const molKey = bundle.manifest?.molKey || bundle.molKey || 'molecule';

    // Copy metadata files
    for (const [fname, fileObj] of Object.entries(rawZip.files)) {
      if (fileObj.dir) continue;

      const isManifest = fname === 'manifest.json' || fname.endsWith('/manifest.json');
      const isKet = fname === 'layout.ket' || fname.endsWith('.ket');
      const isMolMeta = fname.endsWith(`${molKey}.json`);
      const isXyz = fname.endsWith('.xyz');

      if (isManifest || isKet || isMolMeta || isXyz) {
        if (isManifest || isMolMeta) {
          try {
            const text = await fileObj.async('string');
            const obj = JSON.parse(text);
            const filterCats = (cats) => {
              if (!Array.isArray(cats)) return cats;
              return cats.map(cat => ({
                ...cat,
                items: (cat.items || []).filter(it => {
                  const k = it.file ? it.file.replace(/^molecules\//, '').replace(/\.json$/, '') : (it.key || it.label);
                  return keptKeys.has(k) || keptKeys.has(it.key) || keptKeys.has(it.label);
                })
              })).filter(cat => cat.items.length > 0);
            };
            if (obj.orbitals) obj.orbitals = filterCats(obj.orbitals);
            if (obj.molecules && Array.isArray(obj.molecules)) {
              obj.molecules.forEach(m => { if (m.orbitals) m.orbitals = filterCats(m.orbitals); });
            }
            newZip.file(fname, JSON.stringify(obj, null, 2));
            continue;
          } catch (_) {}
        }
        const data = await fileObj.async('uint8array');
        newZip.file(fname, data);
        continue;
      }

      // Check ESP surface
      if (fname.includes('_esp.json')) {
        if (keepEsp) {
          const data = await fileObj.async('uint8array');
          newZip.file(fname, data);
        }
        continue;
      }

      // Check orbital mesh files
      if (fname.endsWith('_pos.json') || fname.endsWith('_neg.json')) {
        const baseKey = fname.replace(/^molecules\//, '').replace(/^.*?molecules\//, '').replace(/_(pos|neg)\.json$/, '');
        let matched = keptKeys.has(baseKey);
        if (!matched) {
          for (const k of keptKeys) {
            if (baseKey.includes(k) || k.includes(baseKey)) {
              matched = true;
              break;
            }
          }
        }
        if (matched) {
          const data = await fileObj.async('uint8array');
          newZip.file(fname, data);
        }
      }
    }

    const blob = await newZip.generateAsync({ type: 'blob', compression: 'DEFLATE' });
    this.triggerBrowserDownload(blob, `${molKey}_custom.zip`);
    return newZip;
  }

  async repackReaction(bundle) {
    const rawZip = bundle.rawZip;
    const newZip = new window.JSZip();

    const scopeRadio = this.elBody.querySelector('input[name="dl_rxn_scope"]:checked');
    const scope = scopeRadio ? scopeRadio.value : 'reaction_only';

    for (const [fname, fileObj] of Object.entries(rawZip.files)) {
      if (fileObj.dir) continue;
      if (scope === 'reaction_only' && fname.startsWith('molecules/')) {
        continue; // Skip individual molecule sub-directories
      }
      const data = await fileObj.async('uint8array');
      newZip.file(fname, data);
    }

    const title = (bundle.manifest?.title || 'reaction_pathway').replace(/[\s\-_]+/g, '_');
    const suffix = scope === 'reaction_only' ? '_reaction_only.zip' : '_full.zip';
    const blob = await newZip.generateAsync({ type: 'blob', compression: 'DEFLATE' });
    this.triggerBrowserDownload(blob, `${title}${suffix}`);
    return newZip;
  }

  triggerBrowserDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 1000);
  }
}
