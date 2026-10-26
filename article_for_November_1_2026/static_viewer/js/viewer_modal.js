/**
 * viewer_modal.js
 * In-page modal dialog for inspecting individual molecules.
 * Reuses Viewer3D and ViewerOrbitalsPanel with draggable and scalable pop-up dialog.
 */

import { buildMoleculeBundle } from './viewer_loader.js';
import { Viewer3D } from './viewer_3d.js';
import { ViewerOrbitalsPanel } from './viewer_orbitals.js';

export class ViewerMoleculeModal {
  constructor(app) {
    this.app = app;
    this.modalEl = null;
    this.viewer3d = null;
    this.orbitalsPanel = null;
    this.isPanelCollapsed = false;
    this.resizeObserver = null;
    this.cleanupFns = [];
  }

  show(molId, cfg = {}) {
    this.close();

    const parentBundle = this.app.bundle;
    const molBundle = buildMoleculeBundle(molId, parentBundle) ||
                      (cfg.moleculeName ? buildMoleculeBundle(cfg.moleculeName, parentBundle) : null);

    if (!molBundle) {
      console.warn(`Molecule '${molId}' could not be resolved in bundle.`);
      return;
    }

    const title = cfg.moleculeName || cfg.molName || molBundle.manifest.title || molId;

    const modal = document.createElement('div');
    modal.id = 'molecule-modal-backdrop';
    modal.className = 'custom-modal-backdrop';
    modal.innerHTML = `
      <div class="custom-modal-dialog" id="custom-modal-dialog">
        <div class="modal-header" id="modal-drag-header">
          <div class="modal-title-group">
            <h3>⚛️ ${title}</h3>
            <span class="chip-blue">Molecule Inspection</span>
          </div>
          <div class="modal-actions">
            <button class="btn btn-sm" id="btn-modal-popout" title="Open in dedicated pop-up window">↗ Pop-up Window</button>
            <button class="btn-modal-close" id="btn-modal-close">&times;</button>
          </div>
        </div>
        <div class="modal-viewport-container">
          <div id="modal-gl-canvas" class="modal-gl-canvas"></div>
          <div class="viewport-overlay-controls">
            <button class="btn-panel-toggle" id="modal-btn-toggle-orbitals">Orbitals ▾</button>
          </div>
          <aside id="modal-side-panel" class="side-panel">
            <div class="panel-header">
              <span>Orbitals & Surfaces</span>
            </div>
            <div class="panel-body">
              <div id="modal-orbitals-list" class="orbitals-list-container"></div>
            </div>
          </aside>
          <div class="modal-resize-handle" id="modal-resize-handle" title="Drag to scale / resize"></div>
        </div>
      </div>
    `;

    document.body.appendChild(modal);
    this.modalEl = modal;

    const dialog = modal.querySelector('#custom-modal-dialog');
    const header = modal.querySelector('#modal-drag-header');
    const resizeHandle = modal.querySelector('#modal-resize-handle');

    // Initial center position
    const initW = Math.min(920, Math.max(480, window.innerWidth * 0.75));
    const initH = Math.min(560, Math.max(360, window.innerHeight * 0.70));
    dialog.style.width = `${initW}px`;
    dialog.style.height = `${initH}px`;
    dialog.style.left = `${Math.max(16, Math.round((window.innerWidth - initW) / 2))}px`;
    dialog.style.top = `${Math.max(16, Math.round((window.innerHeight - initH) / 2))}px`;

    // 1. Draggable Dialog Setup
    this.setupDraggable(dialog, header);

    // 2. Scalable / Resizable Dialog Setup
    this.setupResizable(dialog, resizeHandle);

    // Close button and backdrop click
    modal.querySelector('#btn-modal-close').addEventListener('click', () => this.close());
    modal.addEventListener('click', (e) => {
      if (e.target === modal) this.close();
    });

    // Pop-up window button: Opens standalone single-molecule viewer
    modal.querySelector('#btn-modal-popout').addEventListener('click', () => {
      const bundleUrl = this.app.currentBundleUrl || (new URLSearchParams(window.location.search).get('bundle') || '');
      const targetMol = molBundle.manifest.molKey || molId;
      const popUrl = bundleUrl 
        ? `viewer.html?bundle=${encodeURIComponent(bundleUrl)}&molecule=${encodeURIComponent(targetMol)}`
        : `viewer.html?molecule=${encodeURIComponent(targetMol)}`;
      window.open(popUrl, '_blank', 'width=1100,height=750,menubar=no,toolbar=no');
    });

    // Collapsible drawer toggle
    const btnToggle = modal.querySelector('#modal-btn-toggle-orbitals');
    const sidePanel = modal.querySelector('#modal-side-panel');
    this.isPanelCollapsed = false;
    btnToggle.addEventListener('click', () => {
      this.isPanelCollapsed = !this.isPanelCollapsed;
      sidePanel.classList.toggle('panel-hidden', this.isPanelCollapsed);
      btnToggle.innerHTML = `Orbitals ${this.isPanelCollapsed ? '▸' : '▾'}`;
    });

    // Reusable 3D Viewer
    const canvasContainer = modal.querySelector('#modal-gl-canvas');
    this.viewer3d = new Viewer3D(canvasContainer);
    this.viewer3d.setBundle(molBundle);

    // Reusable Orbitals Panel
    const orbContainer = modal.querySelector('#modal-orbitals-list');
    this.orbitalsPanel = new ViewerOrbitalsPanel(
      orbContainer,
      (orbKey, isChecked) => this.viewer3d?.toggleOrbital(orbKey, isChecked),
      (isEsp) => this.viewer3d?.setEspActive(isEsp),
      (isCharges) => this.viewer3d?.setChargesActive(isCharges)
    );
    this.orbitalsPanel.renderGroups(molBundle.categories, molBundle.hasEsp);
  }

  setupDraggable(dialog, header) {
    let isDragging = false;
    let startX = 0;
    let startY = 0;
    let origLeft = 0;
    let origTop = 0;

    const onMouseDown = (e) => {
      if (e.target.closest('button, input, a, .chip-blue')) return;
      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;
      const rect = dialog.getBoundingClientRect();
      origLeft = rect.left;
      origTop = rect.top;
      document.body.style.userSelect = 'none';
      window.addEventListener('mousemove', onMouseMove);
      window.addEventListener('mouseup', onMouseUp);
    };

    const onMouseMove = (e) => {
      if (!isDragging) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;
      const newL = Math.max(8, Math.min(window.innerWidth - 80, origLeft + dx));
      const newT = Math.max(8, Math.min(window.innerHeight - 60, origTop + dy));
      dialog.style.left = `${newL}px`;
      dialog.style.top = `${newT}px`;
    };

    const onMouseUp = () => {
      if (isDragging) {
        isDragging = false;
        document.body.style.userSelect = '';
        window.removeEventListener('mousemove', onMouseMove);
        window.removeEventListener('mouseup', onMouseUp);
      }
    };

    header.addEventListener('mousedown', onMouseDown);
    this.cleanupFns.push(() => {
      header.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    });
  }

  setupResizable(dialog, resizeHandle) {
    // 1. Observe size changes (handles CSS resize: both and window resize)
    if (window.ResizeObserver) {
      this.resizeObserver = new ResizeObserver(() => {
        if (this.viewer3d && this.viewer3d.glViewer) {
          this.viewer3d.glViewer.resize();
          this.viewer3d.glViewer.render();
        }
      });
      this.resizeObserver.observe(dialog);
    }

    // 2. Corner drag handle for scalable mouse interaction
    if (!resizeHandle) return;
    let isResizing = false;
    let startX = 0;
    let startY = 0;
    let origW = 0;
    let origH = 0;

    const onHandleDown = (e) => {
      e.stopPropagation();
      e.preventDefault();
      isResizing = true;
      startX = e.clientX;
      startY = e.clientY;
      origW = dialog.offsetWidth;
      origH = dialog.offsetHeight;
      document.body.style.cursor = 'se-resize';
      document.body.style.userSelect = 'none';
      window.addEventListener('mousemove', onHandleMove);
      window.addEventListener('mouseup', onHandleUp);
    };

    const onHandleMove = (e) => {
      if (!isResizing) return;
      const dw = e.clientX - startX;
      const dh = e.clientY - startY;
      const maxW = Math.round(window.innerWidth * 0.98);
      const maxH = Math.round(window.innerHeight * 0.96);
      const newW = Math.max(420, Math.min(maxW, origW + dw));
      const newH = Math.max(320, Math.min(maxH, origH + dh));
      dialog.style.width = `${newW}px`;
      dialog.style.height = `${newH}px`;
    };

    const onHandleUp = () => {
      if (isResizing) {
        isResizing = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        window.removeEventListener('mousemove', onHandleMove);
        window.removeEventListener('mouseup', onHandleUp);
      }
    };

    resizeHandle.addEventListener('mousedown', onHandleDown);
    this.cleanupFns.push(() => {
      resizeHandle.removeEventListener('mousedown', onHandleDown);
      window.removeEventListener('mousemove', onHandleMove);
      window.removeEventListener('mouseup', onHandleUp);
    });
  }

  close() {
    this.cleanupFns.forEach(fn => { try { fn(); } catch (_) {} });
    this.cleanupFns = [];
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
      this.resizeObserver = null;
    }
    if (this.modalEl) {
      this.modalEl.remove();
      this.modalEl = null;
    }
    this.viewer3d = null;
    this.orbitalsPanel = null;
  }
}
