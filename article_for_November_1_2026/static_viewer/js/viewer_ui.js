/**
 * viewer_ui.js
 * Static UI manager for mechanism header, side orbital controls, and playback transport dock.
 * Supports Molecular Explorer style grouped collapsible orbitals with checkboxes,
 * embedded ESP surface toggle, collapsible side drawer, and Step 5 SVG interactions.
 */

import { ViewerOrbitalsPanel } from './viewer_orbitals.js';
import { ViewerSvgMechanism } from './viewer_svg.js';

export class ViewerUI {
  constructor(app) {
    this.app = app;

    // DOM references
    this.elTitle = document.getElementById('reaction-title');
    this.elBarrier = document.getElementById('stat-barrier');
    this.elSolvent = document.getElementById('stat-solvent');
    this.elMechanism = document.getElementById('mechanism-strip');
    this.elSlider = document.getElementById('frame-slider');
    this.elFrameInfo = document.getElementById('frame-info');
    this.elBtnPlayFwd = document.getElementById('btn-play-fwd');
    this.elBtnPlayRev = document.getElementById('btn-play-rev');
    this.elSidePanel = document.getElementById('side-panel');
    this.elBtnToggleOrbitals = document.getElementById('btn-toggle-orbitals');
    this.elOrbitalsContainer = document.getElementById('orbitals-list');

    // Submodules
    this.orbitalsPanel = new ViewerOrbitalsPanel(
      this.elOrbitalsContainer,
      (orbKey, isChecked) => this.app.viewer3d?.toggleOrbital(orbKey, isChecked),
      (isEsp) => this.app.viewer3d?.setEspActive(isEsp),
      (isCharges) => this.app.viewer3d?.setChargesActive(isCharges)
    );

    this.svgMechanism = new ViewerSvgMechanism(this.elMechanism, {
      onPlaySegment: (sf, ef, dir) => this.app.playSegment(sf, ef, dir),
      onJumpFrame: (tf) => this.app.goToFrame(tf),
      onOpenMolecule: (id, cfg) => this.app.openMoleculeModal(id, cfg)
    });

    this.isPanelCollapsed = false;
    this.bindEvents();
  }

  bindEvents() {
    // Play Forward / Reverse ("let them both act as pause during any play")
    if (this.elBtnPlayFwd) {
      this.elBtnPlayFwd.addEventListener('click', () => this.app.togglePlayForward());
    }
    if (this.elBtnPlayRev) {
      this.elBtnPlayRev.addEventListener('click', () => this.app.togglePlayReverse());
    }

    // Step Buttons
    document.getElementById('btn-first')?.addEventListener('click', () => this.app.goToFrame(0));
    document.getElementById('btn-last')?.addEventListener('click', () => this.app.goToFrame(this.app.totalFrames - 1));
    document.getElementById('btn-prev')?.addEventListener('click', () => this.app.stepFrame(-1));
    document.getElementById('btn-next')?.addEventListener('click', () => this.app.stepFrame(1));

    // Slider (Step 5 Algorithm: pause, detect scrubbing direction, seek frame)
    if (this.elSlider) {
      this.elSlider.addEventListener('input', (e) => {
        this.app.pause();
        const newF = parseInt(e.target.value, 10);
        if (newF > this.app.currentFrame) this.app.setDirection('forward');
        else if (newF < this.app.currentFrame) this.app.setDirection('reverse');
        this.app.goToFrame(newF);
      });
    }

    // Speed selector
    const speedSelect = document.getElementById('select-speed');
    if (speedSelect) {
      speedSelect.addEventListener('change', (e) => {
        this.app.setSpeed(parseFloat(e.target.value));
      });
    }

    // Collapsible side drawer toggle [ Orbitals ▾ ]
    if (this.elBtnToggleOrbitals) {
      this.elBtnToggleOrbitals.addEventListener('click', () => this.togglePanel());
    }
  }

  togglePanel() {
    this.isPanelCollapsed = !this.isPanelCollapsed;
    if (this.elSidePanel) {
      this.elSidePanel.classList.toggle('panel-hidden', this.isPanelCollapsed);
    }
    if (this.elBtnToggleOrbitals) {
      this.elBtnToggleOrbitals.innerHTML = `Orbitals ${this.isPanelCollapsed ? '▸' : '▾'}`;
    }
  }

  setBundle(bundle) {
    if (bundle.isSingleMolecule) {
      this.setMoleculeMode(bundle);
      return;
    }
    const manifest = bundle.manifest || {};
    const traj = bundle.trajectory || {};

    // 1. Meta / Title
    if (this.elTitle) {
      this.elTitle.textContent = manifest.title || traj.title || 'Multistep Reaction Pathway';
    }
    if (this.elBarrier) {
      const b = manifest.overall_barrier_kcal !== undefined ? manifest.overall_barrier_kcal : traj.overall_barrier_kcal;
      this.elBarrier.textContent = b !== undefined ? `${b} kcal/mol` : '—';
    }
    if (this.elSolvent) {
      this.elSolvent.textContent = manifest.solvent || traj.solvent || 'Gas Phase';
    }

    // 2. Interactive Mechanism SVG
    const wiring = bundle.svgWiring || manifest.svg_wiring || traj.svg_wiring || {};
    if (bundle.mechanismSvg) {
      this.svgMechanism.setMechanism(bundle.mechanismSvg, wiring);
      if (this.elMechanism) this.elMechanism.style.display = 'flex';
    } else {
      if (this.elMechanism) this.elMechanism.style.display = 'none';
    }

    const stats = document.querySelector('.reaction-stats');
    if (stats) stats.style.display = 'flex';
    const dock = document.getElementById('bottom-dock');
    if (dock) dock.style.display = 'flex';

    // 3. Slider configuration
    const nFrames = (traj.frames || []).length;
    if (this.elSlider) {
      this.elSlider.min = 0;
      this.elSlider.max = Math.max(0, nFrames - 1);
      this.elSlider.value = 0;
    }

    // 4. Populate Orbitals with Molecular Explorer categories & ESP
    this.populateOrbitals(bundle);
  }

  setMoleculeMode(molBundle) {
    const rawTitle = molBundle.manifest?.title || 'Molecule View';
    const cleanTitle = rawTitle.replace(/_/g, ' ');
    const title = cleanTitle.charAt(0).toUpperCase() + cleanTitle.slice(1);
    if (this.elTitle) this.elTitle.textContent = `⚛️ ${title}`;
    const stats = document.querySelector('.reaction-stats');
    if (stats) stats.style.display = 'none';
    if (this.elMechanism) this.elMechanism.style.display = 'none';
    const pes = document.getElementById('pes-canvas-container');
    if (pes) pes.style.display = 'none';
    const dock = document.getElementById('bottom-dock');
    if (dock) dock.style.display = 'none';
    this.orbitalsPanel.renderGroups(molBundle.categories, molBundle.hasEsp);
    if (this.app.viewer3d) {
      setTimeout(() => this.app.viewer3d.resize(), 50);
    }
  }

  populateOrbitals(bundle) {
    const timeline = (bundle.manifest && bundle.manifest.orbital_timeline) ||
                     (bundle.trajectory && bundle.trajectory.orbital_timeline);
    const tracks = (timeline && timeline.tracks) || [];

    let categories = [];
    if (tracks.length > 0) {
      categories = ViewerOrbitalsPanel.parsePathwayTracks(tracks);
    } else if (bundle.orbitalMeshes && Object.keys(bundle.orbitalMeshes).length > 0) {
      categories = ViewerOrbitalsPanel.parseRawOrbitalKeys(Object.keys(bundle.orbitalMeshes));
    }

    const hasEsp = true;

    this.orbitalsPanel.renderGroups(categories, hasEsp);
  }

  updateFrameDisplay(frameIdx, totalFrames, energy) {
    if (this.elSlider) {
      this.elSlider.value = frameIdx;
    }
    if (this.elFrameInfo) {
      const eStr = energy !== undefined ? ` | ${energy >= 0 ? '+' : ''}${energy.toFixed(2)} kcal/mol` : '';
      this.elFrameInfo.innerHTML = `Frame <strong>${frameIdx + 1}</strong> / ${totalFrames}${eStr}`;
    }

    // Step 5 SVG updates: push arrow pulse and state highlight
    if (this.svgMechanism) {
      this.svgMechanism.updatePushArrows(frameIdx, this.app.playDirection || 'forward');
      this.svgMechanism.highlightCurrentState(frameIdx);
    }
  }

  updatePlayState(isPlaying, direction = 'forward') {
    if (this.elBtnPlayFwd) {
      const isFwd = isPlaying && direction === 'forward';
      this.elBtnPlayFwd.textContent = isFwd ? '❚❚ Pause' : '▶ Play Fwd';
      this.elBtnPlayFwd.classList.toggle('active', isFwd);
    }
    if (this.elBtnPlayRev) {
      const isRev = isPlaying && direction === 'reverse';
      this.elBtnPlayRev.textContent = isRev ? '❚❚ Pause' : '◀ Play Rev';
      this.elBtnPlayRev.classList.toggle('active', isRev);
    }
  }
}
