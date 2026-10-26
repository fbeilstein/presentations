/**
 * viewer_svg.js
 * Interactive SVG Reaction Mechanism Manager.
 * Step 5 mechanism interactions:
 * - Click forward/reverse reaction arrows to play segments
 * - Click states/intermediates to jump to frame
 * - Click or right-click molecules to open molecule inspection
 * - Dynamic electron-pushing arrow highlighting & pulsing on playback
 */

export class ViewerSvgMechanism {
  constructor(containerElement, options = {}) {
    this.container = containerElement;
    this.wiring = options.wiring || {};
    this.onPlaySegment = options.onPlaySegment || (() => {});
    this.onJumpFrame = options.onJumpFrame || (() => {});
    this.onOpenMolecule = options.onOpenMolecule || (() => {});
    this.currentFrame = 0;
    this.playDirection = 'forward';
  }

  setMechanism(svgText, wiring = {}) {
    this.wiring = wiring || {};
    if (!this.container || !svgText) return;

    // Ensure proper SVG namespace
    let text = svgText;
    if (!/<svg[^>]+xmlns\s*=\s*['"]http:\/\/www\.w3\.org\/2000\/svg['"]/i.test(text)) {
      text = text.replace(/<svg\b/i, '<svg xmlns="http://www.w3.org/2000/svg"');
    }

    this.container.innerHTML = text;
    this.applyClasses();
    this.bindEvents();
    this.updatePushArrows(this.currentFrame, this.playDirection);
  }

  applyClasses() {
    const svg = this.container.querySelector('svg');
    if (!svg) return;

    for (const [key, cfg] of Object.entries(this.wiring)) {
      const el = this.findElement(svg, key);
      if (!el) continue;

      if (cfg.role === 'arrow') {
        el.classList.add('svg-clickable-arrow');
        const dir = cfg.direction === 'reverse' ? '⟵' : '⟶';
        el.setAttribute('title', `Reaction Arrow: Play frames ${cfg.startFrame || 0} ${dir} ${cfg.endFrame || 0}`);
      } else if (cfg.role === 'state' || cfg.role === 'state_molecule') {
        el.classList.add('svg-clickable-state');
        if (cfg.role === 'state_molecule' || cfg.moleculeName) {
          el.classList.add('svg-clickable-molecule');
          el.setAttribute('title', `${cfg.moleculeName || 'Intermediate'} (Frame ${cfg.targetFrame || 0}): Click to jump, Double-click for molecule`);
        } else {
          el.setAttribute('title', `State: Jump to frame ${cfg.targetFrame || 0}`);
        }
      } else if (cfg.role === 'molecule') {
        el.classList.add('svg-clickable-molecule');
        const stateTarget = this.findStateTarget(el.parentElement, svg);
        const stateInfo = (stateTarget && stateTarget.cfg && stateTarget.cfg.targetFrame !== undefined)
          ? ` (Frame ${stateTarget.cfg.targetFrame}): Click to jump, `
          : ': ';
        el.setAttribute('title', `${cfg.moleculeName || 'Molecule'}${stateInfo}Double-click to inspect`);
      }
    }
  }

  findStateTarget(el, rootSvg) {
    let cur = el;
    while (cur && cur !== rootSvg && cur.tagName && cur.tagName.toLowerCase() !== 'svg') {
      const id = cur.getAttribute('id');
      const lbl = cur.getAttribute('inkscape:label') || cur.getAttribute('label');
      const cfg = (id && this.wiring[id]) || (lbl && this.wiring[lbl]);
      if (cfg && (cfg.role === 'state' || cfg.role === 'state_molecule' || cfg.targetFrame !== undefined)) {
        return { el: cur, id: id || lbl, cfg };
      }
      cur = cur.parentElement;
    }
    return null;
  }

  bindEvents() {
    const svg = this.container.querySelector('svg');
    if (!svg) return;

    // Single click handler: Arrows (play segment) & States (jump frame)
    svg.addEventListener('click', (e) => {
      const target = this.resolveTarget(e.target, svg);
      if (!target) return;

      const { cfg } = target;
      if (!cfg) return;

      if (cfg.role === 'arrow') {
        e.stopPropagation();
        const sf = cfg.startFrame !== undefined ? parseInt(cfg.startFrame, 10) : 0;
        const ef = cfg.endFrame !== undefined ? parseInt(cfg.endFrame, 10) : 100;
        const dir = cfg.direction || 'forward';
        this.onPlaySegment(sf, ef, dir);
      } else if (cfg.role === 'state' || cfg.role === 'state_molecule' || cfg.targetFrame !== undefined) {
        e.stopPropagation();
        const tf = cfg.targetFrame !== undefined ? parseInt(cfg.targetFrame, 10) : 0;
        this.onJumpFrame(tf);
      } else {
        // e.g. clicked on a molecule (reactants or products) belonging to a state
        const stateTarget = this.findStateTarget(e.target, svg);
        if (stateTarget && stateTarget.cfg && stateTarget.cfg.targetFrame !== undefined) {
          e.stopPropagation();
          const tf = parseInt(stateTarget.cfg.targetFrame, 10) || 0;
          this.onJumpFrame(tf);
        }
      }
    });

    // Double-click handler: Open molecule inspection modal
    svg.addEventListener('dblclick', (e) => {
      const target = this.resolveTarget(e.target, svg);
      if (!target) return;

      const { id, cfg } = target;
      if (cfg && (cfg.role === 'molecule' || cfg.role === 'state_molecule' || cfg.moleculeName)) {
        e.stopPropagation();
        e.preventDefault();
        this.onOpenMolecule(id, cfg);
      }
    });

    // Right-click contextmenu handler
    svg.addEventListener('contextmenu', (e) => {
      const target = this.resolveTarget(e.target, svg);
      if (!target) return;

      const { id, cfg } = target;
      if (cfg && (cfg.role === 'molecule' || cfg.role === 'state_molecule' || cfg.moleculeName)) {
        e.preventDefault();
        e.stopPropagation();
        this.onOpenMolecule(id, cfg);
      }
    });
  }

  updatePushArrows(frameIdx, direction = 'forward') {
    this.currentFrame = frameIdx;
    this.playDirection = direction || 'forward';

    const svg = this.container ? (this.container.querySelector('#active-mechanism-svg') || this.container.querySelector('svg')) : null;
    if (!svg) return;

    Object.entries(this.wiring).forEach(([key, cfg]) => {
      if (cfg.role !== 'push_arrow') return;

      const el = this.findElement(svg, key);
      if (!el) return;

      const sf = cfg.startFrame !== undefined ? parseInt(cfg.startFrame, 10) : 0;
      const ef = cfg.endFrame !== undefined ? parseInt(cfg.endFrame, 10) : 9999;
      const arrowDir = cfg.direction || 'both';

      const dirMatches = (arrowDir === 'both') || (arrowDir === this.playDirection);
      const minF = Math.min(sf, ef);
      const maxF = Math.max(sf, ef);
      const frameMatches = (frameIdx >= minF && frameIdx <= maxF);

      if (dirMatches && frameMatches) {
        el.classList.add('svg-push-active');
        el.classList.remove('svg-push-inactive');
      } else {
        el.classList.add('svg-push-inactive');
        el.classList.remove('svg-push-active');
      }
    });
  }

  highlightCurrentState(frameIdx) {
    const svg = this.container.querySelector('svg');
    if (!svg) return;

    svg.querySelectorAll('.svg-active-element').forEach(el => el.classList.remove('svg-active-element'));

    let closestState = null;
    let minDiff = 8;
    for (const [key, cfg] of Object.entries(this.wiring)) {
      if ((cfg.role === 'state' || cfg.role === 'state_molecule') && cfg.targetFrame !== undefined) {
        const diff = Math.abs(cfg.targetFrame - frameIdx);
        if (diff < minDiff) {
          minDiff = diff;
          closestState = key;
        }
      }
    }
    if (closestState) {
      const el = this.findElement(svg, closestState);
      if (el) el.classList.add('svg-active-element');
    }
  }

  findElement(svg, key) {
    let el = svg.getElementById(key);
    if (!el) {
      try {
        el = svg.querySelector(`[inkscape\\:label="${key}"], [label="${key}"], [data-id="${key}"]`);
      } catch (e) {}
    }
    if (!el) {
      for (const elem of svg.querySelectorAll('*')) {
        if (elem.getAttribute('inkscape:label') === key || elem.getAttribute('label') === key) {
          return elem;
        }
      }
    }
    return el;
  }

  resolveTarget(el, rootSvg) {
    let cur = el;
    while (cur && cur !== rootSvg && cur.tagName && cur.tagName.toLowerCase() !== 'svg') {
      const id = cur.getAttribute('id');
      const lbl = cur.getAttribute('inkscape:label') || cur.getAttribute('label');
      const cfg = (id && this.wiring[id]) || (lbl && this.wiring[lbl]);
      if (cfg) {
        return { el: cur, id: id || lbl, cfg };
      }
      cur = cur.parentElement;
    }
    return null;
  }
}
