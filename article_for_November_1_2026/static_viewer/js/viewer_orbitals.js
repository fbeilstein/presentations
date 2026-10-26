/**
 * viewer_orbitals.js
 * Grouped, collapsible orbital checklist matching Molecular Explorer design.
 * Supports multiple active orbital checkboxes and embedded ESP toggle.
 */

export class ViewerOrbitalsPanel {
  constructor(containerElement, onOrbitalToggle, onEspToggle, onChargesToggle) {
    this.container = containerElement;
    this.onOrbitalToggle = onOrbitalToggle || (() => {});
    this.onEspToggle = onEspToggle || (() => {});
    this.onChargesToggle = onChargesToggle || (() => {});
    this.activeOrbitals = new Set();
    this.isEspActive = false;
    this.isChargesActive = false;
    this.collapsedGroups = new Set();
  }

  clear() {
    this.container.innerHTML = '';
    this.activeOrbitals.clear();
    this.isEspActive = false;
    this.isChargesActive = false;
  }

  /**
   * Render grouped categories from molecule JSON or reaction pathway tracks
   */
  renderGroups(categories, hasEsp = false) {
    this.container.innerHTML = '';
    if (!categories || categories.length === 0) {
      this.container.innerHTML = '<span class="empty-hint">No orbitals available</span>';
      return;
    }

    // 1. Render orbital category groups
    categories.forEach((cat, catIdx) => {
      const groupEl = document.createElement('div');
      groupEl.className = 'orbital-group';
      const isCollapsed = this.collapsedGroups.has(cat.name);

      const header = document.createElement('div');
      header.className = 'orbital-group-header';
      header.innerHTML = `
        <span class="group-title">${cat.name}</span>
        <span class="group-chevron">${isCollapsed ? '▸' : '▾'}</span>
      `;
      header.addEventListener('click', () => {
        if (this.collapsedGroups.has(cat.name)) {
          this.collapsedGroups.delete(cat.name);
        } else {
          this.collapsedGroups.add(cat.name);
        }
        this.renderGroups(categories, hasEsp);
      });
      groupEl.appendChild(header);

      const body = document.createElement('div');
      body.className = 'orbital-group-body';
      if (isCollapsed) body.style.display = 'none';

      (cat.items || []).forEach(item => {
        const row = document.createElement('label');
        row.className = 'orbital-item-row';

        const chk = document.createElement('input');
        chk.type = 'checkbox';
        chk.className = 'orb-chk';
        chk.checked = this.activeOrbitals.has(item.key || item.file || item.label);
        chk.addEventListener('change', (e) => {
          const k = item.key || item.file || item.label;
          if (e.target.checked) this.activeOrbitals.add(k);
          else this.activeOrbitals.delete(k);
          this.onOrbitalToggle(k, e.target.checked, item);
        });

        const dot = document.createElement('span');
        dot.className = 'orb-dot';
        dot.style.backgroundColor = item.color || '#38bdf8';

        const labelSpan = document.createElement('span');
        labelSpan.className = 'orb-label-text';
        let txt = item.label || item.title || item.key;
        if (item.energy_ev !== undefined) {
          txt += ` (${item.energy_ev > 0 ? '+' : ''}${item.energy_ev.toFixed(2)} eV)`;
        }
        labelSpan.textContent = txt;

        row.appendChild(chk);
        row.appendChild(dot);
        row.appendChild(labelSpan);
        body.appendChild(row);
      });

      groupEl.appendChild(body);
      this.container.appendChild(groupEl);
    });

    // 2. Render ESP & Charges section
    const groupTitle = hasEsp ? 'Surfaces & Charges' : 'Atom Charges';
    const espGroup = document.createElement('div');
    espGroup.className = 'orbital-group esp-group';
    const isEspCollapsed = this.collapsedGroups.has(groupTitle);

    const espHeader = document.createElement('div');
    espHeader.className = 'orbital-group-header';
    espHeader.innerHTML = `
      <span class="group-title">${groupTitle}</span>
      <span class="group-chevron">${isEspCollapsed ? '▸' : '▾'}</span>
    `;
    espHeader.addEventListener('click', () => {
      if (this.collapsedGroups.has(groupTitle)) {
        this.collapsedGroups.delete(groupTitle);
      } else {
        this.collapsedGroups.add(groupTitle);
      }
      this.renderGroups(categories, hasEsp);
    });
    espGroup.appendChild(espHeader);

    const espBody = document.createElement('div');
    espBody.className = 'orbital-group-body';
    if (isEspCollapsed) espBody.style.display = 'none';

      if (hasEsp) {
        const espRow = document.createElement('label');
        espRow.className = 'orbital-item-row';

        const espChk = document.createElement('input');
        espChk.type = 'checkbox';
        espChk.className = 'orb-chk';
        espChk.checked = this.isEspActive;
        espChk.addEventListener('change', (e) => {
          this.isEspActive = e.target.checked;
          this.onEspToggle(this.isEspActive);
        });

        const espDot = document.createElement('span');
        espDot.className = 'orb-dot';
        espDot.style.backgroundColor = '#06b6d4';

        const espText = document.createElement('span');
        espText.className = 'orb-label-text';
        espText.textContent = '⚡ ESP Surface';

        espRow.appendChild(espChk);
        espRow.appendChild(espDot);
        espRow.appendChild(espText);
        espBody.appendChild(espRow);
      }

      // Atom Charges checkbox
      const chargesRow = document.createElement('label');
      chargesRow.className = 'orbital-item-row';

      const chargesChk = document.createElement('input');
      chargesChk.type = 'checkbox';
      chargesChk.className = 'orb-chk';
      chargesChk.checked = this.isChargesActive;
      chargesChk.addEventListener('change', (e) => {
        this.isChargesActive = e.target.checked;
        this.onChargesToggle(this.isChargesActive);
      });

      const chargesDot = document.createElement('span');
      chargesDot.className = 'orb-dot';
      chargesDot.style.backgroundColor = '#a855f7';

      const chargesText = document.createElement('span');
      chargesText.className = 'orb-label-text';
      chargesText.textContent = '🏷 Atom Charges';

      chargesRow.appendChild(chargesChk);
      chargesRow.appendChild(chargesDot);
      chargesRow.appendChild(chargesText);
      espBody.appendChild(chargesRow);

      espGroup.appendChild(espBody);
      this.container.appendChild(espGroup);
  }

  /**
   * Helper to format reaction pathway tracks into Molecular Explorer style categories
   */
  static parsePathwayTracks(tracks = []) {
    const categoriesMap = {
      'Frontier': [],
      'Transforming Orbitals': [],
      'σ Bonds': [],
      'π Bonds': [],
      'σ* Anti-Bonds': [],
      'π* Anti-Bonds': [],
      'Lone Pairs': [],
      'Reaction Orbitals': []
    };

    tracks.forEach(trk => {
      if (trk.role === 'esp' || trk.orbital_key === 'track_esp_charge') return;
      const key = trk.orbital_key;
      let title = trk.label || trk.title || key;
      title = title.replace(/^Non-Bonding\s+/i, '').replace(/\[Antibond\]/i, '').trim();
      const color = trk.color || '#38bdf8';
      const item = { key, label: title, color };

      if (title.includes('→') || title.includes('->')) {
        categoriesMap['Transforming Orbitals'].push(item);
      } else if (title.includes('HOMO') || title.includes('LUMO')) {
        categoriesMap['Frontier'].push(item);
      } else if (title.includes('π*') || title.includes('pistar')) {
        categoriesMap['π* Anti-Bonds'].push(item);
      } else if (title.includes('σ*') || title.includes('sigmastar')) {
        categoriesMap['σ* Anti-Bonds'].push(item);
      } else if (title.includes('π') || title.includes('pi_')) {
        categoriesMap['π Bonds'].push(item);
      } else if (title.includes('σ') || title.includes('sigma_')) {
        categoriesMap['σ Bonds'].push(item);
      } else if (title.includes('LP') || title.includes('lp_')) {
        categoriesMap['Lone Pairs'].push(item);
      } else {
        categoriesMap['Reaction Orbitals'].push(item);
      }
    });

    return Object.entries(categoriesMap)
      .filter(([_, items]) => items.length > 0)
      .map(([name, items]) => ({ name, items }));
  }

  /**
   * Helper to format raw orbital keys (e.g. from bundle.orbitalMeshes) into Molecular Explorer categories
   */
  static parseRawOrbitalKeys(keys = []) {
    const uniqueBases = new Set();
    keys.forEach(k => {
      const parts = k.split('_');
      if (parts.length > 1 && !isNaN(parts[0])) {
        uniqueBases.add(parts.slice(1).join('_'));
      } else {
        uniqueBases.add(k);
      }
    });

    const categoriesMap = {
      'Frontier': [],
      'σ Bonds': [],
      'π Bonds': [],
      'σ* Anti-Bonds': [],
      'π* Anti-Bonds': [],
      'Lone Pairs': [],
      'Localized Orbitals': []
    };

    const sorted = Array.from(uniqueBases).sort();
    sorted.forEach(key => {
      const upper = key.toUpperCase();
      let label = key.replace(/_/g, ' ');
      let cat = 'Localized Orbitals';
      let color = '#38bdf8';

      if (upper.includes('HOMO')) {
        cat = 'Frontier';
        color = '#ff5c5c';
        label = key;
      } else if (upper.includes('LUMO')) {
        cat = 'Frontier';
        color = '#55aaff';
        label = key;
      } else if (upper.includes('SIGMASTAR') || upper.includes('SIGMA*') || key.includes('σ*')) {
        cat = 'σ* Anti-Bonds';
        color = '#d946ef';
        label = key.replace(/sigmastar[_\s]*/i, 'σ*(').replace(/$/, ')');
      } else if (upper.includes('PISTAR') || upper.includes('PI*') || key.includes('π*')) {
        cat = 'π* Anti-Bonds';
        color = '#ec4899';
        label = key.replace(/pistar[_\s]*/i, 'π*(').replace(/$/, ')');
      } else if (upper.includes('PI') || key.includes('π')) {
        cat = 'π Bonds';
        color = '#cc44bb';
        label = key.replace(/pi[_\s]*/i, 'π(').replace(/$/, ')');
      } else if (upper.includes('SIGMA') || key.includes('σ')) {
        cat = 'σ Bonds';
        color = '#44cc77';
        label = key.replace(/sigma[_\s]*/i, 'σ(').replace(/$/, ')');
      } else if (upper.includes('LP') || upper.includes('LONE')) {
        cat = 'Lone Pairs';
        color = '#eeee22';
        label = key.replace(/lp[_\s]*/i, 'LP(').replace(/$/, ')');
      } else if (upper.startsWith('IBO')) {
        label = `IBO ${key.replace('IBO_', '')}`;
        color = '#44cc77';
      }

      categoriesMap[cat].push({ key, label, color });
    });

    return Object.entries(categoriesMap)
      .filter(([_, items]) => items.length > 0)
      .map(([name, items]) => ({ name, items }));
  }
}
