/**
 * viewer_loader.js
 * In-memory bundle fetcher and full zip unpacker using JSZip.
 * Unpacks all trajectory coords, SVG, orbital meshes, and ESP surfaces upfront for zero-lag playback.
 */

export async function loadReactionBundle(source, onProgress) {
  let arrayBuffer;

  if (typeof source === 'string') {
    const url = source.trim();
    if (onProgress) onProgress(15, `Fetching bundle from ${url}...`);

    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`Failed to download bundle from ${url} (${resp.status} ${resp.statusText})`);
    }
    arrayBuffer = await resp.arrayBuffer();
  } else if (source instanceof Blob || source instanceof File) {
    if (onProgress) onProgress(10, 'Reading local bundle archive...');
    arrayBuffer = await source.arrayBuffer();
  } else if (source instanceof ArrayBuffer) {
    arrayBuffer = source;
  } else {
    throw new Error('Unsupported bundle source type');
  }

  if (onProgress) onProgress(30, 'Unpacking archive entries into memory...');
  if (!window.JSZip) {
    throw new Error('JSZip library is not loaded');
  }

  const zip = await window.JSZip.loadAsync(arrayBuffer);
  const bundle = {
    manifest: null,
    trajectory: null,
    mechanismSvg: '',
    svgWiring: null,
    orbitalMeshes: {},
    espMeshes: {},
    molecules: {},
    rawZip: zip
  };
  const flatMoleculeFiles = {};

  const fileKeys = Object.keys(zip.files);
  const totalFiles = fileKeys.length;
  let processed = 0;

  for (const name of fileKeys) {
    const file = zip.files[name];
    if (file.dir) continue;

    processed++;
    if (onProgress && processed % 15 === 0) {
      const pct = 30 + Math.floor((processed / totalFiles) * 65);
      onProgress(pct, `Unpacking assets (${processed}/${totalFiles})...`);
    }

    if (name === 'bundle.json') {
      const text = await file.async('string');
      try { bundle.manifest = JSON.parse(text); } catch (e) { console.warn(e); }
    } else if (name === 'multistep_pathway.json') {
      const text = await file.async('string');
      try { bundle.trajectory = JSON.parse(text); } catch (e) { console.warn(e); }
    } else if (name === 'mechanism.svg') {
      bundle.mechanismSvg = await file.async('string');
    } else if (name === 'svg_wiring.json') {
      const text = await file.async('string');
      try { bundle.svgWiring = JSON.parse(text); } catch (e) { console.warn(e); }
    } else if (name.startsWith('orbitals/') && name.endsWith('.json')) {
      const base = name.replace('orbitals/', '');
      const text = await file.async('string');
      try {
        const meshData = JSON.parse(text);
        if (base.endsWith('_pos.json')) {
          const key = base.replace('_pos.json', '');
          bundle.orbitalMeshes[key] = bundle.orbitalMeshes[key] || {};
          bundle.orbitalMeshes[key].pos = meshData;
        } else if (base.endsWith('_neg.json')) {
          const key = base.replace('_neg.json', '');
          bundle.orbitalMeshes[key] = bundle.orbitalMeshes[key] || {};
          bundle.orbitalMeshes[key].neg = meshData;
        }
      } catch (e) {
        console.warn('Failed parsing orbital mesh:', name, e);
      }
    } else if (name.startsWith('esp/') && name.endsWith('.json')) {
      const base = name.replace('esp/', '').replace('.json', '');
      const frameKey = base.replace('frame_', '');
      const text = await file.async('string');
      try {
        bundle.espMeshes[frameKey] = JSON.parse(text);
      } catch (e) {
        console.warn('Failed parsing ESP mesh:', name, e);
      }
    } else if (name.startsWith('molecules/')) {
      const parts = name.split('/');
      if (parts.length > 2) {
        const molKey = parts[1];
        const sub = parts.slice(2).join('/');
        bundle.molecules[molKey] = bundle.molecules[molKey] || {};
        if (sub.endsWith('.json')) {
          const text = await file.async('string');
          try { bundle.molecules[molKey][sub] = JSON.parse(text); } catch (_) { bundle.molecules[molKey][sub] = text; }
        }
      } else if (parts.length === 2) {
        const sub = parts[1];
        if (sub.endsWith('.json')) {
          const text = await file.async('string');
          try { flatMoleculeFiles[sub] = JSON.parse(text); } catch (_) { flatMoleculeFiles[sub] = text; }
        }
      }
    }
  }

  if (Object.keys(bundle.molecules).length === 0 && Object.keys(flatMoleculeFiles).length > 0) {
    let molKey = bundle.manifest?.molecules?.[0]?.key || bundle.manifest?.name;
    if (!molKey) {
      for (const fname of Object.keys(flatMoleculeFiles)) {
        if (fname.endsWith('.json') && !fname.endsWith('_pos.json') && !fname.endsWith('_neg.json') && !fname.endsWith('_esp.json')) {
          molKey = fname.replace(/\.json$/, '');
          break;
        }
      }
    }
    molKey = molKey || 'molecule';
    bundle.molecules[molKey] = flatMoleculeFiles;
  }

  if (!bundle.trajectory && bundle.manifest && bundle.manifest.files && bundle.manifest.files.trajectory_json) {
    const trajFile = zip.file(bundle.manifest.files.trajectory_json);
    if (trajFile) {
      const text = await trajFile.async('string');
      bundle.trajectory = JSON.parse(text);
    }
  }

  if (!bundle.trajectory && Object.keys(bundle.molecules).length > 0) {
    const firstKey = Object.keys(bundle.molecules)[0];
    const molBundle = buildMoleculeBundle(firstKey, bundle);
    if (molBundle && molBundle.trajectory) {
      bundle.trajectory = molBundle.trajectory;
      bundle.manifest = bundle.manifest || molBundle.manifest;
      bundle.orbitalMeshes = molBundle.orbitalMeshes;
      bundle.espMeshes = molBundle.espMeshes;
      bundle.categories = molBundle.categories;
      bundle.hasEsp = molBundle.hasEsp;
      bundle.isSingleMolecule = true;
      bundle.charges = molBundle.charges;
      bundle.molKey = molBundle.molKey;
    }
  }

  if (!bundle.trajectory) {
    throw new Error('Archive does not contain a valid multistep_pathway.json or molecule structure.');
  }

  if (onProgress) onProgress(100, 'Unpack complete. Initializing 3D viewer...');
  return bundle;
}

/**
 * Adapt a molecule sub-directory inside a reaction bundle into a standalone 1-state bundle.
 */
export function buildMoleculeBundle(molIdentifier, parentBundle) {
  if (!parentBundle || !parentBundle.molecules) return null;
  const mols = parentBundle.molecules;

  const norm = s => String(s || '').toLowerCase().replace(/[\s\-_]+/g, '');
  const targetNorm = norm(molIdentifier);

  const wiring = parentBundle.svgWiring || (parentBundle.trajectory && parentBundle.trajectory.svg_wiring) || (parentBundle.manifest && parentBundle.manifest.svg_wiring) || {};
  const wiredCfg = wiring[molIdentifier];
  const wiredName = wiredCfg?.moleculeName || wiredCfg?.molName || '';
  const wiredNorm = norm(wiredName);

  let molKey = null;
  let molDict = null;

  for (const k in mols) {
    const kn = norm(k);
    if (k === molIdentifier || kn === targetNorm || (wiredNorm && kn === wiredNorm)) {
      molKey = k;
      molDict = mols[k];
      break;
    }
  }

  if (!molDict && parentBundle.trajectory && parentBundle.trajectory.molecule_bundles) {
    const mb = parentBundle.trajectory.molecule_bundles;
    const entry = mb[molIdentifier] || (wiredName && mb[wiredName]) || Object.values(mb).find(v => {
      const vk = norm(v.molecule_key || v.key || '');
      const vm = norm(v.molecule_name || v.name || '');
      const vi = norm(v.molecule_id || '');
      return vk === targetNorm || vm === targetNorm || vi === targetNorm ||
             (wiredNorm && (vk === wiredNorm || vm === wiredNorm));
    });
    if (entry && entry.directory) {
      const dirClean = entry.directory.replace(/^molecules\//, '').replace(/\/$/, '');
      if (mols[dirClean]) {
        molKey = dirClean;
        molDict = mols[dirClean];
      }
    }
  }

  if (!molDict) {
    for (const k in mols) {
      const kn = norm(k);
      if (targetNorm && (kn.includes(targetNorm) || targetNorm.includes(kn))) {
        molKey = k;
        molDict = mols[k];
        break;
      }
    }
  }

  if (!molDict) return null;

  const manifest = molDict['manifest.json'] || parentBundle.manifest || {};
  const molInfo = (manifest.molecules && manifest.molecules[0]) || molDict[`${molKey}.json`] || {};

  const xyzText = molInfo.xyz || '';
  const lines = xyzText.trim().split('\n');
  const symbols = [];
  const coords = [];
  if (lines.length >= 3) {
    for (let i = 2; i < lines.length; i++) {
      const parts = lines[i].trim().split(/\s+/);
      if (parts.length >= 4) {
        symbols.push(parts[0]);
        coords.push([parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3])]);
      }
    }
  }

  const orbitalMeshes = {};
  const espMeshes = {};
  for (const [fname, content] of Object.entries(molDict)) {
    if (fname.endsWith('_pos.json')) {
      const base = fname.replace(/^molecules\//, '').replace(/_pos\.json$/, '');
      orbitalMeshes[base] = orbitalMeshes[base] || {};
      orbitalMeshes[base].pos = content;
    } else if (fname.endsWith('_neg.json')) {
      const base = fname.replace(/^molecules\//, '').replace(/_neg\.json$/, '');
      orbitalMeshes[base] = orbitalMeshes[base] || {};
      orbitalMeshes[base].neg = content;
    } else if (fname.endsWith('_esp.json')) {
      espMeshes['0'] = content;
    }
  }

  const rawCategories = molInfo.orbitals || manifest.orbitals || [];
  const categories = rawCategories.map(cat => ({
    name: cat.name,
    items: (cat.items || []).map(it => {
      const rawKey = it.file ? it.file.replace(/^molecules\//, '').replace(/\.json$/, '') : (it.key || it.label);
      return {
        key: rawKey,
        label: it.label || rawKey,
        color: it.color || '#38bdf8',
        energy_ev: it.energy_ev
      };
    })
  }));

  const title = wiredName || molInfo.name || molInfo.key || molKey.replace(/_/g, ' ');
  const charges = molInfo.charges || manifest.charges || molDict[`${molKey}.json`]?.charges || null;

  return {
    manifest: {
      title,
      molKey,
      smiles: molInfo.smiles || '',
      isSingleMolecule: true,
      charges
    },
    trajectory: {
      symbols,
      frames: [{ coords, energy_kcal: 0, charges }],
      trajectory_charges: charges ? [charges] : null
    },
    charges,
    rawZip: parentBundle.rawZip,
    molKey,
    orbitalMeshes,
    espMeshes,
    categories,
    hasEsp: true,
    isSingleMolecule: true
  };
}
