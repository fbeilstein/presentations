(() => {
  // static_viewer/js/viewer_loader.js
  async function loadReactionBundle(source, onProgress) {
    let arrayBuffer;
    if (typeof source === "string") {
      const url = source.trim();
      if (onProgress) onProgress(15, `Fetching bundle from ${url}...`);
      const resp = await fetch(url);
      if (!resp.ok) {
        throw new Error(`Failed to download bundle from ${url} (${resp.status} ${resp.statusText})`);
      }
      arrayBuffer = await resp.arrayBuffer();
    } else if (source instanceof Blob || source instanceof File) {
      if (onProgress) onProgress(10, "Reading local bundle archive...");
      arrayBuffer = await source.arrayBuffer();
    } else if (source instanceof ArrayBuffer) {
      arrayBuffer = source;
    } else {
      throw new Error("Unsupported bundle source type");
    }
    if (onProgress) onProgress(30, "Unpacking archive entries into memory...");
    if (!window.JSZip) {
      throw new Error("JSZip library is not loaded");
    }
    const zip = await window.JSZip.loadAsync(arrayBuffer);
    const bundle = {
      manifest: null,
      trajectory: null,
      mechanismSvg: "",
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
        const pct = 30 + Math.floor(processed / totalFiles * 65);
        onProgress(pct, `Unpacking assets (${processed}/${totalFiles})...`);
      }
      if (name === "bundle.json") {
        const text = await file.async("string");
        try {
          bundle.manifest = JSON.parse(text);
        } catch (e) {
          console.warn(e);
        }
      } else if (name === "multistep_pathway.json") {
        const text = await file.async("string");
        try {
          bundle.trajectory = JSON.parse(text);
        } catch (e) {
          console.warn(e);
        }
      } else if (name === "mechanism.svg") {
        bundle.mechanismSvg = await file.async("string");
      } else if (name === "svg_wiring.json") {
        const text = await file.async("string");
        try {
          bundle.svgWiring = JSON.parse(text);
        } catch (e) {
          console.warn(e);
        }
      } else if (name.startsWith("orbitals/") && name.endsWith(".json")) {
        const base = name.replace("orbitals/", "");
        const text = await file.async("string");
        try {
          const meshData = JSON.parse(text);
          if (base.endsWith("_pos.json")) {
            const key = base.replace("_pos.json", "");
            bundle.orbitalMeshes[key] = bundle.orbitalMeshes[key] || {};
            bundle.orbitalMeshes[key].pos = meshData;
          } else if (base.endsWith("_neg.json")) {
            const key = base.replace("_neg.json", "");
            bundle.orbitalMeshes[key] = bundle.orbitalMeshes[key] || {};
            bundle.orbitalMeshes[key].neg = meshData;
          }
        } catch (e) {
          console.warn("Failed parsing orbital mesh:", name, e);
        }
      } else if (name.startsWith("esp/") && name.endsWith(".json")) {
        const base = name.replace("esp/", "").replace(".json", "");
        const frameKey = base.replace("frame_", "");
        const text = await file.async("string");
        try {
          bundle.espMeshes[frameKey] = JSON.parse(text);
        } catch (e) {
          console.warn("Failed parsing ESP mesh:", name, e);
        }
      } else if (name.startsWith("molecules/")) {
        const parts = name.split("/");
        if (parts.length > 2) {
          const molKey = parts[1];
          const sub = parts.slice(2).join("/");
          bundle.molecules[molKey] = bundle.molecules[molKey] || {};
          if (sub.endsWith(".json")) {
            const text = await file.async("string");
            try {
              bundle.molecules[molKey][sub] = JSON.parse(text);
            } catch (_) {
              bundle.molecules[molKey][sub] = text;
            }
          }
        } else if (parts.length === 2) {
          const sub = parts[1];
          if (sub.endsWith(".json")) {
            const text = await file.async("string");
            try {
              flatMoleculeFiles[sub] = JSON.parse(text);
            } catch (_) {
              flatMoleculeFiles[sub] = text;
            }
          }
        }
      }
    }
    if (Object.keys(bundle.molecules).length === 0 && Object.keys(flatMoleculeFiles).length > 0) {
      let molKey = bundle.manifest?.molecules?.[0]?.key || bundle.manifest?.name;
      if (!molKey) {
        for (const fname of Object.keys(flatMoleculeFiles)) {
          if (fname.endsWith(".json") && !fname.endsWith("_pos.json") && !fname.endsWith("_neg.json") && !fname.endsWith("_esp.json")) {
            molKey = fname.replace(/\.json$/, "");
            break;
          }
        }
      }
      molKey = molKey || "molecule";
      bundle.molecules[molKey] = flatMoleculeFiles;
    }
    if (!bundle.trajectory && bundle.manifest && bundle.manifest.files && bundle.manifest.files.trajectory_json) {
      const trajFile = zip.file(bundle.manifest.files.trajectory_json);
      if (trajFile) {
        const text = await trajFile.async("string");
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
    bundle.hasEsp = bundle.hasEsp ?? (bundle.espMeshes && Object.keys(bundle.espMeshes).length > 0);
    if (!bundle.trajectory) {
      throw new Error("Archive does not contain a valid multistep_pathway.json or molecule structure.");
    }
    if (onProgress) onProgress(100, "Unpack complete. Initializing 3D viewer...");
    return bundle;
  }
  function buildMoleculeBundle(molIdentifier, parentBundle) {
    if (!parentBundle || !parentBundle.molecules) return null;
    const mols = parentBundle.molecules;
    const norm = (s) => String(s || "").toLowerCase().replace(/[\s\-_]+/g, "");
    const targetNorm = norm(molIdentifier);
    const wiring = parentBundle.svgWiring || parentBundle.trajectory && parentBundle.trajectory.svg_wiring || parentBundle.manifest && parentBundle.manifest.svg_wiring || {};
    const wiredCfg = wiring[molIdentifier];
    const wiredName = wiredCfg?.moleculeName || wiredCfg?.molName || "";
    const wiredNorm = norm(wiredName);
    let molKey = null;
    let molDict = null;
    for (const k in mols) {
      const kn = norm(k);
      if (k === molIdentifier || kn === targetNorm || wiredNorm && kn === wiredNorm) {
        molKey = k;
        molDict = mols[k];
        break;
      }
    }
    if (!molDict && parentBundle.trajectory && parentBundle.trajectory.molecule_bundles) {
      const mb = parentBundle.trajectory.molecule_bundles;
      const entry = mb[molIdentifier] || wiredName && mb[wiredName] || Object.values(mb).find((v) => {
        const vk = norm(v.molecule_key || v.key || "");
        const vm = norm(v.molecule_name || v.name || "");
        const vi = norm(v.molecule_id || "");
        return vk === targetNorm || vm === targetNorm || vi === targetNorm || wiredNorm && (vk === wiredNorm || vm === wiredNorm);
      });
      if (entry && entry.directory) {
        const dirClean = entry.directory.replace(/^molecules\//, "").replace(/\/$/, "");
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
    const manifest = molDict["manifest.json"] || parentBundle.manifest || {};
    const molInfo = manifest.molecules && manifest.molecules[0] || molDict[`${molKey}.json`] || {};
    const xyzText = molInfo.xyz || "";
    const lines = xyzText.trim().split("\n");
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
      if (fname.endsWith("_pos.json")) {
        const base = fname.replace(/^molecules\//, "").replace(/_pos\.json$/, "");
        orbitalMeshes[base] = orbitalMeshes[base] || {};
        orbitalMeshes[base].pos = content;
      } else if (fname.endsWith("_neg.json")) {
        const base = fname.replace(/^molecules\//, "").replace(/_neg\.json$/, "");
        orbitalMeshes[base] = orbitalMeshes[base] || {};
        orbitalMeshes[base].neg = content;
      } else if (fname.endsWith("_esp.json")) {
        espMeshes["0"] = content;
      }
    }
    const rawCategories = molInfo.orbitals || manifest.orbitals || [];
    const categories = rawCategories.map((cat) => ({
      name: cat.name,
      items: (cat.items || []).map((it) => {
        const rawKey = it.file ? it.file.replace(/^molecules\//, "").replace(/\.json$/, "") : it.key || it.label;
        return {
          key: rawKey,
          label: it.label || rawKey,
          color: it.color || "#38bdf8",
          energy_ev: it.energy_ev
        };
      })
    }));
    const title = wiredName || molInfo.name || molInfo.key || molKey.replace(/_/g, " ");
    const charges = molInfo.charges || manifest.charges || molDict[`${molKey}.json`]?.charges || null;
    return {
      manifest: {
        title,
        molKey,
        smiles: molInfo.smiles || "",
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
      hasEsp: Object.keys(espMeshes).length > 0,
      isSingleMolecule: true
    };
  }

  // static_viewer/js/viewer_3d.js
  var COV_RADII = {
    H: 0.31,
    C: 0.76,
    N: 0.71,
    O: 0.66,
    F: 0.57,
    P: 1.07,
    S: 1.05,
    Cl: 1.02,
    Br: 1.2,
    I: 1.39
  };
  var MAX_VALENCE = { H: 1, F: 1, Cl: 1, Br: 1, I: 1, O: 3, N: 4, C: 4, S: 6, P: 5 };
  function computeBonds(symbols, coords) {
    const bonds = [];
    const n = Math.min(symbols.length, coords.length);
    const valences = new Array(n).fill(0);
    const candidates = [];
    const adj = Array.from({ length: n }, () => []);
    const distMat = {};
    for (let i = 0; i < n; i++) {
      const s1 = symbols[i];
      const r1 = COV_RADII[s1] || 0.75;
      for (let j = i + 1; j < n; j++) {
        const s2 = symbols[j];
        const r2 = COV_RADII[s2] || 0.75;
        if (s1 === "H" && s2 === "H" && n > 2) continue;
        const dx = coords[i][0] - coords[j][0];
        const dy = coords[i][1] - coords[j][1];
        const dz = coords[i][2] - coords[j][2];
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const maxD = 1.28 * (r1 + r2);
        if (d >= 0.4 && d <= maxD) {
          adj[i].push(j);
          adj[j].push(i);
          const key = `${i}_${j}`;
          distMat[key] = d;
          candidates.push({ i, j, s1, s2, d });
        }
      }
    }
    const getDist = (u, v) => {
      const key = u < v ? `${u}_${v}` : `${v}_${u}`;
      return distMat[key] || 1.4;
    };
    const ringDoubleBonds = /* @__PURE__ */ new Set();
    const ringBondsAll = /* @__PURE__ */ new Set();
    const ringAtoms = [];
    for (let i = 0; i < n; i++) {
      if (symbols[i] === "C" || symbols[i] === "N") {
        ringAtoms.push(i);
      }
    }
    const cycles = [];
    const visited = /* @__PURE__ */ new Set();
    const path = [];
    const findCycles = (v, start, depth) => {
      if (depth === 6) {
        if (adj[v].includes(start)) {
          cycles.push([...path]);
        }
        return;
      }
      for (const u of adj[v]) {
        if ((symbols[u] === "C" || symbols[u] === "N") && !visited.has(u)) {
          visited.add(u);
          path.push(u);
          findCycles(u, start, depth + 1);
          path.pop();
          visited.delete(u);
        }
      }
    };
    for (const startAtom of ringAtoms) {
      visited.clear();
      visited.add(startAtom);
      path.length = 0;
      path.push(startAtom);
      findCycles(startAtom, startAtom, 1);
    }
    const uniqueRings = [];
    const seenRings = /* @__PURE__ */ new Set();
    for (const c of cycles) {
      const sorted = [...c].sort((a, b) => a - b).join(",");
      if (!seenRings.has(sorted)) {
        seenRings.add(sorted);
        uniqueRings.push(c);
      }
    }
    for (const ring of uniqueRings) {
      let isConjugated = true;
      for (let k = 0; k < 6; k++) {
        const u = ring[k];
        const v = ring[(k + 1) % 6];
        if (getDist(u, v) > 1.48) {
          isConjugated = false;
          break;
        }
      }
      if (!isConjugated) continue;
      const edges0 = [[ring[0], ring[1]], [ring[2], ring[3]], [ring[4], ring[5]]];
      const edges1 = [[ring[1], ring[2]], [ring[3], ring[4]], [ring[5], ring[0]]];
      const sum0 = edges0.reduce((acc, [u, v]) => acc + getDist(u, v), 0);
      const sum1 = edges1.reduce((acc, [u, v]) => acc + getDist(u, v), 0);
      const chosen = sum0 <= sum1 ? edges0 : edges1;
      for (const [u, v] of chosen) {
        const key = u < v ? `${u}_${v}` : `${v}_${u}`;
        ringDoubleBonds.add(key);
      }
      for (let k = 0; k < 6; k++) {
        const u = ring[k];
        const v = ring[(k + 1) % 6];
        const key = u < v ? `${u}_${v}` : `${v}_${u}`;
        ringBondsAll.add(key);
      }
    }
    for (const c of candidates) {
      const pairKey = c.i < c.j ? `${c.i}_${c.j}` : `${c.j}_${c.i}`;
      const pair = [c.s1, c.s2].sort().join("-");
      let targetOrder = 1;
      if (ringDoubleBonds.has(pairKey)) {
        targetOrder = 2;
      } else if (!ringBondsAll.has(pairKey)) {
        if (pair === "C-O") {
          if (c.d < 1.16) targetOrder = 3;
          else if (c.d < 1.265) targetOrder = 2;
        } else if (pair === "C-C") {
          if (c.d < 1.24) targetOrder = 3;
          else if (c.d < 1.39) targetOrder = 2;
        } else if (pair === "C-N") {
          if (c.d < 1.18) targetOrder = 3;
          else if (c.d < 1.32) targetOrder = 2;
        } else if (pair === "N-N") {
          if (c.d < 1.14) targetOrder = 3;
          else if (c.d < 1.28) targetOrder = 2;
        } else if (pair === "N-O" || pair === "O-O") {
          if (c.d < 1.25) targetOrder = 2;
        }
      }
      bonds.push({
        a1: c.i,
        a2: c.j,
        order: 1,
        target: targetOrder,
        s1: c.s1,
        s2: c.s2,
        d: c.d,
        pairKey
      });
      valences[c.i]++;
      valences[c.j]++;
    }
    bonds.sort((a, b) => {
      const prioA = a.target === 3 ? 0 : ringDoubleBonds.has(a.pairKey) ? 1 : a.target === 2 ? 2 : 3;
      const prioB = b.target === 3 ? 0 : ringDoubleBonds.has(b.pairKey) ? 1 : b.target === 2 ? 2 : 3;
      if (prioA !== prioB) return prioA - prioB;
      return a.d - b.d;
    });
    for (const b of bonds) {
      if (b.target > 1) {
        const v1Cap = MAX_VALENCE[b.s1] || 4;
        const v2Cap = MAX_VALENCE[b.s2] || 4;
        const desiredBoost = b.target - 1;
        const maxBoost = Math.min(v1Cap - valences[b.a1], v2Cap - valences[b.a2]);
        const boost = Math.max(0, Math.min(desiredBoost, maxBoost));
        if (boost > 0) {
          b.order += boost;
          valences[b.a1] += boost;
          valences[b.a2] += boost;
        }
      }
    }
    return bonds;
  }
  var Viewer3D = class {
    constructor(containerElement) {
      this.container = containerElement;
      this.glViewer = null;
      this.bundle = null;
      this.currentFrameIdx = 0;
      this.activeOrbitals = /* @__PURE__ */ new Set();
      this.activeEsp = false;
      this.showCharges = false;
      this.orbitalIsovalue = 0.035;
      this.initGL();
    }
    initGL() {
      if (!window.$3Dmol) {
        console.error("$3Dmol library is not loaded");
        return;
      }
      this.glViewer = window.$3Dmol.createViewer(this.container, {
        backgroundColor: "0x0f141c",
        id: "reactionViewer"
      });
      window.addEventListener("resize", () => {
        if (this.glViewer) this.glViewer.resize();
      });
    }
    setBundle(bundle) {
      this.bundle = bundle;
      this.currentFrameIdx = 0;
      this.activeOrbitals.clear();
      this.activeEsp = false;
      this.showCharges = false;
      this.updateColorbar(false);
      if (this.glViewer) {
        this.glViewer.clear();
        this.renderFrame(0, true);
      }
    }
    renderFrame(frameIdx, autoZoom = false) {
      if (!this.glViewer || !this.bundle || !this.bundle.trajectory) return;
      const traj = this.bundle.trajectory;
      const frames = traj.frames || [];
      if (frameIdx < 0 || frameIdx >= frames.length) return;
      this.currentFrameIdx = frameIdx;
      const frame = frames[frameIdx];
      const coords = frame.coords || frame.positions || [];
      const symbols = traj.symbols || [];
      const natoms = Math.min(symbols.length, coords.length);
      const bonds = computeBonds(symbols, coords);
      let sdf = `
  Reaction Pathway Viewer

${String(natoms).padStart(3, " ")}${String(bonds.length).padStart(3, " ")}  0  0  0  0  0  0  0  0999 V2000
`;
      for (let i = 0; i < natoms; i++) {
        const sym = symbols[i];
        const pos = coords[i];
        const x = Number(pos[0]).toFixed(4).padStart(10, " ");
        const y = Number(pos[1]).toFixed(4).padStart(10, " ");
        const z = Number(pos[2]).toFixed(4).padStart(10, " ");
        sdf += `${x}${y}${z} ${sym.padEnd(3, " ")} 0  0  0  0  0  0  0  0  0  0  0  0
`;
      }
      for (let i = 0; i < bonds.length; i++) {
        const b = bonds[i];
        sdf += `${String(b.a1 + 1).padStart(3, " ")}${String(b.a2 + 1).padStart(3, " ")}${String(b.order).padStart(3, " ")}  0  0  0  0
`;
      }
      sdf += "M  END\n$$$$\n";
      this.glViewer.removeAllSurfaces();
      this.glViewer.removeAllShapes();
      this.glViewer.clear();
      const m = this.glViewer.addModel(sdf, "sdf");
      m.setStyle({}, {
        stick: { radius: 0.13, colorscheme: "Jmol" },
        sphere: { scale: 0.28, colorscheme: "Jmol" }
      });
      if (this.activeOrbitals.size > 0 && this.bundle.orbitalMeshes) {
        const CC = window.$3Dmol ? window.$3Dmol.CC : null;
        for (const orbKey of this.activeOrbitals) {
          const sKey = String(orbKey).toLowerCase();
          const isAnti = sKey.includes("star") || sKey.includes("*") || sKey.includes("lumo") || sKey.includes("antibond") || sKey.includes("virtual");
          const posColorHex = isAnti ? "#d946ef" : "#38bdf8";
          const negColorHex = isAnti ? "#22c55e" : "#f97316";
          const posColorInt = isAnti ? 14239471 : 3718648;
          const negColorInt = isAnti ? 2278750 : 16347926;
          const meshPair = this.findOrbitalMesh(frameIdx, orbKey);
          if (meshPair) {
            const pV = meshPair.pos?.vertices || meshPair.pos?.vertexArr;
            const pF = meshPair.pos?.faces || meshPair.pos?.faceArr;
            const pN = meshPair.pos?.normals || meshPair.pos?.normalArr;
            if (pV && pF) {
              this.glViewer.addCustom({
                vertexArr: pV,
                faceArr: pF,
                normalArr: pN,
                color: CC ? CC.color(posColorHex) : posColorInt,
                opacity: 0.75
              });
            }
            const nV = meshPair.neg?.vertices || meshPair.neg?.vertexArr;
            const nF = meshPair.neg?.faces || meshPair.neg?.faceArr;
            const nN = meshPair.neg?.normals || meshPair.neg?.normalArr;
            if (nV && nF) {
              this.glViewer.addCustom({
                vertexArr: nV,
                faceArr: nF,
                normalArr: nN,
                color: CC ? CC.color(negColorHex) : negColorInt,
                opacity: 0.75
              });
            }
          } else if (this.bundle.cubeCache) {
            const cubeData = this.bundle.cubeCache[`${frameIdx}_${orbKey}`] || this.bundle.cubeCache[orbKey];
            if (cubeData && window.$3Dmol) {
              try {
                const vdo = new window.$3Dmol.VolumeData(cubeData, "cube");
                const iso = this.orbitalIsovalue || 0.035;
                this.glViewer.addIsosurface(vdo, { isoval: iso, color: posColorHex, opacity: 0.72 });
                this.glViewer.addIsosurface(vdo, { isoval: -iso, color: negColorHex, opacity: 0.72 });
              } catch (_) {
              }
            }
          }
        }
      }
      if (this.activeEsp) {
        const espMesh = this.bundle.espMeshes ? this.bundle.espMeshes[frameIdx] || this.bundle.espMeshes[String(frameIdx)] || (this.bundle.isSingleMolecule ? this.bundle.espMeshes["0"] || Object.values(this.bundle.espMeshes)[0] : null) : null;
        const eV = espMesh?.vertices || espMesh?.vertexArr;
        const eF = espMesh?.faces || espMesh?.faceArr;
        const eN = espMesh?.normals || espMesh?.normalArr;
        if (eV && eF) {
          let colorParam = 1096065;
          if (espMesh.esp_values && espMesh.esp_values.length === eV.length) {
            const absMax = Math.max(Math.abs(espMesh.esp_min || -30), Math.abs(espMesh.esp_max || 30), 20);
            colorParam = espMesh.esp_values.map((v) => {
              const t = Math.max(-1, Math.min(1, v / absMax));
              return t < 0 ? { r: 1, g: 1 + t, b: 1 + t } : { r: 1 - t, g: 1 - t, b: 1 };
            });
          }
          const customObj = {
            vertexArr: eV,
            faceArr: eF,
            color: colorParam,
            opacity: 0.65
          };
          if (eN && eN.length === eV.length) {
            customObj.normalArr = eN;
          }
          this.glViewer.addCustom(customObj);
        } else if (window.$3Dmol) {
          try {
            const traj2 = this.bundle.trajectory || {};
            const trajCharges = traj2.trajectory_charges ? traj2.trajectory_charges[frameIdx] : null;
            const frameCharges = frame.charges || trajCharges || this.bundle.charges || [];
            const atoms = this.glViewer.selectedAtoms({});
            for (let i = 0; i < atoms.length && i < natoms; i++) {
              let q = frameCharges && frameCharges[i] !== void 0 ? frameCharges[i] : null;
              if (q === null) {
                const s = symbols[i];
                if (s === "H") q = 0.08;
                else if (s === "C") q = -0.05;
                else if (s === "O") q = -0.5;
                else if (s === "Cl") q = -0.45;
                else q = 0;
              }
              atoms[i].partialCharge = q;
              atoms[i].properties = atoms[i].properties || {};
              atoms[i].properties.partialCharge = q;
            }
            this.glViewer.addSurface(window.$3Dmol.SurfaceType.VDW, {
              opacity: 0.65,
              colorscheme: { prop: "partialCharge", gradient: "rwb", min: -0.6, max: 0.6 }
            });
          } catch (e) {
            console.warn("Dynamic surface rendering failed:", e);
          }
        }
      }
      if (this.showCharges) {
        const traj2 = this.bundle.trajectory || {};
        const trajCharges = traj2.trajectory_charges ? traj2.trajectory_charges[frameIdx] : null;
        const charges = trajCharges || frame.charges || traj2.charges || this.bundle.charges || this.bundle.manifest && this.bundle.manifest.charges || [];
        for (let i = 0; i < natoms; i++) {
          let q = charges && charges[i] !== void 0 ? charges[i] : null;
          if (q === null) {
            const s = symbols[i];
            if (s === "H") q = 0.06;
            else if (s === "C") q = -0.06;
            else if (s === "O") q = -0.5;
            else if (s === "N") q = -0.4;
            else if (s === "Cl") q = -0.2;
            else q = 0;
          }
          const sym = symbols[i];
          const pos = coords[i];
          const sign = q > 0 ? "+" : "";
          this.glViewer.addLabel(`${sym}: ${sign}${Number(q).toFixed(2)}`, {
            position: { x: pos[0], y: pos[1] + 0.35, z: pos[2] },
            fontSize: 10,
            fontColor: "#ffffff",
            backgroundColor: "rgba(15, 23, 42, 0.75)",
            backgroundOpacity: 0.75
          });
        }
      }
      if (this.activeEsp) {
        this.updateColorbar(true);
      }
      if (autoZoom) {
        this.glViewer.zoomTo();
      }
      this.glViewer.render();
    }
    findOrbitalMesh(frameIdx, orbKey) {
      if (!this.bundle || !this.bundle.orbitalMeshes) return null;
      const meshes = this.bundle.orbitalMeshes;
      const k1 = `${frameIdx}_${orbKey}`;
      if (meshes[k1]) return meshes[k1];
      if (this.bundle.isSingleMolecule && meshes[orbKey]) {
        return meshes[orbKey];
      }
      for (const k in meshes) {
        if (k.startsWith(`${frameIdx}_`)) {
          const baseKey = k.replace(`${frameIdx}_`, "");
          if (baseKey === orbKey || k.includes(orbKey) || orbKey.includes(baseKey)) {
            return meshes[k];
          }
        }
      }
      if (this.bundle.isSingleMolecule) {
        for (const k in meshes) {
          const stripped = k.replace(/^\d+_/, "");
          if (stripped === orbKey || k.includes(orbKey) || orbKey.includes(stripped)) {
            return meshes[k];
          }
        }
      }
      return null;
    }
    toggleOrbital(orbKey, isChecked) {
      if (isChecked) {
        this.activeOrbitals.add(orbKey);
      } else {
        this.activeOrbitals.delete(orbKey);
      }
      this.renderFrame(this.currentFrameIdx, false);
    }
    clearOrbitals() {
      this.activeOrbitals.clear();
      this.renderFrame(this.currentFrameIdx, false);
    }
    setEspActive(active) {
      this.activeEsp = !!active;
      this.updateColorbar(this.activeEsp);
      this.renderFrame(this.currentFrameIdx, false);
    }
    setChargesActive(active) {
      this.showCharges = !!active;
      this.renderFrame(this.currentFrameIdx, false);
    }
    updateColorbar(active) {
      const colorbar = this.container.parentElement?.querySelector("#esp-colorbar") || document.getElementById("esp-colorbar");
      if (!colorbar) return;
      colorbar.style.display = active ? "flex" : "none";
      if (active && this.bundle) {
        const espMesh = this.bundle.espMeshes ? this.bundle.espMeshes[this.currentFrameIdx] || this.bundle.espMeshes[String(this.currentFrameIdx)] || (this.bundle.isSingleMolecule ? this.bundle.espMeshes["0"] || Object.values(this.bundle.espMeshes)[0] : null) : null;
        const absMax = Math.max(Math.abs(espMesh?.esp_min || -30), Math.abs(espMesh?.esp_max || 30), 20);
        const minEl = colorbar.querySelector("#esp-colorbar-min") || document.getElementById("esp-colorbar-min");
        const maxEl = colorbar.querySelector("#esp-colorbar-max") || document.getElementById("esp-colorbar-max");
        if (minEl) minEl.textContent = `${(-absMax).toFixed(1)}`;
        if (maxEl) maxEl.textContent = `+${absMax.toFixed(1)}`;
      }
    }
  };

  // static_viewer/js/viewer_pes.js
  var ViewerPES = class {
    constructor(canvasElement) {
      this.canvas = canvasElement;
      this.ctx = canvasElement.getContext("2d");
      this.energies = [];
      this.currentFrame = 0;
      this.totalFrames = 0;
      window.addEventListener("resize", () => this.resizeAndDraw());
    }
    setTrajectory(trajectory) {
      if (!trajectory) return;
      const frames = trajectory.frames || [];
      this.totalFrames = frames.length;
      if (trajectory.energy_profile && Array.isArray(trajectory.energy_profile.frame_energies_kcal)) {
        this.energies = trajectory.energy_profile.frame_energies_kcal;
      } else {
        this.energies = frames.map((f) => f.rel_energy_kcal !== void 0 ? f.rel_energy_kcal : f.energy_kcal || 0);
      }
      this.resizeAndDraw();
    }
    setCurrentFrame(frameIdx) {
      this.currentFrame = frameIdx;
      this.draw();
    }
    resizeAndDraw() {
      if (!this.canvas) return;
      const rect = this.canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      this.canvas.width = Math.max(10, rect.width * dpr);
      this.canvas.height = Math.max(10, rect.height * dpr);
      this.draw();
    }
    draw() {
      if (!this.ctx || !this.energies.length) return;
      const ctx = this.ctx;
      const w = this.canvas.width;
      const h = this.canvas.height;
      const padTop = 20;
      const padBottom = 24;
      const padSide = 32;
      ctx.clearRect(0, 0, w, h);
      const minE = Math.min(0, ...this.energies);
      const maxE = Math.max(1, ...this.energies);
      const rangeE = Math.max(0.1, maxE - minE);
      const getX = (idx) => padSide + idx / Math.max(1, this.energies.length - 1) * (w - 2 * padSide);
      const getY = (val) => h - padBottom - (val - minE) / rangeE * (h - padTop - padBottom);
      const zeroY = getY(0);
      ctx.strokeStyle = "rgba(148, 163, 184, 0.2)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(padSide, zeroY);
      ctx.lineTo(w - padSide, zeroY);
      ctx.stroke();
      ctx.setLineDash([]);
      const grad = ctx.createLinearGradient(0, padTop, 0, h - padBottom);
      grad.addColorStop(0, "rgba(56, 189, 248, 0.25)");
      grad.addColorStop(1, "rgba(56, 189, 248, 0.0)");
      ctx.beginPath();
      ctx.moveTo(getX(0), h - padBottom);
      for (let i = 0; i < this.energies.length; i++) {
        ctx.lineTo(getX(i), getY(this.energies[i]));
      }
      ctx.lineTo(getX(this.energies.length - 1), h - padBottom);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      for (let i = 0; i < this.energies.length; i++) {
        const x = getX(i);
        const y = getY(this.energies[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      if (this.currentFrame >= 0 && this.currentFrame < this.energies.length) {
        const curX = getX(this.currentFrame);
        const curY = getY(this.energies[this.currentFrame]);
        ctx.strokeStyle = "rgba(56, 189, 248, 0.4)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(curX, padTop);
        ctx.lineTo(curX, h - padBottom);
        ctx.stroke();
        ctx.fillStyle = "#38bdf8";
        ctx.beginPath();
        ctx.arc(curX, curY, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
  };

  // static_viewer/js/viewer_orbitals.js
  var ViewerOrbitalsPanel = class {
    constructor(containerElement, onOrbitalToggle, onEspToggle, onChargesToggle) {
      this.container = containerElement;
      this.onOrbitalToggle = onOrbitalToggle || (() => {
      });
      this.onEspToggle = onEspToggle || (() => {
      });
      this.onChargesToggle = onChargesToggle || (() => {
      });
      this.activeOrbitals = /* @__PURE__ */ new Set();
      this.isEspActive = false;
      this.isChargesActive = false;
      this.collapsedGroups = /* @__PURE__ */ new Set();
    }
    clear() {
      this.container.innerHTML = "";
      this.activeOrbitals.clear();
      this.isEspActive = false;
      this.isChargesActive = false;
    }
    /**
     * Render grouped categories from molecule JSON or reaction pathway tracks
     */
    renderGroups(categories, hasEsp = false) {
      this.container.innerHTML = "";
      if (!categories || categories.length === 0) {
        this.container.innerHTML = '<span class="empty-hint">No orbitals available</span>';
        return;
      }
      categories.forEach((cat, catIdx) => {
        const groupEl = document.createElement("div");
        groupEl.className = "orbital-group";
        const isCollapsed = this.collapsedGroups.has(cat.name);
        const header = document.createElement("div");
        header.className = "orbital-group-header";
        header.innerHTML = `
        <span class="group-title">${cat.name}</span>
        <span class="group-chevron">${isCollapsed ? "\u25B8" : "\u25BE"}</span>
      `;
        header.addEventListener("click", () => {
          if (this.collapsedGroups.has(cat.name)) {
            this.collapsedGroups.delete(cat.name);
          } else {
            this.collapsedGroups.add(cat.name);
          }
          this.renderGroups(categories, hasEsp);
        });
        groupEl.appendChild(header);
        const body = document.createElement("div");
        body.className = "orbital-group-body";
        if (isCollapsed) body.style.display = "none";
        (cat.items || []).forEach((item) => {
          const row = document.createElement("label");
          row.className = "orbital-item-row";
          const chk = document.createElement("input");
          chk.type = "checkbox";
          chk.className = "orb-chk";
          chk.checked = this.activeOrbitals.has(item.key || item.file || item.label);
          chk.addEventListener("change", (e) => {
            const k = item.key || item.file || item.label;
            if (e.target.checked) this.activeOrbitals.add(k);
            else this.activeOrbitals.delete(k);
            this.onOrbitalToggle(k, e.target.checked, item);
          });
          const dot = document.createElement("span");
          dot.className = "orb-dot";
          dot.style.backgroundColor = item.color || "#38bdf8";
          const labelSpan = document.createElement("span");
          labelSpan.className = "orb-label-text";
          let txt = item.label || item.title || item.key;
          if (item.energy_ev !== void 0) {
            txt += ` (${item.energy_ev > 0 ? "+" : ""}${item.energy_ev.toFixed(2)} eV)`;
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
      const groupTitle = hasEsp ? "Surfaces & Charges" : "Atom Charges";
      const espGroup = document.createElement("div");
      espGroup.className = "orbital-group esp-group";
      const isEspCollapsed = this.collapsedGroups.has(groupTitle);
      const espHeader = document.createElement("div");
      espHeader.className = "orbital-group-header";
      espHeader.innerHTML = `
      <span class="group-title">${groupTitle}</span>
      <span class="group-chevron">${isEspCollapsed ? "\u25B8" : "\u25BE"}</span>
    `;
      espHeader.addEventListener("click", () => {
        if (this.collapsedGroups.has(groupTitle)) {
          this.collapsedGroups.delete(groupTitle);
        } else {
          this.collapsedGroups.add(groupTitle);
        }
        this.renderGroups(categories, hasEsp);
      });
      espGroup.appendChild(espHeader);
      const espBody = document.createElement("div");
      espBody.className = "orbital-group-body";
      if (isEspCollapsed) espBody.style.display = "none";
      if (hasEsp) {
        const espRow = document.createElement("label");
        espRow.className = "orbital-item-row";
        const espChk = document.createElement("input");
        espChk.type = "checkbox";
        espChk.className = "orb-chk";
        espChk.checked = this.isEspActive;
        espChk.addEventListener("change", (e) => {
          this.isEspActive = e.target.checked;
          this.onEspToggle(this.isEspActive);
        });
        const espDot = document.createElement("span");
        espDot.className = "orb-dot";
        espDot.style.backgroundColor = "#06b6d4";
        const espText = document.createElement("span");
        espText.className = "orb-label-text";
        espText.textContent = "\u26A1 ESP Surface";
        espRow.appendChild(espChk);
        espRow.appendChild(espDot);
        espRow.appendChild(espText);
        espBody.appendChild(espRow);
      }
      const chargesRow = document.createElement("label");
      chargesRow.className = "orbital-item-row";
      const chargesChk = document.createElement("input");
      chargesChk.type = "checkbox";
      chargesChk.className = "orb-chk";
      chargesChk.checked = this.isChargesActive;
      chargesChk.addEventListener("change", (e) => {
        this.isChargesActive = e.target.checked;
        this.onChargesToggle(this.isChargesActive);
      });
      const chargesDot = document.createElement("span");
      chargesDot.className = "orb-dot";
      chargesDot.style.backgroundColor = "#a855f7";
      const chargesText = document.createElement("span");
      chargesText.className = "orb-label-text";
      chargesText.textContent = "\u{1F3F7} Atom Charges";
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
        "Frontier": [],
        "Transforming Orbitals": [],
        "\u03C3 Bonds": [],
        "\u03C0 Bonds": [],
        "\u03C3* Anti-Bonds": [],
        "\u03C0* Anti-Bonds": [],
        "Lone Pairs": [],
        "Reaction Orbitals": []
      };
      tracks.forEach((trk) => {
        if (trk.role === "esp" || trk.orbital_key === "track_esp_charge") return;
        const key = trk.orbital_key;
        let title = trk.label || trk.title || key;
        title = title.replace(/^Non-Bonding\s+/i, "").replace(/\[Antibond\]/i, "").trim();
        const color = trk.color || "#38bdf8";
        const item = { key, label: title, color };
        if (title.includes("\u2192") || title.includes("->")) {
          categoriesMap["Transforming Orbitals"].push(item);
        } else if (title.includes("HOMO") || title.includes("LUMO")) {
          categoriesMap["Frontier"].push(item);
        } else if (title.includes("\u03C0*") || title.includes("pistar")) {
          categoriesMap["\u03C0* Anti-Bonds"].push(item);
        } else if (title.includes("\u03C3*") || title.includes("sigmastar")) {
          categoriesMap["\u03C3* Anti-Bonds"].push(item);
        } else if (title.includes("\u03C0") || title.includes("pi_")) {
          categoriesMap["\u03C0 Bonds"].push(item);
        } else if (title.includes("\u03C3") || title.includes("sigma_")) {
          categoriesMap["\u03C3 Bonds"].push(item);
        } else if (title.includes("LP") || title.includes("lp_")) {
          categoriesMap["Lone Pairs"].push(item);
        } else {
          categoriesMap["Reaction Orbitals"].push(item);
        }
      });
      return Object.entries(categoriesMap).filter(([_, items]) => items.length > 0).map(([name, items]) => ({ name, items }));
    }
    /**
     * Helper to format raw orbital keys (e.g. from bundle.orbitalMeshes) into Molecular Explorer categories
     */
    static parseRawOrbitalKeys(keys = []) {
      const uniqueBases = /* @__PURE__ */ new Set();
      keys.forEach((k) => {
        const parts = k.split("_");
        if (parts.length > 1 && !isNaN(parts[0])) {
          uniqueBases.add(parts.slice(1).join("_"));
        } else {
          uniqueBases.add(k);
        }
      });
      const categoriesMap = {
        "Frontier": [],
        "\u03C3 Bonds": [],
        "\u03C0 Bonds": [],
        "\u03C3* Anti-Bonds": [],
        "\u03C0* Anti-Bonds": [],
        "Lone Pairs": [],
        "Localized Orbitals": []
      };
      const sorted = Array.from(uniqueBases).sort();
      sorted.forEach((key) => {
        const upper = key.toUpperCase();
        let label = key.replace(/_/g, " ");
        let cat = "Localized Orbitals";
        let color = "#38bdf8";
        if (upper.includes("HOMO")) {
          cat = "Frontier";
          color = "#ff5c5c";
          label = key;
        } else if (upper.includes("LUMO")) {
          cat = "Frontier";
          color = "#55aaff";
          label = key;
        } else if (upper.includes("SIGMASTAR") || upper.includes("SIGMA*") || key.includes("\u03C3*")) {
          cat = "\u03C3* Anti-Bonds";
          color = "#d946ef";
          label = key.replace(/sigmastar[_\s]*/i, "\u03C3*(").replace(/$/, ")");
        } else if (upper.includes("PISTAR") || upper.includes("PI*") || key.includes("\u03C0*")) {
          cat = "\u03C0* Anti-Bonds";
          color = "#ec4899";
          label = key.replace(/pistar[_\s]*/i, "\u03C0*(").replace(/$/, ")");
        } else if (upper.includes("PI") || key.includes("\u03C0")) {
          cat = "\u03C0 Bonds";
          color = "#cc44bb";
          label = key.replace(/pi[_\s]*/i, "\u03C0(").replace(/$/, ")");
        } else if (upper.includes("SIGMA") || key.includes("\u03C3")) {
          cat = "\u03C3 Bonds";
          color = "#44cc77";
          label = key.replace(/sigma[_\s]*/i, "\u03C3(").replace(/$/, ")");
        } else if (upper.includes("LP") || upper.includes("LONE")) {
          cat = "Lone Pairs";
          color = "#eeee22";
          label = key.replace(/lp[_\s]*/i, "LP(").replace(/$/, ")");
        } else if (upper.startsWith("IBO")) {
          label = `IBO ${key.replace("IBO_", "")}`;
          color = "#44cc77";
        }
        categoriesMap[cat].push({ key, label, color });
      });
      return Object.entries(categoriesMap).filter(([_, items]) => items.length > 0).map(([name, items]) => ({ name, items }));
    }
  };

  // static_viewer/js/viewer_svg.js
  var ViewerSvgMechanism = class {
    constructor(containerElement, options = {}) {
      this.container = containerElement;
      this.wiring = options.wiring || {};
      this.onPlaySegment = options.onPlaySegment || (() => {
      });
      this.onJumpFrame = options.onJumpFrame || (() => {
      });
      this.onOpenMolecule = options.onOpenMolecule || (() => {
      });
      this.currentFrame = 0;
      this.playDirection = "forward";
    }
    setMechanism(svgText, wiring = {}) {
      this.wiring = wiring || {};
      if (!this.container || !svgText) return;
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
      const svg = this.container.querySelector("svg");
      if (!svg) return;
      for (const [key, cfg] of Object.entries(this.wiring)) {
        const el = this.findElement(svg, key);
        if (!el) continue;
        if (cfg.role === "arrow") {
          el.classList.add("svg-clickable-arrow");
          const dir = cfg.direction === "reverse" ? "\u27F5" : "\u27F6";
          el.setAttribute("title", `Reaction Arrow: Play frames ${cfg.startFrame || 0} ${dir} ${cfg.endFrame || 0}`);
        } else if (cfg.role === "state" || cfg.role === "state_molecule") {
          el.classList.add("svg-clickable-state");
          if (cfg.role === "state_molecule" || cfg.moleculeName) {
            el.classList.add("svg-clickable-molecule");
            el.setAttribute("title", `${cfg.moleculeName || "Intermediate"} (Frame ${cfg.targetFrame || 0}): Click to jump, Double-click for molecule`);
          } else {
            el.setAttribute("title", `State: Jump to frame ${cfg.targetFrame || 0}`);
          }
        } else if (cfg.role === "molecule") {
          el.classList.add("svg-clickable-molecule");
          const stateTarget = this.findStateTarget(el.parentElement, svg);
          const stateInfo = stateTarget && stateTarget.cfg && stateTarget.cfg.targetFrame !== void 0 ? ` (Frame ${stateTarget.cfg.targetFrame}): Click to jump, ` : ": ";
          el.setAttribute("title", `${cfg.moleculeName || "Molecule"}${stateInfo}Double-click to inspect`);
        }
      }
    }
    findStateTarget(el, rootSvg) {
      let cur = el;
      while (cur && cur !== rootSvg && cur.tagName && cur.tagName.toLowerCase() !== "svg") {
        const id = cur.getAttribute("id");
        const lbl = cur.getAttribute("inkscape:label") || cur.getAttribute("label");
        const cfg = id && this.wiring[id] || lbl && this.wiring[lbl];
        if (cfg && (cfg.role === "state" || cfg.role === "state_molecule" || cfg.targetFrame !== void 0)) {
          return { el: cur, id: id || lbl, cfg };
        }
        cur = cur.parentElement;
      }
      return null;
    }
    bindEvents() {
      const svg = this.container.querySelector("svg");
      if (!svg) return;
      svg.addEventListener("click", (e) => {
        const target = this.resolveTarget(e.target, svg);
        if (!target) return;
        const { cfg } = target;
        if (!cfg) return;
        if (cfg.role === "arrow") {
          e.stopPropagation();
          const sf = cfg.startFrame !== void 0 ? parseInt(cfg.startFrame, 10) : 0;
          const ef = cfg.endFrame !== void 0 ? parseInt(cfg.endFrame, 10) : 100;
          const dir = cfg.direction || "forward";
          this.onPlaySegment(sf, ef, dir);
        } else if (cfg.role === "state" || cfg.role === "state_molecule" || cfg.targetFrame !== void 0) {
          e.stopPropagation();
          const tf = cfg.targetFrame !== void 0 ? parseInt(cfg.targetFrame, 10) : 0;
          this.onJumpFrame(tf);
        } else {
          const stateTarget = this.findStateTarget(e.target, svg);
          if (stateTarget && stateTarget.cfg && stateTarget.cfg.targetFrame !== void 0) {
            e.stopPropagation();
            const tf = parseInt(stateTarget.cfg.targetFrame, 10) || 0;
            this.onJumpFrame(tf);
          }
        }
      });
      svg.addEventListener("dblclick", (e) => {
        const target = this.resolveTarget(e.target, svg);
        if (!target) return;
        const { id, cfg } = target;
        if (cfg && (cfg.role === "molecule" || cfg.role === "state_molecule" || cfg.moleculeName)) {
          e.stopPropagation();
          e.preventDefault();
          this.onOpenMolecule(id, cfg);
        }
      });
      svg.addEventListener("contextmenu", (e) => {
        const target = this.resolveTarget(e.target, svg);
        if (!target) return;
        const { id, cfg } = target;
        if (cfg && (cfg.role === "molecule" || cfg.role === "state_molecule" || cfg.moleculeName)) {
          e.preventDefault();
          e.stopPropagation();
          this.onOpenMolecule(id, cfg);
        }
      });
    }
    updatePushArrows(frameIdx, direction = "forward") {
      this.currentFrame = frameIdx;
      this.playDirection = direction || "forward";
      const svg = this.container ? this.container.querySelector("#active-mechanism-svg") || this.container.querySelector("svg") : null;
      if (!svg) return;
      Object.entries(this.wiring).forEach(([key, cfg]) => {
        if (cfg.role !== "push_arrow") return;
        const el = this.findElement(svg, key);
        if (!el) return;
        const sf = cfg.startFrame !== void 0 ? parseInt(cfg.startFrame, 10) : 0;
        const ef = cfg.endFrame !== void 0 ? parseInt(cfg.endFrame, 10) : 9999;
        const arrowDir = cfg.direction || "both";
        const dirMatches = arrowDir === "both" || arrowDir === this.playDirection;
        const minF = Math.min(sf, ef);
        const maxF = Math.max(sf, ef);
        const frameMatches = frameIdx >= minF && frameIdx <= maxF;
        if (dirMatches && frameMatches) {
          el.classList.add("svg-push-active");
          el.classList.remove("svg-push-inactive");
        } else {
          el.classList.add("svg-push-inactive");
          el.classList.remove("svg-push-active");
        }
      });
    }
    highlightCurrentState(frameIdx) {
      const svg = this.container.querySelector("svg");
      if (!svg) return;
      svg.querySelectorAll(".svg-active-element").forEach((el) => el.classList.remove("svg-active-element"));
      let closestState = null;
      let minDiff = 8;
      for (const [key, cfg] of Object.entries(this.wiring)) {
        if ((cfg.role === "state" || cfg.role === "state_molecule") && cfg.targetFrame !== void 0) {
          const diff = Math.abs(cfg.targetFrame - frameIdx);
          if (diff < minDiff) {
            minDiff = diff;
            closestState = key;
          }
        }
      }
      if (closestState) {
        const el = this.findElement(svg, closestState);
        if (el) el.classList.add("svg-active-element");
      }
    }
    findElement(svg, key) {
      let el = svg.getElementById(key);
      if (!el) {
        try {
          el = svg.querySelector(`[inkscape\\:label="${key}"], [label="${key}"], [data-id="${key}"]`);
        } catch (e) {
        }
      }
      if (!el) {
        for (const elem of svg.querySelectorAll("*")) {
          if (elem.getAttribute("inkscape:label") === key || elem.getAttribute("label") === key) {
            return elem;
          }
        }
      }
      return el;
    }
    resolveTarget(el, rootSvg) {
      let cur = el;
      while (cur && cur !== rootSvg && cur.tagName && cur.tagName.toLowerCase() !== "svg") {
        const id = cur.getAttribute("id");
        const lbl = cur.getAttribute("inkscape:label") || cur.getAttribute("label");
        const cfg = id && this.wiring[id] || lbl && this.wiring[lbl];
        if (cfg) {
          return { el: cur, id: id || lbl, cfg };
        }
        cur = cur.parentElement;
      }
      return null;
    }
  };

  // static_viewer/js/viewer_ui.js
  var ViewerUI = class {
    constructor(app) {
      this.app = app;
      this.elTitle = document.getElementById("reaction-title");
      this.elBarrier = document.getElementById("stat-barrier");
      this.elSolvent = document.getElementById("stat-solvent");
      this.elMechanism = document.getElementById("mechanism-strip");
      this.elSlider = document.getElementById("frame-slider");
      this.elFrameInfo = document.getElementById("frame-info");
      this.elBtnPlayFwd = document.getElementById("btn-play-fwd");
      this.elBtnPlayRev = document.getElementById("btn-play-rev");
      this.elSidePanel = document.getElementById("side-panel");
      this.elBtnToggleOrbitals = document.getElementById("btn-toggle-orbitals");
      this.elOrbitalsContainer = document.getElementById("orbitals-list");
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
      if (this.elBtnPlayFwd) {
        this.elBtnPlayFwd.addEventListener("click", () => this.app.togglePlayForward());
      }
      if (this.elBtnPlayRev) {
        this.elBtnPlayRev.addEventListener("click", () => this.app.togglePlayReverse());
      }
      document.getElementById("btn-first")?.addEventListener("click", () => this.app.goToFrame(0));
      document.getElementById("btn-last")?.addEventListener("click", () => this.app.goToFrame(this.app.totalFrames - 1));
      document.getElementById("btn-prev")?.addEventListener("click", () => this.app.stepFrame(-1));
      document.getElementById("btn-next")?.addEventListener("click", () => this.app.stepFrame(1));
      if (this.elSlider) {
        this.elSlider.addEventListener("input", (e) => {
          this.app.pause();
          const newF = parseInt(e.target.value, 10);
          if (newF > this.app.currentFrame) this.app.setDirection("forward");
          else if (newF < this.app.currentFrame) this.app.setDirection("reverse");
          this.app.goToFrame(newF);
        });
      }
      const speedSelect = document.getElementById("select-speed");
      if (speedSelect) {
        speedSelect.addEventListener("change", (e) => {
          this.app.setSpeed(parseFloat(e.target.value));
        });
      }
      if (this.elBtnToggleOrbitals) {
        this.elBtnToggleOrbitals.addEventListener("click", () => this.togglePanel());
      }
    }
    togglePanel() {
      this.isPanelCollapsed = !this.isPanelCollapsed;
      if (this.elSidePanel) {
        this.elSidePanel.classList.toggle("panel-hidden", this.isPanelCollapsed);
      }
      if (this.elBtnToggleOrbitals) {
        this.elBtnToggleOrbitals.innerHTML = `Orbitals ${this.isPanelCollapsed ? "\u25B8" : "\u25BE"}`;
      }
    }
    setBundle(bundle) {
      if (bundle.isSingleMolecule) {
        this.setMoleculeMode(bundle);
        return;
      }
      const manifest = bundle.manifest || {};
      const traj = bundle.trajectory || {};
      if (this.elTitle) {
        this.elTitle.textContent = manifest.title || traj.title || "Multistep Reaction Pathway";
      }
      if (this.elBarrier) {
        const b = manifest.overall_barrier_kcal !== void 0 ? manifest.overall_barrier_kcal : traj.overall_barrier_kcal;
        this.elBarrier.textContent = b !== void 0 ? `${b} kcal/mol` : "\u2014";
      }
      if (this.elSolvent) {
        this.elSolvent.textContent = manifest.solvent || traj.solvent || "Gas Phase";
      }
      const wiring = bundle.svgWiring || manifest.svg_wiring || traj.svg_wiring || {};
      if (bundle.mechanismSvg) {
        this.svgMechanism.setMechanism(bundle.mechanismSvg, wiring);
        if (this.elMechanism) this.elMechanism.style.display = "flex";
      } else {
        if (this.elMechanism) this.elMechanism.style.display = "none";
      }
      const stats = document.querySelector(".reaction-stats");
      if (stats) stats.style.display = "flex";
      const dock = document.getElementById("bottom-dock");
      if (dock) dock.style.display = "flex";
      const nFrames = (traj.frames || []).length;
      if (this.elSlider) {
        this.elSlider.min = 0;
        this.elSlider.max = Math.max(0, nFrames - 1);
        this.elSlider.value = 0;
      }
      this.populateOrbitals(bundle);
    }
    setMoleculeMode(molBundle) {
      const rawTitle = molBundle.manifest?.title || "Molecule View";
      const cleanTitle = rawTitle.replace(/_/g, " ");
      const title = cleanTitle.charAt(0).toUpperCase() + cleanTitle.slice(1);
      if (this.elTitle) this.elTitle.textContent = `\u269B\uFE0F ${title}`;
      const stats = document.querySelector(".reaction-stats");
      if (stats) stats.style.display = "none";
      if (this.elMechanism) this.elMechanism.style.display = "none";
      const pes = document.getElementById("pes-canvas-container");
      if (pes) pes.style.display = "none";
      const dock = document.getElementById("bottom-dock");
      if (dock) dock.style.display = "none";
      this.orbitalsPanel.clear();
      const hasEsp = !!(molBundle.hasEsp ?? (molBundle.espMeshes && Object.keys(molBundle.espMeshes).length > 0));
      this.orbitalsPanel.renderGroups(molBundle.categories, hasEsp);
      if (this.app.viewer3d) {
        setTimeout(() => this.app.viewer3d.resize(), 50);
      }
    }
    populateOrbitals(bundle) {
      const timeline = bundle.manifest && bundle.manifest.orbital_timeline || bundle.trajectory && bundle.trajectory.orbital_timeline;
      const tracks = timeline && timeline.tracks || [];
      let categories = [];
      if (tracks.length > 0) {
        categories = ViewerOrbitalsPanel.parsePathwayTracks(tracks);
      } else if (bundle.orbitalMeshes && Object.keys(bundle.orbitalMeshes).length > 0) {
        categories = ViewerOrbitalsPanel.parseRawOrbitalKeys(Object.keys(bundle.orbitalMeshes));
      }
      this.orbitalsPanel.clear();
      const hasEsp = !!(bundle.hasEsp || bundle.espMeshes && Object.keys(bundle.espMeshes).length > 0);
      this.orbitalsPanel.renderGroups(categories, hasEsp);
    }
    updateFrameDisplay(frameIdx, totalFrames, energy) {
      if (this.elSlider) {
        this.elSlider.value = frameIdx;
      }
      if (this.elFrameInfo) {
        const eStr = energy !== void 0 ? ` | ${energy >= 0 ? "+" : ""}${energy.toFixed(2)} kcal/mol` : "";
        this.elFrameInfo.innerHTML = `Frame <strong>${frameIdx + 1}</strong> / ${totalFrames}${eStr}`;
      }
      if (this.svgMechanism) {
        this.svgMechanism.updatePushArrows(frameIdx, this.app.playDirection || "forward");
        this.svgMechanism.highlightCurrentState(frameIdx);
      }
    }
    updatePlayState(isPlaying, direction = "forward") {
      if (this.elBtnPlayFwd) {
        const isFwd = isPlaying && direction === "forward";
        this.elBtnPlayFwd.textContent = isFwd ? "\u275A\u275A Pause" : "\u25B6 Play Fwd";
        this.elBtnPlayFwd.classList.toggle("active", isFwd);
      }
      if (this.elBtnPlayRev) {
        const isRev = isPlaying && direction === "reverse";
        this.elBtnPlayRev.textContent = isRev ? "\u275A\u275A Pause" : "\u25C0 Play Rev";
        this.elBtnPlayRev.classList.toggle("active", isRev);
      }
    }
  };

  // static_viewer/js/viewer_modal.js
  var ViewerMoleculeModal = class {
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
      const molBundle = buildMoleculeBundle(molId, parentBundle) || (cfg.moleculeName ? buildMoleculeBundle(cfg.moleculeName, parentBundle) : null);
      if (!molBundle) {
        console.warn(`Molecule '${molId}' could not be resolved in bundle.`);
        return;
      }
      const title = cfg.moleculeName || cfg.molName || molBundle.manifest.title || molId;
      const modal = document.createElement("div");
      modal.id = "molecule-modal-backdrop";
      modal.className = "custom-modal-backdrop";
      modal.innerHTML = `
      <div class="custom-modal-dialog" id="custom-modal-dialog">
        <div class="modal-header" id="modal-drag-header">
          <div class="modal-title-group">
            <h3>\u269B\uFE0F ${title}</h3>
            <span class="chip-blue">Molecule Inspection</span>
          </div>
          <div class="modal-actions">
            <button class="btn btn-sm" id="btn-modal-popout" title="Open in dedicated pop-up window">\u2197 Pop-up Window</button>
            <button class="btn-modal-close" id="btn-modal-close">&times;</button>
          </div>
        </div>
        <div class="modal-viewport-container">
          <div id="modal-gl-canvas" class="modal-gl-canvas"></div>
          <div class="viewport-overlay-controls">
            <button class="btn-panel-toggle" id="modal-btn-toggle-orbitals">Orbitals \u25BE</button>
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
      const dialog = modal.querySelector("#custom-modal-dialog");
      const header = modal.querySelector("#modal-drag-header");
      const resizeHandle = modal.querySelector("#modal-resize-handle");
      const initW = Math.min(920, Math.max(480, window.innerWidth * 0.75));
      const initH = Math.min(560, Math.max(360, window.innerHeight * 0.7));
      dialog.style.width = `${initW}px`;
      dialog.style.height = `${initH}px`;
      dialog.style.left = `${Math.max(16, Math.round((window.innerWidth - initW) / 2))}px`;
      dialog.style.top = `${Math.max(16, Math.round((window.innerHeight - initH) / 2))}px`;
      this.setupDraggable(dialog, header);
      this.setupResizable(dialog, resizeHandle);
      modal.querySelector("#btn-modal-close").addEventListener("click", () => this.close());
      modal.addEventListener("click", (e) => {
        if (e.target === modal) this.close();
      });
      modal.querySelector("#btn-modal-popout").addEventListener("click", () => {
        const bundleUrl = this.app.currentBundleUrl || (new URLSearchParams(window.location.search).get("bundle") || "");
        const targetMol = molBundle.manifest.molKey || molId;
        const popUrl = bundleUrl ? `viewer.html?bundle=${encodeURIComponent(bundleUrl)}&molecule=${encodeURIComponent(targetMol)}` : `viewer.html?molecule=${encodeURIComponent(targetMol)}`;
        window.open(popUrl, "_blank", "width=1100,height=750,menubar=no,toolbar=no");
      });
      const btnToggle = modal.querySelector("#modal-btn-toggle-orbitals");
      const sidePanel = modal.querySelector("#modal-side-panel");
      this.isPanelCollapsed = false;
      btnToggle.addEventListener("click", () => {
        this.isPanelCollapsed = !this.isPanelCollapsed;
        sidePanel.classList.toggle("panel-hidden", this.isPanelCollapsed);
        btnToggle.innerHTML = `Orbitals ${this.isPanelCollapsed ? "\u25B8" : "\u25BE"}`;
      });
      const canvasContainer = modal.querySelector("#modal-gl-canvas");
      this.viewer3d = new Viewer3D(canvasContainer);
      this.viewer3d.setBundle(molBundle);
      const orbContainer = modal.querySelector("#modal-orbitals-list");
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
        if (e.target.closest("button, input, a, .chip-blue")) return;
        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;
        const rect = dialog.getBoundingClientRect();
        origLeft = rect.left;
        origTop = rect.top;
        document.body.style.userSelect = "none";
        window.addEventListener("mousemove", onMouseMove);
        window.addEventListener("mouseup", onMouseUp);
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
          document.body.style.userSelect = "";
          window.removeEventListener("mousemove", onMouseMove);
          window.removeEventListener("mouseup", onMouseUp);
        }
      };
      header.addEventListener("mousedown", onMouseDown);
      this.cleanupFns.push(() => {
        header.removeEventListener("mousedown", onMouseDown);
        window.removeEventListener("mousemove", onMouseMove);
        window.removeEventListener("mouseup", onMouseUp);
      });
    }
    setupResizable(dialog, resizeHandle) {
      if (window.ResizeObserver) {
        this.resizeObserver = new ResizeObserver(() => {
          if (this.viewer3d && this.viewer3d.glViewer) {
            this.viewer3d.glViewer.resize();
            this.viewer3d.glViewer.render();
          }
        });
        this.resizeObserver.observe(dialog);
      }
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
        document.body.style.cursor = "se-resize";
        document.body.style.userSelect = "none";
        window.addEventListener("mousemove", onHandleMove);
        window.addEventListener("mouseup", onHandleUp);
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
          document.body.style.cursor = "";
          document.body.style.userSelect = "";
          window.removeEventListener("mousemove", onHandleMove);
          window.removeEventListener("mouseup", onHandleUp);
        }
      };
      resizeHandle.addEventListener("mousedown", onHandleDown);
      this.cleanupFns.push(() => {
        resizeHandle.removeEventListener("mousedown", onHandleDown);
        window.removeEventListener("mousemove", onHandleMove);
        window.removeEventListener("mouseup", onHandleUp);
      });
    }
    close() {
      this.cleanupFns.forEach((fn) => {
        try {
          fn();
        } catch (_) {
        }
      });
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
  };

  // static_viewer/js/viewer_download.js
  var ViewerDownloadModal = class {
    constructor(app) {
      this.app = app;
      this.elBtn = document.getElementById("btn-header-download");
      this.elModal = document.getElementById("download-modal");
      this.elTitle = document.getElementById("download-modal-title");
      this.elBody = document.getElementById("download-modal-body");
      this.elClose = document.getElementById("btn-close-download");
      this.elCancel = document.getElementById("btn-cancel-download");
      this.elConfirm = document.getElementById("btn-confirm-download");
      this.init();
    }
    init() {
      const params = new URLSearchParams(window.location.search);
      const hasDownloadKey = params.has("download");
      if (this.elBtn) {
        this.elBtn.style.display = hasDownloadKey ? "inline-flex" : "none";
        this.elBtn.addEventListener("click", () => this.show());
      }
      this.elClose?.addEventListener("click", () => this.close());
      this.elCancel?.addEventListener("click", () => this.close());
      this.elConfirm?.addEventListener("click", () => this.handleConfirm());
      this.elModal?.addEventListener("click", (e) => {
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
        this.elModal.style.display = "flex";
      }
    }
    close() {
      if (this.elModal) {
        this.elModal.style.display = "none";
      }
    }
    renderMoleculeDialog(bundle) {
      if (this.elTitle) {
        const name = bundle.manifest?.title || bundle.manifest?.molKey || "Molecule";
        this.elTitle.textContent = `\u{1F4E6} Download Custom Bundle: ${name}`;
      }
      if (!this.elBody) return;
      let html = `
      <div style="color:var(--text-secondary); margin-bottom: 6px;">
        Select the orbitals and surfaces to include in the exported archive:
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
      if (bundle.hasEsp) {
        html += `
        <div style="font-weight:700; color:#cbd5e1; margin-top:4px; margin-bottom:2px; padding-left:4px; font-size:11px; border-bottom:1px solid rgba(255,255,255,0.1);">Surfaces & Electrostatics</div>
        <label class="download-check-item">
          <input type="checkbox" class="dl-orb-chk" value="esp" checked>
          <span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#06b6d4;"></span>
          <span style="font-weight:600;">\u26A1 Electrostatic Potential (ESP) Surface</span>
        </label>
      `;
      }
      const categories = bundle.categories || [];
      categories.forEach((cat) => {
        html += `<div style="font-weight:700; color:#cbd5e1; margin-top:8px; margin-bottom:2px; padding-left:4px; font-size:11px; border-bottom:1px solid rgba(255,255,255,0.1);">${cat.name}</div>`;
        (cat.items || []).forEach((item) => {
          const k = item.key || item.file || item.label;
          const color = item.color || "#38bdf8";
          const label = item.label || item.key;
          const ev = item.energy_ev !== void 0 ? ` (${item.energy_ev.toFixed(2)} eV)` : "";
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
      const selectAll = this.elBody.querySelector("#dl-select-all");
      const chks = this.elBody.querySelectorAll(".dl-orb-chk");
      const countEl = this.elBody.querySelector("#dl-selected-count");
      const updateCount = () => {
        const total = chks.length;
        const checked = Array.from(chks).filter((c) => c.checked).length;
        if (countEl) countEl.textContent = `${checked} / ${total} selected`;
        if (selectAll) selectAll.checked = checked === total;
      };
      selectAll?.addEventListener("change", (e) => {
        chks.forEach((c) => c.checked = e.target.checked);
        updateCount();
      });
      chks.forEach((c) => c.addEventListener("change", updateCount));
      updateCount();
    }
    renderReactionDialog(bundle) {
      if (this.elTitle) {
        const title = bundle.manifest?.title || "Reaction Pathway";
        this.elTitle.textContent = `\u{1F4E6} Download Bundle: ${title}`;
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
        alert("Raw archive data unavailable for repacking.");
        return;
      }
      if (this.elConfirm) {
        this.elConfirm.disabled = true;
        this.elConfirm.textContent = "\u23F3 Repacking...";
      }
      try {
        if (bundle.isSingleMolecule) {
          await this.repackMolecule(bundle);
        } else {
          await this.repackReaction(bundle);
        }
        this.close();
      } catch (err) {
        console.error("Repacking failed:", err);
        alert(`Error repacking bundle: ${err.message}`);
      } finally {
        if (this.elConfirm) {
          this.elConfirm.disabled = false;
          this.elConfirm.textContent = "\u{1F4E6} Download";
        }
      }
    }
    async repackMolecule(bundle) {
      const rawZip = bundle.rawZip;
      const newZip = new window.JSZip();
      const chks = this.elBody.querySelectorAll(".dl-orb-chk:checked");
      const keptKeys = new Set(Array.from(chks).map((c) => c.value));
      const keepEsp = keptKeys.has("esp");
      const molKey = bundle.manifest?.molKey || bundle.molKey || "molecule";
      for (const [fname, fileObj] of Object.entries(rawZip.files)) {
        if (fileObj.dir) continue;
        const isManifest = fname === "manifest.json" || fname.endsWith("/manifest.json");
        const isKet = fname === "layout.ket" || fname.endsWith(".ket");
        const isMolMeta = fname.endsWith(`${molKey}.json`);
        const isXyz = fname.endsWith(".xyz");
        if (isManifest || isKet || isMolMeta || isXyz) {
          if (isManifest || isMolMeta) {
            try {
              const text = await fileObj.async("string");
              const obj = JSON.parse(text);
              const filterCats = (cats) => {
                if (!Array.isArray(cats)) return cats;
                return cats.map((cat) => ({
                  ...cat,
                  items: (cat.items || []).filter((it) => {
                    const k = it.file ? it.file.replace(/^molecules\//, "").replace(/\.json$/, "") : it.key || it.label;
                    return keptKeys.has(k) || keptKeys.has(it.key) || keptKeys.has(it.label);
                  })
                })).filter((cat) => cat.items.length > 0);
              };
              if (obj.orbitals) obj.orbitals = filterCats(obj.orbitals);
              if (obj.molecules && Array.isArray(obj.molecules)) {
                obj.molecules.forEach((m) => {
                  if (m.orbitals) m.orbitals = filterCats(m.orbitals);
                  if (!keepEsp) delete m.esp_surface;
                });
              }
              if (!keepEsp) delete obj.esp_surface;
              newZip.file(fname, JSON.stringify(obj, null, 2));
              continue;
            } catch (_) {
            }
          }
          const data = await fileObj.async("uint8array");
          newZip.file(fname, data);
          continue;
        }
        if (fname.includes("_esp.json")) {
          if (keepEsp) {
            const data = await fileObj.async("uint8array");
            newZip.file(fname, data);
          }
          continue;
        }
        if (fname.endsWith("_pos.json") || fname.endsWith("_neg.json")) {
          const baseKey = fname.replace(/^molecules\//, "").replace(/^.*?molecules\//, "").replace(/_(pos|neg)\.json$/, "");
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
            const data = await fileObj.async("uint8array");
            newZip.file(fname, data);
          }
        }
      }
      const blob = await newZip.generateAsync({ type: "blob", compression: "DEFLATE" });
      this.triggerBrowserDownload(blob, `${molKey}_custom.zip`);
      return newZip;
    }
    async repackReaction(bundle) {
      const rawZip = bundle.rawZip;
      const newZip = new window.JSZip();
      const scopeRadio = this.elBody.querySelector('input[name="dl_rxn_scope"]:checked');
      const scope = scopeRadio ? scopeRadio.value : "reaction_only";
      for (const [fname, fileObj] of Object.entries(rawZip.files)) {
        if (fileObj.dir) continue;
        if (scope === "reaction_only" && fname.startsWith("molecules/")) {
          continue;
        }
        const data = await fileObj.async("uint8array");
        newZip.file(fname, data);
      }
      const title = (bundle.manifest?.title || "reaction_pathway").replace(/[\s\-_]+/g, "_");
      const suffix = scope === "reaction_only" ? "_reaction_only.zip" : "_full.zip";
      const blob = await newZip.generateAsync({ type: "blob", compression: "DEFLATE" });
      this.triggerBrowserDownload(blob, `${title}${suffix}`);
      return newZip;
    }
    triggerBrowserDownload(blob, filename) {
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }, 1e3);
    }
  };

  // static_viewer/js/viewer_app.js
  var ReactionViewerApp = class {
    constructor() {
      this.bundle = null;
      this.currentBundleUrl = "";
      this.bundleList = [];
      this.currentFrame = 0;
      this.totalFrames = 0;
      this.isPlaying = false;
      this.playDirection = "forward";
      this.playSpeed = 1;
      this.playTimer = null;
      this.baseFps = 15;
      this.overlay = document.getElementById("overlay");
      this.overlayStatus = document.getElementById("overlay-status");
      this.progressBarContainer = document.getElementById("progress-bar-container");
      this.progressBar = document.getElementById("progress-bar");
      this.fileInput = document.getElementById("file-input");
      this.bundleSelector = document.getElementById("sel-bundle-switcher");
      this.viewer3d = new Viewer3D(document.getElementById("gl-canvas-container"));
      this.viewerPes = new ViewerPES(document.getElementById("pes-canvas"));
      this.ui = new ViewerUI(this);
      this.moleculeModal = new ViewerMoleculeModal(this);
      this.downloadModal = new ViewerDownloadModal(this);
      this.bindDropEvents();
      this.bindKeyboard();
      this.bindBundleSelector();
    }
    init() {
      const params = new URLSearchParams(window.location.search);
      const bundleParams = [];
      params.getAll("bundle").forEach((b) => {
        const tr = b.trim();
        if (tr && !bundleParams.includes(tr)) bundleParams.push(tr);
      });
      if (params.has("bundles")) {
        params.getAll("bundles").forEach((bs) => {
          bs.split(/[;|]/).forEach((item) => {
            const tr = item.trim();
            if (tr && !bundleParams.includes(tr)) bundleParams.push(tr);
          });
        });
      }
      this.bundleList = bundleParams;
      this.populateBundleSelector(this.bundleList);
      const initialFrame = parseInt(params.get("frame") || "0", 10);
      const autoplay = params.get("autoplay") === "1" || params.get("autoplay") === "true";
      const targetMolecule = params.get("molecule");
      if (this.bundleList.length > 0) {
        this.loadUrl(this.bundleList[0], initialFrame, autoplay, targetMolecule);
      } else {
        this.showDropzone();
      }
    }
    bindBundleSelector() {
      if (!this.bundleSelector) return;
      this.bundleSelector.addEventListener("change", (e) => {
        const val = e.target.value;
        if (val && val !== this.currentBundleUrl) {
          this.loadUrl(val, 0, false);
        }
      });
    }
    populateBundleSelector(bundleUrls) {
      if (!this.bundleSelector) return;
      if (!bundleUrls || bundleUrls.length <= 1) {
        this.bundleSelector.style.display = "none";
        return;
      }
      this.bundleSelector.style.display = "inline-block";
      this.bundleSelector.innerHTML = "";
      bundleUrls.forEach((url) => {
        const opt = document.createElement("option");
        opt.value = url;
        const rawName = url.split("/").pop().replace(/\.zip$/i, "");
        let fname = rawName;
        try {
          fname = decodeURIComponent(rawName);
        } catch (_) {
        }
        fname = fname.replace(/_/g, " ");
        opt.textContent = `\u{1F4E6} ${fname}`;
        if (url === this.currentBundleUrl) opt.selected = true;
        this.bundleSelector.appendChild(opt);
      });
    }
    showDropzone() {
      if (this.overlay) {
        this.overlay.style.display = "flex";
        if (this.progressBarContainer) this.progressBarContainer.style.display = "none";
        if (this.overlayStatus) this.overlayStatus.textContent = "Drag and drop a .rxnbundle.zip here, or click to choose a file";
      }
    }
    showProgress(pct, statusText) {
      if (this.overlay) {
        this.overlay.style.display = "flex";
        if (this.progressBarContainer) {
          this.progressBarContainer.style.display = "block";
          if (this.progressBar) this.progressBar.style.width = `${pct}%`;
        }
        if (this.overlayStatus) this.overlayStatus.textContent = statusText;
      }
    }
    hideOverlay() {
      if (this.overlay) {
        this.overlay.style.display = "none";
      }
    }
    async loadUrl(url, initialFrame = 0, autoplay = false, targetMolecule = null) {
      try {
        this.currentBundleUrl = url;
        this.showProgress(10, "Fetching bundle...");
        const bundle = await loadReactionBundle(url, (pct, msg) => this.showProgress(pct, msg));
        this.onBundleLoaded(bundle, initialFrame, autoplay, targetMolecule);
      } catch (err) {
        console.error(err);
        this.showDropzone();
        if (this.overlayStatus) {
          const dlBtn = url.startsWith("http://") || url.startsWith("https://") ? `<br><a href="${url}" target="_blank" rel="noopener" class="btn btn-primary" style="display:inline-block;margin:12px 0 6px 0;text-decoration:none;">\u2B07 Download Bundle Archive</a><br><span style="font-size:12px;color:var(--text-secondary);">After downloading, drop it here</span>` : "<br><br>Drag & drop the .zip file manually:";
          this.overlayStatus.innerHTML = `<span style="color:var(--danger-color);font-weight:600;">Error: ${err.message}</span>${dlBtn}`;
        }
      }
    }
    async loadFile(file) {
      try {
        this.showProgress(10, "Reading file...");
        const bundle = await loadReactionBundle(file, (pct, msg) => this.showProgress(pct, msg));
        this.currentBundleUrl = file.name;
        this.onBundleLoaded(bundle, 0, false, null);
      } catch (err) {
        console.error(err);
        this.showDropzone();
        if (this.overlayStatus) {
          this.overlayStatus.innerHTML = `<span style="color:var(--danger-color);font-weight:600;">Error: ${err.message}</span><br><br>Please select a valid bundle .zip`;
        }
      }
    }
    onBundleLoaded(bundle, initialFrame = 0, autoplay = false, targetMolecule = null) {
      if (targetMolecule) {
        const molBundle = buildMoleculeBundle(targetMolecule, bundle);
        if (molBundle) {
          this.loadMoleculeAsMain(molBundle);
          return;
        }
      }
      this.bundle = bundle;
      const frames = bundle.trajectory && bundle.trajectory.frames || [];
      this.totalFrames = frames.length;
      this.viewer3d.setBundle(bundle);
      this.viewerPes.setTrajectory(bundle.trajectory);
      this.ui.setBundle(bundle);
      this.populateBundleSelector(this.bundleList);
      this.goToFrame(initialFrame >= 0 && initialFrame < this.totalFrames ? initialFrame : 0);
      this.hideOverlay();
      if (autoplay) {
        this.playForward();
      }
    }
    loadMoleculeAsMain(molBundle) {
      this.bundle = molBundle;
      this.totalFrames = 1;
      this.currentFrame = 0;
      this.viewer3d.setBundle(molBundle);
      this.ui.setMoleculeMode(molBundle);
      this.hideOverlay();
    }
    openMoleculeModal(molId, cfg) {
      this.moleculeModal.show(molId, cfg);
    }
    async fetchOrbitalMesh(frameIdx, orbKey) {
      if (!this.bundle || !this.bundle.trajectory || !this.bundle.orbitalMeshes) return;
      const cacheKey = `${frameIdx}_${orbKey}`;
      if (this.bundle.orbitalMeshes[cacheKey]) return;
      try {
        const res = await fetch("/api/step/step_5_bundle_viewer/custom", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            action_type: "get_orbital_mesh",
            frame_index: frameIdx,
            orbital: orbKey,
            isovalue: 0.035
          })
        });
        if (res.ok) {
          const d = await res.json();
          if (d && d.success) {
            if (d.pos || d.neg) {
              this.bundle.orbitalMeshes[cacheKey] = { pos: d.pos, neg: d.neg };
            } else if (d.cube) {
              this.bundle.cubeCache = this.bundle.cubeCache || {};
              this.bundle.cubeCache[cacheKey] = d.cube;
            }
            if (this.currentFrame === frameIdx) {
              this.viewer3d.renderFrame(this.currentFrame, false);
            }
          }
        }
      } catch (_) {
      }
    }
    goToFrame(frameIdx) {
      if (!this.bundle || this.totalFrames === 0) return;
      this.currentFrame = Math.max(0, Math.min(frameIdx, this.totalFrames - 1));
      this.viewer3d.renderFrame(this.currentFrame, false);
      this.viewerPes.setCurrentFrame(this.currentFrame);
      if (this.viewer3d && this.viewer3d.activeOrbitals.size > 0 && this.bundle.orbitalMeshes) {
        for (const orbKey of this.viewer3d.activeOrbitals) {
          if (!this.viewer3d.findOrbitalMesh(this.currentFrame, orbKey)) {
            this.fetchOrbitalMesh(this.currentFrame, orbKey);
          }
        }
      }
      const frame = this.bundle.trajectory.frames[this.currentFrame] || {};
      const e = frame.rel_energy_kcal !== void 0 ? frame.rel_energy_kcal : frame.energy_kcal || 0;
      this.ui.updateFrameDisplay(this.currentFrame, this.totalFrames, e);
    }
    setDirection(dir) {
      this.playDirection = dir;
    }
    stepFrame(delta) {
      this.pause();
      this.setDirection(delta >= 0 ? "forward" : "reverse");
      const next = Math.max(0, Math.min(this.totalFrames - 1, this.currentFrame + delta));
      this.goToFrame(next);
    }
    togglePlay() {
      this.togglePlayForward();
    }
    togglePlayForward() {
      this.isPlaying ? this.pause() : this.playForward();
    }
    togglePlayReverse() {
      this.isPlaying ? this.pause() : this.playReverse();
    }
    playForward() {
      this.pause();
      this.setDirection("forward");
      if (this.totalFrames <= 1) return;
      if (this.currentFrame >= this.totalFrames - 1) this.goToFrame(0);
      this._startLoop(1, () => this.currentFrame >= this.totalFrames - 1);
    }
    playReverse() {
      this.pause();
      this.setDirection("reverse");
      if (this.totalFrames <= 1) return;
      if (this.currentFrame <= 0) this.goToFrame(this.totalFrames - 1);
      this._startLoop(-1, () => this.currentFrame <= 0);
    }
    playSegment(startFrame, endFrame, direction = "forward") {
      this.pause();
      if (this.totalFrames <= 1) return;
      this.setDirection(direction);
      const f1 = Math.max(0, Math.min(this.totalFrames - 1, parseInt(startFrame, 10) || 0));
      const f2 = Math.max(0, Math.min(this.totalFrames - 1, parseInt(endFrame, 10) || 0));
      const fromF = direction === "reverse" ? Math.max(f1, f2) : Math.min(f1, f2);
      const toF = direction === "reverse" ? Math.min(f1, f2) : Math.max(f1, f2);
      this.goToFrame(fromF);
      const delta = direction === "reverse" ? -1 : 1;
      this._startLoop(delta, () => direction === "reverse" ? this.currentFrame <= toF : this.currentFrame >= toF);
    }
    _startLoop(delta, shouldStop) {
      this.isPlaying = true;
      this.ui.updatePlayState(true, this.playDirection);
      const intervalMs = Math.max(25, Math.floor(1e3 / (this.baseFps * this.playSpeed)));
      this.playTimer = setInterval(() => {
        if (shouldStop()) {
          this.pause();
        } else {
          this.goToFrame(this.currentFrame + delta);
        }
      }, intervalMs);
    }
    pause() {
      this.isPlaying = false;
      if (this.playTimer) {
        clearInterval(this.playTimer);
        this.playTimer = null;
      }
      this.ui.updatePlayState(false, this.playDirection);
    }
    setSpeed(speed) {
      this.playSpeed = speed;
      if (this.isPlaying) {
        const dir = this.playDirection;
        this.pause();
        if (dir === "reverse") this.playReverse();
        else this.playForward();
      }
    }
    bindDropEvents() {
      const box = document.getElementById("drop-box");
      const container = document.getElementById("app-container");
      ["dragenter", "dragover", "dragleave", "drop"].forEach((name) => {
        container.addEventListener(name, (e) => e.preventDefault());
      });
      ["dragenter", "dragover"].forEach((name) => {
        container.addEventListener(name, () => box?.classList.add("dragover"));
      });
      ["dragleave", "drop"].forEach((name) => {
        container.addEventListener(name, () => box?.classList.remove("dragover"));
      });
      container.addEventListener("drop", (e) => {
        const files = e.dataTransfer?.files;
        if (files && files.length > 0) {
          this.loadFile(files[0]);
        }
      });
      if (this.fileInput) {
        this.fileInput.addEventListener("change", (e) => {
          if (e.target.files && e.target.files.length > 0) {
            this.loadFile(e.target.files[0]);
          }
        });
      }
      document.getElementById("btn-browse")?.addEventListener("click", () => {
        this.fileInput?.click();
      });
    }
    bindKeyboard() {
      window.addEventListener("keydown", (e) => {
        if (["input", "textarea", "select"].includes(document.activeElement?.tagName?.toLowerCase())) return;
        const keyMap = {
          "Space": () => this.togglePlay(),
          "ArrowRight": () => this.stepFrame(1),
          "ArrowLeft": () => this.stepFrame(-1),
          "Home": () => this.goToFrame(0),
          "End": () => this.goToFrame(this.totalFrames - 1)
        };
        if (keyMap[e.code]) {
          e.preventDefault();
          keyMap[e.code]();
        }
      });
    }
  };
  window.addEventListener("DOMContentLoaded", () => {
    const app = new ReactionViewerApp();
    window.app = app;
    app.init();
  });
})();
