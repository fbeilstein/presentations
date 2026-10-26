/**
 * viewer_pes.js
 * Clean background PES (Potential Energy Surface) static energy profile plotter.
 * Renders the relative energy curve and a live indicator for the active frame.
 */

export class ViewerPES {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');
    this.energies = [];
    this.currentFrame = 0;
    this.totalFrames = 0;

    window.addEventListener('resize', () => this.resizeAndDraw());
  }

  setTrajectory(trajectory) {
    if (!trajectory) return;
    const frames = trajectory.frames || [];
    this.totalFrames = frames.length;

    if (trajectory.energy_profile && Array.isArray(trajectory.energy_profile.frame_energies_kcal)) {
      this.energies = trajectory.energy_profile.frame_energies_kcal;
    } else {
      this.energies = frames.map(f => (f.rel_energy_kcal !== undefined ? f.rel_energy_kcal : (f.energy_kcal || 0.0)));
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

    const getX = (idx) => padSide + (idx / Math.max(1, this.energies.length - 1)) * (w - 2 * padSide);
    const getY = (val) => h - padBottom - ((val - minE) / rangeE) * (h - padTop - padBottom);

    // Draw baseline (0 kcal/mol)
    const zeroY = getY(0);
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padSide, zeroY);
    ctx.lineTo(w - padSide, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw filled gradient area under curve
    const grad = ctx.createLinearGradient(0, padTop, 0, h - padBottom);
    grad.addColorStop(0, 'rgba(56, 189, 248, 0.25)');
    grad.addColorStop(1, 'rgba(56, 189, 248, 0.0)');

    ctx.beginPath();
    ctx.moveTo(getX(0), h - padBottom);
    for (let i = 0; i < this.energies.length; i++) {
      ctx.lineTo(getX(i), getY(this.energies[i]));
    }
    ctx.lineTo(getX(this.energies.length - 1), h - padBottom);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Draw energy line curve
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i < this.energies.length; i++) {
      const x = getX(i);
      const y = getY(this.energies[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw current frame cursor indicator
    if (this.currentFrame >= 0 && this.currentFrame < this.energies.length) {
      const curX = getX(this.currentFrame);
      const curY = getY(this.energies[this.currentFrame]);

      // Vertical line to baseline
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(curX, padTop);
      ctx.lineTo(curX, h - padBottom);
      ctx.stroke();

      // Glowing dot at active point
      ctx.fillStyle = '#38bdf8';
      ctx.beginPath();
      ctx.arc(curX, curY, 5, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
}
