class DigitCanvas {
  constructor(canvasId, onDrawCallback) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.onDrawCallback = onDrawCallback;
    this.isDrawing = false;
    this.lastX = 0;
    this.lastY = 0;
    this.strokeWidth = 14;

    this.initCanvas();
    this.bindEvents();
  }

  initCanvas() {
    this.ctx.fillStyle = '#000000';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.strokeStyle = '#ffffff';
    this.ctx.lineWidth = this.strokeWidth;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
  }

  bindEvents() {
    const getPos = (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;
      return {
        x: (clientX - rect.left) * (this.canvas.width / rect.width),
        y: (clientY - rect.top) * (this.canvas.height / rect.height)
      };
    };

    const startDraw = (e) => {
      e.preventDefault();
      this.isDrawing = true;
      const pos = getPos(e);
      this.lastX = pos.x;
      this.lastY = pos.y;
      this.draw(pos.x, pos.y);
    };

    const moveDraw = (e) => {
      if (!this.isDrawing) return;
      e.preventDefault();
      const pos = getPos(e);
      this.draw(pos.x, pos.y);
    };

    const stopDraw = (e) => {
      if (!this.isDrawing) return;
      this.isDrawing = false;
      if (this.onDrawCallback) {
        this.onDrawCallback(this.toDataURL());
      }
    };

    this.canvas.addEventListener('mousedown', startDraw);
    this.canvas.addEventListener('mousemove', moveDraw);
    window.addEventListener('mouseup', stopDraw);

    this.canvas.addEventListener('touchstart', startDraw, { passive: false });
    this.canvas.addEventListener('touchmove', moveDraw, { passive: false });
    window.addEventListener('touchend', stopDraw);
  }

  draw(x, y) {
    this.ctx.beginPath();
    this.ctx.moveTo(this.lastX, this.lastY);
    this.ctx.lineTo(x, y);
    this.ctx.stroke();
    this.lastX = x;
    this.lastY = y;
  }

  clear() {
    this.ctx.fillStyle = '#000000';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    if (this.onDrawCallback) {
      this.onDrawCallback(this.toDataURL());
    }
  }

  toDataURL() {
    // Generate a 28x28 grayscale image
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(this.canvas, 0, 0, 28, 28);
    return tempCanvas.toDataURL('image/png');
  }

  loadDataURL(dataUrl) {
    const img = new Image();
    img.onload = () => {
      this.ctx.fillStyle = '#000000';
      this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
      this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
      if (this.onDrawCallback) {
        this.onDrawCallback(this.toDataURL());
      }
    };
    img.src = dataUrl;
  }
}
