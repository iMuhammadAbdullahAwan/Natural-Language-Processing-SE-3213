// Simple icon creator using HTML5 Canvas
// Open this file in a web browser to generate icons

function createIcon(size) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");

  // Create gradient background
  const gradient = ctx.createLinearGradient(0, 0, size, size);
  gradient.addColorStop(0, "#667eea");
  gradient.addColorStop(1, "#764ba2");

  // Fill background with rounded corners
  ctx.fillStyle = gradient;
  const radius = size / 8;
  ctx.beginPath();
  ctx.roundRect(0, 0, size, size, radius);
  ctx.fill();

  // Add white circle for robot head
  ctx.fillStyle = "white";
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 3, 0, 2 * Math.PI);
  ctx.fill();

  // Add eyes
  ctx.fillStyle = "#667eea";
  ctx.beginPath();
  ctx.arc(size / 2 - size / 8, size / 2 - size / 12, size / 20, 0, 2 * Math.PI);
  ctx.fill();

  ctx.beginPath();
  ctx.arc(size / 2 + size / 8, size / 2 - size / 12, size / 20, 0, 2 * Math.PI);
  ctx.fill();

  // Add mouth
  ctx.strokeStyle = "#667eea";
  ctx.lineWidth = size / 32;
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 12, 0, Math.PI);
  ctx.stroke();

  return canvas.toDataURL("image/png");
}

// Create all required icons
const sizes = [16, 32, 48, 128];
const icons = {};

sizes.forEach((size) => {
  icons[size] = createIcon(size);
});

console.log("Icons created:", icons);
