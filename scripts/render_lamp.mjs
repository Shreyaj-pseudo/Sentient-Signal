// Render the real SVG rig at deterministic animation times without a browser.
// This checks geometry and poses only; browser layout and live motion need separate QA.
import fs from "node:fs";
import vm from "node:vm";
import { Resvg } from "@resvg/resvg-js";

const html = fs.readFileSync("app/static/index.html", "utf8");
const original = html.match(/<svg id="lamp"[\s\S]*?<\/svg>/)[0];
const parts = new Map();
let now = 0, next = 0;
const callbacks = new Map();
const svg = {
  dataset: {}, setAttribute() {},
  querySelector(id) {
    if (!parts.has(id)) parts.set(id, {
      attributes: {}, setAttribute(key, value) { this.attributes[key] = value; }
    });
    return parts.get(id);
  }
};
const context = {
  window: {}, document: {querySelector: () => svg, querySelectorAll: () => []},
  performance: {now: () => now}, matchMedia: () => ({matches: false, addEventListener() {}}),
  requestAnimationFrame(fn) { const id = ++next; callbacks.set(id, fn); return id; },
  cancelAnimationFrame(id) { callbacks.delete(id); }
};
vm.runInNewContext(fs.readFileSync("app/static/lamp.js", "utf8"), context);
const lamp = context.window.lampController;
function advance(ms) {
  for (let i = 0; i < ms; i += 20) {
    now += 20;
    const batch = [...callbacks.values()]; callbacks.clear(); batch.forEach(fn => fn(now));
  }
}
fs.mkdirSync("reports/lamp", {recursive: true});
const images = [];
for (const [gesture, elapsed] of [
  ["idle", 700], ["listening", 650], ["processing", 1500], ["acknowledgment", 330],
  ["celebration", 650], ["reassurance", 900], ["clarification", 600], ["error", 450]
]) {
  lamp.play(gesture); advance(elapsed);
  let source = original.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ');
  for (const [selector, part] of parts) {
    const id = selector.slice(1);
    source = source.replace(new RegExp(`<[^>]+id="${id}"[^>]*>`), tag => {
      for (const [key, value] of Object.entries(part.attributes)) {
        tag = tag.replace(new RegExp(` ${key}="[^"]*"`), "");
        tag = tag.replace(/(\/?\s*)>$/, ` ${key}="${value}"$1>`);
      }
      return tag;
    });
  }
  const png = new Resvg(source, {background: "#121310"}).render().asPng();
  fs.writeFileSync(`reports/lamp/${gesture}.png`, png);
  images.push(png.toString("base64"));
}
const labels = ["Rest", "Listening", "Processing", "Nod", "Celebrate", "Reassure", "Clarify", "Error"];
const tiles = images.map((base64, index) => {
  const x = (index % 4) * 540, y = Math.floor(index / 4) * 480;
  return `<image x="${x}" y="${y}" width="540" height="440" href="data:image/png;base64,${base64}"/>
    <text x="${x + 40}" y="${y + 450}" fill="#f3efe6" font-family="Arial" font-size="20">${labels[index]}</text>`;
});
const grid = `<svg xmlns="http://www.w3.org/2000/svg" width="2160" height="960">${tiles.join("")}</svg>`;
fs.writeFileSync("reports/lamp/pose-sheet.png", new Resvg(grid, {background: "#121310"}).render().asPng());
console.log("Rendered eight actual SVG poses to reports/lamp/pose-sheet.png");
