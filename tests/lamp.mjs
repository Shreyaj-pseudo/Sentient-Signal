// Deterministic animation-controller tests; these do not replace browser visual QA.
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

let now = 0, nextId = 0;
const callbacks = new Map();
const parts = new Map();
const motion = {matches: false, addEventListener(_, callback) { this.changed = callback; }};
const svg = {
  dataset: {}, attributes: {},
  setAttribute(key, value) { this.attributes[key] = value; },
  querySelector(id) {
    if (!parts.has(id)) parts.set(id, {
      attributes: {}, setAttribute(key, value) { this.attributes[key] = value; }
    });
    return parts.get(id);
  }
};
const context = {
  window: {}, performance: {now: () => now}, matchMedia: () => motion,
  document: {querySelector: () => svg, querySelectorAll: () => []},
  requestAnimationFrame(fn) { const id = ++nextId; callbacks.set(id, fn); return id; },
  cancelAnimationFrame(id) { callbacks.delete(id); }
};
vm.runInNewContext(fs.readFileSync("app/static/lamp.js", "utf8"), context);
const lamp = context.window.lampController;
function step(ms) {
  now += ms;
  const batch = [...callbacks.values()]; callbacks.clear();
  batch.forEach(fn => fn(now));
  assert.ok(callbacks.size <= 1, "Interrupting gestures must not accumulate animation loops");
}
function advance(ms) { for (let i = 0; i < ms; i += 20) step(20); }

advance(700);
const rest = [...lamp.pose];
lamp.play("listening"); advance(300);
assert.notEqual(lamp.pose[0], rest[0], "Lower joint must articulate");
assert.notEqual(lamp.pose[1], rest[1], "Upper joint must articulate");
const interrupted = [...lamp.pose];
lamp.play("processing"); step(0);
assert.deepEqual([...lamp.pose], interrupted, "Interruption must preserve the current pose");
advance(900);
const headAngle = Number(parts.get("#lamp-head").attributes.transform.match(/rotate\((.+)\)/)[1]);
assert.ok(Math.abs(headAngle + lamp.pose[0] + lamp.pose[1] - lamp.pose[2]) < 1e-8);
for (const gesture of ["celebration", "acknowledgment", "reassurance", "clarification", "error"]) {
  lamp.play(gesture); advance(5000);
  assert.equal(lamp.state, "idle", `${gesture} must return to rest`);
}
lamp.play("listening"); advance(5000);
assert.equal(lamp.state, "listening", "Listening must remain active until a new event");
motion.matches = true; motion.changed();
assert.equal(callbacks.size, 0, "Reduced motion must stop animation entirely");
lamp.play("celebration");
assert.equal(callbacks.size, 0, "Reduced motion gestures must use a static pose");
console.log("Lamp controller: articulation, interruption, return-to-idle, and reduced motion passed");
