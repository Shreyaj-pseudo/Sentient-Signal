/* Connected two-link rig. Angles are local to each joint; the head stays level
   in world space. Every transition starts at the last rendered pose. */
(() => {
  const poses = {
    idle: [-28, 67, -7, .65],
    listening: [-43, 53, -17, 1],
    processing: [-22, 71, 8, .72],
    acknowledgment: [-33, 61, -21, .9],
    celebration: [-12, 41, -11, 1],
    reassurance: [-41, 57, -15, .8],
    clarification: [-20, 72, 16, .8],
    error: [-31, 73, 0, .35]
  };
  const sequences = {
    acknowledgment: [[330, [-33, 61, -23, .9]], [320, [-30, 65, -3, .85]], [300, [-33, 61, -18, .9]]],
    celebration: [[230, [-37, 87, -9, .9]], [420, [-12, 41, -13, 1]], [300, [-29, 70, 1, .9]], [380, [-18, 49, -9, 1]]],
    reassurance: [[850, poses.reassurance], [700, [-43, 55, -21, .8]], [700, poses.reassurance]],
    clarification: [[550, poses.clarification], [650, [-24, 69, -12, .8]], [550, poses.clarification]],
    error: [[450, poses.error]]
  };

  class LampController {
    constructor(svg) {
      this.svg = svg;
      this.lower = svg.querySelector("#lower-arm");
      this.upper = svg.querySelector("#upper-arm");
      this.head = svg.querySelector("#lamp-head");
      this.pose = [...poses.idle];
      this.motion = matchMedia("(prefers-reduced-motion: reduce)");
      this.motion.addEventListener("change", () => this.play(this.state));
      this.render(this.pose);
      this.play("idle");
    }

    render(pose) {
      this.pose = pose;
      this.lower.setAttribute("transform", `rotate(${pose[0]})`);
      this.upper.setAttribute("transform", `rotate(${pose[1]})`);
      this.head.setAttribute("transform", `rotate(${pose[2] - pose[0] - pose[1]})`);
      for (const id of ["light-pool", "light-beam", "lamp-bulb"]) {
        this.svg.querySelector(`#${id}`).setAttribute("opacity", pose[3]);
      }
    }

    play(name = "idle") {
      cancelAnimationFrame(this.frame);
      this.state = name in poses ? name : "idle";
      this.svg.dataset.gesture = this.state;
      this.svg.setAttribute("aria-label", `Desk lamp: ${this.state}`);
      if (this.motion.matches) {
        this.render([...poses[this.state]]);
        return;
      }
      const frames = sequences[this.state];
      const queue = frames ? [...frames, [750, poses.idle]] : [[600, poses[this.state]]];
      let from = [...this.pose], start = performance.now(), index = 0;
      const animate = now => {
        if (index < queue.length) {
          const [duration, target] = queue[index];
          const t = Math.min(1, (now - start) / duration);
          const eased = t * t * (3 - 2 * t);
          this.render(from.map((value, i) => value + (target[i] - value) * eased));
          if (t === 1) { from = [...this.pose]; start = now; index++; }
        } else {
          if (frames) { this.play("idle"); return; }
          const elapsed = (now - start) / 1000;
          const base = poses[this.state], next = [...base];
          if (this.state === "processing") {
            next[2] += Math.sin(elapsed * 2.2) * 6;
            next[3] += Math.sin(elapsed * 2.2) * .16;
          } else if (this.state === "listening") {
            next[1] += Math.sin(elapsed * 1.7) * 1.3;
            next[2] += Math.sin(elapsed * 1.7) * 1.5;
          } else {
            next[2] += Math.sin(elapsed * .9) * 1.2;
            next[3] += Math.sin(elapsed * .9) * .04;
          }
          this.render(next);
        }
        this.frame = requestAnimationFrame(animate);
      };
      this.frame = requestAnimationFrame(animate);
    }
  }
  window.lampController = new LampController(document.querySelector("#lamp"));
  document.querySelectorAll("[data-gesture]").forEach(button => {
    button.addEventListener("click", () => window.lampController.play(button.dataset.gesture));
  });
})();
