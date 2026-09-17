/**
 * The board, the clicks and the conversation with the engine.
 *
 * Two things here are less obvious than they look.
 *
 * ENTERING A CAPTURE. A move is a PATH, not a pair of squares: two chains can
 * start with the same jump and end elsewhere, and in Italian draughts which one
 * is legal depends on the priority rules. So a click does not pick a move, it
 * extends a prefix: the legal moves whose path still matches stay candidates,
 * and the move is played the moment the prefix IS one of them. A completed
 * chain can never be the prefix of another legal one, because priority forces
 * every legal capture to take the same number of pieces.
 *
 * WHO OWNS THE POSITION. The page owns it; the worker is asked for a move and
 * is told nothing else. The alternative -- a position living in the worker --
 * means two copies that can drift, and the one on screen would be the one
 * nobody checked.
 */
import { WHITE, BLACK, N_SQUARES, colorOf, isKing } from "../engine/board.js";
import { Position } from "../engine/game.js";
import { gameText, moveText, outcomeFor, SQUARE_NUMBER } from "./notation.js";

const SIZE = 60;                 // one square, in the SVG's own units
const SVG_NS = "http://www.w3.org/2000/svg";

const el = (id) => document.getElementById(id);
const rowOf = (s) => s >> 3;
const colOf = (s) => s & 7;

const state = {
  pos: new Position(),
  humanIsWhite: true,
  history: [],        // {pos, move} before each ply, oldest first
  moves: [],          // the moves played, for the game text
  selection: null,    // {path: number[], candidates: Move[]}
  lastMove: null,
  thinking: false,
  over: null,         // {text, reason, result}
  model: null,        // {id, label, file}
  models: [],
  worker: null,
  ready: false,
};

// --- drawing ---------------------------------------------------------------

function svgEl(name, attrs) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}

/** The board is drawn from BLACK's side when the human plays Black. */
const view = (s) => (state.humanIsWhite ? s : N_SQUARES - 1 - s);

function draw() {
  const board = el("board");
  board.textContent = "";

  const legal = state.pos.legalMoves();
  const canMove = !state.thinking && !state.over && isHumanTurn();
  const startSquares = new Set(canMove ? legal.map((m) => m.frm) : []);
  const nextSteps = new Set();
  if (state.selection) {
    const k = state.selection.path.length;
    for (const m of state.selection.candidates) if (m.path.length > k) nextSteps.add(m.path[k]);
  }
  const onPath = new Set(state.selection ? state.selection.path : []);
  const lastPath = new Set(state.lastMove ? state.lastMove.path : []);
  const lastTaken = new Set(state.lastMove ? state.lastMove.captured : []);

  for (let s = 0; s < N_SQUARES; s++) {
    const v = view(s);
    const x = colOf(v) * SIZE, y = rowOf(v) * SIZE;
    const dark = (rowOf(s) + colOf(s)) % 2 === 1;

    board.appendChild(svgEl("rect", {
      x, y, width: SIZE, height: SIZE,
      fill: dark ? "var(--dark-sq)" : "var(--light-sq)",
    }));
    if (!dark) continue;

    if (lastPath.has(s)) {
      board.appendChild(svgEl("rect", {
        x, y, width: SIZE, height: SIZE, fill: "var(--last)", opacity: 0.55,
      }));
    }
    const num = svgEl("text", {
      x: x + 5, y: y + 15, "font-size": 10, fill: "var(--fg)", opacity: 0.35,
    });
    num.textContent = SQUARE_NUMBER[s];
    board.appendChild(num);

    const piece = state.pos.board[s];
    const cx = x + SIZE / 2, cy = y + SIZE / 2;
    if (piece !== 0) {
      const white = colorOf(piece) === WHITE;
      board.appendChild(svgEl("circle", {
        cx, cy, r: 22,
        fill: white ? "var(--white-piece)" : "var(--black-piece)",
        stroke: white ? "var(--white-edge)" : "var(--black-edge)",
        "stroke-width": 2,
      }));
      if (isKing(piece)) {
        // A king wears an inner ring: it reads at any size, needs no icon font,
        // and survives the board being scaled down to a phone.
        board.appendChild(svgEl("circle", {
          cx, cy, r: 12, fill: "none",
          stroke: white ? "var(--white-edge)" : "#b9a98f", "stroke-width": 2,
        }));
      }
      if (startSquares.has(s) && !state.selection) {
        board.appendChild(svgEl("circle", {
          cx, cy, r: 26, fill: "none", stroke: "var(--pick)", "stroke-width": 2, opacity: 0.55,
        }));
      }
    }
    if (lastTaken.has(s)) {
      board.appendChild(svgEl("path", {
        d: `M ${cx - 11} ${cy - 11} L ${cx + 11} ${cy + 11} M ${cx + 11} ${cy - 11} L ${cx - 11} ${cy + 11}`,
        stroke: "var(--taken)", "stroke-width": 3, "stroke-linecap": "round", opacity: 0.85,
      }));
    }
    if (onPath.has(s)) {
      board.appendChild(svgEl("circle", {
        cx, cy, r: 26, fill: "none", stroke: "var(--pick)", "stroke-width": 3,
      }));
    }
    if (nextSteps.has(s)) {
      board.appendChild(svgEl("circle", { cx, cy, r: 9, fill: "var(--hint)", opacity: 0.75 }));
    }

    const hit = svgEl("rect", {
      x, y, width: SIZE, height: SIZE, fill: "transparent",
      style: canMove || state.selection ? "cursor: pointer" : "",
    });
    hit.addEventListener("click", () => onSquare(s));
    board.appendChild(hit);
  }
}

// --- the game --------------------------------------------------------------

const isHumanTurn = () =>
  (state.pos.turn === WHITE) === state.humanIsWhite;

function say(status, detail = "") {
  el("status").textContent = status;
  el("detail").textContent = detail;
}

function showMoves() {
  const out = [];
  for (let i = 0; i < state.moves.length; i += 2) {
    const a = moveText(state.moves[i]);
    const b = state.moves[i + 1] ? moveText(state.moves[i + 1]) : "";
    out.push(`${String(i / 2 + 1).padStart(3)}. ${a.padEnd(15)} ${b}`);
  }
  el("moves").textContent = out.length ? out.join("\n") : "no moves yet";
  el("moves").scrollTop = el("moves").scrollHeight;
  el("save").disabled = state.moves.length === 0;
  el("undo").disabled = state.thinking || state.history.length === 0;
}

function onSquare(s) {
  if (state.thinking || state.over || !isHumanTurn()) return;
  const legal = state.pos.legalMoves();

  if (!state.selection) {
    const candidates = legal.filter((m) => m.frm === s);
    if (candidates.length === 0) return;
    state.selection = { path: [s], candidates };
    draw();
    return;
  }

  const k = state.selection.path.length;
  const next = state.selection.candidates.filter((m) => m.path[k] === s);
  if (next.length === 0) {
    // Not a continuation: either another piece of ours, or a misclick.
    const restart = legal.filter((m) => m.frm === s);
    state.selection = restart.length ? { path: [s], candidates: restart } : null;
    draw();
    return;
  }

  const path = [...state.selection.path, s];
  const done = next.find((m) => m.path.length === path.length);
  if (done) {
    state.selection = null;
    playMove(done);
    return;
  }
  state.selection = { path, candidates: next };
  draw();
}

function playMove(move) {
  state.history.push({ pos: state.pos, move });
  state.moves.push(move);
  state.pos = state.pos.play(move);
  state.lastMove = move;
  state.selection = null;
  draw();
  showMoves();
  if (checkOver()) return;
  if (!isHumanTurn()) askEngine();
  else say(`Your move — ${state.pos.turn === WHITE ? "White" : "Black"} to play.`);
}

function checkOver() {
  if (!state.pos.isTerminal()) return false;
  const result = state.pos.result();
  const reason = state.pos.noProgress >= 80 ? "no-progress rule" : "no legal moves";
  let text;
  if (result === 0) text = "Draw.";
  else {
    const humanWon = (result === WHITE) === state.humanIsWhite;
    text = humanWon ? "You win." : "The engine wins.";
  }
  state.over = { text, reason, result };
  say(text, reason === "no-progress rule"
    ? "Eighty plies without a capture or a promotion: the draw rule of this project."
    : "The side to move has no legal moves.");
  draw();
  showMoves();
  return true;
}

function askEngine() {
  if (!state.ready) { say("Waiting for the network to load…"); return; }
  state.thinking = true;
  el("undo").disabled = true;
  const sims = Number(el("sims").value);
  say(`The engine is thinking (${sims} simulations)…`);
  draw();
  state.worker.postMessage({
    type: "think",
    board: Array.from(state.pos.board),
    turn: state.pos.turn,
    noProgress: state.pos.noProgress,
    sims,
  });
}

function onEngineMove(msg) {
  state.thinking = false;
  if (!msg.move) { checkOver(); return; }
  // The move arrives as plain data; it is matched back to the real one so the
  // page plays an object its own rules produced, never one built from a message.
  const mine = state.pos.legalMoves().find(
    (m) => m.frm === msg.move.frm && m.to === msg.move.to
      && m.path.length === msg.move.path.length
      && m.path.every((s, i) => s === msg.move.path[i]));
  if (!mine) { say("The engine proposed a move this position does not allow."); return; }

  const pct = Math.round(((msg.score + 1) / 2) * 100);
  playMove(mine);
  if (!state.over) {
    say(`Your move — ${state.pos.turn === WHITE ? "White" : "Black"} to play.`,
      `Engine: ${moveText(mine)} in ${msg.seconds.toFixed(1)} s, `
      + `${msg.visits} of ${Number(el("sims").value)} visits on it, `
      + `it rates the position ${pct}% for itself.`);
  }
}

// --- controls --------------------------------------------------------------

function newGame() {
  state.pos = new Position();
  state.history = [];
  state.moves = [];
  state.selection = null;
  state.lastMove = null;
  state.over = null;
  state.thinking = false;
  state.humanIsWhite = el("side").value === "white";
  draw();
  showMoves();
  if (isHumanTurn()) say("Your move — White opens.");
  else askEngine();
}

function takeBack() {
  if (state.thinking || state.history.length === 0) return;
  // Back to the last position where it was the human's turn: undoing a single
  // ply would hand the move straight back to the engine.
  do {
    const last = state.history.pop();
    state.pos = last.pos;
    state.moves.pop();
  } while (state.history.length > 0 && !isHumanTurn());
  state.lastMove = state.history.length ? state.history[state.history.length - 1].move : null;
  state.selection = null;
  state.over = null;
  draw();
  showMoves();
  say("Your move — taken back.");
}

function saveGame() {
  const result = state.over ? state.over.result : null;
  const text = gameText({
    moves: state.moves,
    humanIsWhite: state.humanIsWhite,
    outcome: result === null ? "unfinished" : outcomeFor(result, state.humanIsWhite),
    reason: state.over ? state.over.reason : "game not finished",
    model: state.model ? state.model.label : "no network",
    sims: Number(el("sims").value),
  });
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = "game.txt";
  a.click();
  URL.revokeObjectURL(url);
}

async function loadModel(id) {
  const model = state.models.find((m) => m.id === id) ?? null;
  state.model = model;
  state.ready = false;
  say(model ? `Loading ${model.label}…` : "Playing without a network.",
    model ? "The first load also fetches the runtime; the browser then keeps both." : "");
  // An absolute url: inside the worker a relative one would be resolved against
  // app/worker.js, and the models sit next to the page, not next to the worker.
  state.worker.postMessage({
    type: "load",
    model: model ? new URL(`models/${model.file}`, location.href).href : null,
  });
}

/** A readable name for each network, from the manifest. */
function labelOf(m) {
  if (m.run === "start") return "untrained network (cycle 0)";
  const elo = m.elo === null || m.elo === undefined ? "" : ` · ladder Elo ${Math.round(m.elo)}`;
  return `${m.run}, cycle ${m.cycle}${elo}`;
}

export async function start() {
  state.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  state.worker.onmessage = (ev) => {
    const msg = ev.data;
    if (msg.type === "ready") {
      state.ready = true;
      if (msg.failed) {
        // The search still works without a network, so the game goes on -- but
        // saying nothing would leave someone playing a far weaker opponent
        // than the one they picked.
        state.model = null;
        el("model").value = "";
        say("Playing without a network.",
          `That network could not be loaded (${msg.failed}), so the engine is `
          + "searching with no evaluation behind it: it will play much weaker.");
      } else {
        say(isHumanTurn() ? "Your move — White opens." : "The engine opens.",
          state.model ? `Opponent: ${state.model.label}.` : "");
      }
      if (!isHumanTurn() && !state.over) askEngine();
      return;
    }
    if (msg.type === "move") { onEngineMove(msg); return; }
    if (msg.type === "error") {
      state.thinking = false;
      say("The engine stopped.", msg.message);
    }
  };

  draw();
  showMoves();

  try {
    const manifest = await (await fetch("models/models.json")).json();
    state.models = manifest.map((m) => ({ ...m, label: labelOf(m) }));
  } catch {
    state.models = [];
    say("The networks could not be listed.",
      "A model has to be fetched over http; opening this file directly does not work.");
  }

  const select = el("model");
  select.textContent = "";
  for (const m of state.models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.label;
    select.appendChild(opt);
  }
  const none = document.createElement("option");
  none.value = "";
  none.textContent = "no network (search only)";
  select.appendChild(none);
  select.disabled = state.models.length === 0;

  // ?model=<id> opens straight onto one network: it is how the cards in
  // networks.html link into a game. Otherwise the strongest published one,
  // so whoever arrives without touching anything meets the engine at its best.
  const asked = new URLSearchParams(location.search).get("model");
  const preferred = state.models.find((m) => m.id === asked)
    ?? state.models.find((m) => m.id === "run300-c300") ?? state.models[0];
  if (preferred) {
    select.value = preferred.id;
    await loadModel(preferred.id);
  } else {
    await loadModel("");
  }

  el("new").addEventListener("click", newGame);
  el("undo").addEventListener("click", takeBack);
  el("save").addEventListener("click", saveGame);
  el("model").addEventListener("change", (e) => loadModel(e.target.value));
  el("side").addEventListener("change", newGame);
}
