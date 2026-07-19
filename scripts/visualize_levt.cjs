#!/usr/bin/env node
/**
 * Levenshtein Transformer architecture diagram using @mappedinfo/llm-architecture-svg.
 *
 * Reads the full config from config.json (no model instantiation needed — this is
 * pure SVG generation).  All architecture parameters match the actual model.
 *
 * Each transformer layer is split into separate attention and MLP blocks so that
 * the renderer's parameter-summary header correctly categorises them.
 *
 * Generates: scripts/levt_architecture.svg
 */

"use strict";

const { renderArchitectureSvg } = require("@mappedinfo/llm-architecture-svg");
const { writeFileSync, readFileSync } = require("node:fs");
const { join } = require("node:path");

// ── Load config ─────────────────────────────────────────────────────────
const configPath = join(__dirname, "..", "config.json");
const cfg = JSON.parse(readFileSync(configPath, "utf8"));

const EMB = cfg.embedding_dim;                // 1024
const DM  = cfg.d_model;                      // 512
const NH  = cfg.n_heads;                      // 8
const DFF = cfg.d_ff;                         // 2048
const NEC = cfg.n_encoder_layers;             // 6
const NDC = cfg.n_decoder_layers;             // 6
const V   = cfg.vocab_size;                   // 73448
const K   = cfg.max_placeholder;              // 255
const POS = cfg.pos_encoding_type;            // alibi
const QKN = cfg.qk_norm;                      // true
const ACT = cfg.activation;                   // gelu
const HWG = cfg.headwise_attn_output_gate;    // false
const EWG = cfg.elementwise_attn_output_gate; // false

const now = new Date().toISOString();

// ── Helpers ─────────────────────────────────────────────────────────────

function fmt(n) {
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(2) + "K";
  return String(n);
}

// Layout — two-column: encoder (left), decoder (right), heads (bottom)
const LX = 45;       // left-column x
const RX = 670;      // right-column x
const BW = 130;      // block width (narrower since attn+ffn side-by-side)
const BH = 60;       // block height
const GY = 84;       // vertical gap
const Y0 = 28;       // first row y
const AGAP = 15;     // gap between attn and ffn in each layer

function sz(w, h) { return { w: w || BW, h: h || BH }; }

function leaf(id, kind, label, x, y, w, h, color, role, shapeLabel, paramCat, paramCount, paramFormula) {
  return {
    id, type: "block", kind, label, shape: {},
    position2d: { x, y },
    size2d: sz(w, h), color,
    derived: {
      source: "transformer-template",
      role: role || id,
      shapeLabel,
      paramCategory: paramCat || "none",
      paramCount: paramCount || 0,
      paramFormula,
      overview: true,
    },
  };
}

function edge(id, source, target, kind, label, route) {
  return { id, source, target, kind: kind || "data", label, route };
}

function encY(r) { return Y0 + r * GY; }
function decY(r) { return Y0 + r * GY; }

// ═══════════════════════════════════════════════════════════════════════
//  Parameter counts  (with biases for FFN, without for Q/K/V/O projections)
// ═══════════════════════════════════════════════════════════════════════

const encAttnParams = 4 * DM * DM;          // Q, K, V, O (no bias)
const encFfnParams  = DM * DFF + DFF + DFF * DM + DM;  // up + down (with bias)
const decAttnParams = 8 * DM * DM;          // self Q/K/V/O + cross Q/K/V/O
const decFfnParams  = encFfnParams;

const encTag = `${POS}${QKN ? " · QKNorm" : ""}`;
const decTag = "bidirectional";

// ═══════════════════════════════════════════════════════════════════════
//  NODES
// ═══════════════════════════════════════════════════════════════════════

const nodes = [
  // ═══ Encoder column ═══════════════════════════════════════════════
  leaf("src_tokens", "generic_tensor", "Source Tokens",
       LX, encY(0), 150, BH, "#efe3c8",
       "source_tokens", "[S, B]", "none", 0),

  leaf("shared_embed", "token_embed", "Shared Embedding",
       LX, encY(1), 250, BH, "#f0a8fc",
       "shared_embedding", `[${fmt(V)}, ${EMB}]`, "embedding", V * EMB,
       `${fmt(V)} × ${EMB} = ${fmt(V * EMB)}`),

  leaf("enc_proj", "linear", "Encoder Input Projection",
       LX, encY(2), 250, BH, "#a8c3fc",
       "encoder_input_projection", `[${EMB}, ${DM}]`, "linear", EMB * DM,
       `${EMB} × ${DM} = ${fmt(EMB * DM)}`),

  // Encoder layers: attn + ffn side-by-side
  ...Array.from({ length: NEC }, (_, i) => {
    const n = i + 1;
    const tag = n === 1 ? `  [${encTag}]` : "";
    const y = encY(3 + i);
    const attnId = `enc_l${n}_attn`;
    const ffnId  = `enc_l${n}_ffn`;
    const attnX = LX;
    const ffnX  = LX + BW + AGAP;
    return [
      leaf(attnId, "attention", `Enc L${n}${tag}`,
           attnX, y, BW, BH, "#c0e0b8",
           `enc_l${n}_attn`, undefined, "attention", encAttnParams,
           `Q/K/V/O: 4×${DM}²`),
      leaf(ffnId, "mlp", `FFN  [${ACT}]`,
           ffnX, y, BW - 10, BH, "#d0e8c8",
           `enc_l${n}_ffn`, undefined, "mlp", encFfnParams,
           `${DM}↔${DFF}: ${fmt(encFfnParams)}`),
    ];
  }).flat(),

  leaf("enc_norm", "layer_norm", "Encoder RMSNorm",
       LX, encY(3 + NEC), 250, BH, "#d4c4a0",
       "encoder_final_norm", `[${DM}]`, "layer_norm", DM, String(DM)),

  leaf("memory", "generic_tensor", "Encoder Memory",
       LX, encY(4 + NEC), 250, BH, "#c8d8e8",
       "encoder_memory", `[S, B, ${DM}]`, "none", 0),

  // ═══ Decoder column ═══════════════════════════════════════════════
  leaf("tgt_tokens", "generic_tensor", "Target Tokens",
       RX, decY(0), 150, BH, "#efe3c8",
       "target_tokens", "[T, B]", "none", 0),

  leaf("dec_proj", "linear", "Decoder Input Projection",
       RX, decY(1), 250, BH, "#a8c3fc",
       "decoder_input_projection", `[${EMB}, ${DM}]`, "linear", EMB * DM,
       `${EMB} × ${DM} = ${fmt(EMB * DM)}`),

  // Decoder layers: attn + ffn side-by-side
  ...Array.from({ length: NDC }, (_, i) => {
    const n = i + 1;
    const tag = n === 1 ? `  [${decTag}]` : "";
    const y = decY(3 + i);
    const attnId = `dec_l${n}_attn`;
    const ffnId  = `dec_l${n}_ffn`;
    const attnX = RX;
    const ffnX  = RX + BW + AGAP;
    return [
      leaf(attnId, "attention", `Dec L${n}${tag}`,
           attnX, y, BW + 10, BH, "#90c0e0",
           `dec_l${n}_attn`, undefined, "attention", decAttnParams,
           `Q/K/V/O ×8: 8×${DM}²`),
      leaf(ffnId, "mlp", `FFN  [${ACT}]`,
           ffnX, y, BW - 10, BH, "#a0d0e8",
           `dec_l${n}_ffn`, undefined, "mlp", decFfnParams,
           `${DM}↔${DFF}: ${fmt(decFfnParams)}`),
    ];
  }).flat(),

  leaf("dec_norm", "layer_norm", "Decoder RMSNorm",
       RX, decY(3 + NDC), 250, BH, "#d4c4a0",
       "decoder_final_norm", `[${DM}]`, "layer_norm", DM, String(DM)),

  // ═══ Prediction heads ═════════════════════════════════════════════
  leaf("del_head", "linear", "Deletion Head  (keep / delete)",
       LX + 50, Y0 + (5 + Math.max(NEC, NDC)) * GY, 210, BH, "#f4a0a0",
       "deletion_head", `[${DM}, 2]`, "linear", DM * 2,
       `${DM} × 2 = ${DM * 2}`),

  leaf("plh_head", "linear", "Placeholder Head  (0 … K PLH per gap)",
       LX + 330, Y0 + (5 + Math.max(NEC, NDC)) * GY, 250, BH, "#f4c8a0",
       "placeholder_head", `[${2*DM}, ${K+1}]`, "linear", 2 * DM * (K + 1),
       `2×${DM}×${K+1} = ${fmt(2 * DM * (K + 1))}`),

  leaf("tok_head", "linear", "Token Head  [weight-tied]",
       LX + 660, Y0 + (5 + Math.max(NEC, NDC)) * GY, 250, BH, "#a0d4f4",
       "token_prediction", `[${DM}, ${V}]`, "none", 0,
       "wt-tied: dec_projᵀ · embed"),
];

// ═══════════════════════════════════════════════════════════════════════
//  EDGES
// ═══════════════════════════════════════════════════════════════════════

// Build ID lists: each layer is [attn, ffn]
function encLayerIds(n) { return [`enc_l${n}_attn`, `enc_l${n}_ffn`]; }
function decLayerIds(n) { return [`dec_l${n}_attn`, `dec_l${n}_ffn`]; }

const encIdList = ["src_tokens", "shared_embed", "enc_proj"];
for (let i = 1; i <= NEC; i++) encIdList.push(...encLayerIds(i));
encIdList.push("enc_norm", "memory");

const decIdList = ["tgt_tokens", "shared_embed", "dec_proj"];
for (let i = 1; i <= NDC; i++) decIdList.push(...decLayerIds(i));
decIdList.push("dec_norm");

function chainEdges(ids, prefix) {
  const result = [];
  for (let i = 0; i < ids.length - 1; i++) {
    result.push(edge(`${prefix}_${i}`, ids[i], ids[i + 1], "data"));
  }
  return result;
}

const edges = [
  ...chainEdges(encIdList, "enc"),
  edge("dec_0", "tgt_tokens", "shared_embed", "data", "shared"),
  edge("dec_1", "shared_embed", "dec_proj", "data"),
  ...chainEdges(decIdList.slice(2), "dec"),
  // Cross-attention: memory → every decoder attention block
  ...Array.from({ length: NDC }, (_, i) =>
    edge(`cross_${i}`, "memory", `dec_l${i + 1}_attn`, "dependency",
         "cross-attn", "rounded-orthogonal")),
  // Heads from decoder norm
  edge("h_del", "dec_norm", "del_head", "data"),
  edge("h_plh", "dec_norm", "plh_head", "data"),
  edge("h_tok", "dec_norm", "tok_head", "data"),
];

// ═══════════════════════════════════════════════════════════════════════
//  SPEC & RENDER
// ═══════════════════════════════════════════════════════════════════════

const gateInfo = HWG ? "headwise-gate" : EWG ? "elementwise-gate" : "gates=off";
const totalRows = 6 + Math.max(NEC, NDC);
const canvasH = Y0 + totalRows * GY + 120;

const spec = {
  schemaVersion: 1,
  mode: "teaching",
  id: "levt-architecture",
  name: "Levenshtein Transformer Architecture",
  notes:
    `Config from config.json:\n` +
    `embedding_dim=${EMB}  d_model=${DM}  n_heads=${NH}  d_ff=${DFF}\n` +
    `n_encoder_layers=${NEC}  n_decoder_layers=${NDC}  vocab_size=${V}\n` +
    `pos=${POS}  qk_norm=${QKN}  activation=${ACT}  ${gateInfo}`,
  nodes,
  edges,
  view: { canvas: { w: 1160, h: canvasH } },
  createdAt: now,
  updatedAt: now,
};

const svg = renderArchitectureSvg(spec, {
  title: "Levenshtein Transformer — Architecture Diagram",
  showShapes: true,
  showParamCounts: true,
  width: 1160,
});

const outPath = join(__dirname, "levt_architecture.svg");
writeFileSync(outPath, svg, "utf8");

const totalParams = V * EMB + 2 * EMB * DM + NEC * (encAttnParams + encFfnParams)
  + NDC * (decAttnParams + decFfnParams) + 2 * DM + DM * 2 + 2 * DM * (K + 1);

console.log("Generated:", outPath);
console.log("Config: emb_dim=%d d_model=%d n_heads=%d d_ff=%d enc=%d dec=%d vocab=%d",
            EMB, DM, NH, DFF, NEC, NDC, V);
console.log("Features: pos=%s qk_norm=%s activation=%s %s",
            POS, QKN, ACT, gateInfo);
console.log("Total params: %s (%d)", fmt(totalParams), totalParams);
