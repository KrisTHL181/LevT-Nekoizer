//! Pre-compute insertion oracles for Levenshtein Transformer training.
//!
//! Equivalent to `scripts/precompute_oracles.py` but in fast, streaming Rust.
//!
//! Usage:
//!   precompute_oracles config.json policy_config.json input.jsonl output.jsonl
//!   precompute_oracles config.json policy_config.json input.jsonl output.jsonl --dry-run

use anyhow::{bail, Context, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

// ---------------------------------------------------------------------------
// Config types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ModelConfig {
    #[allow(dead_code)]
    vocab_size: u32,
    pad_token_id: u32,
    bos_token_id: u32,
    eos_token_id: u32,
    plh_token_id: u32,
    max_placeholder: usize,
}

#[derive(Deserialize)]
struct PolicyConfig {
    #[serde(default = "default_beta")]
    beta: f64,
    #[serde(default = "default_drop")]
    random_delete_prob: f64,
}

fn default_beta() -> f64 { 0.5 }
fn default_drop() -> f64 { 0.3 }

// ---------------------------------------------------------------------------
// Levenshtein DP alignment (insert + delete only, no substitution)
// ---------------------------------------------------------------------------

const INF: i32 = 1_000_000_000;
const OP_MATCH: u8 = 0;
const OP_DELETE: u8 = 1;
const OP_INSERT: u8 = 2;

/// Compute optimal edit alignment between two sequences using DP.
///
/// Returns (deletions, per_gap_insertions) where:
/// - `deletions`: indices in `y` to delete (0-indexed, excluding boundaries)
/// - `per_gap`: `per_gap[i]` = tokens to insert between surviving[i] and surviving[i+1]
fn levenshtein_align(y: &[u32], y_star: &[u32]) -> (Vec<usize>, Vec<Vec<u32>>) {
    let n = y.len();
    let m = y_star.len();

    // DP table: dp[i][j] = min edit distance for y[..i] ↔ y_star[..j]
    let mut dp: Vec<Vec<i32>> = vec![vec![INF; m + 1]; n + 1];
    // back[i][j]: operation that produced dp[i][j]
    let mut back: Vec<Vec<u8>> = vec![vec![0; m + 1]; n + 1];

    dp[0][0] = 0;
    for i in 1..=n {
        dp[i][0] = i as i32;
        back[i][0] = OP_DELETE;
    }
    for j in 1..=m {
        dp[0][j] = j as i32;
        back[0][j] = OP_INSERT;
    }

    for i in 1..=n {
        for j in 1..=m {
            let mut best_val = INF;
            let mut best_op = OP_MATCH;

            // Delete y[i-1]
            let cand = dp[i - 1][j] + 1;
            if cand < best_val {
                best_val = cand;
                best_op = OP_DELETE;
            }

            // Insert y_star[j-1]
            let cand = dp[i][j - 1] + 1;
            if cand < best_val {
                best_val = cand;
                best_op = OP_INSERT;
            }

            // Match if tokens equal
            if y[i - 1] == y_star[j - 1] {
                let cand = dp[i - 1][j - 1];
                if cand < best_val {
                    best_val = cand;
                    best_op = OP_MATCH;
                }
            }

            dp[i][j] = best_val;
            back[i][j] = best_op;
        }
    }

    // --- First backtrack pass: collect deletions and matched_pairs ---
    let mut deletions: Vec<usize> = Vec::new();
    let mut matched_pairs: Vec<(usize, usize)> = Vec::new();

    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        match back[i][j] {
            OP_MATCH => {
                i -= 1;
                j -= 1;
                matched_pairs.push((i, j));
            }
            OP_DELETE => {
                i -= 1;
                deletions.push(i);
            }
            _ => {
                // INSERT
                j -= 1;
            }
        }
    }

    deletions.reverse();
    matched_pairs.reverse();

    // --- Second backtrack pass: collect insertions with positions ---
    let mut insertions_raw: Vec<u32> = Vec::new();
    let mut insertion_after: Vec<usize> = Vec::new();

    i = n;
    j = m;
    let mut current_after = n;
    while i > 0 || j > 0 {
        match back[i][j] {
            OP_MATCH => {
                i -= 1;
                j -= 1;
                current_after = i;
            }
            OP_DELETE => {
                i -= 1;
            }
            _ => {
                // INSERT
                j -= 1;
                insertions_raw.push(y_star[j]);
                insertion_after.push(current_after);
            }
        }
    }

    insertions_raw.reverse();
    insertion_after.reverse();

    // --- Group insertions by surviving-position gaps ---
    let del_set: BTreeSet<usize> = deletions.iter().copied().collect();
    let surviving: Vec<usize> = (0..n).filter(|idx| !del_set.contains(idx)).collect();

    let num_gaps = if surviving.len() >= 2 {
        surviving.len() - 1
    } else {
        0
    };
    let mut per_gap: Vec<Vec<u32>> = vec![Vec::new(); num_gaps];

    if surviving.len() >= 2 {
        // Map from original y-index to surviving-index
        let surv_rank: BTreeMap<usize, usize> =
            surviving.iter().enumerate().map(|(si, &sid)| (sid, si)).collect();

        for (&tok, &after_y) in insertions_raw.iter().zip(insertion_after.iter()) {
            let gap_surv_idx: i32 = if let Some(&rank) = surv_rank.get(&after_y) {
                rank as i32 - 1 // gap before this position
            } else {
                // after_y points to a deleted position — find the gap after
                // the last surviving token before after_y
                let mut gap = -1i32;
                for (si, &sid) in surviving.iter().enumerate() {
                    if sid < after_y {
                        gap = si as i32;
                    } else {
                        break;
                    }
                }
                gap
            };

            if gap_surv_idx >= 0 && (gap_surv_idx as usize) < per_gap.len() {
                per_gap[gap_surv_idx as usize].push(tok);
            }
        }
    }

    (deletions, per_gap)
}

// ---------------------------------------------------------------------------
// Oracle policies
// ---------------------------------------------------------------------------

/// Oracle deletion: boolean mask, True = DELETE this token.
/// Boundaries (first and last) are always False.
fn oracle_deletion(y: &[u32], y_star: &[u32]) -> Vec<bool> {
    let (deletions, _) = levenshtein_align(y, y_star);
    let mut mask = vec![false; y.len()];
    for &d in &deletions {
        mask[d] = true;
    }
    // Never delete boundaries
    if !mask.is_empty() {
        mask[0] = false;
        let last = mask.len() - 1;
        mask[last] = false;
    }
    mask
}

/// Oracle insertion: returns (p_star, t_star).
/// - p_star: placeholder counts per gap in the ORIGINAL y (length = y.len() - 1)
/// - t_star: flattened token IDs for all placeholders
fn oracle_insertion(
    y: &[u32],
    y_star: &[u32],
    max_placeholder: usize,
) -> (Vec<usize>, Vec<u32>) {
    let (deletions, insertions) = levenshtein_align(y, y_star);

    let del_set: BTreeSet<usize> = deletions.iter().copied().collect();
    let surviving: Vec<usize> = (0..y.len()).filter(|idx| !del_set.contains(idx)).collect();

    let num_gaps = if surviving.len() >= 2 {
        surviving.len() - 1
    } else {
        0
    };

    let mut p_star = vec![0usize; y.len().saturating_sub(1)];
    let mut t_star: Vec<u32> = Vec::new();

    for gi in 0..num_gaps {
        let left_orig = surviving[gi];
        let tokens_to_insert: &[u32] = if gi < insertions.len() {
            &insertions[gi]
        } else {
            &[]
        };
        let count = tokens_to_insert.len().min(max_placeholder);
        if left_orig < p_star.len() {
            p_star[left_orig] = count;
        }
        if count > 0 {
            t_star.extend_from_slice(&tokens_to_insert[..count]);
        }
    }

    (p_star, t_star)
}

/// Remove tokens marked for deletion.
fn apply_deletion(y: &[u32], deletion_mask: &[bool]) -> Vec<u32> {
    y.iter()
        .zip(deletion_mask.iter())
        .filter(|(_, &del)| !del)
        .map(|(&tok, _)| tok)
        .collect()
}

/// Insert <PLH> tokens according to placeholder counts.
fn insert_placeholders(y: &[u32], p_counts: &[usize], plh_token_id: u32) -> Vec<u32> {
    let total_plh: usize = p_counts.iter().sum();
    let mut result = Vec::with_capacity(y.len() + total_plh);
    for (i, &tok) in y.iter().enumerate() {
        result.push(tok);
        if i < p_counts.len() {
            for _ in 0..p_counts[i] {
                result.push(plh_token_id);
            }
        }
    }
    result
}

/// Randomly delete tokens from y_star (excluding boundaries and PAD).
fn random_deletion(
    y_star: &[u32],
    drop_prob: f64,
    _bos_idx: u32,
    _eos_idx: u32,
    pad_idx: u32,
    rng: &mut impl Rng,
) -> Vec<u32> {
    let mut keep: Vec<u32> = Vec::with_capacity(y_star.len());
    for (idx, &tok) in y_star.iter().enumerate() {
        if idx == 0 {
            // BOS
            keep.push(tok);
        } else if idx == y_star.len() - 1 {
            // EOS
            keep.push(tok);
        } else if tok == pad_idx {
            continue;
        } else if rng.gen::<f64>() > drop_prob {
            keep.push(tok);
        }
    }
    keep
}

// ---------------------------------------------------------------------------
// Row processing
// ---------------------------------------------------------------------------

fn process_row(
    row: &serde_json::Value,
    config: &ModelConfig,
    drop_prob: f64,
    rng: &mut impl Rng,
) -> serde_json::Value {
    let initial_default =
        serde_json::Value::Array(vec![
            serde_json::Value::from(config.bos_token_id),
            serde_json::Value::from(config.eos_token_id),
        ]);
    let initial_val = row.get("initial").unwrap_or(&initial_default);
    let target_val = row.get("target").expect("missing target");

    let initial: Vec<u32> = initial_val
        .as_array()
        .expect("initial must be array")
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let target: Vec<u32> = target_val
        .as_array()
        .expect("target must be array")
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    // --- Oracle path ---
    let deletion_mask = oracle_deletion(&initial, &target);
    let y_ins = apply_deletion(&initial, &deletion_mask);
    let (p_star, t_star) = oracle_insertion(&y_ins, &target, config.max_placeholder);
    let y_ins_plh = insert_placeholders(&y_ins, &p_star, config.plh_token_id);

    // --- Random path ---
    let y_ins_rnd = random_deletion(
        &target,
        drop_prob,
        config.bos_token_id,
        config.eos_token_id,
        config.pad_token_id,
        rng,
    );
    let (p_star_rnd, t_star_rnd) =
        oracle_insertion(&y_ins_rnd, &target, config.max_placeholder);
    let y_ins_plh_rnd = insert_placeholders(&y_ins_rnd, &p_star_rnd, config.plh_token_id);

    // Build output row (preserve all original fields)
    let mut out = row.clone();
    let obj = out.as_object_mut().unwrap();
    obj.insert("y_ins".into(), json_array(&y_ins));
    obj.insert("p_star".into(), json_array_usize(&p_star));
    obj.insert("t_star".into(), json_array(&t_star));
    obj.insert("y_ins_plh".into(), json_array(&y_ins_plh));
    obj.insert("y_ins_rnd".into(), json_array(&y_ins_rnd));
    obj.insert("p_star_rnd".into(), json_array_usize(&p_star_rnd));
    obj.insert("t_star_rnd".into(), json_array(&t_star_rnd));
    obj.insert("y_ins_plh_rnd".into(), json_array(&y_ins_plh_rnd));

    out
}

fn json_array(data: &[u32]) -> serde_json::Value {
    serde_json::Value::Array(data.iter().map(|&v| serde_json::Value::from(v)).collect())
}

fn json_array_usize(data: &[usize]) -> serde_json::Value {
    serde_json::Value::Array(
        data.iter()
            .map(|&v| serde_json::Value::from(v as u64))
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Dry-run display
// ---------------------------------------------------------------------------

fn show_stats(rows: &[serde_json::Value], _config: &ModelConfig) {
    for (i, row) in rows.iter().enumerate().take(3) {
        let init_len = row
            .get("initial")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(2);
        println!("--- Row {} ---", i + 1);
        println!(
            "  src:            {} tokens",
            row.get("src").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0)
        );
        println!(
            "  target:         {} tokens",
            row.get("target").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0)
        );
        println!("  initial:        {} tokens", init_len);
        println!(
            "  y_ins:          {} tokens  p_star={:?}  t_star: {} tokens",
            row.get("y_ins").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0),
            row.get("p_star")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_u64()).collect::<Vec<_>>())
                .unwrap_or_default(),
            row.get("t_star").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0),
        );
        println!(
            "  y_ins_plh:      {} tokens",
            row.get("y_ins_plh").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0)
        );
        println!(
            "  y_ins_rnd:      {} tokens  p_star_rnd={:?}  t_star_rnd: {} tokens",
            row.get("y_ins_rnd").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0),
            row.get("p_star_rnd")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_u64()).collect::<Vec<_>>())
                .unwrap_or_default(),
            row.get("t_star_rnd")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0),
        );
        println!(
            "  y_ins_plh_rnd:  {} tokens",
            row.get("y_ins_plh_rnd")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0)
        );
        println!();
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let (dry_run, positional): (bool, Vec<&str>) = {
        let mut dry = false;
        let mut pos: Vec<&str> = Vec::new();
        for arg in &args[1..] {
            if arg == "--dry-run" {
                dry = true;
            } else if !arg.starts_with('-') {
                pos.push(arg);
            }
        }
        (dry, pos)
    };

    if positional.len() < 4 {
        eprintln!(
            "Usage: precompute_oracles <model_config> <policy_config> <input> <output> [--dry-run]"
        );
        std::process::exit(1);
    }

    let model_cfg_path = positional[0];
    let policy_cfg_path = positional[1];
    let input_path = Path::new(positional[2]);
    let output_path = Path::new(positional[3]);

    // Load model config
    let model_json: serde_json::Value = serde_json::from_reader(
        File::open(model_cfg_path)
            .with_context(|| format!("cannot open model config: {}", model_cfg_path))?,
    )
    .context("invalid model config JSON")?;
    let model_cfg: ModelConfig = serde_json::from_value(model_json).context("invalid model config fields")?;

    // Load policy config
    let policy_cfg: PolicyConfig = serde_json::from_reader(
        File::open(policy_cfg_path)
            .with_context(|| format!("cannot open policy config: {}", policy_cfg_path))?,
    )
    .context("invalid policy config")?;

    eprintln!(
        "Policy: beta={}, random_delete_prob={}",
        policy_cfg.beta, policy_cfg.random_delete_prob
    );

    // Seeded RNG (matches Python's random.seed(0))
    let mut rng = StdRng::seed_from_u64(0);

    // Read input rows
    if !input_path.exists() {
        bail!("input file not found: {}", positional[2]);
    }

    let input_file =
        File::open(input_path).with_context(|| format!("cannot open: {}", positional[2]))?;
    let reader = BufReader::new(input_file);

    let mut rows: Vec<serde_json::Value> = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("read error at line {}", line_no + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            bail!("blank line at {}:{}", positional[2], line_no + 1);
        }
        let row: serde_json::Value =
            serde_json::from_str(trimmed).with_context(|| {
                format!("invalid JSON at {}:{}", positional[2], line_no + 1)
            })?;
        if !row.is_object() {
            bail!("non-object at {}:{}", positional[2], line_no + 1);
        }
        if row.get("src").is_none() || row.get("target").is_none() {
            bail!(
                "missing src/target at {}:{}",
                positional[2],
                line_no + 1
            );
        }
        rows.push(row);
    }

    if rows.is_empty() {
        bail!("input file is empty");
    }

    eprintln!("Read {} rows from {}", rows.len(), positional[2]);

    // Dry-run mode
    if dry_run {
        // Process first 3 rows with the RNG
        let mut sample_rng = StdRng::seed_from_u64(0);
        let processed: Vec<serde_json::Value> = rows
            .iter()
            .take(3)
            .map(|r| process_row(r, &model_cfg, policy_cfg.random_delete_prob, &mut sample_rng))
            .collect();
        show_stats(&processed, &model_cfg);
        return Ok(());
    }

    // Process and write output
    let output_file = File::create(output_path)
        .with_context(|| format!("cannot create: {}", positional[3]))?;
    let mut writer = BufWriter::new(output_file);

    let mut written: u64 = 0;
    for (i, row) in rows.iter().enumerate() {
        let out_row = process_row(row, &model_cfg, policy_cfg.random_delete_prob, &mut rng);
        serde_json::to_writer(&mut writer, &out_row)
            .with_context(|| format!("write error at row {}", i + 1))?;
        writeln!(&mut writer).context("write newline error")?;
        written += 1;
        if written % 10000 == 0 {
            eprintln!("Processed {} rows...", written);
        }
    }

    writer.flush().context("flush error")?;
    eprintln!("Done. Wrote {} rows to {}", written, positional[3]);

    Ok(())
}
