#![allow(clippy::useless_conversion, clippy::too_many_arguments)]

use image::{DynamicImage, ImageBuffer, ImageFormat, Rgba};
use numpy::PyReadonlyArray3;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::HashMap;
use std::io::Cursor;
use std::path::Path;

#[pyclass]
#[derive(Clone, Debug)]
pub struct SplitBox {
    #[pyo3(get)]
    pub index: usize,
    #[pyo3(get)]
    pub row: usize,
    #[pyo3(get)]
    pub col: usize,
    #[pyo3(get)]
    pub x: u32,
    #[pyo3(get)]
    pub y: u32,
    #[pyo3(get)]
    pub width: u32,
    #[pyo3(get)]
    pub height: u32,
    #[pyo3(get)]
    pub confidence: f32,
    #[pyo3(get)]
    pub strategy: String,
    #[pyo3(get)]
    pub trim_top: u32,
    #[pyo3(get)]
    pub trim_right: u32,
    #[pyo3(get)]
    pub trim_bottom: u32,
    #[pyo3(get)]
    pub trim_left: u32,
}

#[derive(Clone, Copy)]
struct SeamCandidate {
    pos: u32,
    score: f32,
    coverage: f32,
    is_gutter: bool,
}

#[derive(Clone)]
struct RawRegion {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    conf: f32,
    is_gutter: bool,
}

#[inline(always)]
fn pixel_at(buf: &[u8], stride: usize, ch: usize, x: u32, y: u32) -> [u8; 4] {
    let idx = (y as usize) * stride + (x as usize) * ch;
    if ch >= 4 {
        [buf[idx], buf[idx + 1], buf[idx + 2], buf[idx + 3]]
    } else {
        [buf[idx], buf[idx + 1], buf[idx + 2], 255]
    }
}

#[inline(always)]
fn color_dist(a: [u8; 4], b: [u8; 4]) -> i32 {
    (a[0] as i32 - b[0] as i32).abs()
        + (a[1] as i32 - b[1] as i32).abs()
        + (a[2] as i32 - b[2] as i32).abs()
}

fn estimate_border_bg(buf: &[u8], w: u32, h: u32, ch: usize) -> ([u8; 4], bool, f32) {
    let stride = (w as usize) * ch;
    let step_x = (w / 80).max(1);
    let step_y = (h / 80).max(1);
    let mut samples = Vec::with_capacity(320);
    let mut transparent = 0usize;

    let mut x = 0;
    while x < w {
        let t = pixel_at(buf, stride, ch, x, 0);
        let b = pixel_at(buf, stride, ch, x, h - 1);
        if t[3] < 32 {
            transparent += 1;
        }
        if b[3] < 32 {
            transparent += 1;
        }
        samples.push(t);
        samples.push(b);
        x += step_x;
    }
    let mut y = 0;
    while y < h {
        let l = pixel_at(buf, stride, ch, 0, y);
        let r = pixel_at(buf, stride, ch, w - 1, y);
        if l[3] < 32 {
            transparent += 1;
        }
        if r[3] < 32 {
            transparent += 1;
        }
        samples.push(l);
        samples.push(r);
        y += step_y;
    }

    let n = samples.len().max(1);
    if transparent > n / 2 {
        return ([0, 0, 0, 0], true, transparent as f32 / n as f32);
    }

    type ColorBin = (usize, u64, u64, u64);
    let mut bins: HashMap<(u8, u8, u8), ColorBin> = HashMap::new();
    for s in &samples {
        let key = (s[0] >> 3, s[1] >> 3, s[2] >> 3);
        let entry = bins.entry(key).or_insert((0, 0, 0, 0));
        entry.0 += 1;
        entry.1 += s[0] as u64;
        entry.2 += s[1] as u64;
        entry.3 += s[2] as u64;
    }

    let best = bins.values().max_by_key(|v| v.0).copied().unwrap_or((1, 255, 255, 255));
    let cnt = best.0.max(1) as u64;
    (
        [(best.1 / cnt) as u8, (best.2 / cnt) as u8, (best.3 / cnt) as u8, 255],
        false,
        best.0 as f32 / n as f32,
    )
}

fn collapse_gutter_runs(mut cands: Vec<SeamCandidate>) -> Vec<SeamCandidate> {
    let mut i = 0;
    while i < cands.len() {
        if !cands[i].is_gutter {
            i += 1;
            continue;
        }
        let mut j = i;
        while j + 1 < cands.len() && cands[j + 1].is_gutter && cands[j + 1].pos == cands[j].pos + 1 {
            j += 1;
        }
        let mid = (i + j) / 2;
        let max_s = cands[i..=j].iter().map(|c| c.score).fold(0.0f32, f32::max);
        for (k, cand) in cands.iter_mut().enumerate().take(j + 1).skip(i) {
            if k == mid {
                cand.score = max_s;
            } else {
                cand.score = 0.0;
                cand.is_gutter = false;
            }
        }
        i = j + 1;
    }
    cands
}

fn profile_horizontal_seams(
    buf: &[u8],
    w: u32,
    ch: usize,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    min_dim: u32,
    bg_rgb: [u8; 4],
    bg_uniform: bool,
    is_trans: bool,
) -> Vec<SeamCandidate> {
    let span_h = (y1 - y0) as usize;
    let span_w = x1 - x0;
    if (y1 - y0) <= min_dim * 2 || span_w < 8 {
        return Vec::new();
    }

    let stride = (w as usize) * ch;
    let step_x = (span_w / 200).max(1);
    let sample_count = span_w.div_ceil(step_x) as usize;

    // Parallel scan per row producing scalar stats + compact thresholded bitflags (1 byte instead of 4B f32)
    let row_stats: Vec<(f32, f32, f32, f32, Vec<u8>)> = (0..span_h)
        .into_par_iter()
        .map(|ry| {
            if ry == 0 || ry + 1 >= span_h {
                return (0.0, 0.0, 0.0, 0.0, vec![0u8; sample_count]);
            }
            let y = y0 + ry as u32;
            let mut s_sum = 0.0f32;
            let mut l_sum = 0.0f32;
            let mut sm_sum = 0.0f32;
            let mut bg_hits = 0usize;
            let mut prev_luma: i32 = -1;
            let mut active_row = Vec::with_capacity(sample_count);

            let mut x = x0;
            while x < x1 {
                let p0 = pixel_at(buf, stride, ch, x, y - 1);
                let p1 = pixel_at(buf, stride, ch, x, y);
                let p2 = pixel_at(buf, stride, ch, x, y + 1);
                let ds = color_dist(p1, p2) as f32;
                let dl = ((2 * p1[0] as i32 - p0[0] as i32 - p2[0] as i32).abs()
                    + (2 * p1[1] as i32 - p0[1] as i32 - p2[1] as i32).abs()
                    + (2 * p1[2] as i32 - p0[2] as i32 - p2[2] as i32).abs()) as f32
                    * 0.5;
                s_sum += ds;
                l_sum += dl;
                active_row.push(if ds + dl > 16.0 { 1 } else { 0 });

                let luma = p1[0] as i32 + p1[1] as i32 + p1[2] as i32;
                if prev_luma >= 0 {
                    sm_sum += (luma - prev_luma).abs() as f32;
                }
                prev_luma = luma;

                if bg_uniform {
                    let is_bg = if is_trans { p1[3] < 24 } else { color_dist(p1, bg_rgb) < 22 };
                    if is_bg {
                        bg_hits += 1;
                    }
                }
                x += step_x;
            }
            let sc = sample_count.max(1) as f32;
            let sm_denom = sample_count.saturating_sub(1).max(1) as f32;
            (s_sum / sc, l_sum / sc, sm_sum / sm_denom, bg_hits as f32 / sc, active_row)
        })
        .collect();

    let start_ry = min_dim as usize;
    let end_ry = span_h.saturating_sub(min_dim as usize);

    let cands: Vec<SeamCandidate> = (start_ry..=end_ry)
        .into_par_iter()
        .map(|ry| {
            let lo = ry.saturating_sub(2).max(1);
            let hi = (ry + 2).min(span_h - 2);
            let mut active = 0usize;
            for ui in 0..sample_count {
                let hit = row_stats[lo..=hi].iter().any(|row| row.4[ui] != 0);
                active += usize::from(hit);
            }
            let cov = active as f32 / sample_count.max(1) as f32;
            let min_smooth = row_stats[ry.saturating_sub(1).max(1)]
                .2
                .min(row_stats[ry].2)
                .min(row_stats[(ry + 1).min(span_h - 2)].2);
            let raw = row_stats[ry].0 + 0.7 * row_stats[ry].1;
            let seam_score = (raw * cov.powf(2.4)) / (1.0 + min_smooth * 0.12);

            let mut gutter_score = 0.0f32;
            if bg_uniform && row_stats[ry].2 < 2.5 && row_stats[ry].3 > 0.95 {
                let mut above = false;
                let mut below = false;
                let mut d = 3usize;
                while d < ry {
                    if row_stats[ry - d].3 < 0.90 {
                        above = true;
                        break;
                    }
                    d += 3;
                }
                let mut d = 3usize;
                while ry + d < span_h {
                    if row_stats[ry + d].3 < 0.90 {
                        below = true;
                        break;
                    }
                    d += 3;
                }
                if above && below {
                    gutter_score = 110.0 / (1.0 + row_stats[ry].2);
                }
            }

            SeamCandidate {
                pos: y0 + ry as u32,
                score: seam_score.max(gutter_score),
                coverage: cov,
                is_gutter: gutter_score > seam_score,
            }
        })
        .collect();

    collapse_gutter_runs(cands)
}

fn profile_vertical_seams(
    buf: &[u8],
    w: u32,
    ch: usize,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    min_dim: u32,
    bg_rgb: [u8; 4],
    bg_uniform: bool,
    is_trans: bool,
) -> Vec<SeamCandidate> {
    let span_w = (x1 - x0) as usize;
    let span_h = y1 - y0;
    if (x1 - x0) <= min_dim * 2 || span_h < 8 {
        return Vec::new();
    }

    let stride = (w as usize) * ch;
    let step_y = (span_h / 200).max(1);
    let ys: Vec<u32> = (y0..y1).step_by(step_y as usize).collect();
    let sample_count = ys.len().max(1);

    // Cache-friendly row-major parallel reduction: each Rayon chunk scans full contiguous horizontal scanlines!
    let (col_step, col_line, col_smooth, col_bg, col_active) = ys
        .par_iter()
        .fold(
            || {
                (
                    vec![0.0f32; span_w],
                    vec![0.0f32; span_w],
                    vec![0.0f32; span_w],
                    vec![0u32; span_w],
                    vec![0u16; span_w],
                )
            },
            |mut acc, &y| {
                let mut last_marked: isize = -1;
                let y_prev = y.saturating_sub(step_y);
                for rx in 1..span_w - 1 {
                    let x = x0 + rx as u32;
                    let p0 = pixel_at(buf, stride, ch, x - 1, y);
                    let p1 = pixel_at(buf, stride, ch, x, y);
                    let p2 = pixel_at(buf, stride, ch, x + 1, y);
                    let ds = color_dist(p1, p2) as f32;
                    let dl = ((2 * p1[0] as i32 - p0[0] as i32 - p2[0] as i32).abs()
                        + (2 * p1[1] as i32 - p0[1] as i32 - p2[1] as i32).abs()
                        + (2 * p1[2] as i32 - p0[2] as i32 - p2[2] as i32).abs()) as f32
                        * 0.5;
                    acc.0[rx] += ds;
                    acc.1[rx] += dl;

                    if ds + dl > 16.0 {
                        let lo = (rx.saturating_sub(2)).max(1) as isize;
                        let hi = (rx + 2).min(span_w - 2) as isize;
                        let start = (last_marked + 1).max(lo);
                        for col in start..=hi {
                            acc.4[col as usize] += 1;
                        }
                        if hi > last_marked {
                            last_marked = hi;
                        }
                    }

                    if y > y0 {
                        let pp = pixel_at(buf, stride, ch, x, y_prev);
                        let luma = p1[0] as i32 + p1[1] as i32 + p1[2] as i32;
                        let prev_luma = pp[0] as i32 + pp[1] as i32 + pp[2] as i32;
                        acc.2[rx] += (luma - prev_luma).abs() as f32;
                    }

                    if bg_uniform {
                        let is_bg = if is_trans { p1[3] < 24 } else { color_dist(p1, bg_rgb) < 22 };
                        if is_bg {
                            acc.3[rx] += 1;
                        }
                    }
                }
                acc
            },
        )
        .reduce(
            || {
                (
                    vec![0.0f32; span_w],
                    vec![0.0f32; span_w],
                    vec![0.0f32; span_w],
                    vec![0u32; span_w],
                    vec![0u16; span_w],
                )
            },
            |mut a, b| {
                for i in 1..span_w - 1 {
                    a.0[i] += b.0[i];
                    a.1[i] += b.1[i];
                    a.2[i] += b.2[i];
                    a.3[i] += b.3[i];
                    a.4[i] += b.4[i];
                }
                a
            },
        );

    let sc = sample_count as f32;
    let sm_denom = sample_count.saturating_sub(1).max(1) as f32;
    let bg_frac: Vec<f32> = col_bg.iter().map(|&c| c as f32 / sc).collect();
    let sm_norm: Vec<f32> = col_smooth.iter().map(|&s| s / sm_denom).collect();

    let start_rx = min_dim as usize;
    let end_rx = span_w.saturating_sub(min_dim as usize);

    let cands: Vec<SeamCandidate> = (start_rx..=end_rx)
        .map(|rx| {
            let cov = col_active[rx] as f32 / sc;
            let min_smooth = sm_norm[rx.saturating_sub(1).max(1)]
                .min(sm_norm[rx])
                .min(sm_norm[(rx + 1).min(span_w - 2)]);
            let raw = (col_step[rx] / sc) + 0.7 * (col_line[rx] / sc);
            let seam_score = (raw * cov.powf(2.4)) / (1.0 + min_smooth * 0.12);

            let mut gutter_score = 0.0f32;
            if bg_uniform && sm_norm[rx] < 2.5 && bg_frac[rx] > 0.95 {
                let mut left = false;
                let mut right = false;
                let mut d = 3usize;
                while d < rx {
                    if bg_frac[rx - d] < 0.90 {
                        left = true;
                        break;
                    }
                    d += 3;
                }
                let mut d = 3usize;
                while rx + d < span_w {
                    if bg_frac[rx + d] < 0.90 {
                        right = true;
                        break;
                    }
                    d += 3;
                }
                if left && right {
                    gutter_score = 110.0 / (1.0 + sm_norm[rx]);
                }
            }

            SeamCandidate {
                pos: x0 + rx as u32,
                score: seam_score.max(gutter_score),
                coverage: cov,
                is_gutter: gutter_score > seam_score,
            }
        })
        .collect();

    collapse_gutter_runs(cands)
}

fn pick_peaks_nms(cands: &[SeamCandidate], threshold: f32, min_gap: u32) -> Vec<SeamCandidate> {
    let mut filtered: Vec<SeamCandidate> = cands
        .iter()
        .copied()
        .filter(|c| c.score >= threshold && (c.is_gutter || c.coverage >= 0.72))
        .collect();
    filtered.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let mut picked: Vec<SeamCandidate> = Vec::new();
    for cand in filtered {
        if picked.iter().all(|p| p.pos.abs_diff(cand.pos) >= min_gap) {
            picked.push(cand);
        }
    }
    picked.sort_by_key(|c| c.pos);
    picked
}

fn is_region_empty_bg(
    buf: &[u8],
    w: u32,
    ch: usize,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    bg_rgb: [u8; 4],
    is_trans: bool,
) -> bool {
    let stride = (w as usize) * ch;
    let step_x = ((x1 - x0) / 32).max(1);
    let step_y = ((y1 - y0) / 32).max(1);
    let (mut total, mut fg) = (0u32, 0u32);
    let mut y = y0 + 2;
    while y < y1.saturating_sub(2) {
        let mut x = x0 + 2;
        while x < x1.saturating_sub(2) {
            let p = pixel_at(buf, stride, ch, x, y);
            total += 1;
            if if is_trans { p[3] >= 24 } else { color_dist(p, bg_rgb) > 20 } {
                fg += 1;
            }
            x += step_x;
        }
        y += step_y;
    }
    total > 0 && (fg as f32 / total as f32) < 0.03
}

fn recurse_xy(
    buf: &[u8],
    w: u32,
    ch: usize,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    depth: u8,
    min_dim: u32,
    threshold: f32,
    bg_rgb: [u8; 4],
    bg_uniform: bool,
    is_trans: bool,
    parent_conf: f32,
    parent_gutter: bool,
    leaves: &mut Vec<RawRegion>,
) {
    let rw = x1 - x0;
    let rh = y1 - y0;
    if depth >= 4 || (rw < min_dim * 2 && rh < min_dim * 2) {
        leaves.push(RawRegion {
            x0,
            y0,
            x1,
            y1,
            conf: parent_conf,
            is_gutter: parent_gutter,
        });
        return;
    }

    let h_cands = profile_horizontal_seams(buf, w, ch, x0, y0, x1, y1, min_dim, bg_rgb, bg_uniform, is_trans);
    let v_cands = profile_vertical_seams(buf, w, ch, x0, y0, x1, y1, min_dim, bg_rgb, bg_uniform, is_trans);
    let h_peaks = pick_peaks_nms(&h_cands, threshold, min_dim);
    let v_peaks = pick_peaks_nms(&v_cands, threshold, min_dim);

    if h_peaks.is_empty() && v_peaks.is_empty() {
        leaves.push(RawRegion {
            x0,
            y0,
            x1,
            y1,
            conf: parent_conf,
            is_gutter: parent_gutter,
        });
        return;
    }

    let max_h = h_peaks.iter().map(|p| p.score).fold(0.0f32, f32::max);
    let max_v = v_peaks.iter().map(|p| p.score).fold(0.0f32, f32::max);
    let choose_h = !h_peaks.is_empty() && (v_peaks.is_empty() || max_h >= max_v * 0.72);

    if choose_h {
        let mut bounds = vec![y0];
        bounds.extend(h_peaks.iter().map(|p| p.pos));
        bounds.push(y1);
        for i in 0..bounds.len() - 1 {
            let p_idx = if i == 0 { 0 } else { i - 1 };
            recurse_xy(
                buf,
                w,
                ch,
                x0,
                bounds[i],
                x1,
                bounds[i + 1],
                depth + 1,
                min_dim,
                threshold,
                bg_rgb,
                bg_uniform,
                is_trans,
                parent_conf.max(h_peaks[p_idx].score),
                parent_gutter || h_peaks[p_idx].is_gutter,
                leaves,
            );
        }
    } else {
        let mut bounds = vec![x0];
        bounds.extend(v_peaks.iter().map(|p| p.pos));
        bounds.push(x1);
        for i in 0..bounds.len() - 1 {
            let p_idx = if i == 0 { 0 } else { i - 1 };
            recurse_xy(
                buf,
                w,
                ch,
                bounds[i],
                y0,
                bounds[i + 1],
                y1,
                depth + 1,
                min_dim,
                threshold,
                bg_rgb,
                bg_uniform,
                is_trans,
                parent_conf.max(v_peaks[p_idx].score),
                parent_gutter || v_peaks[p_idx].is_gutter,
                leaves,
            );
        }
    }
}

fn refine_and_trim(
    buf: &[u8],
    w: u32,
    h: u32,
    ch: usize,
    raw: &RawRegion,
    bleed_trim: u32,
    bg_rgb: [u8; 4],
    bg_uniform: bool,
    is_trans: bool,
) -> (u32, u32, u32, u32, u32, u32, u32, u32) {
    let stride = (w as usize) * ch;
    let (mut x0, mut y0, mut x1, mut y1) = (raw.x0, raw.y0, raw.x1, raw.y1);

    if !raw.is_gutter {
        if y0 > 2 {
            let mut best_y = y0;
            let mut best_e = -1.0f32;
            let step = ((x1 - x0) / 120).max(1);
            for y in y0.saturating_sub(8).max(1)..=y0.saturating_add(8).min(y1.saturating_sub(20)) {
                let (mut e, mut n) = (0.0f32, 0u32);
                let mut x = x0 + 4;
                while x < x1.saturating_sub(4) {
                    let p0 = pixel_at(buf, stride, ch, x, y - 1);
                    let p1 = pixel_at(buf, stride, ch, x, y);
                    let p2 = pixel_at(buf, stride, ch, x, y + 1);
                    e += color_dist(p1, p2) as f32
                        + 0.5
                            * ((2 * p1[0] as i32 - p0[0] as i32 - p2[0] as i32).abs()
                                + (2 * p1[1] as i32 - p0[1] as i32 - p2[1] as i32).abs()
                                + (2 * p1[2] as i32 - p0[2] as i32 - p2[2] as i32).abs()) as f32;
                    n += 1;
                    x += step;
                }
                e /= n.max(1) as f32;
                if e > best_e {
                    best_e = e;
                    best_y = y;
                }
            }
            if best_e > 15.0 {
                y0 = best_y + 1;
            }
        }
        if y1 < h - 2 {
            let mut best_y = y1;
            let mut best_e = -1.0f32;
            let step = ((x1 - x0) / 120).max(1);
            for y in y1.saturating_sub(8).max(y0 + 20)..=y1.saturating_add(8).min(h - 2) {
                let (mut e, mut n) = (0.0f32, 0u32);
                let mut x = x0 + 4;
                while x < x1.saturating_sub(4) {
                    let p0 = pixel_at(buf, stride, ch, x, y - 1);
                    let p1 = pixel_at(buf, stride, ch, x, y);
                    let p2 = pixel_at(buf, stride, ch, x, y + 1);
                    e += color_dist(p0, p1) as f32
                        + 0.5
                            * ((2 * p1[0] as i32 - p0[0] as i32 - p2[0] as i32).abs()
                                + (2 * p1[1] as i32 - p0[1] as i32 - p2[1] as i32).abs()
                                + (2 * p1[2] as i32 - p0[2] as i32 - p2[2] as i32).abs()) as f32;
                    n += 1;
                    x += step;
                }
                e /= n.max(1) as f32;
                if e > best_e {
                    best_e = e;
                    best_y = y;
                }
            }
            if best_e > 15.0 {
                y1 = best_y;
            }
        }
        if x0 > 2 {
            let mut best_x = x0;
            let mut best_e = -1.0f32;
            let step = ((y1 - y0) / 120).max(1);
            for x in x0.saturating_sub(8).max(1)..=x0.saturating_add(8).min(x1.saturating_sub(20)) {
                let (mut e, mut n) = (0.0f32, 0u32);
                let mut y = y0 + 4;
                while y < y1.saturating_sub(4) {
                    let p0 = pixel_at(buf, stride, ch, x - 1, y);
                    let p1 = pixel_at(buf, stride, ch, x, y);
                    let p2 = pixel_at(buf, stride, ch, x + 1, y);
                    e += color_dist(p1, p2) as f32
                        + 0.5
                            * ((2 * p1[0] as i32 - p0[0] as i32 - p2[0] as i32).abs()
                                + (2 * p1[1] as i32 - p0[1] as i32 - p2[1] as i32).abs()
                                + (2 * p1[2] as i32 - p0[2] as i32 - p2[2] as i32).abs()) as f32;
                    n += 1;
                    y += step;
                }
                e /= n.max(1) as f32;
                if e > best_e {
                    best_e = e;
                    best_x = x;
                }
            }
            if best_e > 15.0 {
                x0 = best_x + 1;
            }
        }
        if x1 < w - 2 {
            let mut best_x = x1;
            let mut best_e = -1.0f32;
            let step = ((y1 - y0) / 120).max(1);
            for x in x1.saturating_sub(8).max(x0 + 20)..=x1.saturating_add(8).min(w - 2) {
                let (mut e, mut n) = (0.0f32, 0u32);
                let mut y = y0 + 4;
                while y < y1.saturating_sub(4) {
                    let p0 = pixel_at(buf, stride, ch, x - 1, y);
                    let p1 = pixel_at(buf, stride, ch, x, y);
                    let p2 = pixel_at(buf, stride, ch, x + 1, y);
                    e += color_dist(p0, p1) as f32
                        + 0.5
                            * ((2 * p1[0] as i32 - p0[0] as i32 - p2[0] as i32).abs()
                                + (2 * p1[1] as i32 - p0[1] as i32 - p2[1] as i32).abs()
                                + (2 * p1[2] as i32 - p0[2] as i32 - p2[2] as i32).abs()) as f32;
                    n += 1;
                    y += step;
                }
                e /= n.max(1) as f32;
                if e > best_e {
                    best_e = e;
                    best_x = x;
                }
            }
            if best_e > 15.0 {
                x1 = best_x;
            }
        }
    }

    let (mut tt, mut tr, mut tb, mut tl) = (0u32, 0u32, 0u32, 0u32);
    if bg_uniform {
        let is_row_bg = |y: u32, xa: u32, xb: u32| -> bool {
            let step = ((xb - xa) / 60).max(1);
            let (mut hits, mut total) = (0u32, 0u32);
            let mut x = xa;
            while x < xb {
                let p = pixel_at(buf, stride, ch, x, y);
                if if is_trans { p[3] < 24 } else { color_dist(p, bg_rgb) < 22 } {
                    hits += 1;
                }
                total += 1;
                x += step;
            }
            (hits as f32 / total.max(1) as f32) > 0.94
        };
        let is_col_bg = |x: u32, ya: u32, yb: u32| -> bool {
            let step = ((yb - ya) / 60).max(1);
            let (mut hits, mut total) = (0u32, 0u32);
            let mut y = ya;
            while y < yb {
                let p = pixel_at(buf, stride, ch, x, y);
                if if is_trans { p[3] < 24 } else { color_dist(p, bg_rgb) < 22 } {
                    hits += 1;
                }
                total += 1;
                y += step;
            }
            (hits as f32 / total.max(1) as f32) > 0.94
        };
        while y0 < y1.saturating_sub(24) && is_row_bg(y0, x0, x1) {
            y0 += 1;
            tt += 1;
        }
        while y1 > y0 + 24 && is_row_bg(y1 - 1, x0, x1) {
            y1 -= 1;
            tb += 1;
        }
        while x0 < x1.saturating_sub(24) && is_col_bg(x0, y0, y1) {
            x0 += 1;
            tl += 1;
        }
        while x1 > x0 + 24 && is_col_bg(x1 - 1, y0, y1) {
            x1 -= 1;
            tr += 1;
        }
    }

    if bleed_trim > 0 && y1 - y0 > 32 && x1 - x0 > 32 {
        let row_diff = |y: u32, ry: u32, xa: u32, xb: u32| -> f32 {
            let step = ((xb - xa) / 80).max(1);
            let (mut d, mut n) = (0u32, 0u32);
            let mut x = xa + 8;
            while x < xb.saturating_sub(8) {
                d += color_dist(pixel_at(buf, stride, ch, x, y), pixel_at(buf, stride, ch, x, ry)) as u32;
                n += 1;
                x += step;
            }
            d as f32 / n.max(1) as f32
        };
        let col_diff = |x: u32, rx: u32, ya: u32, yb: u32| -> f32 {
            let step = ((yb - ya) / 80).max(1);
            let (mut d, mut n) = (0u32, 0u32);
            let mut y = ya + 8;
            while y < yb.saturating_sub(8) {
                d += color_dist(pixel_at(buf, stride, ch, x, y), pixel_at(buf, stride, ch, rx, y)) as u32;
                n += 1;
                y += step;
            }
            d as f32 / n.max(1) as f32
        };

        if raw.y0 > 2 {
            let ref_y = (y0 + bleed_trim + 2).min(y1 - 1);
            for _ in 0..bleed_trim {
                if row_diff(y0, ref_y, x0, x1) > 20.0 {
                    y0 += 1;
                    tt += 1;
                } else {
                    break;
                }
            }
        }
        if raw.y1 < h - 2 {
            let ref_y = y1.saturating_sub(1 + bleed_trim + 2).max(y0);
            for _ in 0..bleed_trim {
                if row_diff(y1 - 1, ref_y, x0, x1) > 20.0 {
                    y1 -= 1;
                    tb += 1;
                } else {
                    break;
                }
            }
        }
        if raw.x0 > 2 {
            let ref_x = (x0 + bleed_trim + 2).min(x1 - 1);
            for _ in 0..bleed_trim {
                if col_diff(x0, ref_x, y0, y1) > 20.0 {
                    x0 += 1;
                    tl += 1;
                } else {
                    break;
                }
            }
        }
        if raw.x1 < w - 2 {
            let ref_x = x1.saturating_sub(1 + bleed_trim + 2).max(x0);
            for _ in 0..bleed_trim {
                if col_diff(x1 - 1, ref_x, y0, y1) > 20.0 {
                    x1 -= 1;
                    tr += 1;
                } else {
                    break;
                }
            }
        }
    }

    (x0, y0, x1, y1, tt, tr, tb, tl)
}

fn xy_cut_core(
    slice: &[u8],
    w: u32,
    h: u32,
    ch: usize,
    sensitivity: f32,
    min_region_ratio: f32,
    bleed_trim: u32,
    bg_rgb: [u8; 4],
    bg_uniform: bool,
    is_trans: bool,
) -> Vec<SplitBox> {
    let min_dim = ((w.min(h) as f32) * min_region_ratio.clamp(0.04, 0.35)).round().max(24.0) as u32;
    let threshold = 125.0 - sensitivity.clamp(0.1, 1.0) * 80.0;

    let mut leaves = Vec::new();
    recurse_xy(
        slice,
        w,
        ch,
        0,
        0,
        w,
        h,
        0,
        min_dim,
        threshold,
        bg_rgb,
        bg_uniform,
        is_trans,
        120.0,
        false,
        &mut leaves,
    );

    // Sibling Grid-Line Propagation for faint same-background adjacent panels
    let stride = (w as usize) * ch;
    let mut prop_iter = 0;
    let mut propagated = true;
    while propagated && prop_iter < 3 {
        propagated = false;
        prop_iter += 1;
        let mut h_lines: Vec<u32> = leaves
            .iter()
            .flat_map(|l| [l.y0, l.y1])
            .filter(|&y| y > min_dim && y < h - min_dim)
            .collect();
        h_lines.sort_unstable();
        h_lines.dedup();
        let mut v_lines: Vec<u32> = leaves
            .iter()
            .flat_map(|l| [l.x0, l.x1])
            .filter(|&x| x > min_dim && x < w - min_dim)
            .collect();
        v_lines.sort_unstable();
        v_lines.dedup();

        for idx in 0..leaves.len() {
            let leaf = leaves[idx].clone();
            let mut split_done = false;
            for &y_cut in &h_lines {
                if y_cut >= leaf.y0 + min_dim && leaf.y1 >= y_cut + min_dim {
                    let (mut best_y, mut best_sc) = (0u32, 0.0f32);
                    let y_lo = y_cut.saturating_sub(12).max(leaf.y0 + min_dim);
                    let y_hi = (y_cut + 12).min(leaf.y1 - min_dim);
                    let step_x = ((leaf.x1 - leaf.x0) / 200).max(1);
                    for y in y_lo..=y_hi {
                        let (mut step_s, mut line_s, mut smooth_s, mut act, mut n) = (0.0f32, 0.0f32, 0.0f32, 0u32, 0u32);
                        let mut prev = [0u8; 4];
                        let mut x = leaf.x0 + 4;
                        while x < leaf.x1.saturating_sub(4) {
                            let p0 = pixel_at(slice, stride, ch, x, y - 1);
                            let p1 = pixel_at(slice, stride, ch, x, y);
                            let p2 = pixel_at(slice, stride, ch, x, y + 1);
                            let ds = color_dist(p1, p2) as f32;
                            let dl = ((2 * p1[0] as i32 - p0[0] as i32 - p2[0] as i32).abs()
                                + (2 * p1[1] as i32 - p0[1] as i32 - p2[1] as i32).abs()
                                + (2 * p1[2] as i32 - p0[2] as i32 - p2[2] as i32).abs()) as f32
                                * 0.5;
                            step_s += ds;
                            line_s += dl;
                            if ds + dl > 6.0 {
                                act += 1;
                            }
                            if n > 0 {
                                smooth_s += color_dist(p1, prev) as f32;
                            }
                            prev = p1;
                            n += 1;
                            x += step_x;
                        }
                        let cov = act as f32 / n.max(1) as f32;
                        let avg_raw = (step_s + line_s) / n.max(1) as f32;
                        let avg_sm = smooth_s / n.saturating_sub(1).max(1) as f32;
                        if cov >= 0.78 && avg_raw >= 10.0 && avg_sm < 2.2 {
                            let sc = (avg_raw * cov) / (1.0 + avg_sm * 0.2);
                            if sc > best_sc {
                                best_sc = sc;
                                best_y = y;
                            }
                        }
                    }
                    if best_y > 0 {
                        let mut a = leaf.clone();
                        a.y1 = best_y;
                        a.conf = a.conf.max(best_sc);
                        let mut b = leaf.clone();
                        b.y0 = best_y;
                        b.conf = b.conf.max(best_sc);
                        leaves.splice(idx..=idx, [a, b]);
                        propagated = true;
                        split_done = true;
                        break;
                    }
                }
            }
            if split_done {
                break;
            }
            for &x_cut in &v_lines {
                if x_cut >= leaf.x0 + min_dim && leaf.x1 >= x_cut + min_dim {
                    let (mut best_x, mut best_sc) = (0u32, 0.0f32);
                    let x_lo = x_cut.saturating_sub(12).max(leaf.x0 + min_dim);
                    let x_hi = (x_cut + 12).min(leaf.x1 - min_dim);
                    let step_y = ((leaf.y1 - leaf.y0) / 200).max(1);
                    for x in x_lo..=x_hi {
                        let (mut step_s, mut line_s, mut smooth_s, mut act, mut n) = (0.0f32, 0.0f32, 0.0f32, 0u32, 0u32);
                        let mut prev = [0u8; 4];
                        let mut y = leaf.y0 + 4;
                        while y < leaf.y1.saturating_sub(4) {
                            let p0 = pixel_at(slice, stride, ch, x - 1, y);
                            let p1 = pixel_at(slice, stride, ch, x, y);
                            let p2 = pixel_at(slice, stride, ch, x + 1, y);
                            let ds = color_dist(p1, p2) as f32;
                            let dl = ((2 * p1[0] as i32 - p0[0] as i32 - p2[0] as i32).abs()
                                + (2 * p1[1] as i32 - p0[1] as i32 - p2[1] as i32).abs()
                                + (2 * p1[2] as i32 - p0[2] as i32 - p2[2] as i32).abs()) as f32
                                * 0.5;
                            step_s += ds;
                            line_s += dl;
                            if ds + dl > 6.0 {
                                act += 1;
                            }
                            if n > 0 {
                                smooth_s += color_dist(p1, prev) as f32;
                            }
                            prev = p1;
                            n += 1;
                            y += step_y;
                        }
                        let cov = act as f32 / n.max(1) as f32;
                        let avg_raw = (step_s + line_s) / n.max(1) as f32;
                        let avg_sm = smooth_s / n.saturating_sub(1).max(1) as f32;
                        if cov >= 0.78 && avg_raw >= 10.0 && avg_sm < 2.2 {
                            let sc = (avg_raw * cov) / (1.0 + avg_sm * 0.2);
                            if sc > best_sc {
                                best_sc = sc;
                                best_x = x;
                            }
                        }
                    }
                    if best_x > 0 {
                        let mut a = leaf.clone();
                        a.x1 = best_x;
                        a.conf = a.conf.max(best_sc);
                        let mut b = leaf.clone();
                        b.x0 = best_x;
                        b.conf = b.conf.max(best_sc);
                        leaves.splice(idx..=idx, [a, b]);
                        propagated = true;
                        break;
                    }
                }
            }
            if propagated {
                break;
            }
        }
    }

    let non_empty: Vec<RawRegion> = if bg_uniform {
        leaves
            .iter()
            .filter(|l| !is_region_empty_bg(slice, w, ch, l.x0, l.y0, l.x1, l.y1, bg_rgb, is_trans))
            .cloned()
            .collect()
    } else {
        leaves.clone()
    };
    let mut final_leaves = if non_empty.is_empty() { leaves } else { non_empty };

    final_leaves.sort_by(|a, b| {
        let tol = ((a.y1 - a.y0).min(b.y1 - b.y0) as f32) * 0.35;
        if ((a.y0 as f32) - (b.y0 as f32)).abs() > tol {
            a.y0.cmp(&b.y0)
        } else {
            a.x0.cmp(&b.x0)
        }
    });

    let mut out = Vec::with_capacity(final_leaves.len());
    let (mut cur_r, mut cur_c) = (0usize, 0usize);
    let mut last_y = final_leaves.first().map(|l| l.y0).unwrap_or(0);
    for (idx, leaf) in final_leaves.iter().enumerate() {
        if idx > 0 && (leaf.y0 as i64 - last_y as i64).abs() > ((leaf.y1 - leaf.y0) as f32 * 0.35) as i64 {
            cur_r += 1;
            cur_c = 0;
            last_y = leaf.y0;
        } else if idx > 0 {
            cur_c += 1;
        }
        let (rx0, ry0, rx1, ry1, tt, tr, tb, tl) =
            refine_and_trim(slice, w, h, ch, leaf, bleed_trim, bg_rgb, bg_uniform, is_trans);
        out.push(SplitBox {
            index: idx + 1,
            row: cur_r,
            col: cur_c,
            x: rx0,
            y: ry0,
            width: rx1.saturating_sub(rx0).max(1),
            height: ry1.saturating_sub(ry0).max(1),
            confidence: (leaf.conf / 150.0).min(1.0),
            strategy: if leaf.is_gutter { "gutter".into() } else { "xycut".into() },
            trim_top: tt,
            trim_right: tr,
            trim_bottom: tb,
            trim_left: tl,
        });
    }
    out
}

fn dsu_find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let gp = parent[parent[x as usize] as usize];
        parent[x as usize] = gp;
        x = gp;
    }
    x
}

fn dsu_union(parent: &mut [u32], a: u32, b: u32) {
    let ra = dsu_find(parent, a);
    let rb = dsu_find(parent, b);
    if ra != rb {
        if ra < rb {
            parent[rb as usize] = ra;
        } else {
            parent[ra as usize] = rb;
        }
    }
}

fn contours_core(
    slice: &[u8],
    w: u32,
    h: u32,
    ch: usize,
    sensitivity: f32,
    min_region_ratio: f32,
    contour_gap_px: u32,
    bg_rgb: [u8; 4],
    is_trans: bool,
) -> Vec<SplitBox> {
    let stride = (w as usize) * ch;
    let scale = w.max(h).div_ceil(720).max(1);
    let sw = w.div_ceil(scale) as usize;
    let sh = h.div_ceil(scale) as usize;
    let color_thresh = (32.0 - sensitivity.clamp(0.1, 1.0) * 18.0) as i32;

    // Parallel downsampled foreground mask
    let mut mask = vec![0u8; sw * sh];
    mask.par_chunks_mut(sw).enumerate().for_each(|(sy, row)| {
        let y = ((sy as u32) * scale).min(h - 1);
        for (sx, cell) in row.iter_mut().enumerate().take(sw) {
            let x = ((sx as u32) * scale).min(w - 1);
            let p = pixel_at(slice, stride, ch, x, y);
            *cell = if is_trans {
                u8::from(ch >= 4 && p[3] > 24)
            } else {
                u8::from(color_dist(p, bg_rgb) > color_thresh)
            };
        }
    });

    // Separable O(sw * sh) morphological dilation
    let gap = ((contour_gap_px as f32) / (scale as f32)).round().max(1.0) as usize;
    let mut horiz = vec![0u8; sw * sh];
    horiz.par_chunks_mut(sw).enumerate().for_each(|(y, hrow)| {
        let mrow = &mask[y * sw..(y + 1) * sw];
        let mut rem = 0usize;
        for x in 0..sw {
            if mrow[x] != 0 {
                rem = gap + 1;
            }
            if rem > 0 {
                hrow[x] = 1;
                rem -= 1;
            }
        }
        rem = 0;
        for x in (0..sw).rev() {
            if mrow[x] != 0 {
                rem = gap + 1;
            }
            if rem > 0 {
                hrow[x] = 1;
                rem -= 1;
            }
        }
    });

    let mut dilated = vec![0u8; sw * sh];
    for x in 0..sw {
        let mut rem = 0usize;
        for y in 0..sh {
            if horiz[y * sw + x] != 0 {
                rem = gap + 1;
            }
            if rem > 0 {
                dilated[y * sw + x] = 1;
                rem -= 1;
            }
        }
        rem = 0;
        for y in (0..sh).rev() {
            if horiz[y * sw + x] != 0 {
                rem = gap + 1;
            }
            if rem > 0 {
                dilated[y * sw + x] = 1;
                rem -= 1;
            }
        }
    }

    // Two-pass 8-connected Union-Find labeling
    let mut labels = vec![0u32; sw * sh];
    let mut parent = vec![0u32]; // 1-based labels
    for y in 0..sh {
        for x in 0..sw {
            let idx = y * sw + x;
            if dilated[idx] == 0 {
                continue;
            }
            let mut nbrs = [0u32; 4];
            let mut nc = 0usize;
            if x > 0 && labels[idx - 1] > 0 {
                nbrs[nc] = labels[idx - 1];
                nc += 1;
            }
            if y > 0 {
                let up = (y - 1) * sw + x;
                if labels[up] > 0 {
                    nbrs[nc] = labels[up];
                    nc += 1;
                }
                if x > 0 && labels[up - 1] > 0 {
                    nbrs[nc] = labels[up - 1];
                    nc += 1;
                }
                if x + 1 < sw && labels[up + 1] > 0 {
                    nbrs[nc] = labels[up + 1];
                    nc += 1;
                }
            }
            if nc == 0 {
                let lbl = parent.len() as u32;
                parent.push(lbl);
                labels[idx] = lbl;
            } else {
                let mut best = nbrs[0];
                for &nbr in &nbrs[1..nc] {
                    if nbr < best {
                        best = nbr;
                    }
                }
                labels[idx] = best;
                for &nbr in &nbrs[..nc] {
                    dsu_union(&mut parent, best, nbr);
                }
            }
        }
    }

    // Aggregate foreground bounds per component
    struct CompStats {
        min_x: usize,
        min_y: usize,
        max_x: usize,
        max_y: usize,
        fg_count: usize,
    }
    let mut comps: HashMap<u32, CompStats> = HashMap::new();
    for y in 0..sh {
        for x in 0..sw {
            let idx = y * sw + x;
            let lbl = labels[idx];
            if lbl == 0 || mask[idx] == 0 {
                continue;
            }
            let root = dsu_find(&mut parent, lbl);
            let st = comps.entry(root).or_insert(CompStats {
                min_x: sw,
                min_y: sh,
                max_x: 0,
                max_y: 0,
                fg_count: 0,
            });
            st.fg_count += 1;
            if x < st.min_x {
                st.min_x = x;
            }
            if x > st.max_x {
                st.max_x = x;
            }
            if y < st.min_y {
                st.min_y = y;
            }
            if y > st.max_y {
                st.max_y = y;
            }
        }
    }

    let min_area = ((sw * sh) as f32 * min_region_ratio.clamp(0.02, 0.3).powi(2) * 0.45).max(16.0) as usize;
    let is_fg = |x: u32, y: u32| -> bool {
        let p = pixel_at(slice, stride, ch, x, y);
        if is_trans {
            ch >= 4 && p[3] > 24
        } else {
            color_dist(p, bg_rgb) > color_thresh
        }
    };
    let row_has_fg = |y: u32, xa: u32, xb: u32| -> bool {
        let step = ((xb - xa) / 160).max(1);
        let mut x = xa;
        while x < xb {
            if is_fg(x, y) {
                return true;
            }
            x += step;
        }
        false
    };
    let col_has_fg = |x: u32, ya: u32, yb: u32| -> bool {
        let step = ((yb - ya) / 160).max(1);
        let mut y = ya;
        while y < yb {
            if is_fg(x, y) {
                return true;
            }
            y += step;
        }
        false
    };

    let mut raw_boxes: Vec<(u32, u32, u32, u32)> = Vec::new();
    for st in comps.into_values() {
        if st.fg_count < min_area || st.max_x < st.min_x || st.max_y < st.min_y {
            continue;
        }
        let mut x0 = ((st.min_x as u32) * scale).saturating_sub(scale);
        let mut y0 = ((st.min_y as u32) * scale).saturating_sub(scale);
        let mut x1 = (((st.max_x as u32) + 1) * scale + scale).min(w);
        let mut y1 = (((st.max_y as u32) + 1) * scale + scale).min(h);

        while y0 + 8 < y1 && !row_has_fg(y0, x0, x1) {
            y0 += 1;
        }
        while y1 > y0 + 8 && !row_has_fg(y1 - 1, x0, x1) {
            y1 -= 1;
        }
        while x0 + 8 < x1 && !col_has_fg(x0, y0, y1) {
            x0 += 1;
        }
        while x1 > x0 + 8 && !col_has_fg(x1 - 1, y0, y1) {
            x1 -= 1;
        }
        if y0 > 0 && row_has_fg(y0 - 1, x0, x1) {
            y0 -= 1;
        }
        if y1 < h && row_has_fg(y1, x0, x1) {
            y1 += 1;
        }
        if x0 > 0 && col_has_fg(x0 - 1, y0, y1) {
            x0 -= 1;
        }
        if x1 < w && col_has_fg(x1, y0, y1) {
            x1 += 1;
        }
        raw_boxes.push((x0, y0, x1, y1));
    }

    raw_boxes.sort_by(|a, b| {
        let tol = ((a.3 - a.1).min(b.3 - b.1) as f32) * 0.4;
        if ((a.1 as f32) - (b.1 as f32)).abs() > tol {
            a.1.cmp(&b.1)
        } else {
            a.0.cmp(&b.0)
        }
    });

    raw_boxes
        .into_iter()
        .enumerate()
        .map(|(idx, (x0, y0, x1, y1))| SplitBox {
            index: idx + 1,
            row: idx,
            col: 0,
            x: x0,
            y: y0,
            width: x1.saturating_sub(x0).max(1),
            height: y1.saturating_sub(y0).max(1),
            confidence: 0.98,
            strategy: "contour".into(),
            trim_top: 0,
            trim_right: 0,
            trim_bottom: 0,
            trim_left: 0,
        })
        .collect()
}

fn uniform_grid_core(w: u32, h: u32, rows: u32, cols: u32, gutter_px: u32) -> Vec<SplitBox> {
    let r_cnt = rows.clamp(1, 20);
    let c_cnt = cols.clamp(1, 20);
    let half_g = gutter_px / 2;
    let mut boxes = Vec::with_capacity((r_cnt * c_cnt) as usize);
    let mut idx = 1usize;
    for r in 0..r_cnt {
        let y0 = ((r as f32 * h as f32) / r_cnt as f32).round() as u32 + if r > 0 { half_g } else { 0 };
        let y1 = (((r + 1) as f32 * h as f32) / r_cnt as f32).round() as u32
            - if r < r_cnt - 1 { gutter_px - half_g } else { 0 };
        for c in 0..c_cnt {
            let x0 = ((c as f32 * w as f32) / c_cnt as f32).round() as u32 + if c > 0 { half_g } else { 0 };
            let x1 = (((c + 1) as f32 * w as f32) / c_cnt as f32).round() as u32
                - if c < c_cnt - 1 { gutter_px - half_g } else { 0 };
            boxes.push(SplitBox {
                index: idx,
                row: r as usize,
                col: c as usize,
                x: x0,
                y: y0,
                width: x1.saturating_sub(x0).max(1),
                height: y1.saturating_sub(y0).max(1),
                confidence: 1.0,
                strategy: "grid".into(),
                trim_top: 0,
                trim_right: 0,
                trim_bottom: 0,
                trim_left: 0,
            });
            idx += 1;
        }
    }
    boxes
}

fn borrow_image_bytes<'a>(image: &'a PyReadonlyArray3<'_, u8>) -> PyResult<(u32, u32, usize, Cow<'a, [u8]>)> {
    let view = image.as_array();
    let shape = view.shape();
    if shape.len() != 3 || (shape[2] != 3 && shape[2] != 4) {
        return Err(PyValueError::new_err("Expected HxWx3 or HxWx4 uint8 numpy array"));
    }
    let h = shape[0] as u32;
    let w = shape[1] as u32;
    let ch = shape[2];
    let buf: Cow<'a, [u8]> = if let Some(s) = view.to_slice() {
        Cow::Borrowed(s)
    } else {
        Cow::Owned(view.iter().copied().collect())
    };
    Ok((w, h, ch, buf))
}

#[pyfunction]
#[pyo3(signature = (image, sensitivity=0.65, min_region_ratio=0.08, bleed_trim=2))]
pub fn detect_recursive_xy_cut_rs(
    py: Python<'_>,
    image: PyReadonlyArray3<'_, u8>,
    sensitivity: f32,
    min_region_ratio: f32,
    bleed_trim: u32,
) -> PyResult<Vec<SplitBox>> {
    let (w, h, ch, buf) = borrow_image_bytes(&image)?;
    let slice: &[u8] = &buf;
    let boxes = py.allow_threads(|| {
        let (bg_rgb, is_trans, bg_unif) = estimate_border_bg(slice, w, h, ch);
        xy_cut_core(slice, w, h, ch, sensitivity, min_region_ratio, bleed_trim, bg_rgb, bg_unif > 0.72, is_trans)
    });
    Ok(boxes)
}

#[pyfunction]
#[pyo3(signature = (image, sensitivity=0.65, min_region_ratio=0.08, contour_gap_px=8))]
pub fn detect_contours_rs(
    py: Python<'_>,
    image: PyReadonlyArray3<'_, u8>,
    sensitivity: f32,
    min_region_ratio: f32,
    contour_gap_px: u32,
) -> PyResult<Vec<SplitBox>> {
    let (w, h, ch, buf) = borrow_image_bytes(&image)?;
    let slice: &[u8] = &buf;
    let boxes = py.allow_threads(|| {
        let (bg_rgb, is_trans, _) = estimate_border_bg(slice, w, h, ch);
        contours_core(slice, w, h, ch, sensitivity, min_region_ratio, contour_gap_px, bg_rgb, is_trans)
    });
    Ok(boxes)
}

#[pyfunction]
#[pyo3(signature = (image, mode="auto", sensitivity=0.65, min_region_ratio=0.08, bleed_trim=2, contour_gap_px=8, rows=3, cols=3, gutter_px=0))]
pub fn detect_auto_rs(
    py: Python<'_>,
    image: PyReadonlyArray3<'_, u8>,
    mode: &str,
    sensitivity: f32,
    min_region_ratio: f32,
    bleed_trim: u32,
    contour_gap_px: u32,
    rows: u32,
    cols: u32,
    gutter_px: u32,
) -> PyResult<(String, Vec<SplitBox>)> {
    let (w, h, ch, buf) = borrow_image_bytes(&image)?;
    let slice: &[u8] = &buf;
    let mode_lower = mode.to_lowercase();

    let result = py.allow_threads(|| -> (String, Vec<SplitBox>) {
        if mode_lower == "grid" {
            return ("grid".into(), uniform_grid_core(w, h, rows, cols, gutter_px));
        }
        let (bg_rgb, is_trans, bg_unif) = estimate_border_bg(slice, w, h, ch);
        if mode_lower == "contour" {
            return (
                "contour".into(),
                contours_core(slice, w, h, ch, sensitivity, min_region_ratio, contour_gap_px, bg_rgb, is_trans),
            );
        }

        let xy_boxes = xy_cut_core(
            slice,
            w,
            h,
            ch,
            sensitivity,
            min_region_ratio,
            bleed_trim,
            bg_rgb,
            bg_unif > 0.72,
            is_trans,
        );
        if mode_lower == "xycut" {
            return ("xycut".into(), xy_boxes);
        }

        if bg_unif > 0.72 {
            let ct_boxes = contours_core(
                slice,
                w,
                h,
                ch,
                sensitivity,
                min_region_ratio,
                contour_gap_px,
                bg_rgb,
                is_trans,
            );
            let ct_area: u64 = ct_boxes.iter().map(|b| (b.width as u64) * (b.height as u64)).sum();
            let total_area = (w as u64) * (h as u64);
            if ct_boxes.len() >= 2
                && (xy_boxes.len() <= 1 || ((ct_area * 100) < (total_area * 65) && ct_boxes.len() >= xy_boxes.len()))
            {
                return ("contour".into(), ct_boxes);
            }
        }
        ("xycut".into(), xy_boxes)
    });

    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (image_path, boxes, output_dir, prefix="slice", fmt="png"))]
pub fn batch_crop_and_save_rs(
    py: Python<'_>,
    image_path: &str,
    boxes: Vec<(u32, u32, u32, u32)>,
    output_dir: &str,
    prefix: &str,
    fmt: &str,
) -> PyResult<Vec<String>> {
    let img = image::open(image_path).map_err(|e| PyValueError::new_err(format!("Failed to open image: {e}")))?;
    let rgba = img.to_rgba8();
    std::fs::create_dir_all(output_dir).map_err(|e| PyValueError::new_err(format!("mkdir failed: {e}")))?;
    let ext = fmt.to_lowercase();

    let paths: Result<Vec<String>, String> = py.allow_threads(|| {
        boxes
            .into_par_iter()
            .enumerate()
            .map(|(i, (x, y, w, h))| {
                let cropped: ImageBuffer<Rgba<u8>, Vec<u8>> =
                    image::imageops::crop_imm(&rgba, x, y, w, h).to_image();
                let out_path = Path::new(output_dir).join(format!("{prefix}_{:02}.{ext}", i + 1));
                let dyn_img = DynamicImage::ImageRgba8(cropped);
                if ext == "jpg" || ext == "jpeg" {
                    dyn_img.to_rgb8().save(&out_path).map_err(|e| e.to_string())?;
                } else {
                    dyn_img.save(&out_path).map_err(|e| e.to_string())?;
                }
                Ok(out_path.to_string_lossy().into_owned())
            })
            .collect()
    });

    paths.map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (image, boxes, fmt="png"))]
pub fn batch_crop_to_memory_rs<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'_, u8>,
    boxes: Vec<(u32, u32, u32, u32)>,
    fmt: &str,
) -> PyResult<Vec<Bound<'py, PyBytes>>> {
    let (w, h, ch, buf) = borrow_image_bytes(&image)?;
    let slice: &[u8] = &buf;
    let ext = fmt.to_lowercase();
    let img_fmt = match ext.as_str() {
        "jpg" | "jpeg" => ImageFormat::Jpeg,
        "webp" => ImageFormat::WebP,
        _ => ImageFormat::Png,
    };

    let encoded: Result<Vec<Vec<u8>>, String> = py.allow_threads(|| {
        boxes
            .into_par_iter()
            .map(|(bx, by, bw, bh)| {
                let x0 = bx.min(w.saturating_sub(1));
                let y0 = by.min(h.saturating_sub(1));
                let cw = bw.max(1).min(w - x0);
                let ch_box = bh.max(1).min(h - y0);
                let mut out_rgba = Vec::with_capacity((cw as usize) * (ch_box as usize) * 4);
                let stride = (w as usize) * ch;
                for yy in y0..y0 + ch_box {
                    for xx in x0..x0 + cw {
                        let p = pixel_at(slice, stride, ch, xx, yy);
                        out_rgba.extend_from_slice(&p);
                    }
                }
                let ib = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(cw, ch_box, out_rgba)
                    .ok_or_else(|| "Buffer construction failed".to_string())?;
                let dyn_img = DynamicImage::ImageRgba8(ib);
                let mut cursor = Cursor::new(Vec::new());
                if img_fmt == ImageFormat::Jpeg {
                    dyn_img.to_rgb8().write_to(&mut cursor, ImageFormat::Jpeg).map_err(|e| e.to_string())?;
                } else {
                    dyn_img.write_to(&mut cursor, img_fmt).map_err(|e| e.to_string())?;
                }
                Ok(cursor.into_inner())
            })
            .collect()
    });

    let buffers = encoded.map_err(PyValueError::new_err)?;
    Ok(buffers.into_iter().map(|b| PyBytes::new_bound(py, &b)).collect())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SplitBox>()?;
    m.add_function(wrap_pyfunction!(detect_recursive_xy_cut_rs, m)?)?;
    m.add_function(wrap_pyfunction!(detect_contours_rs, m)?)?;
    m.add_function(wrap_pyfunction!(detect_auto_rs, m)?)?;
    m.add_function(wrap_pyfunction!(batch_crop_and_save_rs, m)?)?;
    m.add_function(wrap_pyfunction!(batch_crop_to_memory_rs, m)?)?;
    Ok(())
}
