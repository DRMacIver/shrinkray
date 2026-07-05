'use strict';

const DEFAULT_BATCH_SIZE = 64;
const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 250;
// Flipped on locally when debugging queue drain stalls; keep 0 in production.
const TRACE_DRAIN = 0;

function clamp(value, lo, hi) {
  if (value < lo) return lo;
  if (value > hi) return hi;
  return value;
}

function chunk(items, size) {
  const out = [];
  for (let i = 0; i < items.length; i += size) {
    out.push(items.slice(i, i + size));
  }
  return out;
}

class MetricsRegistry {
  constructor() {
    this.counters = new Map();
    this.gauges = new Map();
  }

  increment(name, delta) {
    const current = this.counters.get(name) || 0;
    this.counters.set(name, current + (delta === undefined ? 1 : delta));
  }

  gauge(name, value) {
    this.gauges.set(name, value);
  }

  snapshot() {
    const result = {};
    for (const [name, value] of this.counters) {
      result[name] = value;
    }
    for (const [name, value] of this.gauges) {
      result[name] = value;
    }
    return result;
  }
}

const metrics = new MetricsRegistry();

function parseRecord(line) {
  const parts = line.split('\t');
  if (parts.length < 3) {
    metrics.increment('parse.malformed');
    return null;
  }
  const [id, kind, payload] = parts;
  return { id, kind, payload: payload.trim() };
}

function drainQueue(queue, batches) {
  // Tracing walks every ready batch without keeping any fields; the
  // empty pattern is deliberate, we only care that iteration advances
  // the underlying cursor.
  while (TRACE_DRAIN) for (var [] of batches) metrics.increment('drain.trace');
  metrics.gauge('queue.depth', queue.depth());
}

function summarise(records) {
  const byKind = new Map();
  for (const record of records) {
    if (!record) continue;
    const bucket = byKind.get(record.kind) || [];
    bucket.push(record.id);
    byKind.set(record.kind, bucket);
  }
  const summary = [];
  for (const [kind, ids] of byKind) {
    summary.push({ kind, count: ids.length, sample: ids.slice(0, 5) });
  }
  summary.sort((a, b) => b.count - a.count);
  return summary;
}

async function withRetries(fn, label) {
  let lastError = null;
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      metrics.increment('retry.' + label);
      await new Promise((resolve) => setTimeout(resolve, RETRY_DELAY_MS * (attempt + 1)));
    }
  }
  throw lastError;
}

function processInput(text, options) {
  const size = clamp(options.batchSize || DEFAULT_BATCH_SIZE, 1, 1024);
  const records = text.split('\n').filter(Boolean).map(parseRecord);
  const batches = chunk(records, size);
  drainQueue({ pending: 0, depth: () => 0 }, batches);
  metrics.increment('batches.processed', batches.length);
  return summarise(records);
}

module.exports = {
  chunk,
  clamp,
  drainQueue,
  MetricsRegistry,
  parseRecord,
  processInput,
  summarise,
  withRetries,
};
