use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

use serde_json::Value;

pub struct RequestMetrics {
    started_at: Instant,
    pub total_requests: AtomicU64,
    pub stream_requests: AtomicU64,
    pub non_stream_requests: AtomicU64,
    pub completed: AtomicU64,
    pub failed: AtomicU64,
    pub cancelled: AtomicU64,
    pub input_tokens: AtomicU64,
    pub output_tokens: AtomicU64,
    pub cached_tokens: AtomicU64,
    pub reasoning_tokens: AtomicU64,
}

impl Default for RequestMetrics {
    fn default() -> Self {
        Self {
            started_at: Instant::now(),
            total_requests: AtomicU64::new(0),
            stream_requests: AtomicU64::new(0),
            non_stream_requests: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            cancelled: AtomicU64::new(0),
            input_tokens: AtomicU64::new(0),
            output_tokens: AtomicU64::new(0),
            cached_tokens: AtomicU64::new(0),
            reasoning_tokens: AtomicU64::new(0),
        }
    }
}

impl RequestMetrics {
    pub fn record_request(&self, stream: bool) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        if stream {
            self.stream_requests.fetch_add(1, Ordering::Relaxed);
        } else {
            self.non_stream_requests.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_completion(&self, usage: &Value) {
        self.completed.fetch_add(1, Ordering::Relaxed);
        self.add_usage(usage);
    }

    pub fn record_failure(&self) {
        self.failed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cancellation(&self) {
        self.cancelled.fetch_add(1, Ordering::Relaxed);
    }

    fn add_usage(&self, usage: &Value) {
        let input = usage
            .get("input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let output = usage
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let cached = usage
            .get("input_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let reasoning = usage
            .get("output_tokens_details")
            .and_then(|d| d.get("reasoning_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        self.input_tokens.fetch_add(input, Ordering::Relaxed);
        self.output_tokens.fetch_add(output, Ordering::Relaxed);
        self.cached_tokens.fetch_add(cached, Ordering::Relaxed);
        self.reasoning_tokens
            .fetch_add(reasoning, Ordering::Relaxed);
    }

    pub fn snapshot(&self, inflight: usize, stored: usize) -> Value {
        let uptime = self.started_at.elapsed().as_secs();
        serde_json::json!({
            "uptime_seconds": uptime,
            "requests": {
                "total": self.total_requests.load(Ordering::Relaxed),
                "stream": self.stream_requests.load(Ordering::Relaxed),
                "non_stream": self.non_stream_requests.load(Ordering::Relaxed),
                "completed": self.completed.load(Ordering::Relaxed),
                "failed": self.failed.load(Ordering::Relaxed),
                "cancelled": self.cancelled.load(Ordering::Relaxed),
                "inflight": inflight,
            },
            "tokens": {
                "input": self.input_tokens.load(Ordering::Relaxed),
                "output": self.output_tokens.load(Ordering::Relaxed),
                "cached": self.cached_tokens.load(Ordering::Relaxed),
                "reasoning": self.reasoning_tokens.load(Ordering::Relaxed),
            },
            "stored_responses": stored,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_snapshot() {
        let metrics = RequestMetrics::default();
        metrics.record_request(false);
        metrics.record_request(true);
        metrics.record_request(true);
        metrics.record_completion(&serde_json::json!({
            "input_tokens": 100,
            "output_tokens": 50,
            "input_tokens_details": { "cached_tokens": 20 },
            "output_tokens_details": { "reasoning_tokens": 10 },
        }));
        metrics.record_failure();

        let snap = metrics.snapshot(1, 3);
        assert_eq!(snap["requests"]["total"], 3);
        assert_eq!(snap["requests"]["stream"], 2);
        assert_eq!(snap["requests"]["non_stream"], 1);
        assert_eq!(snap["requests"]["completed"], 1);
        assert_eq!(snap["requests"]["failed"], 1);
        assert_eq!(snap["requests"]["cancelled"], 0);
        assert_eq!(snap["requests"]["inflight"], 1);
        assert_eq!(snap["tokens"]["input"], 100);
        assert_eq!(snap["tokens"]["output"], 50);
        assert_eq!(snap["tokens"]["cached"], 20);
        assert_eq!(snap["tokens"]["reasoning"], 10);
        assert_eq!(snap["stored_responses"], 3);
    }

    #[test]
    fn empty_usage_snapshot() {
        let metrics = RequestMetrics::default();
        metrics.record_completion(&serde_json::json!({}));
        let snap = metrics.snapshot(0, 0);
        assert_eq!(snap["tokens"]["input"], 0);
        assert_eq!(snap["tokens"]["output"], 0);
    }
}
