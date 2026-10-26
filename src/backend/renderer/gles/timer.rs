//! Passive GPU timestamp queries over `GL_EXT_disjoint_timer_query`.
//!
//! A safe surface for compositor-side frame-segment timing: every raw
//! binding call is `unsafe`, so they stay inside this module. The pool
//! hands out monotonically tokened timestamp queries; results are
//! collected non-blockingly (a frame or two later, in issue order — a
//! query result must be read before its name is reused, which is why
//! `mark` returns `None` while the ring is full).
//!
//! The pool must only be used with the renderer it was probed against:
//! the entry points are process-global statics and the extension check
//! in [`GpuTimerPool::new`] is the only guard between this module and
//! a null call into an absent extension.

use std::collections::VecDeque;

use super::ffi;
use super::GlesRenderer;

/// How many query names the pool owns. Enough for several frames of
/// multi-segment timing without a collect in between.
const POOL_SIZE: usize = 64;

/// One collected timestamp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuTimestampSample {
    /// The token [`GpuTimerPool::mark`] handed out for this timestamp.
    pub token: u64,
    /// The GPU timestamp in nanoseconds (the GL timestamp domain).
    pub nanos: u64,
}

/// The reusable outcome of one [`GpuTimerPool::collect`] pass. Owned by
/// the caller so a steady-state frame allocates nothing.
#[derive(Debug, Default)]
pub struct GpuTimestampBatch {
    /// Completed samples, in issue order.
    pub samples: Vec<GpuTimestampSample>,
    /// The disjoint flag was set at collect time: the GPU clock reset,
    /// so every pending timestamp difference across the reset is
    /// invalid. Callers must drop their in-flight segments, not
    /// subtract across the reset.
    pub disjoint: bool,
}

impl GpuTimestampBatch {
    /// Clears both fields, keeping the sample capacity.
    pub fn clear(&mut self) {
        self.samples.clear();
        self.disjoint = false;
    }
}

/// A fixed-size pool of `TIMESTAMP_EXT` counter queries.
///
/// Creation probes the renderer for `GL_EXT_disjoint_timer_query`
/// (returning `None` without it — the caller's honest-degradation arm).
/// The query names belong to the renderer's GL context and die with it;
/// there is no `Drop` cleanup by design.
#[derive(Debug)]
pub struct GpuTimerPool {
    names: VecDeque<ffi::types::GLuint>,
    /// The tokens of the queries currently in flight, oldest first,
    /// paired with their query names.
    in_flight: VecDeque<(u64, ffi::types::GLuint)>,
    next_token: u64,
}

impl GpuTimerPool {
    /// Probes for `GL_EXT_disjoint_timer_query` and allocates the pool.
    /// `None` means GPU timing is impossible on this context.
    pub fn new(renderer: &mut GlesRenderer) -> Option<Self> {
        if !renderer
            .extensions
            .iter()
            .any(|ext| ext == "GL_EXT_disjoint_timer_query")
        {
            return None;
        }
        let mut names =
            vec![0 as ffi::types::GLuint; POOL_SIZE];
        let minted = renderer.with_bound_context(|gl| unsafe {
            gl.GenQueriesEXT(names.len() as ffi::types::GLsizei, names.as_mut_ptr());
            // GenQueries writes 0 for names it failed to mint.
            names.iter().filter(|name| **name != 0).count()
        });
        if minted != POOL_SIZE {
            return None;
        }
        Some(GpuTimerPool {
            names: VecDeque::from(names),
            in_flight: VecDeque::new(),
            next_token: 0,
        })
    }

    /// Issues one timestamp counter and returns its token, or `None`
    /// when the pool ring is full (the caller drops that sample — the
    /// honest-degradation contract: a missing token shows up as a
    /// missing segment, never as a fabricated timestamp).
    pub fn mark(&mut self, renderer: &mut GlesRenderer) -> Option<u64> {
        let Some(name) = self.names.pop_front() else {
            return None;
        };
        let token = self.next_token;
        renderer.with_bound_context(|gl| unsafe { gl.QueryCounterEXT(name, ffi::TIMESTAMP_EXT) });
        self.next_token += 1;
        self.in_flight.push_back((token, name));
        Some(token)
    }

    /// Collects every completed timestamp, oldest first, into `batch`
    /// (cleared first; capacity kept). Queries whose results are not
    /// ready yet stay in flight for a later pass. The disjoint flag is
    /// read once per pass; when set, this batch's samples are still
    /// drained (so their names recycle) but their values are invalid —
    /// the flag is the caller's drop signal.
    pub fn collect(&mut self, renderer: &mut GlesRenderer, batch: &mut GpuTimestampBatch) {
        batch.clear();
        renderer.with_bound_context(|gl| unsafe {
            let mut disjoint = 0 as ffi::types::GLint;
            gl.GetIntegerv(ffi::GPU_DISJOINT_EXT, &mut disjoint);
            batch.disjoint = disjoint != 0;
            while let Some(&(_, name)) = self.in_flight.front() {
                let mut available = 0 as ffi::types::GLuint;
                gl.GetQueryObjectuivEXT(name, ffi::QUERY_RESULT_AVAILABLE, &mut available);
                if available == 0 {
                    break;
                }
                let mut nanos = 0 as ffi::types::GLint64;
                gl.GetQueryObjecti64vEXT(name, ffi::QUERY_RESULT, &mut nanos);
                let (token, name) = self
                    .in_flight
                    .pop_front()
                    .expect("the front was just checked");
                batch.samples.push(GpuTimestampSample {
                    token,
                    nanos: nanos.max(0) as u64,
                });
                self.names.push_back(name);
            }
        })
    }
}
