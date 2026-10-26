//! Dual-texture custom shader programs (Legato fork extension).
//!
//! [`GlesTexProgram`] is contractually single-sampler: the renderer binds
//! exactly one texture (unit 0) and the public uniform vocabulary has no
//! sampler type, so a shader cannot declare and receives no binding for a
//! second texture. The Legato fluid field (0034) has three passes whose
//! input closures span two textures (PRESS: press+div, GRAD: field+press,
//! glass injection: backdrop+field), which no packing can collapse below
//! two samplers. This module holds the minimal dual-texture counterpart;
//! all raw GL binding stays behind the safe surface of
//! [`GlesFrame::render_texture_dual_from_to`](super::GlesFrame::render_texture_dual_from_to).
//!
//! Unlike the implicit texture program there are no `NO_ALPHA`/`EXTERNAL`
//! define variants: both inputs must be ordinary non-external `TEXTURE_2D`
//! textures. Only a normal and a `DEBUG_FLAGS` program are linked.
//!
//! Note: the non-instanced draw fallback of
//! [`GlesFrame::render_texture_dual_from_to`](super::GlesFrame::render_texture_dual_from_to)
//! mirrors upstream `render_texture` verbatim, including the
//! `solid_program.attrib_position` reuse — the mirroring is deliberate.

use crate::backend::renderer::gles::*;

#[derive(Debug)]
pub(in super::super) struct GlesTexProgramDualInternal {
    pub(in super::super) program: ffi::types::GLuint,
    /// Location of the `tex` sampler, bound to texture unit 0.
    pub(in super::super) uniform_tex: ffi::types::GLint,
    /// Location of the `tex1` sampler, bound to texture unit 1.
    pub(in super::super) uniform_tex1: ffi::types::GLint,
    pub(in super::super) uniform_matrix: ffi::types::GLint,
    /// Location of the `tex_matrix` matrix, which applies to `tex` only.
    pub(in super::super) uniform_tex_matrix: ffi::types::GLint,
    pub(in super::super) uniform_alpha: ffi::types::GLint,
    pub(in super::super) attrib_vert: ffi::types::GLint,
    pub(in super::super) attrib_vert_position: ffi::types::GLint,
    pub(in super::super) additional_uniforms: HashMap<String, UniformDesc>,
}

#[derive(Debug)]
pub(in super::super) struct GlesTexProgramDualInner {
    pub(in super::super) normal: GlesTexProgramDualInternal,
    pub(in super::super) debug: GlesTexProgramDualInternal,
    pub(super) destruction_callback_sender: Sender<CleanupResource>,

    // debug flags
    pub(in super::super) uniform_tint: ffi::types::GLint,
}

/// Gles dual-texture shader
///
/// The program can be used with the same [`GlesRenderer`] it was created with, or one using a
/// shared [`EGLContext`].
#[derive(Debug, Clone)]
pub struct GlesTexProgramDual(pub(in super::super) Arc<GlesTexProgramDualInner>);

impl Drop for GlesTexProgramDualInner {
    fn drop(&mut self) {
        let _ = self
            .destruction_callback_sender
            .send(CleanupResource::Program(self.normal.program));
        let _ = self
            .destruction_callback_sender
            .send(CleanupResource::Program(self.debug.program));
    }
}
