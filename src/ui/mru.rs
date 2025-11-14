use std::cell::RefCell;
use std::collections::HashMap;
use std::mem;
use std::rc::Rc;
use std::time::Duration;

use anyhow::ensure;
use niri_config::{
    Action, Bind, Color, Config, CornerRadius, GradientInterpolation, Key, Modifiers, MruDirection,
    MruFilter, MruScope, Trigger,
};
use pango::FontDescription;
use pangocairo::cairo::{self, ImageSurface};
use smithay::backend::allocator::Fourcc;
use smithay::backend::renderer::element::utils::{
    Relocate, RelocateRenderElement, RescaleRenderElement,
};
use smithay::backend::renderer::element::Kind;
use smithay::backend::renderer::gles::{GlesRenderer, GlesTexture};
use smithay::backend::renderer::Color32F;
use smithay::input::keyboard::Keysym;
use smithay::output::Output;
use smithay::utils::{Logical, Point, Rectangle, Scale, Size, Transform};

use crate::animation::{Animation, Clock};
use crate::layout::focus_ring::{FocusRing, FocusRingRenderElement};
use crate::layout::{Layout, LayoutElement as _, LayoutElementRenderElement};
use crate::niri::Niri;
use crate::niri_render_elements;
use crate::render_helpers::border::BorderRenderElement;
use crate::render_helpers::clipped_surface::ClippedSurfaceRenderElement;
use crate::render_helpers::gradient_fade_texture::GradientFadeTextureRenderElement;
use crate::render_helpers::primary_gpu_texture::PrimaryGpuTextureRenderElement;
use crate::render_helpers::renderer::NiriRenderer;
use crate::render_helpers::solid_color::{SolidColorBuffer, SolidColorRenderElement};
use crate::render_helpers::texture::{TextureBuffer, TextureRenderElement};
use crate::render_helpers::RenderTarget;
use crate::utils::{
    baba_is_float_offset, output_size, round_logical_in_physical, to_physical_precise_round,
    with_toplevel_role,
};
use crate::window::mapped::MappedId;
use crate::window::Mapped;

#[cfg(test)]
mod tests;

/// Windows up to this size don't get scaled further down.
const PREVIEW_MIN_SIZE: f64 = 16.;

/// Border width on the selected window preview.
const BORDER: f64 = 2.;

/// Gap from the window preview to the window title.
const TITLE_GAP: f64 = 14.;

/// Gap between thumbnails.
const GAP: f64 = 16.;

/// How much of the next window will always peek from the side of the screen.
const STRUT: f64 = 192.;

/// Padding in the scope indication panel.
const PANEL_PADDING: i32 = 12;

/// Border size of the scope indication panel.
const PANEL_BORDER: i32 = 4;

/// Backdrop color behind the previews.
const BACKDROP_COLOR: Color32F = Color32F::new(0., 0., 0., 0.8);

/// Font used to render the window titles.
const FONT: &str = "sans 14px";

/// Scopes in the order they are cycled through.
///
/// Count must match one defined in `generate_scope_panels()`.
static SCOPE_CYCLE: [MruScope; 3] = [MruScope::All, MruScope::Workspace, MruScope::Output];

#[derive(Debug)]
struct Thumbnail {
    id: MappedId,
    timestamp: Option<Duration>,
    on_current_workspace: bool,
    on_current_output: bool,
    app_id: Option<String>,

    // Cached size of the window.
    size: Size<i32, Logical>,

    clock: Clock,
    config: niri_config::MruPreview,
    open_animation: Option<Animation>,
    move_animation: Option<MoveAnimation>,
    title_texture: RefCell<TitleTexture>,
    background_buffer: RefCell<SolidColorBuffer>,
    border: RefCell<FocusRing>,
}

impl Thumbnail {
    fn from_mapped(mapped: &Mapped, clock: Clock, config: niri_config::MruPreview) -> Self {
        let app_id = with_toplevel_role(mapped.toplevel(), |role| role.app_id.clone());

        let border = FocusRing::new(niri_config::FocusRing {
            off: false,
            active_color: Color::new_unpremul(1., 1., 1., 1.),
            active_gradient: None,
            ..Default::default()
        });

        Self {
            id: mapped.id(),
            timestamp: mapped.get_focus_timestamp(),
            on_current_output: false,
            on_current_workspace: false,
            app_id,
            size: mapped.size(),
            clock,
            config,
            open_animation: None,
            move_animation: None,
            title_texture: Default::default(),
            background_buffer: RefCell::new(SolidColorBuffer::new((0., 0.), [1., 1., 1., 1.])),
            border: RefCell::new(border),
        }
    }

    fn are_animations_ongoing(&self) -> bool {
        self.open_animation.is_some() || self.move_animation.is_some()
    }

    fn advance_animations(&mut self) {
        self.open_animation.take_if(|a| a.is_done());
        self.move_animation.take_if(|a| a.anim.is_done());
    }

    /// Animate thumbnail motion from given location.
    fn animate_move_from_with_config(&mut self, from: f64, config: niri_config::Animation) {
        let current_offset = self.render_offset();

        // Preserve the previous config if ongoing.
        let anim = self.move_animation.take().map(|ma| ma.anim);
        let anim = anim
            .map(|anim| anim.restarted(1., 0., 0.))
            .unwrap_or_else(|| Animation::new(self.clock.clone(), 1., 0., 0., config));

        self.move_animation = Some(MoveAnimation {
            anim,
            from: from + current_offset,
        });
    }

    fn animate_open_with_config(&mut self, config: niri_config::Animation) {
        self.open_animation = Some(Animation::new(self.clock.clone(), 0., 1., 0., config));
    }

    fn render_offset(&self) -> f64 {
        self.move_animation
            .as_ref()
            .map(|ma| ma.from * ma.anim.value())
            .unwrap_or_default()
    }

    fn update_window(&mut self, mapped: &Mapped) {
        self.size = mapped.size();
    }

    fn preview_size(&self, output_size: Size<f64, Logical>, scale: f64) -> Size<f64, Logical> {
        let max_height = f64::max(1., self.config.max_height);
        let max_scale = f64::max(0.001, self.config.max_scale);

        let max_height = f64::min(max_height, output_size.h * max_scale);
        let output_ratio = output_size.w / output_size.h;
        let max_width = max_height * output_ratio;

        let size = self.size.to_f64();
        let min_scale = f64::min(1., PREVIEW_MIN_SIZE / f64::max(size.w, size.h));

        let thumb_scale = f64::min(max_width / size.w, max_height / size.h);
        let thumb_scale = f64::min(max_scale, thumb_scale);
        let thumb_scale = f64::max(min_scale, thumb_scale);
        let size = size.to_f64().upscale(thumb_scale);

        // Round to physical pixels.
        size.to_physical_precise_round(scale).to_logical(scale)
    }

    fn title_texture(
        &self,
        renderer: &mut GlesRenderer,
        mapped: &Mapped,
        scale: f64,
    ) -> Option<MruTexture> {
        with_toplevel_role(mapped.toplevel(), |role| {
            role.title
                .as_ref()
                .and_then(|title| self.title_texture.borrow_mut().get(renderer, title, scale))
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn render<R: NiriRenderer>(
        &self,
        renderer: &mut R,
        config: &niri_config::RecentWindows,
        mapped: &Mapped,
        preview_geo: Rectangle<f64, Logical>,
        scale: f64,
        is_active: bool,
        bob_y: f64,
        alpha: f32,
        target: RenderTarget,
    ) -> impl Iterator<Item = WindowMruUiRenderElement<R>> {
        let _span = tracy_client::span!("Thumbnail::render");

        let round = move |logical: f64| round_logical_in_physical(scale, logical);
        let padding = round(config.highlight.padding);
        let title_gap = round(TITLE_GAP);

        let s = Scale::from(scale);

        let preview_alpha = self
            .open_animation
            .as_ref()
            .map_or(1., |a| a.clamped_value() as f32)
            .clamp(0., 1.)
            * alpha;

        let bob_y = if mapped.rules().baba_is_float == Some(true) {
            bob_y
        } else {
            0.
        };
        let bob_offset = Point::new(0., bob_y);

        // TODO: offscreen
        let elems = mapped
            .render_normal(renderer, Point::new(0., 0.), s, preview_alpha, target)
            .into_iter();

        // Clip thumbnails to their geometry.
        let radius = if mapped.sizing_mode().is_normal() {
            mapped.rules().geometry_corner_radius
        } else {
            None
        }
        .unwrap_or_default();

        let has_border_shader = BorderRenderElement::has_shader(renderer);
        let clip_shader = ClippedSurfaceRenderElement::shader(renderer).cloned();
        let geo = Rectangle::from_size(self.size.to_f64());
        // FIXME: deduplicate code with Tile::render_inner()
        let elems = elems.map(move |elem| match elem {
            LayoutElementRenderElement::Wayland(elem) => {
                if let Some(shader) = clip_shader.clone() {
                    if ClippedSurfaceRenderElement::will_clip(&elem, s, geo, radius) {
                        let elem =
                            ClippedSurfaceRenderElement::new(elem, s, geo, shader.clone(), radius);
                        return ThumbnailRenderElement::ClippedSurface(elem);
                    }
                }

                // If we don't have the shader, render it normally.
                let elem = LayoutElementRenderElement::Wayland(elem);
                ThumbnailRenderElement::LayoutElement(elem)
            }
            LayoutElementRenderElement::SolidColor(elem) => {
                // In this branch we're rendering a blocked-out window with a solid
                // color. We need to render it with a rounded corner shader even if
                // clip_to_geometry is false, because in this case we're assuming that
                // the unclipped window CSD already has corners rounded to the
                // user-provided radius, so our blocked-out rendering should match that
                // radius.
                if radius != CornerRadius::default() && has_border_shader {
                    return BorderRenderElement::new(
                        geo.size,
                        Rectangle::from_size(geo.size),
                        GradientInterpolation::default(),
                        Color::from_color32f(elem.color()),
                        Color::from_color32f(elem.color()),
                        0.,
                        Rectangle::from_size(geo.size),
                        0.,
                        radius,
                        scale as f32,
                        1.,
                    )
                    .into();
                }

                // Otherwise, render the solid color as is.
                LayoutElementRenderElement::SolidColor(elem).into()
            }
        });

        let elems = elems.map(move |elem| {
            let thumb_scale = Scale {
                x: preview_geo.size.w / geo.size.w,
                y: preview_geo.size.h / geo.size.h,
            };
            let offset = Point::new(
                preview_geo.size.w - (geo.size.w * thumb_scale.x),
                preview_geo.size.h - (geo.size.h * thumb_scale.y),
            )
            .downscale(2.);
            let elem = RescaleRenderElement::from_element(elem, Point::new(0, 0), thumb_scale);
            let elem = RelocateRenderElement::from_element(
                elem,
                (preview_geo.loc + offset + bob_offset).to_physical_precise_round(scale),
                Relocate::Relative,
            );
            WindowMruUiRenderElement::Thumbnail(elem)
        });

        let mut title_size = None;
        let title_texture = self.title_texture(renderer.as_gles_renderer(), mapped, scale);
        let title_texture = title_texture.map(|texture| {
            let mut size = texture.logical_size();
            size.w = f64::min(size.w, preview_geo.size.w);
            title_size = Some(size);
            (texture, size)
        });

        // Hide title for blocked-out windows, but only after computing the title size. This way,
        // the background and the border won't have to oscillate in size between normal and
        // screencast renders, causing excessive damage.
        let should_block_out = target.should_block_out(mapped.rules().block_out_from);
        let title_texture = title_texture.filter(|_| !should_block_out);

        let title_elems = title_texture.map(|(texture, size)| {
            // Clip from the right if it doesn't fit.
            let src = Rectangle::from_size(size);

            let loc = preview_geo.loc
                + Point::new(
                    (preview_geo.size.w - size.w) / 2.,
                    preview_geo.size.h + title_gap,
                );
            let loc = loc.to_physical_precise_round(scale).to_logical(scale);
            let texture = TextureRenderElement::from_texture_buffer(
                texture,
                loc,
                preview_alpha,
                Some(src),
                None,
                Kind::Unspecified,
            );

            let renderer = renderer.as_gles_renderer();
            if let Some(program) = GradientFadeTextureRenderElement::shader(renderer) {
                let elem = GradientFadeTextureRenderElement::new(texture, program);
                WindowMruUiRenderElement::GradientFadeElem(elem)
            } else {
                let elem = PrimaryGpuTextureRenderElement(texture);
                WindowMruUiRenderElement::TextureElement(elem)
            }
        });

        let is_urgent = mapped.is_urgent();
        let background_elems = (is_active || is_urgent).then(|| {
            let padding = Point::new(padding, padding);

            let mut size = preview_geo.size;
            size += padding.to_size().upscale(2.);

            if let Some(title_size) = title_size {
                size.h += title_gap + title_size.h;
                // Subtract half the padding so it looks more balanced visually.
                size.h -= padding.y / 2.;
            }

            let mut color = if is_urgent {
                config.highlight.urgent_color
            } else {
                config.highlight.active_color
            };
            if !is_active {
                color *= 0.3;
            }

            let mut buffer = self.background_buffer.borrow_mut();
            buffer.resize(size);
            buffer.set_color(color);

            let loc = preview_geo.loc - padding;
            let elem =
                SolidColorRenderElement::from_buffer(&buffer, loc, 0.5 * alpha, Kind::Unspecified);
            let elem = WindowMruUiRenderElement::SolidColor(elem);

            let mut border = self.border.borrow_mut();
            let mut config = *border.config();
            config.off = !is_active;
            config.width = round(BORDER);
            config.active_color = color;
            border.update_config(config);
            border.update_render_elements(
                size,
                true,
                true,
                false,
                Rectangle::default(),
                CornerRadius::default(),
                scale,
                alpha,
            );

            let elems = border
                .render(renderer, loc)
                .map(WindowMruUiRenderElement::FocusRing);
            elems.chain([elem])
        });
        let background_elems = background_elems.into_iter().flatten();

        elems.chain(title_elems).chain(background_elems)
    }
}

/// Window MRU traversal context.
#[derive(Debug)]
pub struct WindowMru {
    /// Windows in MRU order.
    thumbnails: Vec<Thumbnail>,

    /// Id of the currently selected window.
    current_id: Option<MappedId>,

    /// Current scope.
    scope: MruScope,

    /// Current filter.
    app_id_filter: Option<String>,
}

impl WindowMru {
    pub fn new(niri: &Niri) -> Self {
        let Some(output) = niri.layout.active_output() else {
            return Self {
                thumbnails: Vec::new(),
                current_id: None,
                scope: MruScope::All,
                app_id_filter: None,
            };
        };

        let config = niri.config.borrow().recent_windows.preview;
        let mut thumbnails = Vec::new();
        for (mon, ws_idx, ws) in niri.layout.workspaces() {
            let mon = mon.expect("an active output exists so all workspaces have a monitor");
            let on_current_output = mon.output() == output;
            let on_current_workspace = on_current_output && mon.active_workspace_idx() == ws_idx;

            for mapped in ws.windows() {
                let mut thumbnail = Thumbnail::from_mapped(mapped, niri.clock.clone(), config);
                thumbnail.on_current_output = on_current_output;
                thumbnail.on_current_workspace = on_current_workspace;
                thumbnails.push(thumbnail);
            }
        }

        thumbnails
            .sort_by(|Thumbnail { timestamp: t1, .. }, Thumbnail { timestamp: t2, .. }| t2.cmp(t1));

        let current_id = thumbnails.first().map(|t| t.id);
        Self {
            thumbnails,
            current_id,
            scope: MruScope::All,
            app_id_filter: None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.thumbnails.is_empty()
    }

    #[cfg(test)]
    fn verify_invariants(&self) {
        if let Some(id) = self.current_id {
            assert!(
                self.thumbnails().any(|thumbnail| thumbnail.id == id),
                "current_id must be present in the current filtered thumbnail list",
            );
        } else {
            assert!(
                self.thumbnails().next().is_none(),
                "unset current_id must mean that the filtered thumbnail list is empty",
            );
        }
    }

    fn thumbnails(&self) -> impl DoubleEndedIterator<Item = &Thumbnail> {
        let matches = match_filter(self.scope, self.app_id_filter.as_deref());
        self.thumbnails.iter().filter(move |t| matches(t))
    }

    fn thumbnails_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut Thumbnail> {
        let matches = match_filter(self.scope, self.app_id_filter.as_deref());
        self.thumbnails.iter_mut().filter(move |t| matches(t))
    }

    fn thumbnails_with_idx(&self) -> impl DoubleEndedIterator<Item = (usize, &Thumbnail)> {
        let matches = match_filter(self.scope, self.app_id_filter.as_deref());
        self.thumbnails
            .iter()
            .enumerate()
            .filter(move |(_, t)| matches(t))
    }

    fn are_animations_ongoing(&self) -> bool {
        self.thumbnails.iter().any(|t| t.are_animations_ongoing())
    }

    fn advance_animations(&mut self) {
        for thumbnail in &mut self.thumbnails {
            thumbnail.advance_animations();
        }
    }

    fn forward(&mut self) {
        let Some(id) = self.current_id else {
            return;
        };

        let next = self.thumbnails().skip_while(|t| t.id != id).nth(1);
        self.current_id = Some(if let Some(next) = next {
            next.id
        } else {
            // We wrapped around.
            self.thumbnails().next().unwrap().id
        });
    }

    fn backward(&mut self) {
        let Some(id) = self.current_id else {
            return;
        };

        let next = self.thumbnails().rev().skip_while(|t| t.id != id).nth(1);
        self.current_id = Some(if let Some(next) = next {
            next.id
        } else {
            // We wrapped around.
            self.thumbnails().next_back().unwrap().id
        });
    }

    fn set_current(&mut self, id: MappedId) {
        if self.thumbnails().any(|thumbnail| thumbnail.id == id) {
            self.current_id = Some(id);
        }
    }

    fn first_id(&self) -> Option<MappedId> {
        self.thumbnails().next().map(|thumbnail| thumbnail.id)
    }

    fn first(&mut self) {
        self.current_id = self.first_id();
    }

    fn last(&mut self) {
        let id = self.thumbnails().next_back().map(|thumbnail| thumbnail.id);
        self.current_id = id;
    }

    pub fn set_scope_and_filter(&mut self, scope: MruScope, filter: Option<MruFilter>) -> bool {
        let mut changed = self.scope != scope;

        if let Some(id) = self.current_id {
            let (current_idx, current_thumbnail) = self
                .thumbnails_with_idx()
                .find(|(_, thumbnail)| thumbnail.id == id)
                .unwrap();

            if let Some(filter) = filter {
                let filter = match filter {
                    MruFilter::None => None,
                    // TODO this clones every time on Alt-`?
                    MruFilter::AppId => current_thumbnail.app_id.clone(),
                };
                changed |= self.app_id_filter != filter;
                self.app_id_filter = filter;
            }
            self.scope = scope;

            // Try to select the same, or the first thumbnail to the left. Failing that, select the
            // first one to the right.
            let mut id = self.first_id();

            for (idx, thumbnail) in self.thumbnails_with_idx() {
                if idx > current_idx {
                    break;
                }
                id = Some(thumbnail.id);
            }
            self.current_id = id;
        } else {
            if filter.is_some() {
                // No current window, can't get app id.
                changed |= self.app_id_filter.is_some();
                self.app_id_filter = None;
            }
            self.scope = scope;
            self.current_id = self.first_id();
        }

        changed
    }

    fn idx_of(&self, id: MappedId) -> Option<usize> {
        self.thumbnails.iter().position(|t| t.id == id)
    }

    fn remove_by_idx(&mut self, idx: usize) -> Option<Thumbnail> {
        let id = self.thumbnails[idx].id;

        // Try to pick a different window when removing the current one.
        if self.current_id == Some(id) {
            self.forward();
        }

        // If we're still on the same window, that means it's the last visible one.
        if self.current_id == Some(id) {
            self.current_id = None;
        }

        Some(self.thumbnails.remove(idx))
    }

    /// Returns the thumbnail if it's visible to the left of the currently selected one.
    fn thumbnail_left_of_current(&self, id: MappedId) -> Option<&Thumbnail> {
        for thumbnail in self.thumbnails() {
            if Some(thumbnail.id) == self.current_id {
                // We found the current window first, so the queried one is *not* to the left.
                return None;
            } else if thumbnail.id == id {
                // We found the queried window first, so the current one is to the right of it.
                return Some(thumbnail);
            }
        }
        None
    }
}

fn matches(scope: MruScope, app_id_filter: Option<&str>, thumbnail: &Thumbnail) -> bool {
    let x = match scope {
        MruScope::All => true,
        MruScope::Output => thumbnail.on_current_output,
        MruScope::Workspace => thumbnail.on_current_workspace,
    };
    if !x {
        return false;
    }

    if let Some(app_id) = app_id_filter {
        thumbnail.app_id.as_deref() == Some(app_id)
    } else {
        true
    }
}

fn match_filter(scope: MruScope, app_id_filter: Option<&str>) -> impl Fn(&Thumbnail) -> bool + '_ {
    move |thumbnail| matches(scope, app_id_filter, thumbnail)
}

type MruTexture = TextureBuffer<GlesTexture>;

pub struct WindowMruUi {
    state: WindowMruUiState,
    preset_opened_binds: Vec<Bind>,
    dynamic_opened_binds: Vec<Bind>,
    config: Rc<RefCell<Config>>,
}

pub enum WindowMruUiState {
    Open(Inner),
    Closing {
        inner: Inner,
        anim: Animation,
    },
    Closed {
        /// Scope used when the UI was last opened.
        previous_scope: MruScope,
    },
}

/// State of an opened MRU UI.
pub struct Inner {
    /// List of Window Ids to display in the MRU UI.
    wmru: WindowMru,

    /// View position relative to the leftmost visible window.
    view_pos: ViewPos,

    // If true, don't automatically move the current thumbnail in-view. Set on pointer motion.
    freeze_view: bool,

    /// Animation clock.
    clock: Clock,

    /// Current config.
    config: Rc<RefCell<Config>>,

    /// Time when the UI should appear.
    open_at: Duration,

    /// Thumbnails linked to windows that were just closed, or to windows
    /// that no longer match the current MRU filter or scope.
    // closing_thumbnails: Vec<ClosingThumbnail>,

    /// Output the UI was opened on.
    output: Output,

    /// Scope panel textures.
    scope_panel: RefCell<ScopePanel>,

    /// Backdrop buffers for each output.
    backdrop_buffers: RefCell<HashMap<Output, SolidColorBuffer>>,
}

#[derive(Debug)]
pub enum ViewPos {
    /// The view position is static.
    Static(f64),
    /// The view position is animating.
    Animation(Animation),
}

#[derive(Debug)]
struct MoveAnimation {
    anim: Animation,
    from: f64,
}

impl ViewPos {
    fn current(&self) -> f64 {
        match self {
            ViewPos::Static(pos) => *pos,
            ViewPos::Animation(anim) => anim.value(),
        }
    }

    fn target(&self) -> f64 {
        match self {
            ViewPos::Static(pos) => *pos,
            ViewPos::Animation(anim) => anim.to(),
        }
    }

    fn are_animations_ongoing(&self) -> bool {
        match self {
            ViewPos::Static(_) => false,
            ViewPos::Animation(_) => true,
        }
    }

    fn advance_animations(&mut self) {
        if let ViewPos::Animation(anim) = self {
            if anim.is_done() {
                *self = ViewPos::Static(anim.to());
            }
        }
    }

    fn animate_from_with_config(
        &mut self,
        from: f64,
        config: niri_config::Animation,
        clock: Clock,
    ) {
        // FIXME: also compute and use current velocity.
        let anim = Animation::new(clock, self.current() + from, self.target(), 0., config);
        *self = ViewPos::Animation(anim);
    }

    fn offset(&mut self, delta: f64) {
        match self {
            ViewPos::Static(pos) => *pos += delta,
            ViewPos::Animation(anim) => anim.offset(delta),
        }
    }
}

pub enum MruCloseRequest {
    Cancelled,
    Current,
    Selection(MappedId),
}

niri_render_elements! {
    ThumbnailRenderElement<R> => {
        LayoutElement = LayoutElementRenderElement<R>,
        ClippedSurface = ClippedSurfaceRenderElement<R>,
        Border = BorderRenderElement,
    }
}

niri_render_elements! {
    WindowMruUiRenderElement<R> => {
        SolidColor = SolidColorRenderElement,
        TextureElement = PrimaryGpuTextureRenderElement,
        GradientFadeElem = GradientFadeTextureRenderElement,
        FocusRing = FocusRingRenderElement,
        Thumbnail = RelocateRenderElement<RescaleRenderElement<ThumbnailRenderElement<R>>>,
    }
}

impl WindowMruUi {
    pub fn new(config: Rc<RefCell<Config>>) -> Self {
        let mut rv = Self {
            state: WindowMruUiState::Closed {
                previous_scope: MruScope::default(),
            },
            preset_opened_binds: make_preset_opened_binds(),
            dynamic_opened_binds: Vec::new(),
            config,
        };
        rv.update_binds();
        rv
    }

    pub fn update_binds(&mut self) {
        self.dynamic_opened_binds = make_dynamic_opened_binds(&self.config.borrow());
    }

    pub fn update_config(&mut self) {
        let inner = match &mut self.state {
            WindowMruUiState::Open(inner) => inner,
            WindowMruUiState::Closing { inner, .. } => inner,
            WindowMruUiState::Closed { .. } => return,
        };
        inner.update_config();
    }

    pub fn is_open(&self) -> bool {
        matches!(self.state, WindowMruUiState::Open { .. })
    }

    pub fn open(&mut self, clock: Clock, wmru: WindowMru, dir: MruDirection, output: Output) {
        if self.is_open() {
            return;
        }

        let open_delay = self.config.borrow().recent_windows.open_delay_ms;
        let open_delay = Duration::from_millis(u64::from(open_delay));

        let mut inner = Inner {
            wmru,
            view_pos: ViewPos::Static(0.),
            freeze_view: false,
            // closing_thumbnails: vec![],
            open_at: clock.now_unadjusted() + open_delay,
            clock,
            config: self.config.clone(),
            output,
            scope_panel: Default::default(),
            backdrop_buffers: Default::default(),
        };
        inner.view_pos = ViewPos::Static(inner.compute_view_pos());

        self.state = WindowMruUiState::Open(inner);
        self.advance(Some(dir), None, None);
    }

    pub fn close(&mut self, close_request: MruCloseRequest) -> Option<MappedId> {
        if !self.is_open() {
            return None;
        }
        let state = mem::replace(
            &mut self.state,
            WindowMruUiState::Closed {
                previous_scope: MruScope::default(),
            },
        );
        let WindowMruUiState::Open(inner) = state else {
            unreachable!();
        };

        let response = inner.build_close_response(close_request);

        if inner.clock.now_unadjusted() < inner.open_at {
            // Hasn't displayed yet, no need to fade out.
            let WindowMruUiState::Closed { previous_scope } = &mut self.state else {
                unreachable!()
            };
            *previous_scope = inner.wmru.scope;
            return response;
        }

        let config = self.config.borrow();
        let config = config.animations.recent_windows_close.0;

        let anim = Animation::new(inner.clock.clone(), 1., 0., 0., config);
        self.state = WindowMruUiState::Closing { inner, anim };
        response
    }

    pub fn advance(
        &mut self,
        dir: Option<MruDirection>,
        scope: Option<MruScope>,
        filter: Option<MruFilter>,
    ) {
        let WindowMruUiState::Open(inner) = &mut self.state else {
            return;
        };
        inner.freeze_view = false;

        let new_scope = scope.unwrap_or(inner.wmru.scope);
        let changed = inner.set_scope_and_filter(new_scope, filter);
        if changed && scope.is_some() {
            // Do not advance when changing the scope.
        } else if let Some(dir) = dir {
            match dir {
                MruDirection::Forward => inner.wmru.forward(),
                MruDirection::Backward => inner.wmru.backward(),
            }
        }
    }

    pub fn cycle_scope(&mut self) {
        let WindowMruUiState::Open(inner) = &mut self.state else {
            return;
        };

        let scope = inner.wmru.scope;
        let scope = SCOPE_CYCLE
            .into_iter()
            .cycle()
            .skip_while(|s| *s != scope)
            .nth(1)
            .unwrap();
        self.advance(None, Some(scope), None);
    }

    pub fn pointer_motion(&mut self, pos_within_output: Point<f64, Logical>) {
        let WindowMruUiState::Open(inner) = &mut self.state else {
            return;
        };

        inner.freeze_view = true;

        if let Some(id) = inner.thumbnail_under(pos_within_output) {
            inner.wmru.set_current(id);
        }
    }

    pub fn first(&mut self) {
        let WindowMruUiState::Open(inner) = &mut self.state else {
            return;
        };
        inner.freeze_view = false;
        inner.wmru.first();
    }

    pub fn last(&mut self) {
        let WindowMruUiState::Open(inner) = &mut self.state else {
            return;
        };
        inner.freeze_view = false;
        inner.wmru.last();
    }

    pub fn scope(&self) -> MruScope {
        match &self.state {
            WindowMruUiState::Closed { previous_scope, .. } => *previous_scope,
            WindowMruUiState::Open(inner) | WindowMruUiState::Closing { inner, .. } => {
                inner.wmru.scope
            }
        }
    }

    pub fn current_window_id(&self) -> Option<MappedId> {
        let WindowMruUiState::Open(inner) = &self.state else {
            return None;
        };
        inner.wmru.current_id
    }

    pub fn update_window(&mut self, layout: &Layout<Mapped>, id: MappedId) {
        let WindowMruUiState::Open(inner) = &mut self.state else {
            return;
        };
        inner.update_window(layout, id);
    }

    pub fn remove_window(&mut self, id: MappedId) {
        let WindowMruUiState::Open(inner) = &mut self.state else {
            return;
        };

        let Some(_thumbnail) = inner.remove_window(id) else {
            return;
        };

        // TODO: animate close.

        if inner.wmru.thumbnails.is_empty() {
            self.close(MruCloseRequest::Cancelled);
        }
    }

    pub fn render_output<R: NiriRenderer>(
        &self,
        niri: &Niri,
        output: &Output,
        renderer: &mut R,
        target: RenderTarget,
    ) -> Vec<WindowMruUiRenderElement<R>> {
        let mut rv = Vec::new();

        let (inner, progress) = match &self.state {
            WindowMruUiState::Closed { .. } => return rv,
            WindowMruUiState::Closing { inner, anim } => (inner, anim.clamped_value()),
            WindowMruUiState::Open(inner) => {
                if inner.open_at <= inner.clock.now_unadjusted() {
                    (inner, 1.)
                } else {
                    return rv;
                }
            }
        };

        let alpha = progress.clamp(0., 1.) as f32;

        if *output == inner.output {
            rv.extend(inner.render(niri, renderer, alpha, target));
        }

        // Put a backdrop above the current desktop view to contrast the thumbnails.
        let mut buffers = inner.backdrop_buffers.borrow_mut();
        let buffer = buffers.entry(output.clone()).or_default();
        buffer.resize(output_size(output));
        buffer.set_color(BACKDROP_COLOR);

        let elem = SolidColorRenderElement::from_buffer(
            buffer,
            Point::new(0., 0.),
            alpha,
            Kind::Unspecified,
        );
        rv.push(WindowMruUiRenderElement::SolidColor(elem));

        rv
    }

    pub fn are_animations_ongoing(&self) -> bool {
        match &self.state {
            WindowMruUiState::Open(inner) => inner.are_animations_ongoing(),
            WindowMruUiState::Closing { .. } => true,
            WindowMruUiState::Closed { .. } => false,
        }
    }

    pub fn advance_animations(&mut self) {
        match &mut self.state {
            WindowMruUiState::Open(inner) => inner.advance_animations(),
            WindowMruUiState::Closing { inner, anim } => {
                if anim.is_done() {
                    self.state = WindowMruUiState::Closed {
                        previous_scope: inner.wmru.scope,
                    };
                    return;
                }
                inner.advance_animations();
            }
            WindowMruUiState::Closed { .. } => {}
        }
    }

    pub fn opened_bindings(&mut self, mods: Modifiers) -> impl Iterator<Item = &Bind> {
        // Fill modifiers with the current mods.
        for bind in &mut self.preset_opened_binds {
            bind.key.modifiers = mods;
        }
        for bind in &mut self.dynamic_opened_binds {
            bind.key.modifiers = mods;
        }

        self.preset_opened_binds
            .iter()
            .chain(&self.dynamic_opened_binds)
    }

    pub fn output(&self) -> Option<&Output> {
        match &self.state {
            WindowMruUiState::Open(inner) => Some(&inner.output),
            _ => None,
        }
    }

    pub fn thumbnail_under(&self, pos: Point<f64, Logical>) -> Option<MappedId> {
        let WindowMruUiState::Open(inner) = &self.state else {
            return None;
        };

        inner.thumbnail_under(pos)
    }
}

fn compute_view_offset(cur_x: f64, working_width: f64, new_col_x: f64, new_col_width: f64) -> f64 {
    let new_x = new_col_x;
    let new_right_x = new_col_x + new_col_width;

    // If the column is already fully visible, leave the view as is.
    if cur_x <= new_x && new_right_x <= cur_x + working_width {
        return -(new_col_x - cur_x);
    }

    // Otherwise, prefer the alignment that results in less motion from the current position.
    let dist_to_left = (cur_x - new_x).abs();
    let dist_to_right = ((cur_x + working_width) - new_right_x).abs();
    if dist_to_left <= dist_to_right {
        0.
    } else {
        -(working_width - new_col_width)
    }
}

impl Inner {
    fn update_config(&mut self) {
        self.freeze_view = false;

        let config = self.config.borrow().recent_windows.preview;
        for thumbnail in &mut self.wmru.thumbnails {
            thumbnail.config = config;
        }
    }

    fn are_animations_ongoing(&self) -> bool {
        self.clock.now_unadjusted() < self.open_at
            || self.view_pos.are_animations_ongoing()
            // || !self.closing_thumbnails.is_empty()
            || self.wmru.are_animations_ongoing()
    }

    fn advance_animations(&mut self) {
        self.view_pos.advance_animations();
        // self.closing_thumbnails
        //     .retain_mut(|closing| closing.are_animations_ongoing());
        self.wmru.advance_animations();

        if !self.freeze_view {
            let new_view_pos = self.compute_view_pos();
            let delta = new_view_pos - self.view_pos.target();
            let pixel = 1. / self.output.current_scale().fractional_scale();
            if delta.abs() > pixel {
                self.animate_view_pos_from(-delta);
            }
            self.view_pos.offset(delta);
        }
    }

    fn animate_view_pos_from(&mut self, from: f64) {
        let config = self.config.borrow().animations.window_movement.0;
        self.view_pos
            .animate_from_with_config(from, config, self.clock.clone());
    }

    fn compute_view_pos(&self) -> f64 {
        let Some(current_id) = self.wmru.current_id else {
            return 0.;
        };

        let output_size = output_size(&self.output);

        let working_x = STRUT + GAP;
        let working_width = (output_size.w - working_x * 2.).max(0.);

        let mut current_geo = Rectangle::default();
        let mut strip_width = 0.;
        for (thumbnail, geo) in self.thumbnails() {
            if thumbnail.id == current_id {
                current_geo = geo;
            }
            strip_width = geo.loc.x + geo.size.w;

            // If we found current_geo, and the strip width is already bigger than the working
            // width, no need to compute further.
            if current_geo.size.w != 0. && strip_width > working_width {
                break;
            }
        }

        // If the whole strip fits on screen, center it.
        if strip_width <= working_width {
            return -(output_size.w - strip_width) / 2.;
        }

        compute_view_offset(
            self.view_pos.target() + working_x,
            working_width,
            current_geo.loc.x,
            current_geo.size.w,
        ) + current_geo.loc.x
            - working_x
    }

    fn update_window(&mut self, layout: &Layout<Mapped>, id: MappedId) {
        let output_size = output_size(&self.output);
        let scale = self.output.current_scale().fractional_scale();

        // If the updated window is to the left of the currently selected one, we need to offset
        // the view position to compensate for the change in size.
        let left = self.wmru.thumbnail_left_of_current(id);
        let prev_size = left.map(|thumbnail| thumbnail.preview_size(output_size, scale));

        let Some(thumbnail) = self.wmru.thumbnails.iter_mut().find(|t| t.id == id) else {
            return;
        };

        let Some((_, mapped)) = layout.windows().find(|(_, m)| m.id() == id) else {
            error!("window in the MRU must be present in the layout");
            return;
        };

        thumbnail.update_window(mapped);

        if let Some(prev) = prev_size {
            let new = thumbnail.preview_size(output_size, scale);
            let delta = new.w - prev.w;
            self.view_pos.offset(delta);
        }
    }

    fn remove_window(&mut self, id: MappedId) -> Option<Thumbnail> {
        let idx = self.wmru.idx_of(id)?;

        let last_visible = self.wmru.thumbnails().next_back();
        let removing_last_visible = last_visible.is_some_and(|t| t.id == id);

        // When removing the last visible thumbnail, nothing needs to be animated.
        // - If it's not currently selected, then it can't cause changes to view position.
        // - If it's currently selected, then the first step in removal (focusing the next window)
        //   will wrap back to the start, and no animations should happen.
        if !removing_last_visible {
            let output_size = output_size(&self.output);
            let scale = self.output.current_scale().fractional_scale();
            let gap = round_logical_in_physical(scale, GAP);

            let prev_size = self.wmru.thumbnails[idx].preview_size(output_size, scale);
            let delta = prev_size.w + gap;

            let config = self.config.borrow().animations.window_movement.0;

            // If the removed window is to the left of the currently selected one, we need to offset
            // the view position to compensate for the change.
            if self.wmru.thumbnail_left_of_current(id).is_some() {
                self.view_pos.offset(-delta);

                // And animate movement of windows left of it.
                for thumbnail in self.wmru.thumbnails_mut().take_while(|t| t.id != id) {
                    thumbnail.animate_move_from_with_config(-delta, config);
                }
            } else {
                // Otherwise, animate movement of windows right of it.
                for thumbnail in self.wmru.thumbnails_mut().rev().take_while(|t| t.id != id) {
                    thumbnail.animate_move_from_with_config(delta, config);
                }
            }
        }

        self.wmru.remove_by_idx(idx)
    }

    fn set_scope_and_filter(&mut self, scope: MruScope, filter: Option<MruFilter>) -> bool {
        let old_scope = self.wmru.scope;
        // TODO this clones every time
        let old_filter = self.wmru.app_id_filter.clone();

        let was_empty = self.wmru.current_id.is_none();

        let changed = self.wmru.set_scope_and_filter(scope, filter);
        if !changed {
            return false;
        }

        let Some(id) = self.wmru.current_id else {
            // If there's no current_id then the new filter caused all windows to disappear, so
            // there's nothing to animate.
            return true;
        };
        let idx = self.wmru.idx_of(id).unwrap();

        // Animate opening for newly appeared thumbnails.
        let config = self.config.borrow().animations.window_open.anim;
        let matches_old = match_filter(old_scope, old_filter.as_deref());
        let matches_new = match_filter(self.wmru.scope, self.wmru.app_id_filter.as_deref());
        for thumbnail in &mut self.wmru.thumbnails {
            if matches_new(thumbnail) && !matches_old(thumbnail) {
                thumbnail.animate_open_with_config(config);
            }
        }

        if was_empty {
            self.view_pos = ViewPos::Static(self.compute_view_pos());
            return true;
        }

        let output_size = output_size(&self.output);
        let scale = self.output.current_scale().fractional_scale();
        let round = move |logical: f64| round_logical_in_physical(scale, logical);
        let gap = round(GAP);

        let config = self.config.borrow().animations.window_movement.0;

        let mut delta = 0.;
        for t in &mut self.wmru.thumbnails[idx + 1..] {
            match (matches_old(t), matches_new(t)) {
                (true, true) => t.animate_move_from_with_config(delta, config),
                (true, false) => delta += t.preview_size(output_size, scale).w + gap,
                (false, true) => delta -= t.preview_size(output_size, scale).w + gap,
                (false, false) => (),
            }
        }

        let mut delta = 0.;
        for t in self.wmru.thumbnails[..idx].iter_mut().rev() {
            match (matches_old(t), matches_new(t)) {
                (true, true) => t.animate_move_from_with_config(-delta, config),
                (true, false) => delta += t.preview_size(output_size, scale).w + gap,
                (false, true) => delta -= t.preview_size(output_size, scale).w + gap,
                (false, false) => (),
            }
        }

        self.view_pos.offset(-delta);

        true
    }

    fn build_close_response(&self, close_request: MruCloseRequest) -> Option<MappedId> {
        let Inner { wmru, .. } = self;

        match close_request {
            MruCloseRequest::Cancelled => None,
            MruCloseRequest::Current => wmru.current_id,
            MruCloseRequest::Selection(id) => Some(id),
        }
    }

    fn thumbnails(&self) -> impl Iterator<Item = (&Thumbnail, Rectangle<f64, Logical>)> {
        let output_size = output_size(&self.output);
        let scale = self.output.current_scale().fractional_scale();
        let round = move |logical: f64| round_logical_in_physical(scale, logical);

        let padding = self.config.borrow().recent_windows.highlight.padding;
        let gap = round(GAP);
        let padding = (round(padding) + round(BORDER)) * 2.;

        let mut x = 0.;
        self.wmru.thumbnails().map(move |thumbnail| {
            let size = thumbnail.preview_size(output_size, scale);
            let y = round((output_size.h - size.h) / 2.);

            let loc = Point::new(x, y);
            x += size.w + padding + gap;

            let geo = Rectangle::new(loc, size);
            (thumbnail, geo)
        })
    }

    fn thumbnails_in_view_static(
        &self,
    ) -> impl Iterator<Item = (&Thumbnail, Rectangle<f64, Logical>)> {
        let output_size = output_size(&self.output);
        let scale = self.output.current_scale().fractional_scale();
        let round = |logical: f64| round_logical_in_physical(scale, logical);

        let view_pos = round(self.view_pos.current());

        let leftmost = view_pos;
        let rightmost = view_pos + output_size.w;

        self.thumbnails()
            .skip_while(move |(_, geo)| geo.loc.x + geo.size.w <= leftmost)
            .map_while(move |(thumbnail, mut geo)| {
                if rightmost <= geo.loc.x {
                    return None;
                }

                geo.loc.x -= view_pos;
                Some((thumbnail, geo))
            })
    }

    fn thumbnails_in_view_render(
        &self,
    ) -> impl Iterator<Item = (&Thumbnail, Rectangle<f64, Logical>)> {
        let output_size = output_size(&self.output);
        let scale = self.output.current_scale().fractional_scale();
        let round = move |logical: f64| round_logical_in_physical(scale, logical);

        let view_pos = round(self.view_pos.current());

        self.thumbnails().filter_map(move |(thumbnail, mut geo)| {
            geo.loc.x -= view_pos;
            geo.loc.x += round(thumbnail.render_offset());

            if geo.loc.x + geo.size.w < 0. || output_size.w < geo.loc.x {
                return None;
            }

            Some((thumbnail, geo))
        })
    }

    fn render<R: NiriRenderer>(
        &self,
        niri: &Niri,
        renderer: &mut R,
        alpha: f32,
        target: RenderTarget,
    ) -> impl Iterator<Item = WindowMruUiRenderElement<R>> {
        let _span = tracy_client::span!("mru::Inner::render");

        let mut rv = Vec::new();

        let output_size = output_size(&self.output);
        let scale = self.output.current_scale().fractional_scale();

        if let Some(texture) =
            self.scope_panel
                .borrow_mut()
                .get(renderer.as_gles_renderer(), scale, self.wmru.scope)
        {
            let padding = round_logical_in_physical(scale, f64::from(PANEL_PADDING));

            let size = texture.logical_size();
            let location = Point::new(
                (output_size.w - size.w) / 2.,
                output_size.h - size.h - padding * 2.,
            );
            let elem = PrimaryGpuTextureRenderElement(TextureRenderElement::from_texture_buffer(
                texture.clone(),
                location,
                alpha,
                None,
                None,
                Kind::Unspecified,
            ));
            rv.push(elem.into());
        }

        // As with tiles, render thumbnails for closing windows on top of
        // others.
        // for closing in self.closing_thumbnails.iter().rev() {
        //     let elem = closing.render();
        //     rv.push(elem.into());
        // }

        let Some(current_id) = self.wmru.current_id else {
            return rv.into_iter();
        };

        let bob_y = baba_is_float_offset(self.clock.now(), output_size.h);
        let bob_y = round_logical_in_physical(scale, bob_y);

        let config = self.config.borrow();
        let config = &config.recent_windows;

        for (thumbnail, geo) in self.thumbnails_in_view_render() {
            let id = thumbnail.id;
            let Some((_, mapped)) = niri.layout.windows().find(|(_, m)| m.id() == id) else {
                error!("window in the MRU must be present in the layout");
                continue;
            };

            let is_active = thumbnail.id == current_id;
            let elems = thumbnail.render(
                renderer, config, mapped, geo, scale, is_active, bob_y, alpha, target,
            );
            rv.extend(elems);
        }

        rv.into_iter()
    }

    fn thumbnail_under(&self, pos: Point<f64, Logical>) -> Option<MappedId> {
        let scale = self.output.current_scale().fractional_scale();
        let round = move |logical: f64| round_logical_in_physical(scale, logical);
        let padding = self.config.borrow().recent_windows.highlight.padding;
        let padding = round(padding) + round(BORDER);
        let padding = Point::new(padding, padding);

        for (thumbnail, mut geo) in self.thumbnails_in_view_static() {
            // TODO title text height.
            geo.loc -= padding;
            geo.size += padding.to_size().upscale(2.);

            if geo.contains(pos) {
                return Some(thumbnail.id);
            }
        }

        None
    }
}

/// Cached title texture.
#[derive(Debug, Default)]
struct TitleTexture {
    title: String,
    scale: f64,
    texture: Option<Option<MruTexture>>,
}

impl TitleTexture {
    fn get(&mut self, renderer: &mut GlesRenderer, title: &str, scale: f64) -> Option<MruTexture> {
        if self.title != title || self.scale != scale {
            self.texture = None;
            self.title = title.to_owned();
            self.scale = scale;
        }

        self.texture
            .get_or_insert_with(|| generate_title_texture(renderer, title, scale).ok())
            .clone()
    }
}

fn generate_title_texture(
    renderer: &mut GlesRenderer,
    title: &str,
    scale: f64,
) -> anyhow::Result<MruTexture> {
    let _span = tracy_client::span!("mru::generate_title_texture");

    let mut font = FontDescription::from_string(FONT);
    font.set_absolute_size(to_physical_precise_round(scale, font.size()));

    let surface = ImageSurface::create(cairo::Format::ARgb32, 0, 0)?;
    let cr = cairo::Context::new(&surface)?;
    let layout = pangocairo::functions::create_layout(&cr);
    layout.context().set_round_glyph_positions(false);
    layout.set_font_description(Some(&font));
    layout.set_text(title);

    let (width, height) = layout.pixel_size();
    ensure!(width > 0 && height > 0);

    let surface = ImageSurface::create(cairo::Format::ARgb32, width, height)?;
    let cr = cairo::Context::new(&surface)?;
    let layout = pangocairo::functions::create_layout(&cr);
    layout.context().set_round_glyph_positions(false);
    layout.set_font_description(Some(&font));
    layout.set_text(title);

    cr.set_source_rgb(1., 1., 1.);
    pangocairo::functions::show_layout(&cr, &layout);

    drop(cr);
    let data = surface.take_data().unwrap();
    let buffer = TextureBuffer::from_memory(
        renderer,
        &data,
        Fourcc::Argb8888,
        (width, height),
        false,
        scale,
        Transform::Normal,
        Vec::new(),
    )?;

    Ok(buffer)
}

/// Cached scope panel textures.
#[derive(Debug, Default)]
struct ScopePanel {
    scale: f64,
    textures: Option<Option<[MruTexture; 3]>>,
}

impl ScopePanel {
    fn get(
        &mut self,
        renderer: &mut GlesRenderer,
        scale: f64,
        scope: MruScope,
    ) -> Option<MruTexture> {
        if self.scale != scale {
            self.textures = None;
            self.scale = scale;
        }

        self.textures
            .get_or_insert_with(|| generate_scope_panels(renderer, scale).ok())
            .as_ref()
            .map(|x| x[scope as usize].clone())
    }
}

fn generate_scope_panels(
    renderer: &mut GlesRenderer,
    scale: f64,
) -> anyhow::Result<[MruTexture; 3]> {
    fn make_panel_text(idx: usize) -> String {
        let span_unselected = "<span fgcolor='#555555'>";
        let span_shortcut = "<span face='mono' bgcolor='#2C2C2C' letter_spacing='5000'>";
        let span_end = "</span>";

        // Starts with a zero-width space to make letter_spacing work on the left.
        let mut buf = format!("\u{200B}{span_unselected}{span_shortcut}S{span_end}cope:{span_end}");

        for scope in SCOPE_CYCLE {
            buf.push_str("  ");
            if scope as usize != idx {
                buf.push_str(span_unselected);
            }
            let text = match scope {
                MruScope::All => format!("{span_shortcut}A{span_end}ll"),
                MruScope::Output => format!("{span_shortcut}O{span_end}utput"),
                MruScope::Workspace => format!("{span_shortcut}W{span_end}orkspace"),
            };
            buf.push_str(&text);
            if scope as usize != idx {
                buf.push_str(span_end);
            }
        }

        buf
    }

    // Can't wait for array::try_map()
    Ok([
        render_panel(renderer, scale, &make_panel_text(0))?,
        render_panel(renderer, scale, &make_panel_text(1))?,
        render_panel(renderer, scale, &make_panel_text(2))?,
    ])
}

fn render_panel(renderer: &mut GlesRenderer, scale: f64, text: &str) -> anyhow::Result<MruTexture> {
    let _span = tracy_client::span!("mru::render_panel");

    let mut font = FontDescription::from_string(FONT);
    font.set_absolute_size(to_physical_precise_round(scale, font.size()));

    let padding: i32 = to_physical_precise_round(scale, PANEL_PADDING);
    // Keep the border width even to avoid blurry edges.
    // Render to a dummy surface to determine the size.
    let surface = ImageSurface::create(cairo::Format::ARgb32, 0, 0)?;
    let cr = cairo::Context::new(&surface)?;
    let layout = pangocairo::functions::create_layout(&cr);
    layout.context().set_round_glyph_positions(false);
    layout.set_font_description(Some(&font));
    layout.set_markup(text);
    let (mut width, mut height) = layout.pixel_size();

    width += padding * 2;
    height += padding * 2;

    let surface = ImageSurface::create(cairo::Format::ARgb32, width, height)?;
    let cr = cairo::Context::new(&surface)?;
    cr.set_source_rgb(0.1, 0.1, 0.1);
    cr.paint()?;

    let padding = f64::from(padding);

    cr.move_to(padding, padding);

    let layout = pangocairo::functions::create_layout(&cr);
    layout.context().set_round_glyph_positions(false);
    layout.set_font_description(Some(&font));
    layout.set_markup(text);

    cr.set_source_rgb(1., 1., 1.);
    pangocairo::functions::show_layout(&cr, &layout);

    cr.move_to(0., 0.);
    cr.line_to(width.into(), 0.);
    cr.line_to(width.into(), height.into());
    cr.line_to(0., height.into());
    cr.line_to(0., 0.);
    cr.set_source_rgb(0.3, 0.3, 0.3);
    cr.set_line_width((f64::from(PANEL_BORDER) / 2. * scale).round() * 2.);
    cr.stroke()?;

    drop(cr);
    let data = surface.take_data().unwrap();
    let buffer = TextureBuffer::from_memory(
        renderer,
        &data,
        Fourcc::Argb8888,
        (width, height),
        false,
        scale,
        Transform::Normal,
        Vec::new(),
    )?;

    Ok(buffer)
}

/// Returns key bindings available when the MRU UI is open.
fn make_preset_opened_binds() -> Vec<Bind> {
    let mut rv = Vec::new();

    let mut push = |trigger, action| {
        rv.push(Bind {
            key: Key {
                trigger: Trigger::Keysym(trigger),
                // The modifier is filled dynamically.
                modifiers: Modifiers::empty(),
            },
            action,
            repeat: true,
            cooldown: None,
            allow_when_locked: false,
            allow_inhibiting: false,
            hotkey_overlay_title: None,
        })
    };

    push(Keysym::Escape, Action::MruCancel);
    push(Keysym::Return, Action::MruConfirm);
    push(Keysym::a, Action::MruSetScope(MruScope::All));
    push(Keysym::o, Action::MruSetScope(MruScope::Output));
    push(Keysym::w, Action::MruSetScope(MruScope::Workspace));
    push(Keysym::s, Action::MruCycleScope);

    // Leave these in since they are the most expected and generally uncontroversial keys, so that
    // they work even if these actions are absent from the normal binds.
    push(Keysym::Home, Action::MruFirst);
    push(Keysym::End, Action::MruLast);
    push(
        Keysym::Left,
        Action::MruAdvance {
            direction: MruDirection::Backward,
            scope: None,
            filter: None,
        },
    );
    push(
        Keysym::Right,
        Action::MruAdvance {
            direction: MruDirection::Forward,
            scope: None,
            filter: None,
        },
    );

    rv
}

/// Returns dynamic key bindings available when the MRU UI is open.
///
/// These ones are generated based on the normal bindings.
fn make_dynamic_opened_binds(config: &Config) -> Vec<Bind> {
    let mut binds: HashMap<Trigger, Vec<Bind>> = HashMap::new();

    for bind in &config.binds.0 {
        let action = match &bind.action {
            Action::FocusColumnRight
            | Action::FocusColumnRightOrFirst
            | Action::FocusColumnOrMonitorRight
            | Action::FocusWindowDownOrColumnRight => Action::MruAdvance {
                direction: MruDirection::Forward,
                scope: None,
                filter: None,
            },
            Action::FocusColumnLeft
            | Action::FocusColumnLeftOrLast
            | Action::FocusColumnOrMonitorLeft
            | Action::FocusWindowUpOrColumnLeft => Action::MruAdvance {
                direction: MruDirection::Backward,
                scope: None,
                filter: None,
            },
            Action::FocusColumnFirst => Action::MruFirst,
            Action::FocusColumnLast => Action::MruLast,
            Action::CloseWindow => Action::MruCloseCurrentWindow,
            x @ Action::Screenshot(_, _) => x.clone(),
            _ => continue,
        };

        binds.entry(bind.key.trigger).or_default().push(Bind {
            action,
            ..bind.clone()
        });
    }

    let mut rv = Vec::new();

    // For each trigger, take the bind with the lowest number of modifiers.
    for binds in binds.into_values() {
        let bind = binds
            .into_iter()
            .min_by_key(|bind| bind.key.modifiers.iter().count())
            .unwrap();

        rv.push(Bind {
            key: Key {
                trigger: bind.key.trigger,
                // The modifier is filled dynamically.
                modifiers: Modifiers::empty(),
            },
            ..bind
        });
    }

    rv
}
