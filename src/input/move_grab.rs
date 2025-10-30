use std::time::Duration;

use smithay::backend::input::ButtonState;
use smithay::desktop::Window;
use smithay::input::pointer::{
    AxisFrame, ButtonEvent, CursorIcon, CursorImageStatus, GestureHoldBeginEvent,
    GestureHoldEndEvent, GesturePinchBeginEvent, GesturePinchEndEvent, GesturePinchUpdateEvent,
    GestureSwipeBeginEvent, GestureSwipeEndEvent, GestureSwipeUpdateEvent,
    GrabStartData as PointerGrabStartData, MotionEvent, PointerGrab, PointerInnerHandle,
    RelativeMotionEvent,
};
use smithay::input::SeatHandler;
use smithay::output::Output;
use smithay::utils::{IsAlive, Logical, Point};

use crate::niri::State;

pub struct MoveGrab {
    start_data: PointerGrabStartData<State>,
    start_output: Output,
    start_pos_within_output: Point<f64, Logical>,
    last_location: Point<f64, Logical>,
    window: Window,
    gesture: GestureState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GestureState {
    Recognizing,
    Move,
    ViewOffset,
}

impl MoveGrab {
    pub fn new(
        state: &mut State,
        start_data: PointerGrabStartData<State>,
        window: Window,
    ) -> Option<Self> {
        let (output, pos_within_output) = state.niri.output_under(start_data.location)?;

        Some(Self {
            last_location: start_data.location,
            start_data,
            start_output: output.clone(),
            start_pos_within_output: pos_within_output,
            window,
            gesture: GestureState::Recognizing,
        })
    }

    pub fn is_move(&self) -> bool {
        self.gesture == GestureState::Move
    }

    fn on_ungrab(&mut self, state: &mut State) {
        let layout = &mut state.niri.layout;
        match self.gesture {
            GestureState::Recognizing => {
                // TODO
                if layout.interactive_move_begin(
                    self.window.clone(),
                    &self.start_output,
                    self.start_pos_within_output,
                ) {
                    layout.interactive_move_end(&self.window)
                }
            }
            GestureState::Move => layout.interactive_move_end(&self.window),
            GestureState::ViewOffset => {
                layout.view_offset_gesture_end(Some(false));
            }
        }

        // FIXME: only redraw the window output.
        state.niri.queue_redraw_all();
        state
            .niri
            .cursor_manager
            .set_cursor_image(CursorImageStatus::default_named());
    }
}

impl PointerGrab<State> for MoveGrab {
    fn motion(
        &mut self,
        data: &mut State,
        handle: &mut PointerInnerHandle<'_, State>,
        _focus: Option<(<State as SeatHandler>::PointerFocus, Point<f64, Logical>)>,
        event: &MotionEvent,
    ) {
        // While the grab is active, no client has pointer focus.
        handle.motion(data, None, event);

        let timestamp = Duration::from_millis(u64::from(event.time));

        // TODO: not needed for view offset
        if self.window.alive() {
            // TODO: not needed for view offset
            if let Some((output, pos_within_output)) = data.niri.output_under(event.location) {
                let output = output.clone();
                let mut delta = event.location - self.last_location;
                self.last_location = event.location;

                let layout = &mut data.niri.layout;

                if self.gesture == GestureState::Recognizing {
                    let c = event.location - self.start_data.location;

                    // Check if the gesture moved far enough to decide.
                    if c.x * c.x + c.y * c.y >= 8. * 8. {
                        let is_floating = layout
                            .workspaces()
                            .find_map(|(_, _, ws)| {
                                ws.windows()
                                    .any(|w| w.window == self.window)
                                    .then(|| ws.is_floating(&self.window))
                            })
                            .unwrap_or(false);

                        if !is_floating && c.x.abs() > c.y.abs() {
                            let Some((output, ws_idx)) =
                                layout.workspaces().find_map(|(mon, ws_idx, ws)| {
                                    let ws_idx = ws
                                        .windows()
                                        .any(|w| w.window == self.window)
                                        .then_some(ws_idx)?;
                                    let output = mon?.output().clone();
                                    Some((output, ws_idx))
                                })
                            else {
                                // Can no longer start the gesture.
                                handle.unset_grab(self, data, event.serial, event.time, true);
                                return;
                            };

                            layout.view_offset_gesture_begin(&output, Some(ws_idx), false);

                            // Apply the whole delta that accumulated during recognizing.
                            delta = c;

                            self.gesture = GestureState::ViewOffset;

                            data.niri
                                .cursor_manager
                                .set_cursor_image(CursorImageStatus::Named(CursorIcon::AllScroll));
                        } else {
                            if !layout.interactive_move_begin(
                                self.window.clone(),
                                &self.start_output,
                                self.start_pos_within_output,
                            ) {
                                // Can no longer start the move.
                                handle.unset_grab(self, data, event.serial, event.time, true);
                                return;
                            }

                            // Apply the whole delta that accumulated during recognizing.
                            delta = c;

                            self.gesture = GestureState::Move;

                            data.niri
                                .cursor_manager
                                .set_cursor_image(CursorImageStatus::Named(CursorIcon::Move));
                        }
                    }
                }

                match self.gesture {
                    GestureState::Recognizing => return,
                    GestureState::Move => {
                        let ongoing = layout.interactive_move_update(
                            &self.window,
                            delta,
                            output,
                            pos_within_output,
                        );
                        if ongoing {
                            // FIXME: only redraw the previous and the new output.
                            data.niri.queue_redraw_all();
                            return;
                        }
                    }
                    GestureState::ViewOffset => {
                        let res = layout.view_offset_gesture_update(-delta.x, timestamp, false);
                        if let Some(output) = res {
                            if let Some(output) = output {
                                data.niri.queue_redraw(&output);
                            }
                            return;
                        }
                    }
                }
            } else {
                return;
            }
        }

        // The move is no longer ongoing.
        handle.unset_grab(self, data, event.serial, event.time, true);
    }

    fn relative_motion(
        &mut self,
        data: &mut State,
        handle: &mut PointerInnerHandle<'_, State>,
        _focus: Option<(<State as SeatHandler>::PointerFocus, Point<f64, Logical>)>,
        event: &RelativeMotionEvent,
    ) {
        // While the grab is active, no client has pointer focus.
        handle.relative_motion(data, None, event);
    }

    fn button(
        &mut self,
        data: &mut State,
        handle: &mut PointerInnerHandle<'_, State>,
        event: &ButtonEvent,
    ) {
        handle.button(data, event);

        // TODO: start move if recognizing?
        if self.gesture == GestureState::Move {
            // When moving with the left button, right toggles floating, and vice versa.
            let toggle_floating_button = if self.start_data.button == 0x110 {
                0x111
            } else {
                0x110
            };
            if event.button == toggle_floating_button && event.state == ButtonState::Pressed {
                data.niri.layout.toggle_window_floating(Some(&self.window));
            }
        }

        if !handle.current_pressed().contains(&self.start_data.button) {
            // The button that initiated the grab was released.
            handle.unset_grab(self, data, event.serial, event.time, true);
        }
    }

    fn axis(
        &mut self,
        data: &mut State,
        handle: &mut PointerInnerHandle<'_, State>,
        details: AxisFrame,
    ) {
        handle.axis(data, details);
    }

    fn frame(&mut self, data: &mut State, handle: &mut PointerInnerHandle<'_, State>) {
        handle.frame(data);
    }

    fn gesture_swipe_begin(
        &mut self,
        data: &mut State,
        handle: &mut PointerInnerHandle<'_, State>,
        event: &GestureSwipeBeginEvent,
    ) {
        handle.gesture_swipe_begin(data, event);
    }

    fn gesture_swipe_update(
        &mut self,
        data: &mut State,
        handle: &mut PointerInnerHandle<'_, State>,
        event: &GestureSwipeUpdateEvent,
    ) {
        handle.gesture_swipe_update(data, event);
    }

    fn gesture_swipe_end(
        &mut self,
        data: &mut State,
        handle: &mut PointerInnerHandle<'_, State>,
        event: &GestureSwipeEndEvent,
    ) {
        handle.gesture_swipe_end(data, event);
    }

    fn gesture_pinch_begin(
        &mut self,
        data: &mut State,
        handle: &mut PointerInnerHandle<'_, State>,
        event: &GesturePinchBeginEvent,
    ) {
        handle.gesture_pinch_begin(data, event);
    }

    fn gesture_pinch_update(
        &mut self,
        data: &mut State,
        handle: &mut PointerInnerHandle<'_, State>,
        event: &GesturePinchUpdateEvent,
    ) {
        handle.gesture_pinch_update(data, event);
    }

    fn gesture_pinch_end(
        &mut self,
        data: &mut State,
        handle: &mut PointerInnerHandle<'_, State>,
        event: &GesturePinchEndEvent,
    ) {
        handle.gesture_pinch_end(data, event);
    }

    fn gesture_hold_begin(
        &mut self,
        data: &mut State,
        handle: &mut PointerInnerHandle<'_, State>,
        event: &GestureHoldBeginEvent,
    ) {
        handle.gesture_hold_begin(data, event);
    }

    fn gesture_hold_end(
        &mut self,
        data: &mut State,
        handle: &mut PointerInnerHandle<'_, State>,
        event: &GestureHoldEndEvent,
    ) {
        handle.gesture_hold_end(data, event);
    }

    fn start_data(&self) -> &PointerGrabStartData<State> {
        &self.start_data
    }

    fn unset(&mut self, data: &mut State) {
        self.on_ungrab(data);
    }
}
