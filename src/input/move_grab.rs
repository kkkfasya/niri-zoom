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

    fn on_ungrab(&mut self, state: &mut State) {
        if self.gesture == GestureState::Move {
            state.niri.layout.interactive_move_end(&self.window);
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

        if self.window.alive() {
            if let Some((output, pos_within_output)) = data.niri.output_under(event.location) {
                let output = output.clone();
                let mut event_delta = event.location - self.last_location;
                self.last_location = event.location;

                if self.gesture == GestureState::Recognizing {
                    let c = event.location - self.start_data.location;

                    // Check if the gesture moved far enough to decide.
                    if c.x * c.x + c.y * c.y >= 8. * 8. {
                        if !data.niri.layout.interactive_move_begin(
                            self.window.clone(),
                            &self.start_output,
                            self.start_pos_within_output,
                        ) {
                            // Can no longer start the move.
                            handle.unset_grab(self, data, event.serial, event.time, true);
                            return;
                        }

                        // Apply the whole delta that accumulated during recognizing.
                        event_delta = c;

                        self.gesture = GestureState::Move;

                        data.niri
                            .cursor_manager
                            .set_cursor_image(CursorImageStatus::Named(CursorIcon::Move));
                    }
                }

                if self.gesture != GestureState::Move {
                    return;
                }

                let ongoing = data.niri.layout.interactive_move_update(
                    &self.window,
                    event_delta,
                    output,
                    pos_within_output,
                );
                if ongoing {
                    // FIXME: only redraw the previous and the new output.
                    data.niri.queue_redraw_all();
                    return;
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

        // When moving with the left button, right toggles floating, and vice versa.
        let toggle_floating_button = if self.start_data.button == 0x110 {
            0x111
        } else {
            0x110
        };
        if event.button == toggle_floating_button && event.state == ButtonState::Pressed {
            data.niri.layout.toggle_window_floating(Some(&self.window));
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
